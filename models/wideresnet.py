import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
'''看模型如何处理三个不同的输入'''
#打印一些日志信息
logger = logging.getLogger(__name__)

  
def mish(x):
    """Mish: A Self Regularized Non-Monotonic Neural Activation Function (https://arxiv.org/abs/1908.08681)"""
    return x * torch.tanh(F.softplus(x))

#相当于一个自定义的batchNorm函数，好像模型中并没有用到
#可以尝试将普通的bn层换成自定义的bn层
class PSBatchNorm2d(nn.BatchNorm2d):
    """How Does BN Increase Collapsed Neural Network Filters? (https://arxiv.org/abs/2001.11216)"""

    def __init__(self, num_features, alpha=0.1, eps=1e-05, momentum=0.001, affine=True, track_running_stats=True):
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.alpha = alpha

    def forward(self, x):
        return super().forward(x) + self.alpha

#模型结构不复杂时，使用Sequential，模型结构复杂时，使用forward
#模型中的第一个小模块
class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0, activate_before_residual=False):
        super(BasicBlock, self).__init__()
        #定义每个层都是什么
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=0.001)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=0.001)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.drop_rate = drop_rate
        #如果二者相等，那么当前值为true，否则为false
        self.equalInOut = (in_planes == out_planes)
        #如果输入输出的维度不匹配，此时无法直接进行残差链接，需要先进行一个卷积实现维度的匹配
        #之后再进行残差
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                                                                padding=0, bias=False) or None
        self.activate_before_residual = activate_before_residual

    #定义数据x的处理方式，如何流动
    def forward(self, x):
        #如果输入输出维度不同，并且activate_before_residual为true
        # 那么x先经过一个卷积加上一个relu
        '''这里判断是否需要先经过一个卷积和relu'''
        if not self.equalInOut and self.activate_before_residual == True:
            x = self.relu1(self.bn1(x))
        #否则直接就得到out
        else:
            out = self.relu1(self.bn1(x))
            #如果输入输出的维度相同，那么此时经过第二个卷积和relu的值就是out，否则是x
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        #判断是否需要dropout
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)
        #从前两者中选择一个与out相加
        #输入输出的维度相同时直接进行残差，输入输出的维度不相同时，需要先经过卷积调整维度
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

#模型中的第二个小模块
#这里的block是由上面的BasicBlock组成的，相当于NetworkBlock中有nb_layers个BasicBlock
class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate=0.0, activate_before_residual=False):
        super(NetworkBlock, self).__init__()
        #根据给定的参数创建一个小模块，之后直接将x输入到这个模块中
        self.layer = self._make_layer(
            block, in_planes, out_planes, nb_layers, stride, drop_rate, activate_before_residual)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, drop_rate, activate_before_residual):
        layers = []
        #一共有多少层，每一层都是一个BasicBlock
        for i in range(int(nb_layers)):
            #i == 0 and in_planes or out_planes决定了第一个参数
            #i == 0 and stride or 1决定了第三个参数
            layers.append(block(i == 0 and in_planes or out_planes, out_planes,
                                i == 0 and stride or 1, drop_rate, activate_before_residual))
        return nn.Sequential(*layers)
    #最终将输入x经过多个BasicBlock之后得到输出
    def forward(self, x):
        return self.layer(x)

#定义最终的wideresnet模型代码
#一个wideresnet由多个NetworkBlock和其他的池化，relu等模块组成
#一个NetworkBlock由多个BasicBlock组成
class WideResNet(nn.Module):
    def __init__(self, num_classes, depth=28, widen_factor=2, drop_rate=0.0):
        super(WideResNet, self).__init__()
        #输出的维度越来越大
        channels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6 #每个WideResNet中堆叠多少个BasicBlock
        #传递这个类名，后期直接往这个类名中传递参数
        block = BasicBlock
        # 1st conv before any network block
        #输入大小是3，输出大小是16
        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(
            n, channels[0], channels[1], block, 1, drop_rate, activate_before_residual=True)
        # 2nd block
        self.block2 = NetworkBlock(
            n, channels[1], channels[2], block, 2, drop_rate)
        # 3rd block
        self.block3 = NetworkBlock(
            n, channels[2], channels[3], block, 2, drop_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(channels[3], momentum=0.001)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.fc = nn.Linear(channels[3], num_classes)
        #最终的输出是channels[3]的维度
        self.channels = channels[3]
        #对模型中的参数进行初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        out = self.conv1(x)#channels[0]
        out = self.block1(out)#channels[1]
        out = self.block2(out)#channels[2]
        out = self.block3(out)#channels[3]
        out = self.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, 1)
        #最终的维度进行展平，不管是多少通道的，最终都变成一维
        out = out.view(-1, self.channels)
        #将展平之后的维度变成最终的类别数，从而可以进行分类
        #相当于从三维最终到了num_classes维
        return self.fc(out)


def build_wideresnet(depth, widen_factor, dropout, num_classes):
    logger.info(f"Model: WideResNet {depth}x{widen_factor}")
    return WideResNet(depth=depth,
                      widen_factor=widen_factor,
                      drop_rate=dropout,
                      num_classes=num_classes)
