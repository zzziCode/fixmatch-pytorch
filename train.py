import argparse
import logging
import math
import os
import random
import shutil
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset.cifar import DATASET_GETTERS
from utils import AverageMeter, accuracy
from models.wideresnet import WideResNet

logger = logging.getLogger(__name__)
best_acc = 0

# 保存训练过程中的检查点


def save_checkpoint(state, is_best, checkpoint, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    # 是当前最好的效果，就单独保存一份
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint,
                                               'model_best.pth.tar'))

# 设置随机数种子


# 将所有设置随机数种子的地方都给定一个自定义的种子，这个自定义的种子来自命令行传递的参数
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

# 对学习率进行预热，可以对学习率进行余弦退火
# 开始先对学习率进行预热逐渐递增，之后对学习率进行余弦退火逐渐递减


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,  # 根据默认的参数，在第一个epoch训练时对学习率进行预热
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        # 学习率预热阶段逐渐增加学习率的情况
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # 后面的阶段逐渐使用余弦退火减小学习率
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)

# 对数据按照维度进行重排
def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

#对重排之后的数据恢复成原始状况
def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

# 主函数，用来构建数据集加载器，控制训练


def main():
    # 定义一个参数解析器，用来保存解析命令行传递进来的所有参数，部分参数有默认值，部分参数需要重新传递
    parser = argparse.ArgumentParser(description='PyTorch FixMatch Training')
    # 默认使用第二个显卡训练
    parser.add_argument('--gpu-id', default='1', type=int,
                        help='id(s) for CUDA_VISIBLE_DEVICES')
    # 默认使用四个线程读取数据集
    parser.add_argument('--num-workers', type=int, default=4,
                        help='number of workers')
    # 默认使用cifar10训练
    parser.add_argument('--dataset', default='cifar10', type=str,
                        choices=['cifar10', 'cifar100'],
                        help='dataset name')
    # 默认标记数据由4000个，总数据共有50000个训练集
    parser.add_argument('--num-labeled', type=int, default=4000,
                        help='number of labeled data')
    # 效果未知
    parser.add_argument("--expand-labels", action="store_true",
                        help="expand labels to fit eval steps")
    # 类似于模型使用的backbone
    parser.add_argument('--arch', default='wideresnet', type=str,
                        choices=['wideresnet', 'resnext'],
                        help='dataset name')
    # 训练多少步？？？
    parser.add_argument('--total-steps', default=2**15, type=int,
                        help='number of total steps to run')
    # 验证多少步
    parser.add_argument('--eval-step', default=1024, type=int,
                        help='number of eval steps to run')
    # 开始的训练周期
    parser.add_argument('--start-epoch', default=0, type=int,
                        help='manual epoch number (useful on restarts)')
    # 每次取数据集中多少数据参与训练
    parser.add_argument('--batch-size', default=64, type=int,
                        help='train batchsize')
    # 学习率
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                        help='initial learning rate')
    # 在哪个阶段进行学习率的预热
    parser.add_argument('--warmup', default=0, type=float,
                        help='warmup epochs (unlabeled data based)')
    # 权重衰减的比例
    parser.add_argument('--wdecay', default=5e-4, type=float,
                        help='weight decay')
    # 是否使用改进之后的nesterov动量算法来进行梯度下降
    parser.add_argument('--nesterov', action='store_true', default=True,
                        help='use nesterov momentum')
    # 是否对当前模型进行EMA指数移动平均从而得到新的模型
    parser.add_argument('--use-ema', action='store_true', default=True,
                        help='use EMA model')
    # 指数移动平均的比例
    parser.add_argument('--ema-decay', default=0.999, type=float,
                        help='EMA decay rate')
    #
    parser.add_argument('--mu', default=7, type=int,
                        help='coefficient of unlabeled batch size')
    parser.add_argument('--lambda-u', default=1, type=float,
                        help='coefficient of unlabeled loss')
    # 温度系数
    parser.add_argument('--T', default=1, type=float,
                        help='pseudo label temperature')
    # 阈值，超过多少才能成为伪标签
    parser.add_argument('--threshold', default=0.95, type=float,
                        help='pseudo label threshold')
    # 结果的保存路径
    parser.add_argument('--out', default='result',
                        help='directory to output the result')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    # 随机种子
    parser.add_argument('--seed', default=None, type=int,
                        help="random seed")
    # 混合精度训练
    parser.add_argument("--amp", action="store_true",
                        help="use 16-bit (mixed) precision through NVIDIA apex AMP")
    # 混合精度训练的登记
    parser.add_argument("--opt_level", type=str, default="O1",
                        help="apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                        "See details at https://nvidia.github.io/apex/amp.html")
    # 是否是分布式训练
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    # 是否显示进度条
    parser.add_argument('--no-progress', action='store_true',
                        help="don't use progress bar")
    # 所有的参数保存到args中
    args = parser.parse_args()
    global best_acc

    # 在这里设置一共有多少个类，这里假设有标签数据和无标签数据之间的分布一致
    # 对于一个数据集来说，可以由两种网络来处理，所以一共有2*2中选择
    # 选择不同的数据集和模型，对应的深度宽度以及基数都不一样
    #分析模型的深度和宽度为什么这么设置
    if args.dataset == 'cifar10':
        args.num_classes = 10
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 2
        elif args.arch == 'resnext':
            args.model_cardinality = 4
            args.model_depth = 28
            args.model_width = 4

    elif args.dataset == 'cifar100':
        args.num_classes = 100
        if args.arch == 'wideresnet':
            args.model_depth = 28
            args.model_width = 8
        elif args.arch == 'resnext':
            args.model_cardinality = 8
            args.model_depth = 29
            args.model_width = 64
    # 根据不同的参数创建不同的模型，默认创建wideresnet模型
    def create_model(args):
        #以wideresnet为例，查看如何利用三种数据进行训练
        if args.arch == 'wideresnet':
            import models.wideresnet as models
            # 使用这个函数构建一个wideresnet函数
            model = models.build_wideresnet(depth=args.model_depth,
                                            widen_factor=args.model_width,
                                            dropout=0,
                                            num_classes=args.num_classes)
            #直接使用类名=创建模型也可以
            # model = WideResNet(depth=args.model_depth,
            #                                 widen_factor=args.model_width,
            #                                 dropout=0,
            #                                 num_classes=args.num_classes)
        elif args.arch == 'resnext':
            import models.resnext as models
            model = models.build_resnext(cardinality=args.model_cardinality,
                                         depth=args.model_depth,
                                         width=args.model_width,
                                         num_classes=args.num_classes)
        # 显示模型的总参数量
        logger.info("Total params: {:.2f}M".format(
            sum(p.numel() for p in model.parameters())/1e6))
        return model

    # 配置多卡训练
    # 不在分布式环境中
    if args.local_rank == -1:
        # 选择指定的显卡训练
        device = torch.device('cuda', args.gpu_id)
        # 记录一个训练进程，gpu的可用数量
        args.world_size = 1
        args.n_gpu = torch.cuda.device_count()
    # 在分布式环境中
    else:
        # 根据指定的范围选择显卡设备
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        # 多卡训练之间进行通信
        torch.distributed.init_process_group(backend='nccl')
        # 分布式训练的总进程数
        args.world_size = torch.distributed.get_world_size()
        # 每个进程的可用gpu数量
        args.n_gpu = 1
    # 至少有一个设备用于训练
    args.device = device

    # 配置训练时记录日志的格式
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    # 输出一个警告信息，提示现在是不是在分布式训练，在哪个设备上训练，训练的显卡数是多少
    logger.warning(
        f"Process rank: {args.local_rank}, "
        f"device: {args.device}, "
        f"n_gpu: {args.n_gpu}, "
        f"distributed training: {bool(args.local_rank != -1)}, "
        f"16-bits training: {args.amp}",)
    # 记录模型训练时从命令行传递进来的参数都有哪些，用于辨别模型的训练配置
    logger.info(dict(args._get_kwargs()))

    # 固定模型中所有有随机性的地方，从而使得模型结果可以复现
    if args.seed is not None:
        set_seed(args)

    # 记录当前的结果，并且显示到tensorboard中
    if args.local_rank in [-1, 0]:
        os.makedirs(args.out, exist_ok=True)
        args.writer = SummaryWriter(args.out)
    # 等到进行到达一个同步点才开始下一步训练
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    # 从字典中拿到数据集，并且经过增强
    #具体是通过键值的映射得到数据集的，底层数据集经过转换和选择，最终形成了三个数据集
    labeled_dataset, unlabeled_dataset, test_dataset = DATASET_GETTERS[args.dataset](
        args, './data')

    # 等到进行到达一个同步点才开始下一步训练
    if args.local_rank == 0:
        torch.distributed.barrier()
    # 定义采样方式，判断训练方式，从而可以知道是随机采样还是分布式采样
    train_sampler = RandomSampler if args.local_rank == -1 else DistributedSampler

    labeled_trainloader = DataLoader(
        labeled_dataset,  # 数据集
        sampler=train_sampler(labeled_dataset),  # 对数据集进行采样，决定数据采样的方式
        batch_size=args.batch_size,  # 一次取多少个数据
        num_workers=args.num_workers,  # 一次几个进程取数据
        drop_last=True)  # 最后一点数据如果不足一个批次是摒弃掉还是直接用

    unlabeled_trainloader = DataLoader(
        unlabeled_dataset,
        sampler=train_sampler(unlabeled_dataset),
        batch_size=args.batch_size*args.mu,
        num_workers=args.num_workers,
        drop_last=True)

    test_loader = DataLoader(
        test_dataset,
        sampler=SequentialSampler(test_dataset),
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    # 分布式的话，需要经常进程同步
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    model = create_model(args)

    if args.local_rank == 0:
        torch.distributed.barrier()
    # 将定义的模型放到显卡总进行训练（显卡可用）
    model.to(args.device)

    # 不进行参数衰减的层
    no_decay = ['bias', 'bn']
    grouped_parameters = [
        # 模型的偏置和bn层之外的层进行参数衰减
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': args.wdecay},
        # 模型的偏置和bn层不进行参数衰减
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    # 优化器，nesterov动量算法
    optimizer = optim.SGD(grouped_parameters, lr=args.lr,
                          momentum=0.9, nesterov=args.nesterov)

    args.epochs = math.ceil(args.total_steps / args.eval_step)
    # 学习率的调整（预热和退火）
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, args.warmup, args.total_steps)

    # 根据当前模型创建一个EMA之后的模型
    # 之后有两个模型，一个是原始模型，一个是EMA之后的模型
    if args.use_ema:
        from models.ema import ModelEMA
        ema_model = ModelEMA(args, model, args.ema_decay)

    args.start_epoch = 0

    # 是否要从之前的检查点恢复训练
    if args.resume:
        logger.info("==> Resuming from checkpoint..")
        #恢复训练的检查点文件不存在时，
        assert os.path.isfile(
            args.resume), "Error: no checkpoint directory found!"
        #指定输出目录
        args.out = os.path.dirname(args.resume)
        #指定加载检查点的目录
        checkpoint = torch.load(args.resume)
        #得到最佳精度
        best_acc = checkpoint['best_acc']
        #从哪个epoch开始恢复训练
        args.start_epoch = checkpoint['epoch']
        #加载模型的已有参数
        model.load_state_dict(checkpoint['state_dict'])
        #如果使用了指数移动平均，另外一个模型也需要加载参数
        if args.use_ema:
            ema_model.ema.load_state_dict(checkpoint['ema_state_dict'])
        #优化器和学习率调整器也需要调整参数
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
    #混合精度训练
    if args.amp:
        from apex import amp
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.opt_level)
    #进行分布式的训练
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank],
            output_device=args.local_rank, find_unused_parameters=True)

    # 训练开始之前进行提示，只会提示一次
    logger.info("***** Running training *****")
    logger.info(f"  Task = {args.dataset}@{args.num_labeled}")
    logger.info(f"  Num Epochs = {args.epochs}")
    logger.info(f"  Batch size per GPU = {args.batch_size}")
    logger.info(
        f"  Total train batch size = {args.batch_size*args.world_size}")
    logger.info(f"  Total optimization steps = {args.total_steps}")

    #不管是不是恢复训练，训练前都先将模型的梯度清零
    model.zero_grad()
    train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, optimizer, ema_model, scheduler)
    
    #查看模型结构
    dummy_input = torch.rand(20, 1, 28, 28)
    with SummaryWriter(comment='model') as w:
        w.add_graph(model, (dummy_input,))

# 模型训练的函数逻辑
'''三种样本通道结合在一起进行训练，最后得到最终的结果之后再分开'''
def train(args, labeled_trainloader, unlabeled_trainloader, test_loader,
          model, optimizer, ema_model, scheduler):
    #混合精度训练
    if args.amp:
        from apex import amp
    global best_acc
    test_accs = []
    end = time.time()
    #如果在进行分布式训练，设置分布式训练的进程如何获取数据集中的样本
    if args.world_size > 1:
        labeled_epoch = 0
        unlabeled_epoch = 0
        labeled_trainloader.sampler.set_epoch(labeled_epoch)
        unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)

    #得到数据集的加载器
    labeled_iter = iter(labeled_trainloader)
    unlabeled_iter = iter(unlabeled_trainloader)

    model.train()
    #训练指定的epoch轮
    for epoch in range(args.start_epoch, args.epochs):
        #用来更新不同的指标
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        #这是什么
        mask_probs = AverageMeter()
        #如果训练时不显示进度条，就自定义一个进度条
        if not args.no_progress:
            p_bar = tqdm(range(args.eval_step),
                         disable=args.local_rank not in [-1, 0])
        for batch_idx in range(args.eval_step):
            try:
                #尝试从有标签数据集的加载器中拿到数据和标签，[64,3,32,32]
                inputs_x, targets_x = labeled_iter.next()
                # error occurs ↓
                # inputs_x, targets_x = next(labeled_iter)
            #try中的代码出现异常
            except:
                if args.world_size > 1:
                    labeled_epoch += 1
                    labeled_trainloader.sampler.set_epoch(labeled_epoch)
                #从新的周期开始
                labeled_iter = iter(labeled_trainloader)
                inputs_x, targets_x = labeled_iter.next()
                # error occurs ↓
                # inputs_x, targets_x = next(labeled_iter)

            try:
                #尝试从无标记数据集的加载器中拿到数据，强弱增强的图像放在一起训练
                #从无标签数据集中只取样本，不取数据，[448,3,32,32]
                (inputs_u_w, inputs_u_s), _ = unlabeled_iter.next()
                # error occurs ↓
                # (inputs_u_w, inputs_u_s), _ = next(unlabeled_iter)
            except:
                #解决分布式训练中存在的问题
                if args.world_size > 1:
                    unlabeled_epoch += 1
                    unlabeled_trainloader.sampler.set_epoch(unlabeled_epoch)
                unlabeled_iter = iter(unlabeled_trainloader)
                (inputs_u_w, inputs_u_s), _ = unlabeled_iter.next()
                # error occurs ↓
                # (inputs_u_w, inputs_u_s), _ = next(unlabeled_iter)

            data_time.update(time.time() - end)
            #记录有标签的样本的个数
            batch_size = inputs_x.shape[0]
            #将获取到的一个有标签样本，一个弱增强无标签样本，一个强增强无标签样本进行拼接
            #形成一个完整的输入，通道上进行拼接，[960,3,32,32]
            inputs = interleave(
                torch.cat((inputs_x, inputs_u_w, inputs_u_s)), 2*args.mu+1).to(args.device)
            targets_x = targets_x.to(args.device)
            #得到模型最终的预测
            '''查看模型怎么利用三种数据进行训练，强弱数据之间是怎么联合训练的'''
            '''模型直接将三个输入拼接，看成三个维度的东西，最终得到分类的结果'''
            logits = model(inputs)#[960,10]
            #模型最终的预测应该分为三部分，前64作为有标签的输出，后896分成两个448
            logits = de_interleave(logits, 2*args.mu+1)
            #得到关于三个不同样本的预测，前64作为有标签的输出
            logits_x = logits[:batch_size]#前batch_size个样本有标签
            logits_u_w, logits_u_s = logits[batch_size:].chunk(2)#无标签数据的输出，分为两个448
            del logits

            #第一个损失
            Lx = F.cross_entropy(logits_x, targets_x, reduction='mean')

            #大于阈值的才能成为伪标签，此时选择使用弱增强的输出作来产生伪标签，这一步将其转化成概率
            pseudo_label = torch.softmax(logits_u_w.detach()/args.T, dim=-1)
            #每个样本属于对应伪标签类别的概率，以及伪标签所属类别的序号
            #此时的每个样本都有一个伪标签，但是伪标签不一定靠谱，下一步需要过滤，只有置信度超过阈值的才能与伪标签之间计算损失
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            #定义哪些才能成为伪标签，置信度大于阈值的才能成为伪标签
            #这里只是定义了一个mask，规定哪些损失最终可以留下
            mask = max_probs.ge(args.threshold).float()
            #无标签数据的伪标签损失，计算完损失之后，之后有伪标签的损失才能作数，进行了一步mask
            #这里其实448个样本都计算了损失，但是mask过后，相当于大于阈值的才有伪标签，才能计算损失
            Lu = (F.cross_entropy(logits_u_s, targets_u,
                                  reduction='none') * mask).mean()
            #两个损失进行加权
            loss = Lx + args.lambda_u * Lu

            
            #不管怎样，都会进行梯度更新
            if args.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            #更新计算得到的损失
            losses.update(loss.item())
            losses_x.update(Lx.item())
            losses_u.update(Lu.item())
            #进行参数更新
            optimizer.step()
            scheduler.step()
            #更新ema模型的参数，最终使用EMA计算之后的模型来进行测试
            #这里的目的是为了模型最终的预测结果更加平衡
            if args.use_ema:
                ema_model.update(model)
            model.zero_grad()

            batch_time.update(time.time() - end)
            end = time.time()
            mask_probs.update(mask.mean().item())
            #如果没有定义进度条
            if not args.no_progress:
                p_bar.set_description("Train Epoch: {epoch}/{epochs:4}. Iter: {batch:4}/{iter:4}. LR: {lr:.4f}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. Loss_x: {loss_x:.4f}. Loss_u: {loss_u:.4f}. Mask: {mask:.2f}. ".format(
                    epoch=epoch + 1,
                    epochs=args.epochs,
                    batch=batch_idx + 1,
                    iter=args.eval_step,
                    lr=scheduler.get_last_lr()[0],
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    loss_x=losses_x.avg,
                    loss_u=losses_u.avg,
                    mask=mask_probs.avg))#最后一个参数标识了那些样本才能成为伪标签
                p_bar.update()

        if not args.no_progress:
            p_bar.close()

        #EMA在测试阶段使用
        if args.use_ema:
            test_model = ema_model.ema
        else:
            test_model = model
        #每一个epoch结束都需要测试一次
        if args.local_rank in [-1, 0]:
            #测试时使用的是EMA处理过的模型
            test_loss, test_acc = test(args, test_loader, test_model, epoch)

            #将测试的结果展示到tensorboard中
            args.writer.add_scalar('train/1.train_losses', losses.avg, epoch)
            args.writer.add_scalar('train/2.train_loss_x', losses_x.avg, epoch)
            args.writer.add_scalar('train/3.train_loss_u', losses_u.avg, epoch)
            args.writer.add_scalar('train/4.mask', mask_probs.avg, epoch)
            args.writer.add_scalar('test/1.test_acc', test_acc, epoch)
            args.writer.add_scalar('test/2.test_loss', test_loss, epoch)
            
            print(f"train_losses:{losses.avg}\t train_loss_x{losses_x.avg}\t train_loss_u{losses_u.avg}\ttest_acc{test_acc}\ttest_loss{test_loss}")

            is_best = test_acc > best_acc
            best_acc = max(test_acc, best_acc)

            model_to_save = model.module if hasattr(model, "module") else model
            if args.use_ema:
                ema_to_save = ema_model.ema.module if hasattr(
                    ema_model.ema, "module") else ema_model.ema
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_to_save.state_dict(),
                'ema_state_dict': ema_to_save.state_dict() if args.use_ema else None,
                'acc': test_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, is_best, args.out)

            test_accs.append(test_acc)
            logger.info('Best top-1 acc: {:.2f}'.format(best_acc))
            logger.info('Mean top-1 acc: {:.2f}\n'.format(
                np.mean(test_accs[-20:])))

    if args.local_rank in [-1, 0]:
        args.writer.close()

# 模型测试的函数逻辑
#测试时使用的模型是训练时经过EMA平滑之后的模型
def test(args, test_loader, model, epoch):
    #为了实时更新
    batch_time = AverageMeter()
    data_time = AverageMeter()  
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    #没有进度条自己定义一个
    if not args.no_progress:
        test_loader = tqdm(test_loader,
                           disable=args.local_rank not in [-1, 0])

    #测试时不需要梯度
    with torch.no_grad():
        #从10000个测试集样本中取出样本测试模型的性能
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            data_time.update(time.time() - end)
            model.eval()

            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)

            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            #更新这三个监控指标
            losses.update(loss.item(), inputs.shape[0])
            top1.update(prec1.item(), inputs.shape[0])
            top5.update(prec5.item(), inputs.shape[0])
            batch_time.update(time.time() - end)
            end = time.time()
            #没有进度条时，直接打印这些信息
            if not args.no_progress:
                test_loader.set_description("Test Iter: {batch:4}/{iter:4}. Data: {data:.3f}s. Batch: {bt:.3f}s. Loss: {loss:.4f}. top1: {top1:.2f}. top5: {top5:.2f}. ".format(
                    batch=batch_idx + 1,
                    iter=len(test_loader),
                    data=data_time.avg,
                    bt=batch_time.avg,
                    loss=losses.avg,
                    top1=top1.avg,
                    top5=top5.avg,
                ))
        if not args.no_progress:
            test_loader.close()

    #显示当前精度
    logger.info("top-1 acc: {:.2f}".format(top1.avg))
    logger.info("top-5 acc: {:.2f}".format(top5.avg))
    return losses.avg, top1.avg


# 从main函数进入当前文件，开始训练
if __name__ == '__main__':
    main()
