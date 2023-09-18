import logging
import math

import numpy as np
from PIL import Image
from torchvision import datasets
from torchvision import transforms
#引入数据增强的所有方法
from .randaugment import RandAugmentMC

logger = logging.getLogger(__name__)

#用于对数据集进行归一化
cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)

#以cifar10数据集的处理为例进行详解
#对cifar10数据集进行处理，从而可以得到三个新的数据集
#有标签训练集(4000),无标签训练集，测试集(10000)
def get_cifar10(args, root):
    # 定义数据增强和转换函数，用于有标签数据集
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),  # 随机裁剪
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)  # 标准化
    ])
    
    # 定义验证集的转换函数
    transform_val = transforms.Compose([
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)  # 标准化
    ])
    
    # 下载或加载 CIFAR-10 数据集，也就是最原始的数据集(50000)，之后将其进行转换切分
    base_dataset = datasets.CIFAR10(root, train=True, download=True)

    # 将数据集拆分为有标签和无标签数据索引，这里只是得到了索引
    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, base_dataset.targets)

    '''一个函数就可以通过给定的索引将已有的数据集进行拆分'''
    '''拆分之后的数据集都有标签，只是无标签数据的标签忽略了'''
    # 创建有标签数据集，应用定义的转换函数
    train_labeled_dataset = CIFAR10SSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled)#这里的有标签数据的转换函数是之前定义好的，比较简单

    # 创建无标签数据集，应用自定义的转换函数（TransformFixMatch）
    #这里应该是包含两个样本和一个标签，两个样本是一个原始样本经过强弱增强之后形成的
    train_unlabeled_dataset = CIFAR10SSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch(mean=cifar10_mean, std=cifar10_std))

    # 创建测试数据集，应用验证集的转换函数
    #这里的测试数据集的转换函数是之前定义的
    test_dataset = datasets.CIFAR10(
        root, train=False, transform=transform_val, download=False)

    # 返回有标签数据集、无标签数据集和测试数据集
    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


#原理与cifar10类似，将原始数据集分割成三部分
#根据计算得到的索引，应用CIFAR100SSL函数分割
def get_cifar100(args, root):

    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    base_dataset = datasets.CIFAR100(
        root, train=True, download=True)

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, base_dataset.targets)

    train_labeled_dataset = CIFAR100SSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled)

    train_unlabeled_dataset = CIFAR100SSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch(mean=cifar100_mean, std=cifar100_std))

    test_dataset = datasets.CIFAR100(
        root, train=False, transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


#相当于给定所有的标签，从每一类中选择部分样本组成有标签样本，剩下的全是无标签样本
def x_u_split(args, labels):
    # 计算每个类别的有标签样本数量，相当于是一个比例
    label_per_class = args.num_labeled // args.num_classes
    
    # 将标签列表转换为 NumPy 数组
    labels = np.array(labels)
    
    labeled_idx = []  # 用于存储有标签样本的索引
    # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
    # 所有样本都被视为无标签样本的索引，然后从无标签样本中选择部分样本作为有标签数据
    unlabeled_idx = np.array(range(len(labels)))  
    
    # 针对每个类别，随机选择一定数量的样本作为有标签样本
    for i in range(args.num_classes):
        idx = np.where(labels == i)[0]  # 找到属于当前类别的样本的索引
        idx = np.random.choice(idx, label_per_class, False)  # 随机选择样本，不允许重复
        labeled_idx.extend(idx)  # 将选择的有标签样本索引添加到列表中
    labeled_idx = np.array(labeled_idx)  # 将列表转换为 NumPy 数组
    #意思是只取固定数量的有标签样本
    assert len(labeled_idx) == args.num_labeled  # 确保有标签样本数量正确

    # 如果需要扩展有标签样本或者有标签样本数量小于批量大小，进行扩展
    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labeled)  # 计算扩展的倍数
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])  # 复制有标签样本索引
    np.random.shuffle(labeled_idx)  # 随机打乱有标签样本索引顺序
    #根据命令行参数中规定的有标签样本的数量从而将数据集分割成两部分
    return labeled_idx, unlabeled_idx  # 返回有标签样本索引和无标签样本索引


#无标签数据增强转换的函数
class TransformFixMatch(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),#这里的增强方式是自定义的
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32*0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        #一次返回两种数据，同一个数据会经过强增强和弱增强
        return self.normalize(weak), self.normalize(strong)

#构造新数据集的函数
class CIFAR10SSL(datasets.CIFAR10):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        #先对原始数据集进行初步的转换
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        #按照给定的索引取出样本和标签
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        #如果有自定义的转换函数，需要进一步转换
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        #返回分割之后的数据和标签
        return img, target

#原理与CIFAR10SSL一致
#从原始数据集中切分出想要的数据集并进行转换
class CIFAR100SSL(datasets.CIFAR100):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

#这个字典的键是一个字符串，值是通过函数形成的
DATASET_GETTERS = {'cifar10': get_cifar10,
                   'cifar100': get_cifar100}
