from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import argparse
import os
import random
import time
import hubconf  # noqa: F401
import copy
import datetime

from quant import (
    block_reconstruction,
    layer_reconstruction,
    BaseQuantBlock,
    QuantModule,
    QuantModel,
    set_weight_quantize_params,
)
from data.imagenet import build_imagenet_data

from pointnet import model as point_model

import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from pointnet.dataset import ShapeNetDataset, ModelNetDataset
from pointnet.model import PointNetCls, feature_transform_regularizer
import torch.nn.functional as F
from tqdm import tqdm


def seed_all(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

@torch.no_grad()
def validate_model(val_loader, model, device=None, print_freq=100):
    if device is None:
        device = next(model.parameters()).device
    else:
        model.to(device)
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (images, target) in enumerate(val_loader):
        images = images.to(device)
        target = target.to(device)

        # compute output
        output = model(images)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            progress.display(i)

    print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

    return top1.avg

def get_train_samples(train_loader, num_samples):
    train_data, target = [], []
    for batch in train_loader:
        train_data.append(batch[0])
        target.append(batch[1])
        if len(train_data) * batch[0].size(0) >= num_samples:
            break
    return torch.cat(train_data, dim=0)[:num_samples], torch.cat(target, dim=0)[:num_samples]


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='running parameters',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # general parameters for data and model
    parser.add_argument('--seed', default=1005, type=int, help='random seed for results reproduction')
    parser.add_argument('--arch', default='resnet18', type=str, help='model name',
                        choices=['resnet18', 'resnet50', 'mobilenetv2', 'regnetx_600m', 'regnetx_3200m', 'mnasnet'])
    parser.add_argument('--batch_size', default=64, type=int, help='mini-batch size for data loader')
    # parser.add_argument('--workers', default=4, type=int, help='number of workers for data loader')
    """
        数据集路径
    """
    parser.add_argument('--data_path', default='/home/nku524/dl/dataset/imageNet-1k', type=str, help='path to ImageNet data')
    # quantization parameters
    """
        量化参数
        n_bits_w ： 权重量化位宽
        n_bits_a ： 激活量化位宽
        channel_wise ： 权重是否按通道量化，默认为True
    """
    parser.add_argument('--n_bits_w', default=8, type=int, help='bitwidth for weight quantization')
    parser.add_argument('--channel_wise', default=True, help='apply channel_wise quantization for weights')
    parser.add_argument('--n_bits_a', default=8, type=int, help='bitwidth for activation quantization')
    parser.add_argument('--disable_8bit_head_stem', action='store_true')

    """权重校准参数"""
    # weight calibration parameters
    parser.add_argument('--num_samples', default=1024, type=int, help='size of the calibration dataset')
    parser.add_argument('--iters_w', default=20000, type=int, help='number of iteration for adaround')

    """
        舍入成本与重建损失的权重
    """
    parser.add_argument('--weight', default=0.01, type=float, help='weight of rounding cost vs the reconstruction loss.')
    parser.add_argument('--keep_cpu', action='store_true', help='keep the calibration data on cpu')

    parser.add_argument('--b_start', default=20, type=int, help='temperature at the beginning of calibration')
    parser.add_argument('--b_end', default=2, type=int, help='temperature at the end of calibration')
    parser.add_argument('--warmup', default=0.2, type=float, help='in the warmup period no regularization is applied')

    """激活校准参数"""
    # activation calibration parameters
    parser.add_argument('--lr', default=4e-5, type=float, help='learning rate for LSQ')
    """

    """
    parser.add_argument('--init_wmode', default='mse', type=str, choices=['minmax', 'mse', 'minmax_scale'],
                        help='init opt mode for weight')
    parser.add_argument('--init_amode', default='mse', type=str, choices=['minmax', 'mse', 'minmax_scale'],
                        help='init opt mode for activation')

    parser.add_argument('--prob', default=0.5, type=float)
    parser.add_argument('--input_prob', default=0.5, type=float)
    """
        正则化的超参数
    """
    parser.add_argument('--lamb_r', default=0.1, type=float, help='hyper-parameter for regularization')
    """
        KL发散的温度系数
    """
    parser.add_argument('--T', default=4.0, type=float, help='temperature coefficient for KL divergence')
    parser.add_argument('--bn_lr', default=1e-3, type=float, help='learning rate for DC')
    """
        DC的超参数
    """
    parser.add_argument('--lamb_c', default=0.02, type=float, help='hyper-parameter for DC')



    # TODO pointnet需要的参数
    parser.add_argument(
        '--batchSize', type=int, default=32, help='input batch size')
    parser.add_argument(
        '--num_points', type=int, default=2500, help='input batch size')
    parser.add_argument(
        '--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument(
        '--nepoch', type=int, default=2, help='number of epochs to train for')
    parser.add_argument('--outf', type=str, default='cls', help='output folder')
    parser.add_argument('--model', type=str, default='', help='model path')
    # parser.add_argument('--dataset', type=str, required=True, help="dataset path")
    parser.add_argument('--dataset', type=str,
                        default='/home/nku524/dl/dataset/shapenetcore_partanno_segmentation_benchmark_v0',
                        help="dataset path")
    parser.add_argument('--dataset_type', type=str, default='shapenet', help="dataset type shapenet|modelnet40")
    parser.add_argument('--feature_transform', action='store_true', help="use feature transform")

    parser.add_argument(
        '--setgpu', type=int, help='cuda:0?1', default=0)
    parser.add_argument(
        '--a_count', type=int, help='', default=1)
    parser.add_argument(
        '--record', type=str, help='', default="")
    parser.add_argument(
        '--pointnet_num_samples', type=int, help='', default=32)

    args = parser.parse_args()

    torch.cuda.set_device(args.setgpu)

    seed_all(args.seed)

    # build imagenet data loader
    train_loader, test_loader = build_imagenet_data(batch_size=args.batch_size, workers=args.workers,
                                                    data_path=args.data_path)

    #TODO pointnet 数据集 准备

    blue = lambda x: '\033[94m' + x + '\033[0m'
    args.manualSeed = random.randint(1, 10000)  # fix seed
    print("Random Seed: ", args.manualSeed)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)

    if args.dataset_type == 'shapenet':
        dataset = ShapeNetDataset(
            root=args.dataset,
            classification=True,
            npoints=args.num_points)

        test_dataset = ShapeNetDataset(
            root=args.dataset,
            classification=True,
            split='test',
            npoints=args.num_points,
            data_augmentation=False)
    elif args.dataset_type == 'modelnet40':
        dataset = ModelNetDataset(
            root=args.dataset,
            npoints=args.num_points,
            split='trainval')

        test_dataset = ModelNetDataset(
            root=args.dataset,
            split='test',
            npoints=args.num_points,
            data_augmentation=False)
    else:
        exit('wrong dataset type')

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batchSize,
        shuffle=True,
        num_workers=int(args.workers))

    testdataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batchSize,
        shuffle=True,
        num_workers=int(args.workers))

    print(len(dataset), len(test_dataset))
    num_classes = len(dataset.classes)
    print('classes', num_classes)

    try:
        os.makedirs(args.outf)
    except OSError:
        pass

    # load model
    # 加载模型
    """
        加载需要量化的模型
       eval是一个内置的Python函数，用于执行字符串中的Python表达式并返回其结果。例如，eval("2 + 2")将返回4。最终，eval函数执行了如下的代码：
       hubconf.resnet18(pretrained=True)（如果args.arch是resnet18）。
       这意味着我们正在从hubconf模块中调用一个名为resnet18的函数，并且传递pretrained=True作为它的参数。
    """

    # TODO 加载pointnet模型
    cnn = point_model.PointNetCls(k=num_classes, feature_transform=args.feature_transform)
    args.model = "/home/nku524/dl/codebase/pointnet/utils/cls/cls_model_249.pth"
    if args.model != '':
        cnn.load_state_dict(torch.load(args.model, map_location=torch.device('cuda:0')))
    print(cnn)
    # cnn = eval('hubconf.{}(pretrained=True)'.format(args.arch))

    cnn.cuda()  # 将模型移动到GPU上
    cnn.eval()  # 设置模型为评估模式

    fp_model = copy.deepcopy(cnn)   # 深度复制模型
    fp_model.cuda()   # 将复制的模型移动到GPU上
    fp_model.eval()   # 设置复制的模型为评估模式

    """
        权重量化参数
        激活量化参数
    """
    # build quantization parameters
    wq_params = {'n_bits': args.n_bits_w, 'channel_wise': args.channel_wise, 'scale_method': args.init_wmode}
    aq_params = {'n_bits': args.n_bits_a, 'channel_wise': False, 'scale_method': args.init_amode,
                 'leaf_param': True, 'prob': args.prob}

    """
        对模型结构进行量化
        is_fusing=False 不进行BN fold，关闭量化状态

        层融合目的是将多个连续的层操作融合成一个操作，以减少计算开销和提高模型推理速度。
        这通常在量化之前进行，作为模型优化的一部分。
    """

    fp_model = QuantModel(model=fp_model, weight_quant_params=wq_params, act_quant_params=aq_params, is_fusing=False)
    fp_model.cuda()  # 将量化模型移动到GPU上
    fp_model.eval()  # 设置量化模型为评估模式
    """fp_model只是用来对比，不需要开启量化"""
    fp_model.set_quant_state(False, False)   # 关闭量化状态

    """
       qnn是经过BN fold和开启量化状态的 （is_fusing=True）
    """
    qnn = QuantModel(model=cnn, weight_quant_params=wq_params, act_quant_params=aq_params, is_fusing=False)
    qnn.cuda()
    qnn.eval()

    if not args.disable_8bit_head_stem:
        print('Setting the first and the last layer to 8-bit')
        qnn.set_first_last_layer_to_8bit()

    """禁用神经网络中最后一个量化模块（QuantModule）的激活量化。"""
    qnn.disable_network_output_quantization()

    """得到量化后的模型"""
    print('the quantized model is below!')
    print(qnn)


    pointnet_cali_data, pointnet_cali_target = get_train_samples(dataloader, num_samples=args.pointnet_num_samples)

    # cali_data, cali_target = get_train_samples(train_loader, num_samples=args.num_samples)
    device = next(qnn.parameters()).device

    # Kwargs for weight rounding calibration
    """
        用于权重舍入校准的Kwargs
    """
    kwargs = dict(cali_data=pointnet_cali_data, iters=args.iters_w, weight=args.weight,
                b_range=(args.b_start, args.b_end), warmup=args.warmup, opt_mode='mse',
                lr=args.lr, input_prob=args.input_prob, keep_gpu=not args.keep_cpu,
                lamb_r=args.lamb_r, T=args.T, bn_lr=args.bn_lr, lamb_c=args.lamb_c, a_count = args.a_count)


    '''
    #     init weight quantizer
    #
    #     初始化权重量化器，对于每个module中的权重计算出scale和zero_point，后续每个权重数值都学习一个 \theta
    #
    #     遍历每一个module，针对每一个module，如果是QuantModule,
    #     1. 开始初始化权重量化器： module.weight_quantizer.set_inited(False)
    #     2. 遍历每一个module,则用每个module的weight_quantizer计算一个scale和zero_point
    #         module.weight_quantizer(module.weight)，
    #         对于如果是QuantModule的weight_quantizer，后续传入数据到就会得到量化结果
    #     3. 设置量化初始化为完成： module.weight_quantizer.set_inited(True)
    #
    #     计算出最优的min和max，然后根据min和max计算出S和zero_point
    #     然后根据计算出来的S和zero_point对每个module的权重进行量化和反量化操作
    # '''

    set_weight_quantize_params(qnn)

    # tmp = []
    # for name, named_module in qnn.named_modules():
    #     if isinstance(named_module,QuantModule):
    #         print("---------------------------------------------------------")
    #         tmp += named_module
    # first_module = tmp[0]
    # print("org_weight:{}".format(first_module.org_weight))
    # print(first_module.weight_quantizer.delta, first_module.weight_quantizer.zero_point)

    """
        重建: 重建就是让量化模型和FP模型的输出尽量保持一致,对量化模型的算子进行了重建,因为直接量化性能下降很多
    """
    def set_weight_act_quantize_params(module, fp_module):
        if isinstance(module, QuantModule):
            """
                传入完整的qnn、fp_model和当前的module、fp_module
            """
            layer_reconstruction(qnn, fp_model, module, fp_module, **kwargs)
        elif isinstance(module, BaseQuantBlock):
            block_reconstruction(qnn, fp_model, module, fp_module, **kwargs)
        else:
            raise NotImplementedError

    """
        区块重建。对于第一层和最后一层，我们只能应用层重建。
    """
    # a_count = 0
    def recon_model(model: nn.Module, fp_model: nn.Module):
        """
            Block reconstruction. For the first and last layers, we can only apply layer reconstruction.
        """
        for (name, module), (_, fp_module) in zip(model.named_children(), fp_model.named_children()):
            if isinstance(module, QuantModule):
                print('Reconstruction for layer {}'.format(name))
                set_weight_act_quantize_params(module, fp_module)
            elif isinstance(module, BaseQuantBlock):
                """比如对于ResNet里的包含conv BN Relu的block，在block层面再做一次"""
                print('Reconstruction for block {}'.format(name))
                set_weight_act_quantize_params(module, fp_module)
            else:
                recon_model(module, fp_module)

    """
        开始校准
    """
    # Start calibration
    recon_model(qnn, fp_model)
    #
    """qnn设置量化状态为True"""
    qnn.set_quant_state(weight_quant=True, act_quant=True)

    """
        qnn完成了reconstruction,使用测试集测试输出精度
    """
    # print('Full quantization (W{}A{}) accuracy: {}'.format(args.n_bits_w, args.n_bits_a,
    #                                                        validate_model(test_loader, qnn)))

    # TODO 测试量化后pointnet
    total_correct = 0
    total_testset = 0
    for i, data in tqdm(enumerate(testdataloader, 0)):
        points, target = data
        target = target[:, 0]
        points = points.transpose(2, 1)
        points, target = points.cuda(), target.cuda()
        # qnn = qnn.eval()
        # fp_model = fp_model.eval()
        qnn = qnn.eval()
        # pred, _, _ = fp_model(points)
        pred, _, _ = qnn(points)
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        total_correct += correct.item()
        total_testset += points.size()[0]

    print("Pointnet after quantization (W{}A{}) final accuracy {}".format(args.n_bits_w, args.n_bits_a, total_correct / float(total_testset)))

    # # 获取当前时间
    # current_time = datetime.datetime.now()
    # # 将时间转化为字符串
    # time_str = current_time.strftime("%Y-%m-%d %H:%M:%S")
    # record_str = "num_samples{} Pointnet after quantization (W{}A{}) final accuracy {}".format(args.pointnet_num_samples, args.n_bits_w, args.n_bits_a, total_correct / float(total_testset)) + "  ***  " + time_str + "\n"
    # with open(args.record, 'a', encoding='utf-8') as file:
    #     # 写入数据到文件
    #     file.write(record_str)
    # print("数据已成功写入到文件")

if __name__ == '__main__':
    pass
