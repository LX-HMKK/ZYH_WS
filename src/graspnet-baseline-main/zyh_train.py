""" Training routine for GraspNet baseline model. 
"""
#GraspNet 通常是一个神经网络，用于从点云（point cloud）或深度图像（depth image）中预测抓取姿势（grasp poses）

#标准 Python 模块
import os
import sys
import numpy as np
from datetime import datetime
import argparse

import torch
import torch.nn as nn  #神经网络模块
import torch.optim as optim #优化器
from torch.optim import lr_scheduler #学习率调度
from torch.utils.data import DataLoader #数据加载器
from torch.utils.tensorboard import SummaryWriter #TensorBoard 日志记录，用于可视化训练过程

#获取当前 Python 脚本所在的绝对路径的目录名
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
#引入对应的模块函数
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))

#导入了 GraspNet 项目 中的关键模块和工具，主要用于 模型定义、数据加载、损失计算和训练优化
from graspnet import GraspNet, get_loss
from pytorch_utils import BNMomentumScheduler
from graspnet_dataset import GraspNetDataset, collate_fn, load_grasp_labels
from label_generation import process_grasp_labels

#-----设置变量----命令行来命名参数
parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', required=True, help='Dataset root') #指定 GraspNet 数据集的根目录路径。
parser.add_argument('--camera', required=True, help='Camera split [realsense/kinect]')#选择使用的相机类型

parser.add_argument('--checkpoint_path', default=None, help='Model checkpoint path [default: None]') #加载预训练模型的路径 None（从头训练）
parser.add_argument('--log_dir', default='log', help='Dump dir to save model checkpoint [default: log]')#保存模型检查点（checkpoint）和日志的目录

parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]') #从原始点云中随机采样的点数
parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')#视角数量或批处理中的视图限制
parser.add_argument('--max_epoch', type=int, default=18, help='Epoch to run [default: 18]')#最大训练轮数
parser.add_argument('--batch_size', type=int, default=2, help='Batch Size during training [default: 2]')#训练时的批大小点云模型通常显存占用较高，需较小 batch size

parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')#Adam 优化器的初始学习率
parser.add_argument('--weight_decay', type=float, default=0, help='Optimization L2 weight decay [default: 0]')#L2 正则化系数（防止过拟合）
parser.add_argument('--bn_decay_step', type=int, default=2, help='Period of BN decay (in epochs) [default: 2]')
parser.add_argument('--bn_decay_rate', type=float, default=0.5, help='Decay rate for BN decay [default: 0.5]')
parser.add_argument('--lr_decay_steps', default='8,12,16', help='When to decay the learning rate (in epochs) [default: 8,12,16]')#定义学习率衰减的 时间表
parser.add_argument('--lr_decay_rates', default='0.1,0.1,0.1', help='Decay rates for lr decay [default: 0.1,0.1,0.1]')
#lr_decay_steps='8,12,16' 表示在 epoch 8、12、16 时衰减学习率。
#lr_decay_rates='0.1,0.1,0.1' 表示每次衰减为当前的 10%

cfgs = parser.parse_args()#会读取命令行参数，并将其与 add_argument() 定义的规则匹配

# ------------------------------------------------------------------------- GLOBAL CONFIG BEG


EPOCH_CNT = 0
LR_DECAY_STEPS = [int(x) for x in cfgs.lr_decay_steps.split(',')]#整数列表（如 [8, 12, 16]）
LR_DECAY_RATES = [float(x) for x in cfgs.lr_decay_rates.split(',')]#浮点数列表（如 [0.1, 0.01, 0.001]）
assert(len(LR_DECAY_STEPS)==len(LR_DECAY_RATES))#确保二者长度相等

#制定预训练权重路径：默认和用户确定两种情况
DEFAULT_CHECKPOINT_PATH = os.path.join(cfgs.log_dir, 'checkpoint.tar')
CHECKPOINT_PATH = cfgs.checkpoint_path if cfgs.checkpoint_path is not None \
    else DEFAULT_CHECKPOINT_PATH

if not os.path.exists(cfgs.log_dir):
    os.makedirs(cfgs.log_dir)#创建日志目录
LOG_FOUT = open(os.path.join(cfgs.log_dir, 'log_train.txt'), 'a')#避免覆盖历史日志
LOG_FOUT.write(str(cfgs)+'\n')
def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

# Init datasets and dataloaders 多进程数据加载时设置随机种子，确保数据增强的可复现性。
def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    pass

# Create Dataset and Dataloader 创建训练集和测试集
valid_obj_idxs, grasp_labels = load_grasp_labels(cfgs.dataset_root)
TRAIN_DATASET = GraspNetDataset(cfgs.dataset_root, valid_obj_idxs, grasp_labels, camera=cfgs.camera, split='train', num_points=cfgs.num_point, remove_outlier=True, augment=True)
TEST_DATASET = GraspNetDataset(cfgs.dataset_root, valid_obj_idxs, grasp_labels, camera=cfgs.camera, split='test_seen', num_points=cfgs.num_point, remove_outlier=True, augment=False)

print(len(TRAIN_DATASET), len(TEST_DATASET))
TRAIN_DATALOADER = DataLoader(TRAIN_DATASET, 
                              batch_size=cfgs.batch_size, 
                              shuffle=True,
                              num_workers=4, 
                              worker_init_fn=my_worker_init_fn, 
                              collate_fn=collate_fn)
TEST_DATALOADER = DataLoader(TEST_DATASET, 
                             batch_size=cfgs.batch_size, 
                             shuffle=False,
                             num_workers=4, 
                             worker_init_fn=my_worker_init_fn, 
                             collate_fn=collate_fn)
print(len(TRAIN_DATALOADER), len(TEST_DATALOADER))
# Init the model and optimzier
net = GraspNet(input_feature_dim=0, num_view=cfgs.num_view, num_angle=12, num_depth=4,
                        cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04])

#如果有GPU就使用GPU，没有则使用CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)


# Load the Adam optimizer
optimizer = optim.Adam(net.parameters(), lr=cfgs.learning_rate, weight_decay=cfgs.weight_decay)
# Load checkpoint if there is any
it = -1 # for the initialize value of `LambdaLR` and `BNMomentumScheduler`
start_epoch = 0
if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    log_string("-> loaded checkpoint %s (epoch: %d)"%(CHECKPOINT_PATH, start_epoch))
# Decay Batchnorm momentum from 0.5 to 0.999
# note: pytorch's BN momentum (default 0.1)= 1 - tensorflow's BN momentum
BN_MOMENTUM_INIT = 0.5
BN_MOMENTUM_MAX = 0.001
bn_lbmd = lambda it: max(BN_MOMENTUM_INIT * cfgs.bn_decay_rate**(int(it / cfgs.bn_decay_step)), BN_MOMENTUM_MAX)
bnm_scheduler = BNMomentumScheduler(net, bn_lambda=bn_lbmd, last_epoch=start_epoch-1)


def get_current_lr(epoch):
    lr = cfgs.learning_rate
    for i,lr_decay_epoch in enumerate(LR_DECAY_STEPS):
        if epoch >= lr_decay_epoch:
            lr *= LR_DECAY_RATES[i]
    return lr

def adjust_learning_rate(optimizer, epoch):
    lr = get_current_lr(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# TensorBoard Visualizers
TRAIN_WRITER = SummaryWriter(os.path.join(cfgs.log_dir, 'train'))
TEST_WRITER = SummaryWriter(os.path.join(cfgs.log_dir, 'test'))

# ------------------------------------------------------------------------- GLOBAL CONFIG END

def train_one_epoch():
    stat_dict = {} # collect statistics
    adjust_learning_rate(optimizer, EPOCH_CNT)
    bnm_scheduler.step() # decay BN momentum
    # set model to training mode
    net.train()
    for batch_idx, batch_data_label in enumerate(TRAIN_DATALOADER):
        for key in batch_data_label:
            if 'list' in key:
                for i in range(len(batch_data_label[key])):
                    for j in range(len(batch_data_label[key][i])):
                        batch_data_label[key][i][j] = batch_data_label[key][i][j].to(device)
            else:
                batch_data_label[key] = batch_data_label[key].to(device)

        # Forward pass
        end_points = net(batch_data_label)

        # Compute loss and gradients, update parameters.
        loss, end_points = get_loss(end_points)
        loss.backward()
        if (batch_idx+1) % 1 == 0:
            optimizer.step()
            optimizer.zero_grad()

        # Accumulate statistics and print out
        for key in end_points:
            if 'loss' in key or 'acc' in key or 'prec' in key or 'recall' in key or 'count' in key:
                if key not in stat_dict: stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()

        batch_interval = 10
        if (batch_idx+1) % batch_interval == 0:
            log_string(' ---- batch: %03d ----' % (batch_idx+1))
            for key in sorted(stat_dict.keys()):
                TRAIN_WRITER.add_scalar(key, stat_dict[key]/batch_interval, (EPOCH_CNT*len(TRAIN_DATALOADER)+batch_idx)*cfgs.batch_size)
                log_string('mean %s: %f'%(key, stat_dict[key]/batch_interval))
                stat_dict[key] = 0

def evaluate_one_epoch():
    stat_dict = {} # collect statistics
    # set model to eval mode (for bn and dp)
    net.eval()
    for batch_idx, batch_data_label in enumerate(TEST_DATALOADER):
        if batch_idx % 10 == 0:
            print('Eval batch: %d'%(batch_idx))
        for key in batch_data_label:
            if 'list' in key:
                for i in range(len(batch_data_label[key])):
                    for j in range(len(batch_data_label[key][i])):
                        batch_data_label[key][i][j] = batch_data_label[key][i][j].to(device)
            else:
                batch_data_label[key] = batch_data_label[key].to(device)
        
        # Forward pass
        with torch.no_grad():
            end_points = net(batch_data_label)

        # Compute loss
        loss, end_points = get_loss(end_points)

        # Accumulate statistics and print out
        for key in end_points:
            if 'loss' in key or 'acc' in key or 'prec' in key or 'recall' in key or 'count' in key:
                if key not in stat_dict: stat_dict[key] = 0
                stat_dict[key] += end_points[key].item()

    for key in sorted(stat_dict.keys()):
        TEST_WRITER.add_scalar(key, stat_dict[key]/float(batch_idx+1), (EPOCH_CNT+1)*len(TRAIN_DATALOADER)*cfgs.batch_size)
        log_string('eval mean %s: %f'%(key, stat_dict[key]/(float(batch_idx+1))))

    mean_loss = stat_dict['loss/overall_loss']/float(batch_idx+1)
    return mean_loss


def train(start_epoch):
    global EPOCH_CNT 
    min_loss = 1e10
    loss = 0
    for epoch in range(start_epoch, cfgs.max_epoch):
        EPOCH_CNT = epoch
        log_string('**** EPOCH %03d ****' % (epoch))
        log_string('Current learning rate: %f'%(get_current_lr(epoch)))
        log_string('Current BN decay momentum: %f'%(bnm_scheduler.lmbd(bnm_scheduler.last_epoch)))
        log_string(str(datetime.now()))
        # Reset numpy seed.
        # REF: https://github.com/pytorch/pytorch/issues/5059
        np.random.seed()

        #训练一轮
        train_one_epoch()
        
        loss = evaluate_one_epoch()
        # Save checkpoint
        save_dict = {'epoch': epoch+1, # after training one epoch, the start_epoch should be epoch+1
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    }
        try: # with nn.DataParallel() the net is added as a submodule of DataParallel
            save_dict['model_state_dict'] = net.module.state_dict()
        except:
            save_dict['model_state_dict'] = net.state_dict()
        torch.save(save_dict, os.path.join(cfgs.log_dir, 'checkpoint.tar'))

if __name__=='__main__':
    train(start_epoch)
