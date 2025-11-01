import os
import sys
from tqdm import tqdm
import shutil
import argparse
import logging
import time
import random
import numpy as np
import logging

import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.nn import BCEWithLogitsLoss, MSELoss, CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers.factories import Act, Conv, Dropout, Norm, split_args
from monai.utils import deprecated_arg

from monai.transforms import MapTransform
import SimpleITK as sitk

from monai.utils import first, set_determinism
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    RandAffined,
    Rand3DElasticd,
    RandGridDistortiond,
    Rotate90d,
    SaveImaged,
    ScaleIntensityRanged,
    Spacingd,
    EnsureTyped,
    EnsureType,
    Invertd,
    ToTensord,
    RandFlipd,
    RandRotate90d
)
from monai.handlers.utils import from_engine
from monai.networks.layers import Norm
from monai.networks.nets import UNet
from monai.networks.nets import DenseNet121
from monai.metrics import DiceMetric
from monai.metrics import ConfusionMatrixMetric
from monai.metrics import MeanIoU
from monai.losses import DiceLoss
from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.config import print_config
from monai.apps import download_and_extract
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
import random
import tarfile
import shutil
import re
import pandas as pd
import glob
from tqdm import tqdm
from einops import rearrange,repeat,reduce
from models.MVF_final import MVF_Net


# 加载数据
raw_files = os.listdir('/course75/XXL/raw/')
raw_files = sorted(raw_files)[4:]
# 删除没有标签的样本
skip_numbers = [107,143,149,153,156,16,23,26,33,41,66,7,70,72,75,79,81,85,86,93,94]
skip_pattern = re.compile(r"\b(" + "|".join(map(str, skip_numbers))+r")\.nii\.gz\b")
raw_files = [file for file in raw_files if not skip_pattern.search(file) and file.endswith('.nii.gz')]
train_images = [os.path.join('/course75/XXL/raw/',file) for file in raw_files]
train_images = sorted(train_images)

mask_files = os.listdir('/course75/XXL/mask/')
mask_files = sorted(mask_files)[1:]
# 删除没有标签的样本
mask_files = [file for file in mask_files if not skip_pattern.search(file) and file.endswith('.nii.gz')]
train_labels = [os.path.join('/course75/XXL/mask/',file) for file in mask_files]
train_labels = sorted(train_labels)

#处理标签
class_label = pd.read_csv('/JingTeam/Shuying/multi_task_XXL/class_label2.csv')
type_dict = dict(zip(class_label['Folder_Name'], class_label['WHO_histologic_type']))

#数据集
data_dicts = [
    {"image": image_name, "label": label_name, "target": type_dict.get(os.path.basename(image_name))}
    for image_name, label_name in zip(train_images, train_labels)
]
train_images2 = sorted(
        glob.glob(os.path.join('/JingTeam/Shuying/classification_XXL/data/spacing_raw_96/', "*.nii.gz")))
train_labels2 = sorted(
        glob.glob(os.path.join('/JingTeam/Shuying/classification_XXL/data/spacing_mask_96/', "*.nii.gz")))
data_dicts2 = [
    {"image": image_name, "label": label_name, "target": type_dict.get(os.path.basename(image_name))}
    for image_name, label_name in zip(train_images2, train_labels2)
]

#分割训练集和测试集90:31
def group_data(data):
    grouped_data = {0: [], 1: []}
    for item in data:
        grouped_data[item["target"]].append(item)
    return grouped_data

grouped_data_dict = group_data(data_dicts)
grouped_data_dicts2 = group_data(data_dicts2)

train_files = []
val_files = []

for target in grouped_data_dict.keys():
    train_size = len(grouped_data_dict[target]) // 4 * 3
    test_size = len(grouped_data_dict[target])-train_size
    train_files.extend(grouped_data_dict[target][:train_size])
    val_files.extend(grouped_data_dicts2[target][:test_size])

random.shuffle(train_files)
random.shuffle(val_files)

# 变换
train_transforms = Compose(
[
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Spacingd(
        keys=["image", "label"],
        pixdim=(1.0, 1.0, 1.0),
        mode=("bilinear", "nearest"),
    ),
    ScaleIntensityRanged(
        keys=["image"], a_min=-210, a_max=290, b_min=0.0, b_max=1.0, clip=True
    ),
    CropForegroundd(keys=["image", "label"], source_key="image"), #按前景裁剪
    RandCropByPosNegLabeld( #按阴阳比例裁剪，每个CT裁剪出4个图，但实际输出差异不大
        keys=["image", "label"],
        label_key="label",
        spatial_size=(96,96,96),
        pos=1,# 裁剪中心的正例、负例占比，目前是1:1
        neg=0,
        num_samples=1,
        image_key="image",
        image_threshold=0,
    ),
    #一些数据增强
    RandFlipd(
        keys=["image", "label"],
        spatial_axis=[0],
        prob=0.40,
    ),
    RandFlipd(
        keys=["image", "label"],
        spatial_axis=[1],
        prob=0.40,
    ),
    RandFlipd(
        keys=["image", "label"],
        spatial_axis=[2],
        prob=0.40,
    ),
    RandRotate90d(
        keys=["image", "label"],
        prob=0.20,
        max_k=3,
    ),
    # RandAffined(
    #     keys=['image', 'label'],
    #     mode=('bilinear', 'nearest'),
    #     prob=1.0, spatial_size=(96,96,96),
    #     rotate_range=(np.pi/36, np.pi/36, np.pi/4),
    #     scale_range=(0.15, 0.15, 0.15)), #随机仿射变换
    # Rand3DElasticd(
    #     keys=["image","label"],
    #     mode=("bilinear","nearest"),
    #     prob=1.0,
    #     sigma_range=(5,8),
    #     magnitude_range=(100,200),
    #     spatial_size=(96,96,96),
    #     rotate_range=(np.pi/36, np.pi/36, np.pi),
    #     scale_range=(0.15,0.15,0.15)), #随机弹性变换
    # RandGridDistortiond(
    #     keys=["image","label"],
    #     num_cells=4, prob=1.0,
    #     distort_limit=(-0.2,0.2),
    #     mode=['bilinear','nearest']),
    ToTensord(keys=["image", "label"]),
])

val_transforms = Compose(
[
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    # Spacingd(
    #     keys=["image", "label"],
    #     pixdim=(1.0, 1.0, 1.0),
    #     mode=("bilinear", "nearest"),
    # ),
    ScaleIntensityRanged(
        keys=["image"], a_min=-210, a_max=290, b_min=0.0, b_max=1.0, clip=True
    ),
    #CropForegroundd(keys=["image", "label"], source_key="image"),
    ToTensord(keys=["image", "label"]),
])



train_ds = CacheDataset(
    data=train_files,
    transform=train_transforms,
    cache_rate=1.0,
    num_workers=4
)
train_loader = DataLoader(
    train_ds, batch_size=1, shuffle=True, num_workers=4, pin_memory=True
)

val_ds = CacheDataset(
    data=val_files, transform=val_transforms,cache_rate=1.0, num_workers=4
)
val_loader = DataLoader(
    val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True
)


class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum.cuda()
    

# 验证函数
def evaluate(val_loader, model, device, epoch:int):
    model.eval()
    
    clf_acc = 0
    post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=2)]) #作用于预测值上
    post_label = Compose([EnsureType(), AsDiscrete(to_onehot=2)])
    num_correct = 0.0
    clf_list = []
    target_list = []

    with torch.no_grad():
        for val_data in val_loader:
            inputs, val_labels, val_targets = (val_data["image"].to(device),
                                     val_data["label"].to(device),
                                     val_data["target"])
            seg, clf = model(inputs)
            #评价分割
            val_outputs = [post_pred(i) for i in decollate_batch(seg)]
            val_labels = [post_label(i) for i in decollate_batch(val_labels)]
            dice_metric(y_pred=val_outputs, y=val_labels)
            #评价分类
            predictions = np.argmax(clf.detach().cpu().numpy(), axis=1)
            clf_list.extend(predictions)
            target_list.extend(val_targets.detach().numpy())
                
        #dice平均值
        metric = dice_metric.aggregate().item()
        dice_metric.reset()
        #accuracy
        accuracy = accuracy_score(target_list, clf_list)
        precision = precision_score(target_list, clf_list)
        recall = recall_score(target_list, clf_list)
        f1 = f1_score(target_list, clf_list)
        auc = roc_auc_score(target_list, clf_list)
                
        return clf_list, target_list, metric, accuracy, precision, recall, f1, auc

    
# 环境配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MVF_Net(
    spatial_dims=3, 
    num_classes=2, 
    act=('elu', {'inplace': True}),
    dropout_prob_down=0.5,
    dropout_prob_up=(0.5,0.5),
    dropout_dim=3,
    bias=False).to(device)


#focal_loss定义
class MultiClassFocalLossWithAlpha(nn.Module):
    def __init__(self, alpha=[0.18, 0.26, 0.56], gamma=2, reduction='mean'):
        """
        :param alpha: 权重系数列表，三分类中第0类权重0.2，第1类权重0.3，第2类权重0.5
        :param gamma: 困难样本挖掘的gamma
        :param reduction:
        """
        super(MultiClassFocalLossWithAlpha, self).__init__()
        self.alpha = torch.tensor(alpha).to(device)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        alpha = self.alpha[target]  # 为当前batch内的样本，逐个分配类别权重，shape=(bs), 一维向量
        log_softmax = torch.log_softmax(pred, dim=1) # 对模型裸输出做softmax再取log, shape=(bs, 3)
        logpt = torch.gather(log_softmax, dim=1, index=target.view(-1, 1))  # 取出每个样本在类别标签位置的log_softmax值, shape=(bs, 1)
        logpt = logpt.view(-1)  # 降维，shape=(bs)
        ce_loss = -logpt  # 对log_softmax再取负，就是交叉熵了
        pt = torch.exp(logpt)  #对log_softmax取exp，把log消了，就是每个样本在类别标签位置的softmax值了，shape=(bs)
        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss  # 根据公式计算focal loss，得到每个样本的loss值，shape=(bs)
        if self.reduction == "mean":
            return torch.mean(focal_loss)
        if self.reduction == "sum":
            return torch.sum(focal_loss)
        return focal_loss
    
#第二种focal_loss定义
class focal_loss_multi(nn.Module):
    def __init__(self, alpha=[0.18,0.26,0.56], gamma = 2, num_classes=3,size_average=True):
        super(focal_loss_multi, self).__init__()
        self.size_average = size_average
        if isinstance(alpha,(float,int)): #只设置第一类别的权重
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha) #self.alpha = [0.25,0.75,0.75,0.75,0.75]
        if isinstance(alpha, list): #全部权重自己设置
            self.alpha = torch.Tensor(alpha)
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        alpha = self.alpha.to(device)
        N = inputs.size(0)
        C = inputs.size(1)
        #下面这些只是为了获取四个样本的概率probs
        #如模型中有softmax，则不需要下一行代码
        P = F.softmax(inputs, dim=1)
        #P = inputs
        class_mask = inputs.data.new(N,C).fill_(0) #生成和input一样shape的tensor
        class_mask = class_mask.requires_grad_() #加入梯度计算
        ids = targets.view(-1,1) #获取目标的索引
        alpha = alpha.gather(0,ids.view(-1))
        #one hot 
        class_mask.data.scatter_(1,ids.data,1.) #利用scatter将索引丢给mask
        probs = (P * class_mask).sum(1).view(-1,1)
        #focal loss公式
        log_p = probs.log()
        loss = torch.pow((1 - probs), self.gamma) * log_p
        batch_loss = (-alpha*loss).t()
        
        #batch loss求平均
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

#二分类focal loss
class BCEFocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=5/6, reduction='mean'):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, predict, target):
        pt = torch.sigmoid(predict) # sigmoide获取概率
        eps = 1e-7
        #在原始ce上增加动态权重因子，注意alpha的写法，下面多类时不能这样使用
        loss = - self.alpha * (1 - pt) ** self.gamma * target * torch.log(pt + eps) - (1 - self.alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt + eps)

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss
    
    
seg_loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
#clf_loss_function = MultiClassFocalLossWithAlpha()
clf_loss_function = BCEFocalLoss(gamma=2, alpha=5/6, reduction='mean')

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0001)
#optimizer = torch.optim.Adam(model.parameters(),lr=0.1,weight_decay = 1e-4)
dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5000)

# for name, param in model.named_parameters():
#     # if 'inc1' in name or 'att1' in name or 'weights' in name or 'seg' in name:
#     #     param.requires_grad = False
#     if 'inc2' in name or 'inc3' in name:
#         param.requires_grad = False
#     if 'att2' in name or 'att3' in name or 'weights' in name or 'clf' in name:
#         param.requires_grad = False


val_interval = 5
max_epochs = 5000
autoweight = AutomaticWeightedLoss(2)
epoch_loss_values = []
epoch_seg_loss_values = []
epoch_clf_loss_values = []
best_acc = -1
best_metric = -1
best_metric_epoch = -1
best_acc_epoch = -1

save_path = '/JingTeam/Shuying/multi_task_XXL/savemodel/'

# logging.basicConfig(filename=save_path+"/log.txt", level=logging.INFO,
#                     format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
# logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

import csv
def storfile(data,filename):
    data=list(map(lambda x:[x],data))
    with open(filename,'w',newline='') as f:
        mywrite=csv.writer(f)
        for i in data:
            mywrite.writerow(i)
            
# 模型训练
for epoch in range(max_epochs):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{max_epochs}")
    model.train()
    epoch_loss = 0
    epoch_seg_loss = 0
    epoch_clf_loss = 0
    step = 0
    print(optimizer.state_dict()['param_groups'][0]['lr']) #打印学习率
    for batch_data in train_loader:
        step += 1
        inputs, labels, targets = batch_data["image"].to(device), batch_data["label"].to(device), batch_data["target"].to(device)
        seg, clf = model(inputs)
        seg_loss = seg_loss_function(seg, labels)
        clf_loss = clf_loss_function(clf, targets)
        loss = autoweight.to(device)(seg_loss, clf_loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_seg_loss += seg_loss.item()
        epoch_clf_loss += clf_loss.item()
        # logging.info('iteration %d : loss : %f loss_seg : %f loss_clf : %f' %
        #              (epoch, loss.item(), seg_loss.item(), clf_loss.item()))
    epoch_loss /= step
    epoch_seg_loss /= step
    epoch_clf_loss /= step
    epoch_loss_values.append(epoch_loss)
    epoch_seg_loss_values.append(epoch_seg_loss)
    epoch_clf_loss_values.append(epoch_clf_loss)
    lr_scheduler.step()
    print(model.weights1.data)
    print(model.weights2.data)
    print(f"epoch {epoch + 1} seg loss: {epoch_seg_loss:.4f} clf loss: {epoch_clf_loss:.4f} average loss: {epoch_loss:.4f}")
    
    
    if (epoch + 1) % val_interval == 0:
        clf_list, target_list, metric, accuracy, precision, recall, f1, auc = evaluate(val_loader, model, device, epoch + 1)
        
        print(clf_list)
        print(target_list)
        
        if metric > best_metric: 
            best_metric = metric
            best_metric_epoch = epoch + 1
            model_save_path = os.path.join(
                save_path, 
                f"epoch{best_metric_epoch}-{metric:.4f}-{auc:.4f}.pth"
            )
            torch.save(model.state_dict(), model_save_path)
            print("saved new best metric model")
            
        if auc > best_acc: 
            best_acc = auc
            best_acc_epoch = epoch + 1
            model_save_path = os.path.join(
                save_path, 
                f"epoch{best_acc_epoch}-{metric:.4f}-{auc:.4f}.pth"
            )
            torch.save(model.state_dict(), model_save_path)
            print("saved new best auc model")
                        
        print(
            "current epoch: {} current dice: {:.4f} current auc: {:.4f} "
            "best dice: {:.4f} best auc: {:.4f}".format(
                epoch + 1, metric, auc, best_metric, best_acc
            )
        )
        print("accuracy: {:.4f} precision: {:.4f} recall: {:.4f} f1: {:.4f}".format(accuracy, precision, recall, f1))
        torch.cuda.empty_cache()
        
    # if (epoch + 1) % 50 == 0:
    #     storfile(epoch_loss_values,'/JingTeam/Shuying/multi_task_XXL/savemodel/loss_new.csv')
    #     storfile(epoch_seg_loss_values,'/JingTeam/Shuying/multi_task_XXL/savemodel/seg_loss_new.csv')
    #     storfile(epoch_clf_loss_values,'/JingTeam/Shuying/multi_task_XXL/savemodel/clf_loss_new.csv')
        
print(
    f"train completed, best_metric: {best_metric:.4f} best_acc: {best_acc:.4f}")