from __future__ import annotations
from typing import Union
from typing import Sequence
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import OrderedDict
from monai.networks.layers.factories import Act, Conv, Dropout, Norm, split_args, Pool
from monai.networks.nets import UNet
from torch.nn import Module, Sequential
from torch.nn import Conv3d, ConvTranspose3d, BatchNorm3d, MaxPool3d, AvgPool1d
from torch.nn import ReLU, Sigmoid
from monai.networks.blocks.convolutions import Convolution
from monai.utils import deprecated_arg
import re
from collections.abc import Callable, Sequence
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.utils.module import look_up_option

#unet
class Conv3D_Block(nn.Module):

    def __init__(self, inp_feat, out_feat, kernel=3, stride=1, padding=1, residual=None):

        super(Conv3D_Block, self).__init__()

        self.conv1 = Sequential(
                        Conv3d(inp_feat, out_feat, kernel_size=kernel,
                                    stride=stride, padding=padding, bias=True),
                        BatchNorm3d(out_feat),
                        ReLU())

        self.conv2 = Sequential(
                        Conv3d(out_feat, out_feat, kernel_size=kernel,
                                    stride=stride, padding=padding, bias=True),
                        BatchNorm3d(out_feat))

        self.residual = residual

        if self.residual is not None:
            self.residual_upsampler = Conv3d(inp_feat, out_feat, kernel_size=1, bias=False)

    def forward(self, x):

        res = x

        if not self.residual:
            return self.conv2(self.conv1(x))
        else:
            return self.conv2(self.conv1(x)) + self.residual_upsampler(res)


class Deconv3D_Block(nn.Module):

    def __init__(self, inp_feat, out_feat, kernel=4, stride=2, padding=1):

        super(Deconv3D_Block, self).__init__()

        self.deconv = Sequential(
                        #3D反卷积
                        ConvTranspose3d(inp_feat, out_feat, kernel_size=kernel,
                                    stride=stride, padding=padding, output_padding=0, bias=True),
                        ReLU())

    def forward(self, x):

        return self.deconv(x)
    


# inception block
class Inception(nn.Module):
    def __init__(self,in_c,c1,c2,c3,c4): #in_c=1, c1=8, c2=[8,8], c3=[8,12,12], c4=4
        super(Inception,self).__init__()
        #线路1 1*1的卷积层
        self.p1 = nn.Sequential(
            nn.Conv3d(in_c,c1,kernel_size=1),
            nn.BatchNorm3d(c1), nn.ReLU()
        )
        #线路2 1*1卷积层后接3*3的卷积
        self.p2 = nn.Sequential(
            nn.Conv3d(in_c,c2[0],kernel_size=1),
            nn.BatchNorm3d(c2[0]), nn.ReLU(),
            nn.Conv3d(c2[0], c2[1], kernel_size=3,padding=1),
            nn.BatchNorm3d(c2[1]), nn.ReLU()
        )
        #线路3 1*1卷积层后接两个3*3的卷积层
        self.p3 = nn.Sequential(
            nn.Conv3d(in_c, c3[0], kernel_size=1),
            nn.BatchNorm3d(c3[0]), nn.ReLU(),
            nn.Conv3d(c3[0], c3[1], kernel_size=3,padding=1),
            nn.BatchNorm3d(c3[1]), nn.ReLU(),
            nn.Conv3d(c3[1], c3[2], kernel_size=3,padding=1),
            nn.BatchNorm3d(c3[2]), nn.ReLU()
        )
        #线路4 3*3最大池化后接1*1卷积层
        self.p4 = nn.Sequential(
            nn.MaxPool3d(kernel_size=3,stride=1,padding=1),
            nn.Conv3d(in_c,c4,kernel_size=1),
            nn.BatchNorm3d(c4), nn.ReLU()
        )
    def forward(self, x):
        p1 = self.p1(x)
        p2 = self.p2(x)
        p3 = self.p3(x)
        p4 = self.p4(x)
        return torch.cat((p1,p2,p3,p4),dim=1)

# 注意力机制
class Channel_Attention_Module_Conv(nn.Module):
    def __init__(self, channels, gamma = 2, b = 1):
        super(Channel_Attention_Module_Conv, self).__init__()
        kernel_size = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.avg_pooling = nn.AdaptiveAvgPool3d(1)
        self.max_pooling = nn.AdaptiveMaxPool3d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size = kernel_size, padding = (kernel_size - 1) // 2, bias = False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_x = self.avg_pooling(x)
        max_x = self.max_pooling(x)
        avg_out = self.conv(avg_x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        max_out = self.conv(max_x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        v = self.sigmoid(avg_out + max_out)
        return x * v
    
class Channel_Attention_Module_FC(nn.Module):
    def __init__(self, channels, ratio):
        super(Channel_Attention_Module_FC, self).__init__()
        self.avg_pooling = nn.AdaptiveAvgPool3d(1)
        self.max_pooling = nn.AdaptiveMaxPool3d(1)
        self.fc_layers = nn.Sequential(
            nn.Linear(in_features = channels, out_features = channels // ratio, bias = False),
            nn.ReLU(),
            nn.Linear(in_features = channels // ratio, out_features = channels, bias = False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, d, h, w = x.shape
        avg_x = self.avg_pooling(x).view(b, c)
        max_x = self.max_pooling(x).view(b, c)
        v = self.fc_layers(avg_x) + self.fc_layers(max_x)
        v = self.sigmoid(v).view(b, c, 1, 1, 1)
        return x * v
    
    
class CPCABlock(nn.Module):
    def __init__(self, channel_attention_mode: str, channels: int = None,
                 ratio: int = None, gamma: int = None, b: int = None):
        super(CPCABlock, self).__init__()
        if channel_attention_mode == "FC":
            assert channels != None and ratio != None and channel_attention_mode == "FC", \
                "FC channel attention block need feature maps' channels, ratio"
            self.channel_attention_block = Channel_Attention_Module_FC(channels = channels, ratio = ratio)
        elif channel_attention_mode == "Conv":
            assert channels != None and gamma != None and b != None and channel_attention_mode == "Conv", \
                "Conv channel attention block need feature maps' channels, gamma, b"
            self.channel_attention_block = Channel_Attention_Module_Conv(channels = channels, gamma = gamma, b = b)
        else:
            assert channel_attention_mode in ["FC", "Conv"], \
                "channel attention block must be 'FC' or 'Conv'"
        self.dconv5_5 = nn.Conv3d(channels,channels,kernel_size=5,padding=2,groups=channels)
        self.dconv1_7 = nn.Conv3d(channels,channels,kernel_size=(1,1,7),padding=(0,0,3),groups=channels)
        self.dconv7_1 = nn.Conv3d(channels,channels,kernel_size=(7,7,1),padding=(3,3,0),groups=channels)
        self.dconv1_11 = nn.Conv3d(channels,channels,kernel_size=(1,1,11),padding=(0,0,5),groups=channels)
        self.dconv11_1 = nn.Conv3d(channels,channels,kernel_size=(11,11,1),padding=(5,5,0),groups=channels)
        self.dconv1_21 = nn.Conv3d(channels,channels,kernel_size=(1,1,21),padding=(0,0,10),groups=channels)
        self.dconv21_1 = nn.Conv3d(channels,channels,kernel_size=(21,21,1),padding=(10,10,0),groups=channels)
        self.conv = nn.Conv3d(channels,channels,kernel_size=1,padding=0)


    def forward(self, x):
        inputs = self.channel_attention_block(x)
        x_init = self.dconv5_5(inputs)
        x_1 = self.dconv1_7(x_init)
        x_1 = self.dconv7_1(x_1)
        x_2 = self.dconv1_11(x_init)
        x_2 = self.dconv11_1(x_2)
        x_3 = self.dconv1_21(x_init)
        x_3 = self.dconv21_1(x_3)
        x = x_1 + x_2 + x_3 + x_init
        spatial_att = self.conv(x)
        out = spatial_att * inputs
        out = self.conv(out)

        return out

    
#vnet
def get_acti_layer(act: tuple[str, dict] | str, nchan: int = 0):
    if act == "prelu":
        act = ("prelu", {"num_parameters": nchan})
    act_name, act_args = split_args(act)
    act_type = Act[act_name]
    return act_type(**act_args)


class LUConv(nn.Module):

    def __init__(self, spatial_dims: int, nchan: int, act: tuple[str, dict] | str, bias: bool = False):
        super().__init__()

        self.act_function = get_acti_layer(act, nchan)
        self.conv_block = Convolution(
            spatial_dims=spatial_dims,
            in_channels=nchan,
            out_channels=nchan,
            kernel_size=5,
            act=None,
            norm=Norm.BATCH,
            bias=bias,
        )

    def forward(self, x):
        out = self.conv_block(x)
        out = self.act_function(out)
        return out


def _make_nconv(spatial_dims: int, nchan: int, depth: int, act: tuple[str, dict] | str, bias: bool = False):
    layers = []
    for _ in range(depth):
        layers.append(LUConv(spatial_dims, nchan, act, bias))
    return nn.Sequential(*layers)


class InputTransition(nn.Module):

    def __init__(
        self, spatial_dims: int, in_channels: int, out_channels: int, act: tuple[str, dict] | str, bias: bool = False
    ):
        super().__init__()

        if out_channels % in_channels != 0:
            raise ValueError(
                f"out channels should be divisible by in_channels. Got in_channels={in_channels}, out_channels={out_channels}."
            )

        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act_function = get_acti_layer(act, out_channels)
        self.conv_block = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=5,
            act=None,
            norm=Norm.BATCH,
            bias=bias,
        )

    def forward(self, x):
        out = self.conv_block(x)
        repeat_num = self.out_channels // self.in_channels
        x16 = x.repeat([1, repeat_num, 1, 1, 1][: self.spatial_dims + 2])
        out = self.act_function(torch.add(out, x16))
        return out


class DownTransition(nn.Module):

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        nconvs: int,
        act: tuple[str, dict] | str,
        dropout_prob: float | None = None,
        dropout_dim: int = 3,
        bias: bool = False,
    ):
        super().__init__()

        conv_type: type[nn.Conv2d | nn.Conv3d] = Conv[Conv.CONV, spatial_dims]
        norm_type: type[nn.BatchNorm2d | nn.BatchNorm3d] = Norm[Norm.BATCH, spatial_dims]
        dropout_type: type[nn.Dropout | nn.Dropout2d | nn.Dropout3d] = Dropout[Dropout.DROPOUT, dropout_dim]

        out_channels = 2 * in_channels
        self.down_conv = conv_type(in_channels, out_channels, kernel_size=2, stride=2, bias=bias)
        self.bn1 = norm_type(out_channels)
        self.act_function1 = get_acti_layer(act, out_channels)
        self.act_function2 = get_acti_layer(act, out_channels)
        self.ops = _make_nconv(spatial_dims, out_channels, nconvs, act, bias)
        self.dropout = dropout_type(dropout_prob) if dropout_prob is not None else None

    def forward(self, x):
        down = self.act_function1(self.bn1(self.down_conv(x)))
        if self.dropout is not None:
            out = self.dropout(down)
        else:
            out = down
        out = self.ops(out)
        att = torch.add(out, down)
        out = self.act_function2(torch.add(out, down))
        return out, att


class UpTransition(nn.Module):

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        nconvs: int,
        act: tuple[str, dict] | str,
        dropout_prob: tuple[float | None, float] = (None, 0.5),
        dropout_dim: int = 3,
    ):
        super().__init__()

        conv_trans_type: type[nn.ConvTranspose2d | nn.ConvTranspose3d] = Conv[Conv.CONVTRANS, spatial_dims]
        norm_type: type[nn.BatchNorm2d | nn.BatchNorm3d] = Norm[Norm.BATCH, spatial_dims]
        dropout_type: type[nn.Dropout | nn.Dropout2d | nn.Dropout3d] = Dropout[Dropout.DROPOUT, dropout_dim]

        self.up_conv = conv_trans_type(in_channels, out_channels // 2, kernel_size=2, stride=2)
        self.bn1 = norm_type(out_channels // 2)
        self.dropout = dropout_type(dropout_prob[0]) if dropout_prob[0] is not None else None
        self.dropout2 = dropout_type(dropout_prob[1])
        self.act_function1 = get_acti_layer(act, out_channels // 2)
        self.act_function2 = get_acti_layer(act, out_channels)
        self.ops = _make_nconv(spatial_dims, out_channels, nconvs, act)

    def forward(self, x, skipx):
        if self.dropout is not None:
            out = self.dropout(x)
        else:
            out = x
        skipxdo = self.dropout2(skipx)
        out = self.act_function1(self.bn1(self.up_conv(out)))
        xcat = torch.cat((out, skipxdo), 1)
        out = self.ops(xcat)
        out = self.act_function2(torch.add(out, xcat))
        return out


class OutputTransition(nn.Module):

    def __init__(
        self, spatial_dims: int, in_channels: int, out_channels: int, act: tuple[str, dict] | str, bias: bool = False
    ):
        super().__init__()

        conv_type: type[nn.Conv2d | nn.Conv3d] = Conv[Conv.CONV, spatial_dims]

        self.act_function1 = get_acti_layer(act, out_channels)
        self.conv_block = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=5,
            act=None,
            norm=Norm.BATCH,
            bias=bias,
        )
        self.conv2 = conv_type(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # convolve 32 down to 2 channels
        out = self.conv_block(x)
        out = self.act_function1(out)
        out = self.conv2(out)
        return out


#dense模块
class _DenseLayer(nn.Module):

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        growth_rate: int,
        bn_size: int,
        dropout_prob: float,
        act: str | tuple = ("relu", {"inplace": True}),
        norm: str | tuple = "batch",
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions of the input image.
            in_channels: number of the input channel.
            growth_rate: how many filters to add each layer (k in paper).
            bn_size: multiplicative factor for number of bottle neck layers.
                (i.e. bn_size * k features in the bottleneck layer)
            dropout_prob: dropout rate after each dense layer.
            act: activation type and arguments. Defaults to relu.
            norm: feature normalization type and arguments. Defaults to batch norm.
        """
        super().__init__()

        out_channels = bn_size * growth_rate
        conv_type: Callable = Conv[Conv.CONV, spatial_dims]
        dropout_type: Callable = Dropout[Dropout.DROPOUT, spatial_dims]

        self.layers = nn.Sequential()

        self.layers.add_module("norm1", get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels))
        self.layers.add_module("relu1", get_act_layer(name=act))
        self.layers.add_module("conv1", conv_type(in_channels, out_channels, kernel_size=1, bias=False))

        self.layers.add_module("norm2", get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=out_channels))
        self.layers.add_module("relu2", get_act_layer(name=act))
        self.layers.add_module("conv2", conv_type(out_channels, growth_rate, kernel_size=3, padding=1, bias=False))

        if dropout_prob > 0:
            self.layers.add_module("dropout", dropout_type(dropout_prob))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        new_features = self.layers(x)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):

    def __init__(
        self,
        spatial_dims: int,
        layers: int,
        in_channels: int,
        bn_size: int,
        growth_rate: int,
        dropout_prob: float,
        act: str | tuple = ("relu", {"inplace": True}),
        norm: str | tuple = "batch",
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions of the input image.
            layers: number of layers in the block.
            in_channels: number of the input channel.
            bn_size: multiplicative factor for number of bottle neck layers.
                (i.e. bn_size * k features in the bottleneck layer)
            growth_rate: how many filters to add each layer (k in paper).
            dropout_prob: dropout rate after each dense layer.
            act: activation type and arguments. Defaults to relu.
            norm: feature normalization type and arguments. Defaults to batch norm.
        """
        super().__init__()
        for i in range(layers):
            layer = _DenseLayer(spatial_dims, in_channels, growth_rate, bn_size, dropout_prob, act=act, norm=norm)
            in_channels += growth_rate
            self.add_module("denselayer%d" % (i + 1), layer)


class _Transition(nn.Sequential):

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        act: str | tuple = ("relu", {"inplace": True}),
        norm: str | tuple = "batch",
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions of the input image.
            in_channels: number of the input channel.
            out_channels: number of the output classes.
            act: activation type and arguments. Defaults to relu.
            norm: feature normalization type and arguments. Defaults to batch norm.
        """
        super().__init__()

        conv_type: Callable = Conv[Conv.CONV, spatial_dims]
        pool_type: Callable = Pool[Pool.AVG, spatial_dims]

        self.add_module("norm", get_norm_layer(name=norm, spatial_dims=spatial_dims, channels=in_channels))
        self.add_module("relu", get_act_layer(name=act))
        self.add_module("conv", conv_type(in_channels, out_channels, kernel_size=1, bias=False))
        self.add_module("pool", pool_type(kernel_size=2, stride=2))

                
        
# diy model    
class MVF_Net(nn.Module):
    def __init__(
            self,
            spatial_dims: int,
            num_classes: int,
            act: tuple[str, dict] | str = ("elu", {"inplace": True}),
            dropout_prob_down: float | None = 0.5,
            dropout_prob_up: tuple[float | None, float] = (0.5, 0.5),
            dropout_dim: int = 3,
            bias: bool = False,
            kernel_size: Union[Sequence[int], int] = 3,
            **kwargs) -> None:
        super().__init__()

        conv_type: Callable = Conv[Conv.CONV, 3]
        
        #inception block
        #axial view
        self.inc1 = Inception(in_c=1, c1=8, c2=[8,8], c3=[8,12,12], c4=4)
        
        #coronal view
        self.inc2 = Inception(in_c=1, c1=8, c2=[8,8], c3=[8,12,12], c4=4)
        
        #sagittal view
        self.inc3 = Inception(in_c=1, c1=8, c2=[8,8], c3=[8,12,12], c4=4)
        
        #fusion
        self.att1 = CPCABlock("FC", channels = 32, ratio = 4)
        self.att2 = CPCABlock("FC", channels = 32, ratio = 4)
        self.att3 = CPCABlock("FC", channels = 32, ratio = 4)
        self.weights1 = nn.Parameter(torch.tensor([1.0, 0.0, 0.0]))
        self.weights2 = nn.Parameter(torch.tensor([0.0, 0.5, 0.5]))
        
        #分割
        self.seg_in = InputTransition(spatial_dims, 32, 32, act, bias=bias)
        self.seg_down64 = DownTransition(spatial_dims, 32, 1, act, bias=bias)
        self.seg_down128 = DownTransition(spatial_dims, 64, 2, act, bias=bias)
        self.seg_down256 = DownTransition(spatial_dims, 128, 3, act, dropout_prob=dropout_prob_down, bias=bias)
        self.seg_down512 = DownTransition(spatial_dims, 256, 2, act, dropout_prob=dropout_prob_down, bias=bias)
        self.seg_up512 = UpTransition(spatial_dims, 512, 512, 2, act, dropout_prob=dropout_prob_up)
        self.seg_up256 = UpTransition(spatial_dims, 512, 256, 2, act, dropout_prob=dropout_prob_up)
        self.seg_up128 = UpTransition(spatial_dims, 256, 128, 1, act)
        self.seg_up64 = UpTransition(spatial_dims, 128, 64, 1, act)
        self.seg_out = OutputTransition(spatial_dims, 64, 2, act, bias=bias)

        
        #分类
        conv_type: type[nn.Conv1d | nn.Conv2d | nn.Conv3d] = Conv[Conv.CONV, spatial_dims]
        pool_type: type[nn.MaxPool1d | nn.MaxPool2d | nn.MaxPool3d] = Pool[Pool.MAX, spatial_dims]
        avg_pool_type: type[nn.AdaptiveAvgPool1d | nn.AdaptiveAvgPool2d | nn.AdaptiveAvgPool3d] = Pool[
            Pool.ADAPTIVEAVG, spatial_dims]
        self.clf_in = conv_type(32, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.clf_dense1 = _DenseBlock(spatial_dims=spatial_dims, layers=6, in_channels=64, bn_size=4, growth_rate=32,
                dropout_prob=0.2, act=("relu", {"inplace": True}), norm="batch")
        self.clf_trans1 = _Transition(spatial_dims, in_channels=256, out_channels=128, act=("relu", {"inplace": True}), norm="batch")
        self.clf_dense2 = _DenseBlock(spatial_dims=spatial_dims, layers=12, in_channels=128, bn_size=4, growth_rate=32,
                dropout_prob=0.2, act=("relu", {"inplace": True}), norm="batch")
        self.clf_trans2 = _Transition(spatial_dims, in_channels=512, out_channels=256, act=("relu", {"inplace": True}), norm="batch")
        self.clf_dense3 = _DenseBlock(spatial_dims=spatial_dims, layers=24, in_channels=256, bn_size=4, growth_rate=32,
                dropout_prob=0.2, act=("relu", {"inplace": True}), norm="batch")
        self.clf_trans3 = _Transition(spatial_dims, in_channels=1024, out_channels=512, act=("relu", {"inplace": True}), norm="batch")
        self.clf_dense4 = _DenseBlock(spatial_dims=spatial_dims, layers=16, in_channels=512, bn_size=4, growth_rate=32,
                dropout_prob=0.2, act=("relu", {"inplace": True}), norm="batch")
        self.clf_fc = nn.Sequential(get_norm_layer(name="batch", spatial_dims=spatial_dims, channels=1024),
                                   get_act_layer(name=("relu", {"inplace": True})),
                                   avg_pool_type(1), nn.Flatten(1), nn.Linear(1024, num_classes))
        
        
        for m in self.modules():
            if isinstance(m, conv_type):
                nn.init.kaiming_normal_(torch.as_tensor(m.weight))
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                nn.init.constant_(torch.as_tensor(m.weight), 1)
                nn.init.constant_(torch.as_tensor(m.bias), 0)
            # elif isinstance(m, nn.Linear):
            #     nn.init.constant_(torch.as_tensor(m.bias), 0)
    
    
    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):

        #三视图
        x1 = x #axial view
        x2 = x.permute(0,1,4,2,3) #coronal view
        x3 = x.permute(0,1,4,3,2) #sagittal view
        
        #3D inception block
        #x1
        x1 = self.inc1(x1)
        #x2
        x2 = self.inc2(x2)
        #x3
        x3 = self.inc3(x3) # (B, 32, 96, 96, 96)
        
        #特征融合
        x1 = self.att1(x1)
        x2 = self.att2(x2)
        x3 = self.att3(x3)
        x2 = x2.permute(0,1,3,4,2)
        x3 = x3.permute(0,1,4,3,2)
        w11, w12, w13 = torch.softmax(self.weights1, dim=0)
        w21, w22, w23 = torch.softmax(self.weights2, dim=0)
        x_seg = w11*x1 + w12*x2 + w13*x3
        x_clf = w21*x1 + w22*x2 + w23*x3
        
        
        #分割分类
        #initial
        seg_out1 = self.seg_in(x_seg)
        clf_out = self.clf_in(x_clf)
        #block1
        seg_out2, att = self.seg_down64(seg_out1)
        atten_out = torch.softmax(att, dim=1)
        clf_out = clf_out*atten_out
        #block2
        seg_out3, att = self.seg_down128(seg_out2)
        atten_out = torch.softmax(att, dim=1)
        clf_out = self.clf_dense1(clf_out)
        clf_out = self.clf_trans1(clf_out)
        clf_out = clf_out*atten_out
        #block3
        seg_out4, att = self.seg_down256(seg_out3)
        atten_out = torch.softmax(att, dim=1)
        clf_out = self.clf_dense2(clf_out)
        clf_out = self.clf_trans2(clf_out)
        clf_out = clf_out*atten_out
        #block4
        seg_out5, att = self.seg_down512(seg_out4)
        atten_out = torch.softmax(att, dim=1)
        clf_out = self.clf_dense3(clf_out)
        clf_out = self.clf_trans3(clf_out)
        clf_out = clf_out*atten_out
        #解码器
        seg_result = self.seg_up512(seg_out5, seg_out4)
        seg_result = self.seg_up256(seg_result, seg_out3)
        seg_result = self.seg_up128(seg_result, seg_out2)
        seg_result = self.seg_up64(seg_result, seg_out1)
        seg_result = self.seg_out(seg_result)
        
        clf_out = self.clf_dense4(clf_out)
        clf_result = self.clf_fc(clf_out)
        
        
        return seg_result, clf_result
   