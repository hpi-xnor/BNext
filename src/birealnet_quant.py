import torch
import torch.nn as nn
from torchinfo import summary
import torchvision
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from utils_quant import *

def conv3x3(in_planes, out_planes, kernel_size = 3, stride=1, groups = 1, dilation = 1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=kernel_size//2, dilation = dilation, groups = groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding = 0, bias=False)

class firstconv3x3(nn.Module):
    def __init__(self, inp, oup, stride, quant = False):
        super(firstconv3x3, self).__init__()
        if quant:
            self.conv1 = nn.Conv2d(inp, oup, 3, stride, 1, bias=False)
        else:
            self.conv1 = QuantizeConv(inp, oup, 3, stride, 1, 1, 1, False, activation_bits = 8, weight_bits = 8) 

        self.bn1 = nn.BatchNorm2d(oup)
        self.prelu = nn.PReLU(oup, oup)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.prelu(out)

        return out

class BinaryActivation(nn.Module):
    def __init__(self):
        super(BinaryActivation, self).__init__()

    def forward(self, x):
        out_forward = torch.sign(x)
        mask1 = x < -1
        mask2 = x < 0
        mask3 = x < 1
        out1 = (-1) * mask1.type(torch.float32) + (x*x + 2*x) * (1-mask1.type(torch.float32))
        out2 = out1 * mask2.type(torch.float32) + (-x*x + 2*x) * (1-mask2.type(torch.float32))
        out3 = out2 * mask3.type(torch.float32) + 1 * (1- mask3.type(torch.float32))
        out = out_forward.detach() - out3.detach() + out3

        return out

class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1,out_chn,1,1), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out


class HardSign(nn.Module):
    def __init__(self, range=[-1, 1], progressive=False):
        super(HardSign, self).__init__()
        self.range = range
        self.progressive = progressive
        self.register_buffer("temperature", torch.ones(1))

    def adjust(self, x, scale=0.1):
        self.temperature.mul_(scale)

    def forward(self, x, scale=None):
        if scale == None:
            scale = torch.ones_like(x)

        replace = x.clamp(self.range[0], self.range[1]) + scale
        x = x.div(self.temperature.clamp(min=1e-8)).clamp(-1, 1)
        if not self.progressive:
            sign = x.sign() * scale
        else:
            sign = x * scale
        return (sign - replace).detach() + replace


class HardBinaryConv(nn.Module):
    def __init__(self, in_chn, out_chn, kernel_size=3, stride=1, padding=1, groups=1):
        super(HardBinaryConv, self).__init__()
        self.stride = stride
        self.padding = kernel_size // 2
        self.groups = groups
        self.number_of_weights = in_chn // groups * out_chn * kernel_size * kernel_size
        self.shape = (out_chn, in_chn // groups, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.randn((self.shape)) * 0.001, requires_grad=True)
        # self.weight_bias = nn.Parameter(torch.zeros(out_chn, in_chn, 1, 1))
        self.register_buffer("temperature", torch.ones(1))

    def forward(self, x):
        self.weight.data.clamp_(-1.5, 1.5)
        real_weights = self.weight

        binary_weights_no_grad = (real_weights / self.temperature.clamp(min=1e-8)).clamp(-1, 1)
        if self.temperature < 1e-5:
            binary_weights_no_grad = binary_weights_no_grad.sign()
        cliped_weights = real_weights
        binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        y = F.conv2d(x, binary_weights, stride=self.stride, padding=self.padding, groups=self.groups)

        return y


class SqueezeAndExpand(nn.Module):
    def __init__(self, channels, planes, ratio=8, attention_mode="hard_sigmoid", quant = False, bits = 8):
        super(SqueezeAndExpand, self).__init__()
        if not quant:
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Conv2d(channels, channels // ratio, kernel_size=1, padding=0),
                nn.ReLU(channels // ratio),
                nn.Conv2d(channels // ratio, planes, kernel_size=1, padding=0),
            )
        else:
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d((1,1)),
                QuantizeConv(channels, channels // ratio, 1, 1, 0, 1, 1, True, activation_bits = bits*2, weight_bits = bits),
                nn.ReLU(),
                 QuantizeConv(channels //ratio, channels, 1, 1, 0, 1, 1, True, activation_bits = bits*2, weight_bits = bits),
            )
        if attention_mode == "sigmoid":
            self.attention = nn.Sigmoid()

        elif attention_mode == "hard_sigmoid":
            self.attention = HardSigmoid()

        else:
            self.attention = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.se(x)
        x = self.attention(x)
        return x


class Attention(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0.1, gamma=1e-6, groups=1, quant = False, se_bits = 8):
        super(Attention, self).__init__()

        self.inplanes = inplanes
        self.planes = planes

        self.move = LearnableBias(inplanes)
        self.binary_activation = HardSign(range=[-1.5, 1.5])
        self.binary_conv = HardBinaryConv(inplanes, planes, kernel_size=3, stride=stride, groups=groups)

        self.norm1 = nn.BatchNorm2d(planes)
        self.norm2 = nn.BatchNorm2d(planes)

        self.dropout_pre = nn.Dropout2d(drop_rate) if (drop_rate > 0 and planes >= 384) else nn.Identity()
        self.dropout_aft = nn.Dropout2d(drop_rate) if (drop_rate > 0 and planes >= 384) else nn.Identity()

        self.activation1 = nn.PReLU(inplanes)
        self.activation2 = nn.PReLU(planes)

        self.downsample = downsample
        self.stride = stride
        if stride == 2:
            self.pooling = nn.AvgPool2d(2, 2)

        self.se = SqueezeAndExpand(planes, planes, attention_mode="sigmoid", quant = quant, bits = se_bits)
        self.scale = nn.Parameter(torch.ones(1, planes, 1, 1) * 0.5)

    def forward(self, input):
        if self.training:
            self.scale.data.clamp_(0, 1)

        residual = self.activation1(input)

        if self.downsample is not None:
            residual = self.downsample(residual)

        x = self.move(input)
        x = self.binary_activation(x, scale=None)
        x = self.binary_conv(x)
        x = self.norm1(x)
        x = self.activation2(x)

        mix = self.scale * residual + x * (1 - self.scale)
        x = self.se(mix) * x

        x = x * residual
        x = self.norm2(x)
        x = x + residual

        return x


class FFN_3x3(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0.1, gamma=1e-8, groups=1, quant = False, se_bits = 8):
        super(FFN_3x3, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride

        self.move = LearnableBias(inplanes)
        self.binary_activation = HardSign(range=[-1.5, 1.5])
        self.binary_conv = HardBinaryConv(inplanes, planes, kernel_size=3, stride=stride, groups=groups)

        self.norm1 = nn.BatchNorm2d(planes)
        self.norm2 = nn.BatchNorm2d(planes)

        self.activation1 = nn.PReLU(inplanes)
        self.activation2 = nn.PReLU(planes)

        self.dropout_pre = nn.Dropout2d(drop_rate) if (drop_rate > 0 and planes >= 384) else nn.Identity()
        self.dropout_aft = nn.Dropout2d(drop_rate) if (drop_rate > 0 and planes >= 384) else nn.Identity()

        if stride == 2:
            self.pooling = nn.AvgPool2d(2, 2)

        self.downsample = downsample

        self.se = SqueezeAndExpand(inplanes, planes, attention_mode="sigmoid", quant = quant, bits = se_bits)
        self.scale = nn.Parameter(torch.ones(1, planes, 1, 1) * 0.5)

    def forward(self, input):
        self.scale.data.clamp_(0, 1)

        residual = input

        if self.stride == 2:
            residual = self.downsample(residual)

        x = self.move(input)
        x = self.binary_activation(x, scale=None)
        x = self.binary_conv(x)
        x = self.norm1(x)
        x = self.activation2(x)
        mix = self.scale * residual + (1 - self.scale) * x
        x = self.se(mix) * x

        x = self.norm2(x)
        x = x + residual

        return x

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate = 0.1, mode = "scale", quant = False, se_bits = 8):
        super(BasicBlock, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        
        self.Attention = Attention(inplanes, planes, stride, downsample, drop_rate = drop_rate, groups = 1, quant = quant, se_bits = se_bits)
         
        self.FFN  = FFN_3x3(planes, planes, 1, None, drop_rate = drop_rate, groups = 1, quant = quant, se_bits = se_bits)    
          
    def forward(self, input):
        x = self.Attention(input)
        y = self.FFN(x)

        return y
      

class birealnet18(nn.Module):
    def __init__(self, num_classes=1000, block = BasicBlock, layers = [2, 2, 2, 2], quant = 8, se_bits = 8):
        super(birealnet18, self).__init__()
        drop_rate = 0.2 if num_classes == 100 else 0.
        width = 1
        self.inplanes = 64

        if num_classes == 1000:
            if not quant:
                self.conv1 = nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
            else:
                self.conv1 = QuantizeConv(3, 64, 7, 2, 3, 1, 1, False, activation_bits = 8, weight_bits = 8, clip_val = 1000)
            self.bn1 = nn.BatchNorm2d(64)
            self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
            self.maxpool = lambda x:x

        self.layer0 = self._make_layer(block, int(width*64), layers[0], quant = quant, se_bits = se_bits)
        self.layer1 = self._make_layer(block, int(width*128), layers[1], stride=2, quant = quant, se_bits = se_bits)
        self.layer2 = self._make_layer(block, int(width*256), layers[2], stride=2, quant = quant, se_bits = se_bits)
        self.layer3 = self._make_layer(block, int(width*512), layers[3], stride=2, quant = quant, se_bits = se_bits)

        self.prelu = nn.PReLU(512)
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        if not quant:
            self.fc = nn.Linear(512, num_classes)
        else:
            self.fc = QuantizeLinear(512, num_classes, activation_bits = 8, weight_bits = 8)

    def _make_layer(self, block, planes, blocks, stride=1, quant = False, se_bits = 8):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if not quant:
                downsample = nn.Sequential(
                                nn.AvgPool2d(kernel_size=2, stride=stride),
                                conv1x1(self.inplanes, planes * block.expansion),
                                nn.BatchNorm2d(planes * block.expansion))
            else:
                downsample = nn.Sequential(
                                nn.AvgPool2d(kernel_size=2, stride=stride),
                                QuantizeConv(self.inplanes, planes*block.expansion, 1, 1, 0, 1, 1, True, activation_bits = 8, weight_bits = 8, clip_val = 1000),
                                nn.BatchNorm2d(planes * block.expansion)
                                )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, drop_rate = 0, quant = quant, se_bits = se_bits))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, drop_rate = 0, quant = quant, se_bits = se_bits))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)
        
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.prelu(x)
        x = self.pool1(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x




if __name__ == "__main__":
    input = torch.randn(1, 3, 224, 224).cuda()
    model = birealnet18(num_classes = 1000, quant = True).cuda()
    print(model(input).size())
    #summary(model, input_size=(1, 3, 224, 224))
    
