import torch
import torch.nn as nn
from torchinfo import summary
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from timm.models.layers import trunc_normal_, DropPath
from utils_quant import *

# stage ratio: 1:1:3:1
stage_out_channel_tiny = [32] + [64] + [128] * 2 + [256] * 2 + [512] * 6 + [1024] * 2

# stage ratio 1:1:3:1
stage_out_channel_small = [48] + [96] + [192] * 2 + [384] * 2 + [768] * 6 + [1536] * 2

# stage ratio 2:2:4:2
stage_out_channel_middle = [48] + [96] + [192] * 4 + [384] * 4 + [768] * 8 + [1536] * 4

# stage ratio 2:2:4:2
#stage_out_channel_large = [48] + [96] + [192] * 4 + [384] * 4 + [768] * 16 + [1536] * 4

# stage ratio 2:2:8:2
stage_out_channel_super = [64] + [128] + [256] * 4 + [512] * 4 + [1024] * 16 + [2048] * 4


def conv3x3(in_planes, out_planes, kernel_size=3, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=kernel_size // 2, dilation=dilation, groups=groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class HardSigmoid(nn.Module):
    def __init__(self, ):
        super(HardSigmoid, self).__init__()

    def forward(self, x):
        return F.relu6(x + 3) / 6


class firstconv3x3(nn.Module):
    def __init__(self, inp, oup, stride, quant = False):
        super(firstconv3x3, self).__init__()

        if not quant:
            self.conv1 = nn.Conv2d(inp, oup, 3, stride, 1, bias=False)
        else:
            self.conv1 = QuantizeConv(inp, oup, 3, stride, 1, 1, 1, False, activation_bits=8, weight_bits=8,
                                      clip_val=1000)

        self.bn1 = nn.BatchNorm2d(oup)
        self.prelu = nn.PReLU(oup, oup)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        
        return out

class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1, out_chn, 1, 1), requires_grad=True)

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

    def forward(self, x):
        
        replace = x.clamp(self.range[0], self.range[1])
        x = x.div(self.temperature.clamp(min=1e-8)).clamp(-1, 1)
        if not self.progressive:
            sign = x.sign() 
        else:
            sign = x
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
        self.register_buffer("temperature", torch.ones(1))

    def forward(self, x):
        if self.training:
            self.weight.data.clamp_(-1.5, 1.5)
        
        real_weights = self.weight

        if self.temperature < 1e-7:
            binary_weights_no_grad = real_weights.sign()
        else:
            binary_weights_no_grad = (real_weights / self.temperature.clamp(min=1e-8)).clamp(-1, 1)
        cliped_weights = real_weights
        
        if self.training:
            binary_weights = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        else:
            binary_weights = binary_weights_no_grad

        y = F.conv2d(x, binary_weights, stride=self.stride, padding=self.padding, groups=self.groups)

        return y


class SqueezeAndExpand(nn.Module):
    def __init__(self, channels, planes, ratio=8, attention_mode="hard_sigmoid", quant = False, bits = 8):
        super(SqueezeAndExpand, self).__init__()
        if not quant:
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d((1,1)),
                nn.Conv2d(channels, channels // ratio, kernel_size=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(channels // ratio, planes, kernel_size=1, padding=0),
            )

        else:
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                QuantizeConv(channels, channels // ratio, 1, 1, 0, 1, 1, True, activation_bits=bits, weight_bits=bits),
                nn.ReLU(),
                QuantizeConv(channels // ratio, channels, 1, 1, 0, 1, 1, True, activation_bits=bits, weight_bits=bits)

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

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0.1, groups=1, quant = False, se_bits = 32):
        super(Attention, self).__init__()

        self.inplanes = inplanes
        self.planes = planes

        self.move = LearnableBias(inplanes)
        self.binary_activation = HardSign(range=[-1.5, 1.5])
        self.binary_conv = HardBinaryConv(inplanes, planes, kernel_size=3, stride=stride, groups=groups)

        self.norm1 = nn.BatchNorm2d(planes)
        self.norm2 = nn.BatchNorm2d(planes)

        self.activation1 = nn.PReLU(inplanes)
        self.activation2 = nn.PReLU(planes)

        self.downsample = downsample
        self.stride = stride
        if stride == 2:
            self.pooling = nn.AvgPool2d(2, 2)

        self.se = SqueezeAndExpand(planes, planes, attention_mode="sigmoid", quant=quant, bits = se_bits)
        self.scale = nn.Parameter(torch.ones(1, planes, 1, 1) * 0.5)

    def forward(self, input):

        residual = self.activation1(input)

        if self.stride == 2:
            residual = self.pooling(residual)

        x = self.move(input)
        x = self.binary_activation(x)
        x = self.binary_conv(x)
        x = self.norm1(x)
        x = self.activation2(x)

        if self.training:
            scale = self.scale.data.clamp_(0, 1)
        else:
            scale = self.scale
        
        if self.stride == 2:
            input = self.pooling(input)
        
        mix = input* scale + x * (1 - scale)
        x = self.se(mix) * x
        x = x * residual
        x = self.norm2(x)
        x = x + residual

        return x


class FFN_3x3(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0.1, groups=1, quant = False, se_bits = 32):
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

        if stride == 2:
            self.pooling = nn.AvgPool2d(2, 2)

        self.se = SqueezeAndExpand(inplanes, planes, attention_mode="sigmoid", quant=quant, bits = se_bits)
        self.scale = nn.Parameter(torch.ones(1, planes, 1, 1) * 0.5)

    def forward(self, input):

        residual = input

        if self.stride == 2:
            residual = self.pooling(residual)

        x = self.move(input)
        x = self.binary_activation(x)
        x = self.binary_conv(x)
        x = self.norm1(x)
        x = self.activation2(x)

        if self.training:
            scale = self.scale.data.clamp_(0, 1)
        else:
            scale = self.scale
        
        if self.stride == 2:
            input = self.pooling(input)

        mix = input * scale + x * (1 - scale)
        x = self.se(mix) * x
        x = self.norm2(x)

        x = x + residual

        return x


class FFN_1x1(nn.Module):
    def __init__(self, inplanes, planes, stride=1, drop_rate=0.1, quant = False, se_bits = 32):
        super(FFN_1x1, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride

        self.move = LearnableBias(inplanes)
        self.binary_activation = HardSign(range=[-1.5, 1.5])
        self.binary_conv = HardBinaryConv(inplanes, planes, kernel_size=1, stride=stride, padding=0)

        self.norm1 = nn.BatchNorm2d(planes)
        self.norm2 = nn.BatchNorm2d(planes)

        self.activation1 = nn.PReLU(inplanes)
        self.activation2 = nn.PReLU(planes)

        if stride == 2:
            self.pooling = nn.AvgPool2d(2, 2)

        self.se = SqueezeAndExpand(inplanes, planes, attention_mode="sigmoid", quant = quant, bits = se_bits)
        self.scale = nn.Parameter(torch.ones(1, planes, 1, 1) * 0.5)

    def forward(self, input):

        residual = input
        if self.stride == 2:
            residual = self.pooling(residual)

        x = self.move(input)
        x = self.binary_activation(x)
        x = self.binary_conv(x)
        x = self.norm1(x)
        x = self.activation2(x)

        if self.training:
            scale = self.scale.data.clamp_(0, 1)
        else:
            scale = self.scale
        mix = input * scale + x * (1 - scale)
        x = self.se(mix) * x
        x = self.norm2(x)
        x = x + residual

        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0.1, mode="scale", quant = False, se_bits = 32):
        super(BasicBlock, self).__init__()
        self.inplanes = inplanes
        self.planes = planes

        if mode == "scale":
            self.Attention = Attention(inplanes, inplanes, stride, None, drop_rate=drop_rate, groups=1, quant = quant, se_bits = se_bits)
        else:
            self.Attention = FFN_3x3(inplanes, inplanes, stride, None, drop_rate=drop_rate, groups=1, quant = quant, se_bits = se_bits)

        if inplanes == planes:
            self.FFN = FFN_1x1(inplanes, inplanes, drop_rate=drop_rate, quant = quant, se_bits = se_bits)

        else:
            self.FFN_1 = FFN_1x1(inplanes, inplanes, drop_rate=drop_rate, quant = quant, se_bits = se_bits)

            self.FFN_2 = FFN_1x1(inplanes, inplanes, drop_rate=drop_rate, quant = quant, se_bits = se_bits)

    def forward(self, input):
        x = self.Attention(input)

        if self.inplanes == self.planes:
            y = self.FFN(x)

        else:
            y_1 = self.FFN_1(x)
            y_2 = self.FFN_2(x)
            y = torch.cat((y_1, y_2), dim=1)

        return y


class BNext(nn.Module):
    def __init__(self, num_classes=1000, size="tiny", quant = False, se_bits = 32):
        super(BNext, self).__init__()
        drop_rate = 0.2 if num_classes == 100 else 0.0

        if size == "tiny":
            stage_out_channel = stage_out_channel_tiny
        elif size == "small":
            stage_out_channel = stage_out_channel_small
        elif size == "middle":
            stage_out_channel = stage_out_channel_middle
        elif size == "large":
            stage_out_channel = stage_out_channel_large
        elif size == "super":
            stage_out_channel = stage_out_channel_super
        else:
            raise ValueError("The size is not defined!")

        self.feature = nn.ModuleList()
        drop_rates = [x.item() for x in torch.linspace(0, drop_rate, (len(stage_out_channel)))]
        for i in range(len(stage_out_channel)):
            if i == 0:
                self.feature.append(firstconv3x3(3, stage_out_channel[i], 1 if num_classes != 1000 else 2, quant = quant))
            elif i == 1:
                self.feature.append((BasicBlock(stage_out_channel[i - 1], stage_out_channel[i], 1,
                                                drop_rate=drop_rates[i], mode="bias", quant = quant, se_bits = se_bits)))
            elif stage_out_channel[i - 1] != stage_out_channel[i] and stage_out_channel[i] != stage_out_channel[1]:
                self.feature.append(
                    BasicBlock(stage_out_channel[i - 1], stage_out_channel[i], 2, drop_rate=drop_rates[i],
                               mode="scale" if i % 2 == 0 else "bias", quant = quant, se_bits = se_bits))
            else:
                self.feature.append(
                    BasicBlock(stage_out_channel[i - 1], stage_out_channel[i], 1, drop_rate=drop_rates[i],
                               mode="scale" if i % 2 == 0 else "bias", quant = quant, se_bits = se_bits))

        self.prelu = nn.PReLU(stage_out_channel[-1])
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        if not quant:
            self.fc = nn.Linear(stage_out_channel[-1], num_classes)
        else:
            self.fc = QuantizeLinear(stage_out_channel[-1], num_classes, activation_bits=8, weight_bits=8)

    def forward(self, x):
        for i, block in enumerate(self.feature):
            x = block(x)
        x = self.prelu(x)
        x = self.pool1(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


if __name__ == "__main__":
    model = nn.DataParallel(BNext(num_classes=1000, size="super", quant=False)).cpu()
    model_dir = "/data/scratch/nianhui.guo/BitReMixNet/mobilenet/USI_Version/models/ImageNet_reactnet_super_optimizer_AdamW_mixup_0.0_cutmix_0.0_aug_repeats_1_KD_4_assistant_3_EfficientNet_B0_HK_True_Instance_aa_rand-m7-mstd0.5-inc1__elm_True_recoup_True_0/checkpoint.pth.tar"
    checkpoints = torch.load(model_dir, map_location = "cpu")
    model.load_state_dict(checkpoints['state_dict'], strict = False)
    print(model.eval().cuda(0)(torch.randn(1, 3, 224, 224).cuda(0)))
    #summary(model, input_size=(1, 3, 224, 224))

