import torch
import torch.nn as nn
import pdb
import matplotlib.pyplot as plt
import seaborn as sns
import math
from torch.nn import Parameter
import torch.nn.functional as F
import numpy as np


class BinaryQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.sign(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input[0].ge(1)] = 0
        grad_input[input[0].le(-1)] = 0
        return grad_input


class ZMeanBinaryQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        out = torch.sign(input)
        out[out == -1] = 0
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input[0].ge(1)] = 0
        grad_input[input[0].le(-1)] = 0
        return grad_input


class SymQuantizer(torch.autograd.Function):
    """
        uniform quantization
    """

    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise, type=None):
        """
        :param ctx:
        :param input: tensor to be quantized
        :param clip_val: clip the tensor before quantization
        :param quant_bits: number of bits
        :return: quantized tensor
        """
        ctx.save_for_backward(input, clip_val)
        input = torch.where(input < clip_val[1], input, clip_val[1])
        input = torch.where(input > clip_val[0], input, clip_val[0])
        if layerwise:
            max_input = torch.max(torch.abs(input)).expand_as(input)
        else:
            if input.ndimension() <= 3:
                max_input = torch.max(torch.abs(input), dim=-1, keepdim=True)[0].expand_as(input).detach()
            elif input.ndimension() == 4:
                tmp = input.view(input.shape[0], input.shape[1], -1)
                max_input = torch.max(torch.abs(tmp), dim=-1, keepdim=True)[0].unsqueeze(-1).expand_as(input).detach()
            else:
                raise ValueError
        s = (2 ** (num_bits - 1) - 1) / max_input
        output = torch.round(input * s)
        # print(torch.max(output))
        output = output.div(s)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: saved non-clipped full-precision tensor and clip_val
        :param grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        input, clip_val = ctx.saved_tensors  # unclipped input
        grad_input = grad_output.clone()
        grad_input[input.ge(clip_val[1])] = 0
        grad_input[input.le(clip_val[0])] = 0
        return grad_input, None, None, None, None


class AsymQuantizer(torch.autograd.Function):
    """
        min-max quantization
    """

    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise, type=None):
        """
        :param ctx:
        :param input: tensor to be quantized
        :param clip_val: clip the tensor before quantization
        :param quant_bits: number of bits
        :return: quantized tensor
        """
        ctx.save_for_backward(input, clip_val)
        input = torch.where(input < clip_val[1], input, clip_val[1])
        input = torch.where(input > clip_val[0], input, clip_val[0])
        if layerwise:
            alpha = (input.max() - input.min()).detach()
            beta = input.min().detach()
        else:
            if input.ndimension() <= 3:
                alpha = (input.max(dim=-1, keepdim=True)[0] - input.min(dim=-1, keepdim=True)[0]).expand_as(
                    input).detach()
                beta = input.min(dim=-1, keepdim=True)[0].expand_as(input).detach()
            elif input.ndimension() == 4:
                tmp = input.view(input.shape[0], input.shape[1], -1)
                alpha = (tmp.max(dim=-1, keepdim=True)[0].unsqueeze(-1) - \
                         tmp.min(dim=-1, keepdim=True)[0].unsqueeze(-1)).expand_as(input).detach()
                beta = tmp.min(dim=-1, keepdim=True)[0].unsqueeze(-1).expand_as(input).detach()
            else:
                raise ValueError
        input_normalized = (input - beta) / (alpha + 1e-8)
        s = (2 ** num_bits - 1)
        quant_input = torch.round(input_normalized * s)
        quant_input = quant_input.div(s)
        output = quant_input * (alpha + 1e-8) + beta

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: saved non-clipped full-precision tensor and clip_val
        :param grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        input, clip_val = ctx.saved_tensors  # unclipped input
        grad_input = grad_output.clone()
        grad_input[input.ge(clip_val[1])] = 0
        grad_input[input.le(clip_val[0])] = 0
        return grad_input, None, None, None, None


class TwnQuantizer(torch.autograd.Function):
    """Ternary Weight Networks (TWN)
    Ref: https://arxiv.org/abs/1605.04711
    """

    @staticmethod
    def forward(ctx, input, clip_val, num_bits, layerwise, type=None):
        """
        :param input: tensor to be ternarized
        :return: quantized tensor
        """
        ctx.save_for_backward(input, clip_val)
        input = torch.where(input < clip_val[1], input, clip_val[1])
        input = torch.where(input > clip_val[0], input, clip_val[0])
        if layerwise:
            m = input.norm(p=1).div(input.nelement())
            thres = 0.7 * m
            pos = (input > thres).float()
            neg = (input < -thres).float()
            mask = (input.abs() > thres).float()
            alpha = (mask * input).abs().sum() / mask.sum()
            result = alpha * pos - alpha * neg
        else:  # row-wise only for embed / weight
            n = input[0].nelement()
            m = input.data.norm(p=1, dim=1).div(n)
            thres = (0.7 * m).view(-1, 1).expand_as(input)
            pos = (input > thres).float()
            neg = (input < -thres).float()
            mask = (input.abs() > thres).float()
            alpha = ((mask * input).abs().sum(dim=1) / mask.sum(dim=1)).view(-1, 1)
            result = alpha * pos - alpha * neg

        return result

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: saved non-clipped full-precision tensor and clip_val
        :param grad_output: gradient ert the quantized tensor
        :return: estimated gradient wrt the full-precision tensor
        """
        input, clip_val = ctx.saved_tensors  # unclipped input
        grad_input = grad_output.clone()
        grad_input[input.ge(clip_val[1])] = 0
        grad_input[input.le(clip_val[0])] = 0
        return grad_input, None, None, None, None


class QuantizeConv(nn.Conv2d):
    def __init__(self, *kargs, activation_bits=8, weight_bits=8, clip_val=1000.0):
        super(QuantizeConv, self).__init__(*kargs)
        self.activation_bits = activation_bits
        self.activation_quantizer = AsymQuantizer if self.activation_bits != 1 else BinaryQuantizer

        self.weight_bits = weight_bits
        self.weight_quantizer = AsymQuantizer if self.weight_bits != 1 else BinaryQuantizer

        self.register_buffer("weight_clip_val", torch.tensor([-clip_val, clip_val]))
        self.register_buffer("activation_clip_val", torch.tensor([-clip_val, clip_val]))

    def forward(self, input):
        if self.weight_bits != 1:
            scaling_factor =2* torch.mean(torch.mean(torch.mean(abs(self.weight),dim=3,keepdim=True),dim=2,keepdim=True),dim=1,keepdim=True).detach()
            
            weight = self.weight/scaling_factor
            
            if self.weight.size(-1) == self.weight.size(-2) and self.weight.size(-1) == 1:
                weight = weight.squeeze(-1).squeeze(-1)
            else:
                weight = weight
            
            weight = self.weight_quantizer.apply(weight, self.weight_clip_val, self.weight_bits, False)

            if self.weight.size(-1) == self.weight.size(-2) and self.weight.size(-1) == 1:
                weight = weight.unsqueeze(-1).unsqueeze(-1)
            
            weight = weight * scaling_factor

        else:
            scaling_factor = torch.mean(abs(self.weight), dim=1, keepdim=True).detach()
            weight = self.weight_quantizer.apply(self.weight) * scaling_factor

        if not ((self.in_channels == 3 and self.activation_bits == 8) or self.activation_bits == 32):
            size = input.size()
            if size[-1] == 1:
                activation = input.squeeze(-1).squeeze(-1)
            else:
                activation = input
            activation = self.activation_quantizer.apply(activation, self.activation_clip_val, self.activation_bits,
                                                         False)

            if size[-1] == 1:
                activation = activation.unsqueeze(-1).unsqueeze(-1)
        else:
            activation = input

        out = nn.functional.conv2d(activation, weight, stride=self.stride, padding=self.padding, groups=self.groups,
                                   bias=self.bias)

        return out


class QuantizeLinear(nn.Linear):
    def __init__(self, *kargs, bias=True, activation_bits=8, weight_bits=8):
        super(QuantizeLinear, self).__init__(*kargs, bias=True)
        self.quantize_act = True
        self.weight_bits = 8
        self.quantize_act = True
        clip_val = 25
        if self.weight_bits == 2:
            self.weight_quantizer = TwnQuantizer
        elif self.weight_bits == 1:
            self.weight_quantizer = BinaryQuantizer
        else:
            self.weight_quantizer = AsymQuantizer
        self.register_buffer('weight_clip_val', torch.tensor([-clip_val, clip_val]))
        self.init = True

        if self.quantize_act:
            self.input_bits = activation_bits
            if self.input_bits == 1:
                self.act_quantizer = BinaryQuantizer
            elif self.input_bits == 2:
                self.act_quantizer = TwnQuantizer
            else:
                self.act_quantizer = AsymQuantizer
            self.register_buffer('act_clip_val', torch.tensor([-clip_val, clip_val]))
        self.register_parameter('scale', Parameter(torch.Tensor([0.0]).squeeze()))

    def forward(self, input, type=None):
        if self.weight_bits == 1:
            scaling_factor = torch.mean(abs(self.weight), dim=1, keepdim=True)
            scaling_factor = scaling_factor.detach()
            real_weights = self.weight - torch.mean(self.weight, dim=-1, keepdim=True)
            binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
            cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
            weight = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        else:
            scaling_factor = 2*torch.mean(abs(self.weight), dim=1,keepdim=True).detach()
            weight = self.weight/scaling_factor
            
            weight = self.weight_quantizer.apply(weight, self.weight_clip_val, self.weight_bits, False)*scaling_factor

        if self.input_bits == 1:
            binary_input_no_grad = torch.sign(input)
            cliped_input = torch.clamp(input, -1.0, 1.0)
            ba = binary_input_no_grad.detach() - cliped_input.detach() + cliped_input
        else:
            ba = self.act_quantizer.apply(input, self.act_clip_val, self.input_bits, False)

        out = nn.functional.linear(ba, weight)

        if not self.bias is None:
            out += self.bias.view(1, -1).expand_as(out)

        return out


class QuantizeEmbedding(nn.Embedding):
    def __init__(self, *kargs, padding_idx=None, config=None, type=None):
        super(QuantizeEmbedding, self).__init__(*kargs, padding_idx=padding_idx)
        self.weight_bits = config.weight_bits
        self.layerwise = False
        if self.weight_bits == 2:
            self.weight_quantizer = TwnQuantizer
        elif self.weight_bits == 1:
            self.weight_quantizer = BinaryQuantizer
        else:
            self.weight_quantizer = SymQuantizer
        self.init = True
        self.register_buffer('weight_clip_val', torch.tensor([-config.clip_val, config.clip_val]))

    def forward(self, input, type=None):
        if self.weight_bits == 1:
            scaling_factor = torch.mean(abs(self.weight), dim=1, keepdim=True)
            scaling_factor = scaling_factor.detach()
            real_weights = self.weight - torch.mean(self.weight, dim=-1, keepdim=True)
            binary_weights_no_grad = scaling_factor * torch.sign(real_weights)
            cliped_weights = torch.clamp(real_weights, -1.0, 1.0)
            weight = binary_weights_no_grad.detach() - cliped_weights.detach() + cliped_weights
        else:
            weight = self.weight_quantizer.apply(self.weight, self.weight_clip_val, self.weight_bits, self.layerwise)
        out = nn.functional.embedding(
            input, weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)
        return out
