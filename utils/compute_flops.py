# Code from https://github.com/simochen/model-tools.
import numpy as np

import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
from args import args

def print_model_param_nums(model=None):
    total = 0
    for n, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            total += m.weight.nelement()
    print(' + Number of params: %.2fM' % (total))
    return total

def print_model_param_nums_sparse(model=None):
    sparse_total = 0
    for n, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            sparse_total += m.weight.nelement()*(m.subnet.sum()/m.subnet.nelement())
    print(' Sparse + Number of params: %.2fM' % (sparse_total))
    return sparse_total

def print_model_param_flops(model=None, c = 3, input_res=32, multiply_adds=False):

    prods = {}
    def save_hook(name):
        def hook_per(self, input, output):
            prods[name] = np.prod(input[0].shape)
        return hook_per

    list_1=[]
    def simple_hook(self, input, output):
        list_1.append(np.prod(input[0].shape))

    list_2={}
    def simple_hook2(self, input, output):
        list_2['names'] = np.prod(input[0].shape)

    list_conv_param = []
    list_conv=[]
    def conv_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups)
        # print("kernel ops", self.kernel_size[0], self.kernel_size[1], self.in_channels, self.groups)
        bias_ops = 1 if self.bias is not None else 0

        params = output_channels * (kernel_ops + bias_ops)

        flops = (kernel_ops * (2 if multiply_adds else 1) + bias_ops) * output_channels * output_height * output_width * batch_size
        list_conv_param.append(self.weight.nelement())
        list_conv.append(flops)

    list_linear=[]
    def linear_hook(self, input, output):
        batch_size = input[0].size(0) if input[0].dim() == 2 else 1

        weight_ops = self.weight.nelement() * (2 if multiply_adds else 1)
        bias_ops = self.bias.nelement()

        flops = batch_size * (weight_ops + bias_ops)
        list_linear.append(flops)

    list_bn=[]
    def bn_hook(self, input, output):
        list_bn.append(input[0].nelement() * 2)

    list_relu=[]
    def relu_hook(self, input, output):
        list_relu.append(input[0].nelement())

    list_pooling=[]
    def pooling_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size * self.kernel_size
        bias_ops = 0
        params = 0
        flops = (kernel_ops + bias_ops) * output_channels * output_height * output_width * batch_size

        list_pooling.append(flops)

    list_upsample=[]
    # For bilinear upsample
    def upsample_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        flops = output_height * output_width * output_channels * batch_size * 12
        list_upsample.append(flops)

    def foo(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d):
                net.register_forward_hook(conv_hook)
            if isinstance(net, torch.nn.Linear):
                net.register_forward_hook(linear_hook)
            if isinstance(net, torch.nn.BatchNorm2d):
                net.register_forward_hook(bn_hook)
            if isinstance(net, torch.nn.ReLU):
                net.register_forward_hook(relu_hook)
            if isinstance(net, torch.nn.MaxPool2d) or isinstance(net, torch.nn.AvgPool2d):
                net.register_forward_hook(pooling_hook)
            if isinstance(net, torch.nn.Upsample):
                net.register_forward_hook(upsample_hook)
            return
        for c in childrens:
            foo(c)

    if model == None:
        model = torchvision.models.alexnet()
    foo(model)
    input = Variable(torch.rand(c,input_res,input_res).unsqueeze(0), requires_grad = True)
    out = model(input)

    total_flops = (sum(list_conv) + sum(list_linear) + sum(list_bn) + sum(list_relu) + sum(list_pooling) + sum(list_upsample))
    conv_flops = sum(list_conv)

    print('  + Number of FLOPs: %.2f' % (total_flops))
    print('  + Number of conv FLOPs: %.2f' % (conv_flops))
    print('  + Number of conv params: %.2f' % (sum(list_conv_param)))
    print("conv flops list", list_conv)

    return list_conv, total_flops


def print_model_param_flops_sparse(model=None, c = 3, input_res=32, multiply_adds=False):

    prods = {}
    def save_hook(name):
        def hook_per(self, input, output):
            prods[name] = np.prod(input[0].shape)
        return hook_per

    list_1=[]
    def simple_hook(self, input, output):
        list_1.append(np.prod(input[0].shape))

    list_2={}
    def simple_hook2(self, input, output):
        list_2['names'] = np.prod(input[0].shape)

    list_conv_param = []
    list_conv_param_dense = []
    list_conv_flops=[]
    list_conv_flops_dense = []
    flops_dict = {}
    def conv_hook(self, input, output):
        if not hasattr(self, 'subnet'):
            subnet_sparsity = 1
        else:
            subnet_sparsity = self.subnet.sum()/self.subnet.nelement()
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()
        input_reduced = torch.sum(input[0].squeeze(0), dim=(1, 2))
        # print(input_reduced.size())
        input_sparsity = float(torch.sum(torch.abs(input_reduced)>1e-20))/input_reduced.size()[0]
        # print(torch.sum(torch.abs(input_reduced)>1e-20), input_reduced.size()[0], input_sparsity)
        kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups)
        # print("kernel ops", self.kernel_size[0], self.kernel_size[1], self.in_channels, self.groups)
        bias_ops = 1 if self.bias is not None else 0

        params = output_channels * (kernel_ops + bias_ops)
        flops = (kernel_ops * (2 if multiply_adds else 1) + bias_ops) * output_channels * output_height * output_width * batch_size * subnet_sparsity * input_sparsity

        list_conv_flops.append(flops)
        list_conv_flops_dense.append((kernel_ops * (2 if multiply_adds else 1) + bias_ops) * output_channels * output_height * output_width * batch_size)
        # print("in conv", kernel_ops, output_channels, output_height, output_width, batch_size)
        list_conv_param.append(self.weight.nelement() * subnet_sparsity * input_sparsity)
        list_conv_param_dense.append(self.weight.nelement())

    list_linear_param = []
    list_linear_param_dense = []
    list_linear_flops = []
    list_linear_flops_dense = []
    def linear_hook(self, input, output):
        input_reduced = torch.sum(input[0].squeeze(0), dim=(0, 1)) if input[0].dim() > 2 else input[0].squeeze(0)
        input_sparsity = float(torch.sum(torch.abs(input_reduced) > 1e-20)) / input_reduced.size()[0]
        weight_ops = input[0].nelement()*self.out_features * (2 if multiply_adds else 1) * (self.subnet.sum()/self.subnet.nelement()) * input_sparsity
        list_linear_flops.append(weight_ops)
        list_linear_flops_dense.append(input[0].nelement()*self.out_features * (2 if multiply_adds else 1))
        list_linear_param.append(self.weight.nelement() * (self.subnet.sum()/self.subnet.nelement()) * input_sparsity)
        list_linear_param_dense.append(self.weight.nelement() )

    list_bn=[]
    def bn_hook(self, input, output):
        list_bn.append(input[0].nelement() * 2)

    list_relu=[]
    def relu_hook(self, input, output):
        list_relu.append(input[0].nelement())

    list_pooling=[]
    def pooling_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        kernel_ops = self.kernel_size * self.kernel_size
        bias_ops = 0
        params = 0
        flops = (kernel_ops + bias_ops) * output_channels * output_height * output_width * batch_size

        list_pooling.append(flops)

    list_upsample=[]
    # For bilinear upsample
    def upsample_hook(self, input, output):
        batch_size, input_channels, input_height, input_width = input[0].size()
        output_channels, output_height, output_width = output[0].size()

        flops = output_height * output_width * output_channels * batch_size * 12
        list_upsample.append(flops)

    handles = []
    def foo(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d):
                handles.append(net.register_forward_hook(conv_hook))
            if isinstance(net, torch.nn.Linear):
                handles.append(net.register_forward_hook(linear_hook))
            if isinstance(net, torch.nn.BatchNorm2d):
                handles.append(net.register_forward_hook(bn_hook))
            if isinstance(net, torch.nn.ReLU):
                handles.append(net.register_forward_hook(relu_hook))
            if isinstance(net, torch.nn.MaxPool2d) or isinstance(net, torch.nn.AvgPool2d):
                handles.append(net.register_forward_hook(pooling_hook))
            if isinstance(net, torch.nn.Upsample):
                handles.append(net.register_forward_hook(upsample_hook))
            return
        for c in childrens:
            foo(c)

    def unfoo():
        for h in handles:
            h.remove()

    temp_list = []
    for n, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            temp_list.append(m.bias.clone())
            m.bias.data = torch.zeros_like(m.bias)

    if model == None:
        model = torchvision.models.alexnet()
    foo(model)
    input = Variable(torch.rand(c,input_res,input_res).unsqueeze(0))
    out = model(input)

    idx = 0
    for n, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            m.bias.data = temp_list[idx].clone()
            idx += 1
    unfoo()
    total_flops = (sum(list_conv_flops) + sum(list_linear_flops) + sum(list_bn) + sum(list_relu) + sum(list_pooling) + sum(list_upsample))
    conv_flops = sum(list_conv_flops)
    linear_flops = sum(list_linear_flops)
    conv_flops_dense = sum(list_conv_flops_dense)
    linear_flops_dense = sum(list_linear_flops_dense)
    conv_params = sum(list_conv_param)
    conv_params_dense = sum(list_conv_param_dense)
    linear_params = sum(list_linear_param)
    linear_params_dense = sum(list_linear_param_dense)
    print('  + Number of FLOPs: %.2f' % (total_flops))
    print('  + Number of conv FLOPs: %.2f' % (conv_flops))
    print('  + Number of dense conv FLOPs: %.2f' % (conv_flops_dense))
    print("conv list:", list_conv_flops_dense)
    print("conv flops reduction ratio:", conv_flops/conv_flops_dense)
    print('  + Number of linear FLOPs: %.2f' % (linear_flops))
    print('  + Number of dense linear FLOPs: %.2f' % (linear_flops_dense))
    if linear_flops_dense != 0:
        print("linear flops reduction ratio:", linear_flops / linear_flops_dense)

    print('  + Number of conv param: %.2f' % (conv_params))
    print('  + Number of dense conv param: %.2f' % (conv_params_dense))
    print("conv param reduction ratio:", conv_params/conv_params_dense)
    print('  + Number of linear param: %.2f' % (linear_params))
    print('  + Number of dense linear param: %.2f' % (linear_params_dense))
    if linear_params_dense != 0:
        print("linear param reduction ratio:", linear_params / linear_params_dense)
    if args.conv_type.endswith("Linear"):
        return linear_flops / linear_flops_dense
    else:
        return conv_flops / conv_flops_dense