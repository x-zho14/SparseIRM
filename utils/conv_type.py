import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import math
from args import args as parser_args

DenseConv = nn.Conv2d
DenseLinear = nn.Linear

class GetScoreImp(autograd.Function):
    @staticmethod
    def forward(ctx, scores):
        scores_imp = torch.where(scores > 0.9, torch.ones_like(scores), scores * parser_args.scaling_para)
        ctx.save_for_backward(scores)
        return scores_imp

    @staticmethod
    def backward(ctx, grad_outputs):
        scores, = ctx.saved_variables
        grad_ori = torch.where(scores > 0.9, grad_outputs, grad_outputs*parser_args.scaling_para)
        return grad_ori

class GetScoreImpNoGrad(autograd.Function):
    @staticmethod
    def forward(ctx, scores):
        scores_imp = torch.where(scores > 0.9, torch.ones_like(scores), scores * parser_args.scaling_para)
        return scores_imp

    @staticmethod
    def backward(ctx, grad_outputs):
        return torch.zeros_like(grad_outputs)

class ReinforceLOOVRWeightIMP(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        self.register_buffer('subnet', torch.zeros_like(self.scores))
        self.train_weights = False
        if parser_args.score_init_constant is not None:
            self.scores.data = (
                    torch.ones_like(self.scores) * parser_args.score_init_constant
            )
        else:
            nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))
        self.prune = True
        self.register_buffer("stored_mask_0", torch.zeros_like(self.scores))
        self.register_buffer("stored_mask_1", torch.zeros_like(self.scores))

    @property
    def clamped_scores(self):
        scores_imp = torch.where(self.scores > 0.9, torch.ones_like(self.scores), self.scores * parser_args.scaling_para)
        return scores_imp

    def fix_subnet(self):
        self.subnet = (torch.rand_like(self.scores) < self.clamped_scores).float()

    def forward(self, x):
        if parser_args.obtain_prior_prob_with_snip:
            return F.conv2d(x, self.weight*self.ones_mask, self.bias, self.stride, self.padding, self.dilation, self.groups)
        else:
            if self.prune:
                if not self.train_weights:
                    scores_imp = GetScoreImpNoGrad.apply(self.scores)
                    self.subnet = StraightThroughBinomialSampleNoGrad.apply(scores_imp)
                    if parser_args.j == 0:
                        self.stored_mask_0.data = (self.subnet-self.scores)/torch.sqrt((self.scores+1e-20)*(1-self.scores+1e-20))
                    else:
                        self.stored_mask_1.data = (self.subnet-self.scores)/torch.sqrt((self.scores+1e-20)*(1-self.scores+1e-20))
                    w = self.weight * self.subnet
                    x = F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
                else:
                    w = self.weight * self.subnet
                    x = F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
            else:
                x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
            return x

class ProbMaskConvIMP(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        self.register_buffer('subnet', torch.zeros_like(self.scores))
        self.train_weights = False
        self.prune = True
        if parser_args.score_init_constant is not None:
            self.scores.data = (
                    torch.ones_like(self.scores) * parser_args.score_init_constant
            )
        else:
            nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

    @property
    def clamped_scores(self):
        scores_imp = torch.where(self.scores > 0.9, torch.ones_like(self.scores), self.scores * parser_args.scaling_para)
        return scores_imp

    def fix_subnet(self):
        self.subnet = (torch.rand_like(self.scores) < self.clamped_scores).float()

    def forward(self, x):
        if parser_args.obtain_prior_prob_with_snip:
            return F.conv2d(x, self.weight*self.ones_mask, self.bias, self.stride, self.padding, self.dilation, self.groups)
        else:
            if not self.train_weights:
                if not parser_args.discrete:
                    eps = 1e-20
                    temp = parser_args.T
                    uniform0 = torch.rand_like(self.scores)
                    uniform1 = torch.rand_like(self.scores)
                    noise = -torch.log(torch.log(uniform0 + eps) / torch.log(uniform1 + eps) + eps)
                    scores_imp = GetScoreImp.apply(self.scores)
                    self.subnet = torch.sigmoid((torch.log(scores_imp + eps) - torch.log(1.0 - scores_imp + eps) + noise) * temp)
                else:
                    self.subnet = (torch.rand_like(self.scores) < self.clamped_scores).float()
                w = self.weight * self.subnet
                x = F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
            else:
                w = self.weight * self.subnet
                x = F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
            return x

class ProbMaskConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))  #Probability
        self.register_buffer('subnet', torch.zeros_like(self.scores))
        self.train_weights = False
        self.prune = True
        if parser_args.score_init_constant is not None:
            self.scores.data = (
                    torch.ones_like(self.scores) * parser_args.score_init_constant
            )
        else:
            nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))
        self.ones_mask = nn.Parameter(torch.ones_like(self.weight))

    @property
    def clamped_scores(self):
        return self.scores

    def fix_subnet(self):
        self.subnet = (torch.rand_like(self.scores) < self.clamped_scores).float()

    def fix_subnet_others(self):
        self.subnet = (self.clamped_scores > 0.5)

    def forward(self, x):
        if parser_args.obtain_prior_prob_with_snip:
            return F.conv2d(x, self.weight*self.ones_mask, self.bias, self.stride, self.padding, self.dilation, self.groups)
        else:
            if not self.train_weights:                                      #training
                if not parser_args.discrete:
                    eps = 1e-20
                    temp = parser_args.T
                    uniform0 = torch.rand_like(self.scores)
                    uniform1 = torch.rand_like(self.scores)
                    noise = -torch.log(torch.log(uniform0 + eps) / torch.log(uniform1 + eps) + eps)
                    self.subnet = torch.sigmoid((torch.log(self.clamped_scores + eps) - torch.log(1.0 - self.clamped_scores + eps) + noise) * temp)
                else:
                    self.subnet = (torch.rand_like(self.scores) < self.clamped_scores).float()
                w = self.weight * self.subnet
                x = F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
            else:                                                           #testing
                w = self.weight * self.subnet
                x = F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
            return x

class ProbMaskConvLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))  #Probability
        self.register_buffer('subnet', torch.zeros_like(self.scores))
        self.train_weights = False
        self.prune = True
        if parser_args.score_init_constant is not None:
            self.scores.data = (
                    torch.ones_like(self.scores) * parser_args.score_init_constant
            )
        else:
            nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))
        self.ones_mask = nn.Parameter(torch.ones_like(self.weight))

    @property
    def clamped_scores(self):
        return self.scores

    def fix_subnet_others(self):
        self.subnet = (self.clamped_scores > 0.5).float()

    def fix_subnet(self):
        self.subnet = (torch.rand_like(self.scores) < self.clamped_scores).float()

    def forward(self, x):
        if parser_args.obtain_prior_prob_with_snip:
            return F.linear(x, self.weight*self.ones_mask, self.bias)
        else:
            if not self.train_weights:                                      #training
                if not parser_args.discrete:
                    eps = 1e-20
                    temp = parser_args.T
                    uniform0 = torch.rand_like(self.scores)
                    uniform1 = torch.rand_like(self.scores)
                    noise = -torch.log(torch.log(uniform0 + eps) / torch.log(uniform1 + eps) + eps)
                    self.subnet = torch.sigmoid((torch.log(self.clamped_scores + eps) - torch.log(1.0 - self.clamped_scores + eps) + noise) * temp)
                else:
                    self.subnet = (torch.rand_like(self.scores) < self.clamped_scores).float()
                w = self.weight * self.subnet
                x = F.linear(x, w, self.bias)
            else:                                                           #testing
                w = self.weight * self.subnet
                x = F.linear(x, w, self.bias)
            return x

class ReinforceLOOVRLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()[0], 1))
        self.register_buffer('subnet', torch.zeros_like(self.scores))
        self.train_weights = False
        if parser_args.score_init_constant is not None:
            self.scores.data = (
                    torch.ones_like(self.scores) * parser_args.score_init_constant
            )
        else:
            nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))
        if self.out_features == 10 or self.out_features == 100:
            self.prune = False
            self.subnet = torch.ones_like(self.scores)
        else:
            self.prune = True
        self.register_buffer("stored_mask_0", torch.zeros_like(self.scores))
        self.register_buffer("stored_mask_1", torch.zeros_like(self.scores))

    @property
    def clamped_scores(self):
        return self.scores

    def fix_subnet(self):
        self.subnet = (torch.rand_like(self.scores) < self.clamped_scores).float()

    def fix_subnet_others(self):
        self.subnet = (self.clamped_scores > 0.5)

    def flip_one_channel(self):
        self.subnet.data[0] = 1 - self.subnet.data[0]

    def forward(self, x):
        if parser_args.obtain_prior_prob_with_snip:
            self.ones_mask = torch.ones_like(self.weight)
            return F.conv2d(x, self.weight*self.ones_mask, self.bias, self.stride, self.padding, self.dilation, self.groups)
        else:
            if self.prune:
                if not self.train_weights:
                    self.subnet = StraightThroughBinomialSampleNoGrad.apply(self.scores)
                    if parser_args.j == 0:
                        self.stored_mask_0.data = (self.subnet-self.scores)/torch.sqrt((self.scores+1e-20)*(1-self.scores+1e-20))
                    else:
                        self.stored_mask_1.data = (self.subnet-self.scores)/torch.sqrt((self.scores+1e-20)*(1-self.scores+1e-20))
                    w = self.weight * self.subnet
                    x = F.linear(x, w, self.bias)
                else:
                    w = self.weight * self.subnet
                    x = F.linear(x, w, self.bias)
            else:
                x = F.linear(x, self.weight, self.bias)
            return x


class ReinforceLOO(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()[0], 1, 1, 1))
        self.register_buffer('subnet', torch.zeros_like(self.scores))
        self.train_weights = False
        if parser_args.score_init_constant is not None:
            self.scores.data = (
                    torch.ones_like(self.scores) * parser_args.score_init_constant
            )
        else:
            nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))
        if self.out_channels == 10 or self.out_channels == 100:
            self.prune = False
            self.subnet = torch.ones_like(self.scores)
        else:
            self.prune = True
        self.register_buffer("stored_mask_0", torch.zeros_like(self.scores))
        self.register_buffer("stored_mask_1", torch.zeros_like(self.scores))

    @property
    def clamped_scores(self):
        return self.scores

    def fix_subnet(self):
        self.subnet = (torch.rand_like(self.scores) < self.scores).float()

    def forward(self, x):
        if parser_args.obtain_prior_prob_with_snip:
            self.ones_mask = torch.ones_like(self.weight)
            return F.conv2d(x, self.weight*self.ones_mask, self.bias, self.stride, self.padding, self.dilation, self.groups)
        else:
            if self.prune:
                if not self.train_weights:
                    self.subnet = StraightThroughBinomialSampleNoGrad.apply(self.scores)
                    if parser_args.j == 0:
                        self.stored_mask_0.data = (self.subnet-self.scores)/((self.scores+1e-20)*(1-self.scores+1e-20))
                    else:
                        self.stored_mask_1.data = (self.subnet-self.scores)/((self.scores+1e-20)*(1-self.scores+1e-20))
                    w = self.weight * self.subnet
                    x = F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
                else:
                    w = self.weight * self.subnet
                    x = F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
            else:
                x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
            return x

class ReinforceLOOVRWeight(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        self.register_buffer('subnet', torch.zeros_like(self.scores))
        self.train_weights = False
        if parser_args.score_init_constant is not None:
            self.scores.data = (
                    torch.ones_like(self.scores) * parser_args.score_init_constant
            )
        else:
            nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))
        self.prune = True
        self.register_buffer("stored_mask_0", torch.zeros_like(self.scores))
        self.register_buffer("stored_mask_1", torch.zeros_like(self.scores))
        self.ones_mask = nn.Parameter(torch.ones_like(self.weight))

    @property
    def clamped_scores(self):
        return self.scores

    def fix_subnet(self):
        self.subnet = (torch.rand_like(self.scores) < self.clamped_scores).float()

    def flip_one_channel(self):
        self.subnet.data[0] = 1 - self.subnet.data[0]

    def forward(self, x):
        if parser_args.obtain_prior_prob_with_snip:
            return F.conv2d(x, self.weight*self.ones_mask, self.bias, self.stride, self.padding, self.dilation, self.groups)
        else:
            if self.prune:
                if not self.train_weights:
                    self.subnet = StraightThroughBinomialSampleNoGrad.apply(self.scores)
                    if parser_args.j == 0:
                        self.stored_mask_0.data = (self.subnet-self.scores)/torch.sqrt((self.scores+1e-20)*(1-self.scores+1e-20))
                    else:
                        self.stored_mask_1.data = (self.subnet-self.scores)/torch.sqrt((self.scores+1e-20)*(1-self.scores+1e-20))
                    w = self.weight * self.subnet
                    x = F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
                else:
                    w = self.weight * self.subnet
                    x = F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
            else:
                x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
            return x




class ReinforceLOOVR(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()[0], 1, 1, 1))
        self.register_buffer('subnet', torch.zeros_like(self.scores))
        self.train_weights = False
        if parser_args.score_init_constant is not None:
            self.scores.data = (
                    torch.ones_like(self.scores) * parser_args.score_init_constant
            )
        else:
            nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))
        if self.out_channels == 10 or self.out_channels == 100:
            self.prune = False
            self.subnet = torch.ones_like(self.scores)
        else:
            self.prune = True
        self.register_buffer("stored_mask_0", torch.zeros_like(self.scores))
        self.register_buffer("stored_mask_1", torch.zeros_like(self.scores))

    @property
    def clamped_scores(self):
        return self.scores

    def fix_subnet(self):
        self.subnet = (torch.rand_like(self.scores) < self.clamped_scores).float()

    def flip_one_channel(self):
        self.subnet.data[0] = 1 - self.subnet.data[0]

    def forward(self, x):
        if parser_args.obtain_prior_prob_with_snip:
            self.ones_mask = torch.ones_like(self.weight)
            return F.conv2d(x, self.weight*self.ones_mask, self.bias, self.stride, self.padding, self.dilation, self.groups)
        else:
            if self.prune:
                if not self.train_weights:
                    self.subnet = StraightThroughBinomialSampleNoGrad.apply(self.scores)
                    if parser_args.j == 0:
                        self.stored_mask_0.data = (self.subnet-self.scores)/torch.sqrt((self.scores+1e-20)*(1-self.scores+1e-20))
                    else:
                        self.stored_mask_1.data = (self.subnet-self.scores)/torch.sqrt((self.scores+1e-20)*(1-self.scores+1e-20))
                    w = self.weight * self.subnet
                    x = F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
                else:
                    w = self.weight * self.subnet
                    x = F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
            else:
                x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
            return x

class Reinforce(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()[0], 1, 1, 1))
        self.register_buffer('subnet', torch.zeros_like(self.scores))
        self.train_weights = False
        if parser_args.score_init_constant is not None:
            self.scores.data = (
                    torch.ones_like(self.scores) * parser_args.score_init_constant
            )
        else:
            nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))
        if self.out_channels == 10 or self.out_channels == 100:
            self.prune = False
            self.subnet = torch.ones_like(self.scores)
        else:
            self.prune = True
        self.register_buffer("stored_mask_0", torch.zeros_like(self.scores))
        self.register_buffer("stored_mask_1", torch.zeros_like(self.scores))

    @property
    def clamped_scores(self):
        return self.scores

    def fix_subnet(self):
        self.subnet = (torch.rand_like(self.scores) < self.clamped_scores).float()

    def forward(self, x):
        if parser_args.obtain_prior_prob_with_snip:
            self.ones_mask = torch.ones_like(self.weight)
            return F.conv2d(x, self.weight*self.ones_mask, self.bias, self.stride, self.padding, self.dilation, self.groups)
        else:
            if self.prune:
                if not self.train_weights:
                    self.stored_mask_dict[parser_args.j] = (self.subnet-self.scores)/(self.scores+1e-20)*(1-self.scores+1e-20)
                    self.stored_mask_dict_vr[parser_args.j] = (self.subnet-self.scores)/(self.scores+1e-20)*(1-self.scores+1e-20)
                    self.subnet = StraightThroughBinomialSampleNoGrad.apply(self.scores)
                    if parser_args.j == 0:
                        self.stored_mask_0.data = (self.subnet-self.scores)/(self.scores+1e-20)*(1-self.scores+1e-20)
                    else:
                        self.stored_mask_1.data = (self.subnet-self.scores)/(self.scores+1e-20)*(1-self.scores+1e-20)
                    w = self.weight * self.subnet
                    x = F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
                else:
                    w = self.weight * self.subnet
                    x = F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
            else:
                x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
            return x

class StraightThroughBinomialSampleNoGrad(autograd.Function):
    @staticmethod
    def forward(ctx, scores):
        output = (torch.rand_like(scores) < scores).float()
        return output

    @staticmethod
    def backward(ctx, grad_outputs):
        return torch.zeros_like(grad_outputs)