import os
import pathlib
import shutil
import math
import torch
import torch.nn as nn
from args import args as parser_args
import tqdm

def stablize_bn(model, train_loader):
    for i, (train_x, train_y, train_g, train_c) in tqdm.tqdm(
            enumerate(train_loader), ascii=True, total=len(train_loader)
    ):
        model(train_x)

def save_checkpoint(state, is_best, filename="checkpoint.pth", save=False, finetune=False):
    filename = pathlib.Path(filename)
    if not filename.parent.exists():
        os.makedirs(filename.parent)
    torch.save(state, filename)
    if is_best:
        if finetune:
            shutil.copyfile(filename, str(filename.parent / "model_best_finetune.pth"))
        else:
            shutil.copyfile(filename, str(filename.parent / "model_best.pth"))
        if not save:
            os.remove(filename)

def get_lr(optimizer):
    return optimizer.param_groups[0]["lr"]

def freeze_model_weights(model):
    print("=> Freezing model weights")
    for n, m in model.named_modules():
        if hasattr(m, "weight") and m.weight is not None:
            print(f"==> No gradient to {n}.weight")
            m.weight.requires_grad = False
            if m.weight.grad is not None:
                print(f"==> Setting gradient of {n}.weight to None")
                m.weight.grad = None

            if hasattr(m, "bias") and m.bias is not None:
                print(f"==> No gradient to {n}.bias")
                m.bias.requires_grad = False

                if m.bias.grad is not None:
                    print(f"==> Setting gradient of {n}.bias to None")
                    m.bias.grad = None

def freeze_model_subnet(model):
    print("=> Freezing model subnet")
    for n, m in model.named_modules():
        if hasattr(m, "scores"):
            m.scores.requires_grad = False
            print(f"==> No gradient to {n}.scores")
            if m.scores.grad is not None:
                print(f"==> Setting gradient of {n}.scores to None")
                m.scores.grad = None

def fix_model_subnet(model):
    for n, m in model.named_modules():
        if hasattr(m, "scores"):
            if m.prune:
                m.fix_subnet()
                m.train_weights = True
                # print("after fixing, mean:", torch.mean(m.subnet))


def fix_model_subnet_others(model):
    for n, m in model.named_modules():
        if hasattr(m, "scores"):
            if m.prune:
                m.fix_subnet_others()
                m.train_weights = True
                # print("after fixing, mean:", torch.sum(m.scores<0.5))

def unfix_model_subnet(model):
    for n, m in model.named_modules():
        if hasattr(m, "scores"):
            if m.prune:
                m.train_weights = False

def unfreeze_model_weights(model):
    print("=> Unfreezing model weights")
    for n, m in model.named_modules():
        if hasattr(m, "weight") and m.weight is not None:
            print(f"==> Gradient to {n}.weight")
            m.weight.requires_grad = True
            if hasattr(m, "bias") and m.bias is not None:
                print(f"==> Gradient to {n}.bias")
                m.bias.requires_grad = True

def unfreeze_model_subnet(model):
    print("=> Unfreezing model subnet")
    for n, m in model.named_modules():
        if hasattr(m, "scores"):
            print(f"==> Gradient to {n}.scores")
            m.scores.requires_grad = True

def set_model_prune_rate(model, prune_rate):
    print(f"==> Setting prune rate of network to {prune_rate}")
    for n, m in model.named_modules():
        if hasattr(m, "set_prune_rate"):
            m.set_prune_rate(prune_rate)
            print(f"==> Setting prune rate of {n} to {prune_rate}")

def solve_v(x):
    k = x.nelement() * parser_args.prune_rate
    def f(v):
        return (x - v).clamp(0, 1).sum() - k
    if f(0) < 0:
        return 0, 0
    a, b = 0, x.max()
    itr = 0
    while (1):
        itr += 1
        v = (a + b) / 2
        obj = f(v)
        if abs(obj) < 1e-3 or itr > 20:
            break
        if obj < 0:
            b = v
        else:
            a = v
    v = max(0, v)
    return v, itr


def solve_v_total(model, total):
    k = total * parser_args.prune_rate
    a, b = 0, 0
    for n, m in model.named_modules():
        if hasattr(m, "scores") and m.prune:
            b = max(b, m.scores.max())
    def f(v):
        s = 0
        for n, m in model.named_modules():
            if hasattr(m, "scores") and m.prune:
                s += (m.scores - v).clamp(0, 1).sum()
        return s - k
    if f(0) < 0:
        return 0, 0
    itr = 0
    while (1):
        itr += 1
        v = (a + b) / 2
        obj = f(v)
        if abs(obj) < 1e-3 or itr > 20:
            break
        if obj < 0:
            b = v
        else:
            a = v
    v = max(0, v)
    return v, itr


def constrainScore(model, args, v_meter, max_score_meter):
    for n, m in model.named_modules():
        if hasattr(m, "scores"):
            if m.prune:
                if args.center:
                    m.scores.clamp_(-0.5, 0.5)
                else:
                    max_score_meter.update(m.scores.max())
                    v, itr = solve_v(m.scores)
                    v_meter.update(v)
                    m.scores.sub_(v).clamp_(0, 1)

def constrainScoreByWhole(model, v_meter, max_score_meter):
    total = 0
    for n, m in model.named_modules():
        if hasattr(m, "scores"):
            if m.prune:
                total += m.scores.nelement()
                max_score_meter.update(m.scores.max())
    v, itr = solve_v_total(model, total)
    v_meter.update(v)
    for n, m in model.named_modules():
        if hasattr(m, "scores"):
            if m.prune:
                m.scores.sub_(v).clamp_(0, 1)