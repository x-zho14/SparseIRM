import os
import pathlib
import random
import time
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import copy
import numpy as np
from utils.logging import AverageMeter, ProgressMeter
from utils.net_utils import freeze_model_subnet, save_checkpoint, get_lr, fix_model_subnet, fix_model_subnet_others, unfix_model_subnet, freeze_model_weights, stablize_bn
from utils.schedulers import get_policy, assign_learning_rate
from args import args
import importlib
import data
import models
from utils.compute_flops import print_model_param_flops_sparse, print_model_param_flops
from utils.irm_utils import concat_envs, mean_nll_class, mean_nll_multi_class, make_mnist_envs, CMNIST_LYDP, CIFAR_LYPD, COCOcolor_LYPD, CMNISTFULL_LYDP
from models.irm_models import EBD
import shutil
import wandb
from sklearn.linear_model import LogisticRegression
from utils.irm_utils import eval_acc_class, eval_acc_multi_class


wandb.init(project="irm", name=args.runs_name, config=args)

def main():
    print(args)
    torch.backends.cudnn.benchmark = True

    try:
        main_worker(args)
    except KeyboardInterrupt as e:
        print("rep_count: ", args.rep_count)

def main_worker(args):
    args.finetuning = False
    args.stablizing = False
    args.obtain_prior_prob_with_snip = False
    args.scaling_para = 1
    train, validate, modifier = get_trainer(args)
    model = get_model(args)
    model = set_gpu(args, model)
    
    optimizer, weight_opt = get_optimizer(args, model)
    # data = get_dataset(args)
    if args.set == "mnist":
        dp = CMNIST_LYDP(args)
    elif args.set == "mnistfull":
        dp = CMNISTFULL_LYDP(args)
    args.arch = "EBD"
    ebd = get_model(args)
    ebd = set_gpu(args, ebd)

    args.acc_list = []
    lr_policy = get_policy(args.lr_policy)(optimizer, args)
    if args.set != "mnistfull":
        criterion = mean_nll_class
        args.eval_fn = eval_acc_class
    else:
        criterion = mean_nll_multi_class
        args.eval_fn = eval_acc_multi_class

    args.gpu = args.multigpu[0]
    args.sparsity_best, args.sparsity_best_test = 0, 0
    best_acc, best_min_acc, best_maj_acc, best_train_acc, best_train_min_acc, best_train_maj_acc, best_min_train_test_acc, best_train_corr, best_test_corr = 0, 0, 0, 0, 0, 0, 0, 0, 0
    run_base_dir, ckpt_base_dir, log_base_dir = get_directories(args)
    args.ckpt_base_dir = ckpt_base_dir
    writer = SummaryWriter(log_dir=log_base_dir)
    epoch_time = AverageMeter("epoch_time", ":.4f", write_avg=False)
    validation_time = AverageMeter("validation_time", ":.4f", write_avg=False)
    train_time = AverageMeter("train_time", ":.4f", write_avg=False)
    progress_overall = ProgressMeter(
        1, [epoch_time, validation_time, train_time], prefix="Overall Timing"
    )
    end_epoch = time.time()

    args.start_epoch = args.start_epoch or 0
    acc1 = None

    save_checkpoint(
        {"epoch": 0, "arch": args.arch, "state_dict": model.state_dict(), "best_acc": best_acc,
         "best_min_acc": best_min_acc, "best_maj_acc": best_maj_acc, "best_train_acc": best_train_acc,
         "best_train_min_acc": best_train_min_acc, "best_train_maj_acc": best_train_maj_acc,
         "optimizer": optimizer.state_dict(), "test_acc": None}, False,
        filename=ckpt_base_dir / f"initial.state", save=False, )

    # Save the initial state
    flops_reduction_list = []
    pr_target = args.prune_rate
    ts = int(args.ts * args.epochs)
    te = int(args.te * args.epochs)
    pr_start = args.pr_start
    args.steps = 0
    record_test_best = None

    for epoch in range(args.start_epoch, args.epochs):
        if args.iterative:
            if epoch < ts:
                args.prune_rate = 1
            elif epoch < te:
                args.prune_rate = pr_target + (pr_start - pr_target) * (1 - (epoch - ts) / (te - ts)) ** 3
            else:
                args.prune_rate = pr_target
        if args.TA:
            args.T = 1 / ((1 - 0.03) * (1 - epoch / args.epochs) + 0.03)
        print("current lr: ", get_lr(optimizer))
        if weight_opt is not None:
            print("current weight lr: ", weight_opt.param_groups[0]["lr"])
        print("current prune rate: ", args.prune_rate)
        start_train = time.time()
        train_acc, train_minacc, train_majacc, train_corr = train(dp.get_train_loader(), model, ebd, criterion, optimizer, epoch, args, writer, weight_opt)
        iter = 0
        while (iter < 1):
            fix_model_subnet(model)
            train_acc, train_minacc, train_majacc, _, train_corr = validate(dp.get_train_loader(), model, criterion,
                                                                            args, writer, epoch)
            train_time.update((time.time() - start_train) / 60)
            start_validation = time.time()
            test_acc, test_minacc, test_majacc, losses, test_corr = validate(dp.get_test_loader(), model, criterion,
                                                                             args, writer, epoch)
            validation_time.update((time.time() - start_validation) / 60)

            is_best = (test_acc > best_acc) and (train_acc > 0.6)
            if is_best:
                best_acc = test_acc
                record_test_best = (train_acc, test_acc)
            if is_best or epoch == args.epochs - 1:
                if is_best:
                    print(f"==> New best, saving at {ckpt_base_dir / 'model_best.pth'}")
            epoch_time.update((time.time() - end_epoch) / 60)
            progress_overall.display(epoch)
            progress_overall.write_to_tensorboard(writer, prefix="diagnostics", global_step=epoch)
            end_epoch = time.time()
            print("record: (train_acc, test_acc)", record_test_best)

            iter += 1
        unfix_model_subnet(model)



def get_trainer(args):
    trainer = importlib.import_module(f"trainers.{args.trainer}")
    return trainer.train, trainer.validate, trainer.modifier

def set_gpu(args, model):
    assert torch.cuda.is_available(), "CPU-only experiments currently unsupported"
    print(f"=> Parallelizing on {args.multigpu} gpus")
    torch.cuda.set_device(args.multigpu[0])
    model = torch.nn.DataParallel(model, device_ids=args.multigpu).cuda(
        args.multigpu[0]
    )
    return model

def get_dataset(args):
    print(f"=> Getting {args.set} dataset")
    dataset = getattr(data, args.set)(args)
    return dataset

def get_model(args):
    print("=> Creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch]()
    print(model)
    return model

def get_optimizer(args, model):
    for n, v in model.named_parameters():
        if v.requires_grad:
            print("<DEBUG> gradient to", n)

        if not v.requires_grad:
            print("<DEBUG> no gradient to", n)
    if args.optimizer == "adamw":
        parameters = list(model.named_parameters())
        for n, v in parameters:
            print(n, "weight_para")
        weight_params = [(n, v) for n, v in parameters if v.requires_grad]
        selected_weight_params_by_wd = add_weight_decay(weight_params, 0.05)
        optimizer = torch.optim.AdamW(
            selected_weight_params_by_wd,
            2e-3,
        )
        return optimizer, None
    elif args.optimizer == "sgd":
        if not args.train_weights_at_the_same_time:
            parameters = list(model.named_parameters())
            bn_params = [v for n, v in parameters if ("bn" in n) and v.requires_grad]
            rest_params = [v for n, v in parameters if ("bn" not in n) and v.requires_grad]
            optimizer = torch.optim.SGD(
                [
                    {
                        "params": bn_params,
                        "weight_decay": 0 if args.no_bn_decay else args.weight_decay,
                    },
                    {"params": rest_params, "weight_decay": args.weight_decay},
                ],
                args.lr,
                momentum=0.9,
                weight_decay=0,
                nesterov=args.nesterov,
            )
        else:
            parameters = list(model.named_parameters())
            for n, v in parameters:
                if ("score" not in n) and v.requires_grad:
                    print(n, "weight_para")
            for n, v in parameters:
                if ("score" in n) and v.requires_grad:
                    print(n, "score_para")
            weight_params = [v for n, v in parameters if ("score" not in n) and v.requires_grad]
            score_params = [v for n, v in parameters if ("score" in n) and v.requires_grad]
            optimizer1 = torch.optim.SGD(
                score_params, lr=0.1, weight_decay=1e-6, momentum=0.9
            )
            optimizer2 = torch.optim.SGD(
                weight_params,
                args.weight_opt_lr,
                momentum=0.9,
                nesterov=args.nesterov,
            )
            return optimizer1, optimizer2

    elif args.optimizer == "adam":
        if not args.train_weights_at_the_same_time:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay
            )
        elif args.weight_opt == "sgd":
            parameters = list(model.named_parameters())
            for n, v in parameters:
                if ("score" not in n) and v.requires_grad:
                    print(n, "weight_para")
            for n, v in parameters:
                if ("score" in n) and v.requires_grad:
                    print(n, "score_para")
            weight_params = [v for n, v in parameters if ("score" not in n) and v.requires_grad]
            score_params = [v for n, v in parameters if ("score" in n) and v.requires_grad]
            optimizer1 = torch.optim.Adam(
                score_params, lr=args.lr, weight_decay=args.weight_decay
            )
            optimizer2 = torch.optim.SGD(
                weight_params,
                args.weight_opt_lr,
                momentum=0.9,
                nesterov=args.nesterov,
            )
            return optimizer1, optimizer2
        elif args.weight_opt == "adamw":
            parameters = list(model.named_parameters())
            for n, v in parameters:
                if ("score" not in n) and v.requires_grad:
                    print(n, "weight_para")
            for n, v in parameters:
                if ("score" in n) and v.requires_grad:
                    print(n, "score_para")
            weight_params = [(n, v)  for n, v in parameters if ("score" not in n) and v.requires_grad]
            score_params = [v for n, v in parameters if ("score" in n) and v.requires_grad]
            selected_weight_params_by_wd = add_weight_decay(weight_params, 0.05)
            optimizer1 = torch.optim.Adam(
                score_params, lr=args.lr, weight_decay=args.weight_decay
            )
            optimizer2 = torch.optim.AdamW(
                selected_weight_params_by_wd,
                2e-3,
            )
            # print("opt1, opt2", optimizer1, optimizer2)
            return optimizer1, optimizer2
        elif args.weight_opt == "adam":
            parameters = list(model.named_parameters())
            for n, v in parameters:
                if ("score" not in n) and v.requires_grad:
                    print(n, "weight_para")
            for n, v in parameters:
                if ("score" in n) and v.requires_grad:
                    print(n, "score_para")
            weight_params = [v for n, v in parameters if ("score" not in n) and v.requires_grad]
            score_params = [v for n, v in parameters if ("score" in n) and v.requires_grad]
            optimizer1 = torch.optim.Adam(
                score_params, lr=args.lr, weight_decay=args.weight_decay
            )
            optimizer2 = torch.optim.Adam(
                weight_params,
                args.weight_opt_lr,
            )
            # print("opt1, opt2", optimizer1, optimizer2)
            return optimizer1, optimizer2
    return optimizer, None

def add_weight_decay(weight_params, weight_decay=0.05, skip_list=()):
    decay = []
    no_decay = []
    for name, param in weight_params:
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


def _run_dir_exists(run_base_dir):
    log_base_dir = run_base_dir / "logs"
    ckpt_base_dir = run_base_dir / "checkpoints"
    return log_base_dir.exists() or ckpt_base_dir.exists()

def get_directories(args):
    if args.config is None or args.name is None:
        raise ValueError("Must have name and config")

    config = pathlib.Path(args.config).stem
    if args.log_dir is None:
        run_base_dir = pathlib.Path(
            f"runs/{config}/{args.name}/prune_rate={args.prune_rate}"
        )
    else:
        run_base_dir = pathlib.Path(
            f"{args.log_dir}/{config}/{args.name}/prune_rate={args.prune_rate}"
        )
    if args.width_mult != 1.0:
        run_base_dir = run_base_dir / "width_mult={}".format(str(args.width_mult))

    args.rep_count = "/"
    if _run_dir_exists(run_base_dir):
        rep_count = 0
        while _run_dir_exists(run_base_dir / str(rep_count)):
            rep_count += 1
        args.rep_count = "/" + str(rep_count)
        run_base_dir = run_base_dir / str(rep_count)

    log_base_dir = run_base_dir / "logs"
    ckpt_base_dir = run_base_dir / "checkpoints"

    if not run_base_dir.exists():
        os.makedirs(run_base_dir)

    (run_base_dir / "settings.txt").write_text(str(args))

    return run_base_dir, ckpt_base_dir, log_base_dir

if __name__ == "__main__":
    main()
