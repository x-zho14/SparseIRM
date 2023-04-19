import torch
import tqdm
from utils.eval_utils import accuracy
from utils.logging import AverageMeter, ProgressMeter
from utils.net_utils import constrainScoreByWhole
from torch.utils.tensorboard import SummaryWriter
import shutil
import numpy as np

writer = SummaryWriter()
__all__ = ["train", "validate", "modifier"]

def calculateGrad(model, fn_avg, fn_list, args):
    for n, m in model.named_modules():
        if hasattr(m, "scores") and m.prune:
            m.scores.grad.data += 1/(args.K-1)*(fn_list[0] - fn_avg)*getattr(m, 'stored_mask_0') + 1/(args.K-1)*(fn_list[1] - fn_avg)*getattr(m, 'stored_mask_1')
            if "IMP" in args.conv_type:
                # print("process grad in another way")
                m.scores.grad.data = torch.where(m.scores > 0.9, m.scores.grad.data, m.scores.grad.data * args.scaling_para)

def calculateGrad_pge(model, fn_avg, fn_list, args):
    for n, m in model.named_modules():
        if hasattr(m, "scores") and m.prune:
            m.scores.grad.data += 1/args.K*(fn_list[0]*getattr(m, 'stored_mask_0')) + 1/args.K*(fn_list[1]*getattr(m, 'stored_mask_1'))
            if "IMP" in args.conv_type:
                m.scores.grad.data = torch.where(m.scores > 0.9, m.scores.grad.data, m.scores.grad.data * args.scaling_para)

def calScalingPara(model, args):
    remaining_part = 0
    original_part = 0
    for n, m in model.named_modules():
        if hasattr(m, "scores") and m.prune:
            original_part += (torch.lt(m.scores, 0.9).float() * m.scores).sum().item()
            ge_loc = torch.ge(m.scores, 0.9).float()
            remaining_part += (m.scores.sum() - ge_loc.sum()).item()
    if remaining_part < 0:
        # print("remaining_part negative", remaining_part)
        remaining_part = 0
    if original_part == 0:
        args.scaling_para = 0
    else:
        args.scaling_para = remaining_part/original_part

def train(train_loader, model, ebd, criterion, optimizer, epoch, args, writer, weight_opt):
    loss_meter = AverageMeter("Loss", ":.3f")
    train_nll_meter = AverageMeter("train_nll", ":6.2f")
    train_penalty_meter = AverageMeter("train_penalty", ":6.2f")
    weight_norm_meter = AverageMeter("weight_norm", ":6.2f")
    train_acc_meter = AverageMeter("train_acc", ":6.2f")
    train_minacc_meter = AverageMeter("train_minacc", ":6.2f")
    train_majacc_meter = AverageMeter("train_majacc", ":6.2f")
    train_corr_meter = AverageMeter("train_corr", ":6.2f")
    v_meter = AverageMeter("v", ":6.4f")
    max_score_meter = AverageMeter("max_score", ":6.4f")
    l = [loss_meter, train_nll_meter, train_penalty_meter, weight_norm_meter, train_acc_meter, train_minacc_meter, train_majacc_meter, train_corr_meter]
    progress = ProgressMeter(
        len(train_loader),
        l,
        prefix=f"Epoch: [{epoch}]",
    )
    model.train()
    args.discrete = False
    args.val_loop = False
    args.num_batches = len(train_loader)

    for i, (train_x, train_y, train_g, train_c) in tqdm.tqdm(
            enumerate(train_loader), ascii=True, total=len(train_loader)
    ):
        train_x, train_y, train_g, train_c = train_x.cuda(), train_y.cuda().float(), train_g.cuda(), train_c.cuda()
        print(train_x.size(), train_y.size())
        train_c_label = (2*train_y-1)*train_c - train_y +1

        l, tn, tp, wn, t_acc, t_min_acc, t_maj_acc, t_corr = 0, 0, 0, 0, 0, 0, 0, 0
        if optimizer is not None:
            optimizer.zero_grad()
        if weight_opt is not None:
            weight_opt.zero_grad()
        fn_list = []

        for j in range(args.K):
            args.j = j
            if args.irm_type == "rex":
                loss_list = []
                train_logits = model(train_x)
                train_nll = 0
                for i in range(int(train_g.max()) + 1):
                    ei = (train_g == i).view(-1)
                    ey = train_y[ei]
                    el = train_logits[ei]
                    enll = criterion(el, ey)
                    train_nll += enll / (train_g.max() + 1)
                    loss_list.append(enll)
                loss_t = torch.stack(loss_list)
                train_penalty = ((loss_t - loss_t.mean()) ** 2).mean()
            elif args.irm_type == "irmv1":
                train_logits = ebd(train_g).view(-1, 1) * model(train_x)
                train_nll = criterion(train_logits, train_y)
                grad = torch.autograd.grad(
                    train_nll * args.envs_num, ebd.parameters(),
                    create_graph=True)[0]
                train_penalty = torch.mean(grad ** 2)
            elif args.irm_type == "irmv1b":
                e1 = (train_g == 0).view(-1).nonzero().view(-1)
                e2 = (train_g == 1).view(-1).nonzero().view(-1)
                e1 = e1[torch.randperm(len(e1))]
                e2 = e2[torch.randperm(len(e2))]
                s1 = torch.cat([e1[::2], e2[::2]])
                s2 = torch.cat([e1[1::2], e2[1::2]])
                train_logits = ebd(train_g).view(-1, 1) * model(train_x)

                train_nll1 = criterion(train_logits[s1], train_y[s1])
                train_nll2 = criterion(train_logits[s2], train_y[s2])
                train_nll = train_nll1 + train_nll2
                grad1 = torch.autograd.grad(train_nll1 * args.envs_num, ebd.parameters(), create_graph=True)[0]
                grad2 = torch.autograd.grad(train_nll2 * args.envs_num, ebd.parameters(), create_graph=True)[0]
                train_penalty = torch.mean(torch.abs(grad1 * grad2))
            else:
                raise Exception

            train_acc, train_minacc, train_majacc = args.eval_fn(train_logits, train_y, train_c)
            # t_c = np.corrcoef(torch.cat((torch.sigmoid(train_logits), train_c_label), 1).t().detach().cpu().numpy())[0,1]
            weight_norm = torch.tensor(0.).cuda()
            for n, m in model.named_modules():
                if hasattr(m, "weight") and m.weight is not None:
                    weight_norm += m.weight.norm().pow(2)
                if hasattr(m, "bias") and m.bias is not None:
                    weight_norm += m.bias.norm().pow(2)
            # for w in model.parameters():
            #     weight_norm += w.norm().pow(2)
            print("args.step args.penalty_anneal_iters", args.steps, args.penalty_anneal_iters)
            penalty_weight = args.penalty_weight if args.steps >= args.penalty_anneal_iters else 0.0
            print("penalty weights", penalty_weight)

            loss = train_nll + args.l2_regularizer_weight * weight_norm + penalty_weight * train_penalty
            if penalty_weight > 1.0:
                loss /= (1. + penalty_weight)
            loss = loss / args.K
            fn_list.append(loss.item()*args.K)
            loss.backward()
            # for n, m in model.named_modules():
            #     if hasattr(m, "scores"):
            #         print("pr grad mean", n, m.scores.grad.mean().item())
            #         print(m.train_weights)
            l = l + loss.item()
            tn = tn + train_nll.item() / args.K
            tp = tp + train_penalty / args.K
            wn = wn + args.l2_regularizer_weight * weight_norm / args.K
            t_acc = t_acc + train_acc.item() / args.K
            t_min_acc = t_min_acc + train_minacc.item() / args.K
            t_maj_acc = t_maj_acc + train_majacc.item() / args.K
            # t_corr = t_corr + t_c.item() / args.K

        fn_avg = l
        if not args.finetuning:
            if "ReinforceLOO" in args.conv_type:
                calculateGrad(model, fn_avg, fn_list, args)
            if args.conv_type == "Reinforce":
                calculateGrad_pge(model, fn_avg, fn_list, args)
        loss_meter.update(l, train_x.size(0))
        train_nll_meter.update(tn, train_x.size(0))
        train_penalty_meter.update(tp, train_x.size(0))
        weight_norm_meter.update(wn, train_x.size(0))
        train_acc_meter.update(t_acc, train_x.size(0))
        train_minacc_meter.update(t_min_acc, train_x.size(0))
        train_majacc_meter.update(t_maj_acc, train_x.size(0))
        train_corr_meter.update(t_corr, train_x.size(0))
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 3)

        if optimizer is not None:
            if "Dense" not in args.conv_type and not args.fix_subnet:
                if args.steps >= len(train_loader)*args.epochs*args.ts:
                    print("args.steps >= len(train_loader)*args.epochs*args.ts", args.steps, len(train_loader)*args.epochs*args.ts)
                    optimizer.step()
            else:
                optimizer.step()
        if weight_opt is not None:
            weight_opt.step()
        args.steps += 1
        if "Dense" not in args.conv_type:
            if not args.finetuning:
                with torch.no_grad():
                    constrainScoreByWhole(model, v_meter, max_score_meter)
                    if "IMP" in args.conv_type:
                        calScalingPara(model, args)
                        t = (len(train_loader) * epoch + i)
                        writer.add_scalar(f"train/scaling_para", args.scaling_para, global_step=t)
                        # print("scalingpara at this batch", args.scaling_para)
    progress.display(len(train_loader))
    progress.write_to_tensorboard(writer, prefix="train" if not args.finetuning else "train_ft", global_step=epoch)
    return train_acc_meter.avg, train_minacc_meter.avg, train_majacc_meter.avg, train_corr_meter.avg

def validate(val_loader, model, criterion, args, writer, epoch):
    loss_meter = AverageMeter("Loss", ":.3f")
    test_acc_meter = AverageMeter("test_acc", ":6.2f")
    test_minacc_meter = AverageMeter("test_minacc", ":6.2f")
    test_majacc_meter = AverageMeter("test_majacc", ":6.2f")

    loss_meter_d = AverageMeter("Loss_d", ":.3f")
    test_acc_meter_d = AverageMeter("test_acc_d", ":6.2f")
    test_minacc_meter_d = AverageMeter("test_minacc_d", ":6.2f")
    test_majacc_meter_d = AverageMeter("test_majacc_d", ":6.2f")
    test_corr_meter_d = AverageMeter("test_corr_d", ":6.2f")

    progress = ProgressMeter(
        len(val_loader), [loss_meter, test_acc_meter, test_minacc_meter, test_majacc_meter, loss_meter_d, test_acc_meter_d, test_minacc_meter_d, test_majacc_meter_d, test_corr_meter_d], prefix="Test: "
    )
    args.val_loop = True
    if args.use_running_stats:
        model.eval()
    if writer is not None:
        for n, m in model.named_modules():
            if hasattr(m, "scores") and m.prune:
                writer.add_histogram(n, m.scores)
    with torch.no_grad():
        for i, (test_x, test_y, test_g, test_c) in tqdm.tqdm(enumerate(val_loader), ascii=True, total=len(val_loader)):
            test_x, test_y, test_g, test_c = test_x.cuda(), test_y.cuda().float(), test_g.cuda(), test_c.cuda()
            test_c_label = (2 * test_y - 1) * test_c - test_y + 1
            args.discrete = False
            test_logits = model(test_x)
            loss = criterion(test_logits, test_y)
            test_acc, test_minacc, test_majacc = args.eval_fn(test_logits, test_y, test_c)
            loss_meter.update(loss.item(), test_x.size(0))
            test_acc_meter.update(test_acc.item(), test_x.size(0))
            test_minacc_meter.update(test_minacc.item(), test_x.size(0))
            test_majacc_meter.update(test_majacc.item(), test_x.size(0))
            args.discrete = True
            test_logits_d = model(test_x)
            loss_d = criterion(test_logits_d, test_y)
            test_acc_d, test_minacc_d, test_majacc_d = args.eval_fn(test_logits_d, test_y, test_c)
            loss_meter_d.update(loss_d.item(), test_x.size(0))
            test_acc_meter_d.update(test_acc_d.item(), test_x.size(0))
            test_minacc_meter_d.update(test_minacc_d.item(), test_x.size(0))
            test_majacc_meter_d.update(test_majacc_d.item(), test_x.size(0))
            test_corr_meter_d.update(np.corrcoef(torch.cat((torch.sigmoid(test_logits_d), test_c_label), 1).t().detach().cpu().numpy())[0,1])
            if i % args.print_freq == 0:
                progress.display(i)
        progress.display(len(val_loader))
        if writer is not None:
            progress.write_to_tensorboard(writer, prefix="test" if not args.finetuning else "test_ft", global_step=epoch)
    return test_acc_meter_d.avg, test_minacc_meter_d.avg, test_majacc_meter_d.avg, loss_meter_d.avg, test_corr_meter_d.avg


def modifier(args, epoch, model):
    return
