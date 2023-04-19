import argparse
import sys
import yaml

from configs import parser as _parser

args = None


def parse_arguments():
    parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")

    # General Config
    parser.add_argument(
        "--data", help="path to dataset base directory", default="/mnt/disk1/datasets"
    )
    parser.add_argument("--optimizer", help="Which optimizer to use", default="sgd")
    parser.add_argument("--weight_opt", help="Which optimizer to use for weight", default="sgd")
    parser.add_argument("--set", help="name of dataset", type=str, default="ImageNet")
    parser.add_argument(
        "-a", "--arch", metavar="ARCH", default="ResNet18", help="model architecture"
    )
    parser.add_argument(
        "--config", help="Config file to use (see configs dir)", default=None
    )
    parser.add_argument(
        "--log-dir", help="Where to save the runs. If None use ./runs", default=None
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=1,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 1)",
    )
    parser.add_argument(
        "--epochs",
        default=90,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--start-epoch",
        default=None,
        type=int,
        metavar="N",
        help="manual epoch number (useful on restarts)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=256,
        type=int,
        metavar="N",
        help="mini-batch size (default: 256), this is the total "
             "batch size of all GPUs on the current node when "
             "using Data Parallel or Distributed Data Parallel",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.1,
        type=float,
        metavar="LR",
        help="initial learning rate",
        dest="lr",
    )
    parser.add_argument(
        "--warmup_length", default=0, type=int, help="Number of warmup iterations"
    )
    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="momentum"
    )
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "-p",
        "--print-freq",
        default=10,
        type=int,
        metavar="N",
        help="print frequency (default: 10)",
    )
    parser.add_argument("--num-classes", default=10, type=int)
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--resume_train_weights",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "-e",
        "--evaluate",
        dest="evaluate",
        action="store_true",
        help="evaluate model on validation set",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        default=None,
        type=str,
        help="use pre-trained model",
    )


    parser.add_argument(
        "--pretrained_distill",
        dest="pretrained_distill",
        default=None,
        type=str,
        help="use pre-trained model",
    )
    parser.add_argument(
        "--alpha_distill",
        default=0.95,
        type=float,
        metavar="W",
        help="alpha_distill",
    )

    parser.add_argument(
        "--T_distill",
        default=2,
        type=float,
        metavar="W",
        help="T_distill",
    )

    parser.add_argument(
        "--seed", default=None, type=int, help="seed for initializing training. "
    )
    parser.add_argument(
        "--multigpu",
        default=None,
        type=lambda x: [int(a) for a in x.split(",")],
        help="Which GPUs to use for multigpu training",
    )

    # Learning Rate Policy Specific
    parser.add_argument(
        "--lr-policy", default="constant_lr", help="Policy for the learning rate."
    )
    parser.add_argument(
        "--lr-adjust", default=30, type=int, help="Interval to drop lr"
    )
    parser.add_argument(
        "--lr-gamma", default=0.1, type=int, help="Multistep multiplier"
    )
    parser.add_argument(
        "--name", default=None, type=str, help="Experiment name to append to filepath"
    )
    parser.add_argument(
        "--save-every", default=-1, type=int, help="Save every ___ epochs"
    )
    parser.add_argument(
        "--prune-rate",
        default=0.0,
        help="Amount of pruning to do during sparse training",
        type=float,
    )

    parser.add_argument(
        "--param-prune-rate",
        default=0.0,
        help="Amount of param pruning to do during sparse training",
        type=float,
    )

    parser.add_argument(
        "--pr_start",
        default=1.0,
        help="Amount of pruning rate for start",
        type=float,
    )

    parser.add_argument(
        "--low-data", default=1, help="Amount of data to use", type=float
    )
    parser.add_argument(
        "--width-mult",
        default=1.0,
        help="How much to vary the width of the network.",
        type=float,
    )
    parser.add_argument(
        "--nesterov",
        default=False,
        action="store_true",
        help="Whether or not to use nesterov for SGD",
    )
    parser.add_argument(
        "--threetimes",
        default=False,
        action="store_true",
        help="Whether use threetimes",
    )
    parser.add_argument(
        "--random-subnet",
        action="store_true",
        help="Whether or not to use a random subnet when fine tuning for lottery experiments",
    )
    parser.add_argument(
        "--one-batch",
        action="store_true",
        help="One batch train set for debugging purposes (test overfitting)",
    )
    parser.add_argument(
        "--conv-type", type=str, default=None, help="What kind of sparsity to use"
    )
    parser.add_argument(
        "--freeze-weights",
        action="store_true",
        help="Whether or not to train only subnet (this freezes weights)",
    )
    parser.add_argument(
        "--st",
        action="store_true",
        help="st",
    )

    parser.add_argument("--mode", default="fan_in", help="Weight initialization mode")
    parser.add_argument(
        "--nonlinearity", default="relu", help="Nonlinearity used by initialization"
    )
    parser.add_argument("--bn-type", default=None, help="BatchNorm type")
    parser.add_argument(
        "--init", default="kaiming_normal", help="Weight initialization modifications"
    )
    parser.add_argument(
        "--no-bn-decay", action="store_true", default=False, help="No batchnorm decay"
    )
    parser.add_argument(
        "--scale-fan", action="store_true", default=False, help="scale fan"
    )
    parser.add_argument(
        "--first-layer-dense", action="store_true", help="First layer dense or sparse"
    )
    parser.add_argument(
        "--last-layer-dense", action="store_true", help="Last layer dense or sparse"
    )
    parser.add_argument(
        "--approx", action="store_true", help="Use Approx Discrete Mode"
    )
    parser.add_argument(
        "--zero", action="store_true", help="Use Zero Discrete Mode"
    )
    parser.add_argument(
        "--flip", action="store_true", help="Use Flip Mask Randomly Mode"
    )
    parser.add_argument(
        "--bs", action="store_true", help="Sample from Score"
    )
    parser.add_argument(
        "--gumbel_sample", action="store_true", help="Sample from gumbel"
    )
    parser.add_argument(
        "--multiply_prob", action="store_true", help="multiply probability"
    )
    parser.add_argument(
        "--no_multiply", action="store_true", help="no multiply probability"
    )
    parser.add_argument(
        "--multiply_cont", action="store_true", help="multiply continuous mask"
    )
    parser.add_argument(
        "--multiply_prob_bs", action="store_true", help="multiply probability and bs"
    )
    parser.add_argument(
        "--indiv", action="store_true", help="individual temperature for different channels"
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        help="Label smoothing to use, default 0.0",
        default=None,
    )
    parser.add_argument(
        "--first-layer-type", type=str, default=None, help="Conv type of first layer"
    )
    parser.add_argument(
        "--trainer", type=str, default="default", help="cs, ss, or standard training"
    )
    parser.add_argument(
        "--score-init-constant",
        type=float,
        default=None,
        help="Sample Baseline Subnet Init",
    )

    parser.add_argument(
        "--K",
        type=int,
        default=20,
        help="Sample K nets",
    )

    parser.add_argument(
        "--update_freq",
        type=int,
        default=20,
        help="Update freq",
    )

    parser.add_argument(
        "--D",
        type=float,
        default=0.01,
        help="Min difference",
    )

    parser.add_argument(
        "--T",
        type=float,
        default=1,
        help="Temperature for gumbel training",
    )

    parser.add_argument(
        "--TA",
        default=False,
        action="store_true",
        help="Tempearature annealing",
    )

    parser.add_argument(
        "--TA2",
        default=False,
        action="store_true",
        help="Tempearature annealing2",
    )

    parser.add_argument(
        "--TA_grow",
        default=False,
        action="store_true",
        help="Tempearature annealing grow",
    )

    parser.add_argument(
        "--center",
        default=False,
        action="store_true",
        help="Score center at (-0.5, 0.5) or (0, 1)",
    )

    parser.add_argument(
        "--straight_through",
        default=False,
        action="store_true",
        help="Whether ignore gradient from sigmoid",
    )

    parser.add_argument(
        "--PLA-factor",
        type=float,
        default=0.1,
        help="PLA-factor",
    )

    parser.add_argument(
        "--PLA-patience",
        type=float,
        default=10,
        help="PLA-patience",
    )

    parser.add_argument(
        "--gradient-loss-para",
        type=float,
        default=0,
        help="gradient_loss_para",
    )

    parser.add_argument(
        "--abs-loss-para",
        type=float,
        default=0,
        help="abs-loss-para",
    )

    parser.add_argument(
        "--thres",
        type=float,
        default=0.9,
        help="thres",
    )

    parser.add_argument(
        "--runs-name",
        type=str,
        default="name_random",
        help="name",
    )

    parser.add_argument(
        "--resume-compare-loss1",
        type=str,
        default="",
        help="resume-compare-loss1",
    )

    parser.add_argument(
        "--resume-compare-loss2",
        type=str,
        default="",
        help="resume-compare-loss2",
    )

    parser.add_argument(
        "--init_weights",
        type=str,
        default="",
        help="init weights loc",
    )

    parser.add_argument(
        "--trained_mask",
        type=str,
        default="",
        help="trained mask loc",
    )

    parser.add_argument(
        "--weight_rescaling",
        default=False,
        action="store_true",
        help="Whether use weight_rescaling",
    )

    parser.add_argument(
        "--constrain_by_layer",
        default=False,
        action="store_true",
        help="Whether constrain by layer",
    )

    parser.add_argument(
        "--weight_rescaling_data",
        default=False,
        action="store_true",
        help="Whether use weight_rescaling_data",
    )

    parser.add_argument(
        "--use_running_stats",
        default=False,
        action="store_true",
        help="Whether use bn running stats",
    )

    parser.add_argument(
        "--not_clipping",
        default=False,
        action="store_true",
        help="Whether use clipping",
    )

    parser.add_argument(
        "--rescaling_para",
        default=False,
        action="store_true",
        help="Whether use rescaling para",
    )

    parser.add_argument(
        "--lasso_para",
        type=float,
        default=0,
        help="lasso para",
    )

    parser.add_argument(
        "--dont_freeze_weights",
        default=False,
        action="store_true",
        help="Whether not freeze weights",
    )

    parser.add_argument(
        "--iterative",
        default=False,
        action="store_true",
        help="Whether use iterative pruning",
    )

    parser.add_argument(
        "--prob_by_weight",
        default=False,
        action="store_true",
        help="Whether use probability by weight assignment",
    )

    parser.add_argument(
        "--rescale_at_fix_subnet",
        default=False,
        action="store_true",
        help="Whether rescale weights at fix subnet",
    )

    parser.add_argument(
        "--train_weights_at_the_same_time",
        default=False,
        action="store_true",
        help="Whether train_weights at the same time",
    )

    parser.add_argument(
        "--sample_from_training_set",
        default=False,
        action="store_true",
        help="Whether sample from training set",
    )

    parser.add_argument(
        "--load_true_para",
        default=False,
        action="store_true",
        help="Whether load true para",
    )

    parser.add_argument(
        "--distill",
        default=False,
        action="store_true",
        help="Whether distill",
    )

    parser.add_argument(
        "--finetune",
        default=False,
        action="store_true",
        help="Whether finetune",
    )

    parser.add_argument(
        "--stablize",
        default=False,
        action="store_true",
        help="Whether stablize",
    )

    parser.add_argument(
        "--prev_best",
        type=float,
        default=0,
        help="previous best acc1",
    )

    parser.add_argument(
        "--weight_opt_lr",
        type=float,
        default=0.1,
        help="lr for weight training at the same time",
    )

    parser.add_argument(
        "--n",
        type=int,
        default=500,
        help="n",
    )

    parser.add_argument(
        "--ts",
        type=float,
        default=0.16,
        help="ts",
    )

    parser.add_argument(
        "--te",
        type=float,
        default=0.6,
        help="te",
    )

    parser.add_argument(
        "--d",
        type=int,
        default=20000,
        help="d",
    )

    parser.add_argument(
        "--s",
        type=int,
        default=80,
        help="ns",
    )

    parser.add_argument(
        "--c",
        type=float,
        default=0.75,
        help="c",
    )

    parser.add_argument(
        "--init_prob",
        default=False,
        action="store_true",
        help="Whether init prob",
    )

    parser.add_argument(
        "--thres_before",
        type=float,
        default=1e-3,
        help="thres_before",
    )

    parser.add_argument(
        "--wide_ratio",
        type=float,
        default=1e-3,
        help="wide_ratio",
    )

    parser.add_argument(
        "--noise",
        type=float,
        default=1,
        help="noise",
    )

    parser.add_argument(
        "--cal_p_q",
        default=False,
        action="store_true",
        help="Whether cal p q ratio",
    )

    parser.add_argument(
        "--just_finetune",
        default=False,
        action="store_true",
        help="Whether just finetune",
    )

    parser.add_argument(
        "--snip",
        default=False,
        action="store_true",
        help="Whether use snip",
    )

    parser.add_argument('--envs_num', type=int, default=2)
    parser.add_argument('--classes_num', type=int, default=2)
    parser.add_argument('--l2_regularizer_weight', type=float, default=0.001)
    parser.add_argument('--data_num', type=int, default=50000)
    parser.add_argument('--env_type', default="linear", type=str, choices=["2_group", "cos", "linear"])
    parser.add_argument('--irm_type', default="irmv1", type=str)
    parser.add_argument('--hidden_dim', type=int, default=390)
    parser.add_argument('--penalty_anneal_iters', type=int, default=200)
    parser.add_argument('--penalty_weight', type=float, default=10000.0)
    parser.add_argument('--grayscale_model', type=int, default=0)
    parser.add_argument('--weight_lr_schedule', default=False, action="store_true")

    parser.add_argument('--fix_subnet', default=False, action="store_true")
    parser.add_argument('--freeze_weight', default=False, action="store_true")
    parser.add_argument('--step', default="ours", type=str)
    parser.add_argument('--runs_name', type=str)
    parser.add_argument('--prior_sd_coef', type=float, default=0)
    parser.add_argument('--dim_inv', type=int, default=2)
    parser.add_argument('--variance_gamma', type=float, default=1.0)
    parser.add_argument('--dim_spu', type=int, default=10)
    parser.add_argument('--image_scale', type=int, default=32)
    parser.add_argument('--cons_ratio', type=str, default="0.999_0.7_0.1")
    parser.add_argument('--noise_ratio', type=float, default=0.05)
    parser.add_argument('--step_gamma', type=float, default=0.1)
    parser.add_argument('--step_round', type=int, default=3)
    parser.add_argument('--inner_steps', type=int, default=1)

    args = parser.parse_args()

    # Allow for use from notebook without config file
    if len(sys.argv) > 1:
        get_config(args)

    return args


def get_config(args):
    # get commands from command line
    override_args = _parser.argv_to_vars(sys.argv)

    # load yaml file
    yaml_txt = open(args.config).read()

    # override args
    loaded_yaml = yaml.load(yaml_txt, Loader=yaml.FullLoader)
    for v in override_args:
        loaded_yaml[v] = getattr(args, v)

    print(f"=> Reading YAML config from {args.config}")
    args.__dict__.update(loaded_yaml)


def run_args():
    global args
    if args is None:
        args = parse_arguments()


run_args()
