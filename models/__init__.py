from models.resnet import ResNet18, ResNet50, ResNet101, WideResNet50_2, WideResNet101_2
from models.resnet_cifar import cResNet18, cResNet50, cResNet101
from models.frankle import FC, Conv2, Conv4, Conv6, Conv4Wide, Conv8, Conv6Wide, Conv2MNIST
from models.vgg_cifar import vgg11, vgg11_bn
from models.vgg_cifar_new import vgg11_new, vgg11_bn_new, vgg16_new, vgg19_new, vgg19_bn_new
from models.vgg_cifar_new_fc import vgg11_new_fc, vgg11_bn_new_fc, vgg16_new_fc, vgg16_bn_new_fc, vgg19_new_fc, vgg19_bn_new_fc
from models.resnet_cifar_new import resnet32, resnet20
from models.mobilenetv1 import MobileNetV1
from models.mobilenetv3 import mobilenetv3_large, mobilenetv3_small
from models.mobilenetv3_origin import mobilenetv3_large_origin, mobilenetv3_small_origin
from models.vgg_cifar_new_fc_dropout import vgg11_bn_new_fc_drop, vgg16_bn_new_fc_drop
from models.lenet5 import LeNet5
from models.resnet_cifar_new_1x import resnet20_1x, resnet32_1x
from models.vgg_relu_bn import vgg19_relu_bn
from models.resnet_b import resnet20_1w1a

from models.irm_models import MLP, EBD, resnet18_sepfc_us, resnet50_sepfc_us, MLPFull
__all__ = [
    "ResNet18",
    "ResNet50",
    "ResNet101",
    "cResNet18",
    "cResNet50",
    "WideResNet50_2",
    "WideResNet101_2",
    "FC",
    "Conv2",
    "Conv2MNIST",
    "Conv4",
    "Conv6",
    "Conv4Wide",
    "Conv8",
    "Conv6Wide",
    "vgg11",
    "vgg11_bn",
    "vgg11_new",
    "vgg11_bn_new",
    "vgg16_new",
    "vgg16_bn_new_fc",
    "vgg19_new",
    "vgg19_bn_new",
    "resnet32",
    "vgg19_bn_new_fc",
    "Linear",
    "MobileNetV1",
    "mobilenetv3_large",
    "mobilenetv3_origin",
    "vgg11_bn_speed_up",
    "vgg11_speed_up",
    "vgg11_reg",
    "vgg11_bn_new_fc_drop",
    "vgg16_bn_new_fc_drop",
    "LeNet5",
    "resnet20",
    "wideresnet2810",
    "wideresnet285",
    "wideresnet28_wr"
    "resnet20_1x",
    "resnet32_1x",
    "StackedLSTMMask",
    "vgg19_relu_bn",
    "resnet20_1w1a",
    "vip_s7",
    "vip_s1_c10",
    "vip_s1_c10_dense",
    "vip_s1_c10_div6",
    "vip_s7_linear",
    "vip_s1_c10_linear",
    "vip_s1_c10_div6_linear",
    "MLP",
    "EBD",
    "resnet18_sepfc_us",
    "resnet50_sepfc_us",
    "MLPFull"
]
