import torch
from heapq import nsmallest
from operator import itemgetter
from resnet_kuangliu import ResNet18_kuangliu_c
from torchvision.models import ResNet
from torch.autograd import Variable
import lrp_utils
import torch.nn as nn
import copy

PYTORCH_ENABLE_MPS_FALLBACK = 1
EXTREMELY_HIGH_VALUE = 99999999

# This code is from the authors of the paper "Pruning by Explaining: A Novel Criterion for  Deep Neural Network Pruning"
# with small modifications to fit the current codebase
# https://github.com/seulkiyeom/LRP_Pruning_toy_example

lrp_layer2method = {
    "nn.ReLU": lrp_utils.relu_wrapper_fct,
    "nn.BatchNorm2d": lrp_utils.relu_wrapper_fct,
    "nn.Conv2d": lrp_utils.conv2d_beta0_wrapper_fct,
    "nn.Linear": lrp_utils.linearlayer_eps_wrapper_fct,
    "nn.AdaptiveAvgPool2d": lrp_utils.adaptiveavgpool2d_wrapper_fct,
    "nn.MaxPool2d": lrp_utils.maxpool2d_wrapper_fct,
    "sum_stacked2": lrp_utils.eltwisesum_stacked2_eps_wrapper_fct,
}


def get_candidates_to_prune(model, pruning_ratio, X_test, y_test_true):

    lrp_model = copy.deepcopy(model)
    if isinstance(model, ResNet):
        wrapper_model = ResNet18_kuangliu_c()
        lrp_params = {
            "conv2d_ignorebias": True,
            "eltwise_eps": 1e-6,
            "linear_eps": 1e-6,
            "pooling_eps": 1e-6,
            "use_zbeta": True,
        }

        wrapper_model.copyfromresnet(
            model, lrp_params=lrp_params, lrp_layer2method=lrp_layer2method
        )
        pruner = FilterPruner(wrapper_model)
        train_epoch(X_test, y_test_true, pruner, lrp_model, rank_filters=True)
    else:
        pruner = FilterPruner(lrp_model)
        output = pruner.forward_lrp(X_test)

        T = torch.zeros_like(output)
        for ii in range(y_test_true.size(0)):
            T[ii, y_test_true[ii]] = 1.0
        pruner.backward_lrp(output * T)
        pruner.filter_ranks = {
            pruner.activation_to_layer[i]: v for i, v in pruner.filter_ranks.items()
        }

    pruner.normalize_ranks_per_layer()
    return pruner.get_pruning_plan(pruning_ratio)


def train_epoch(X, y, pruner, model, optimizer=None, rank_filters=True):

    data, target = Variable(X), Variable(y)
    train_batch(pruner, model, optimizer, 0, data, target, rank_filters)


def train_batch(pruner, model, optimizer, batch_idx, batch, label, rank_filters):
    model.train()
    model.zero_grad()
    if optimizer is not None:
        optimizer.zero_grad()

    with torch.enable_grad():
        output = model(batch)

    if rank_filters:  # for pruning
        batch.requires_grad = True

        with torch.enable_grad():
            output = pruner.model(batch)

        print("Computing LRP")

        # Map the original targets to [0, num_selected_classes)
        T = torch.zeros_like(output)
        for ii in range(len(label)):
            T[ii, label[ii]] = 1.0

        # Multiply output with target
        lrp_anchor = output * T / (output * T).sum(dim=1, keepdim=True)
        output.backward(lrp_anchor, retain_graph=True)

        pruner.compute_filter_criterion("conv", criterion="lrp")


def fhook(self, input, output):
    self.input = input[0]
    self.output = output.data


class FilterPruner:
    def __init__(self, model):
        self.model = model
        self.reset()
        self.cuda = torch.cuda.is_available()
        self.relevance = False
        self.method_type = "ICLR"  # "ICLR", "grad", "weight", "taylor"
        self.norm = False

    def reset(self):
        self.filter_ranks = {}
        self.forward_hook()

    def forward_hook(self):
        # For Forward Hook
        for module in self.model.modules():
            if len(list(module.children())) != 0:
                continue
            module.register_forward_hook(fhook)

    def forward_lrp(self, x):
        self.activation_to_layer = {}
        self.grad_index = 0

        self.activation_index = 0
        for name, module in self.model.named_modules():
            if len(list(module.children())) != 0:
                continue
            if name == "classifier.0":
                x = torch.flatten(x, 1)
                x = self.model.classifier(x)
                break
            elif name == "fc":
                x = torch.flatten(x, 1)
                x = module(x)
            else:
                x = module(x)
                if isinstance(module, torch.nn.modules.conv.Conv2d):
                    self.activation_to_layer[self.activation_index] = name
                    self.activation_index += 1

        return x

    def backward_lrp(self, R, relevance_method="z"):
        for index, module in enumerate(list(self.model.modules())[::-1]):  # 접근 방법
            if len(list(module.children())) != 0:
                continue
            if isinstance(module, torch.nn.modules.conv.Conv2d):  # !!!
                activation_index = self.activation_index - self.grad_index - 1

                values = (
                    torch.sum(R, dim=0, keepdim=True)
                    .sum(dim=2, keepdim=True)
                    .sum(dim=3, keepdim=True)[0, :, 0, 0]
                    .data
                )
                values = values.cuda() if self.cuda else values

                if activation_index not in self.filter_ranks:
                    self.filter_ranks[activation_index] = (
                        torch.FloatTensor(R.size(1)).zero_().cuda()
                        if self.cuda
                        else torch.FloatTensor(R.size(1)).zero_()
                    )

                self.filter_ranks[activation_index] += values
                self.grad_index += 1

            R = lrp(module, R.data, relevance_method, 1)

    def compute_filter_criterion(
        self, layer_type="conv", relevance_method="z", criterion="lrp"
    ):

        self.activation_index = 0
        for name, module in self.model.named_modules():
            if criterion == "lrp" or criterion == "weight":
                if hasattr(module, "module"):
                    if isinstance(module.module, nn.Conv2d) and layer_type == "conv":
                        if name not in self.filter_ranks:
                            self.filter_ranks[name] = torch.zeros(
                                module.relevance.shape[1]
                            )
                        # else:
                        #     print("here")

                        if criterion == "lrp":
                            values = torch.sum(module.relevance.abs(), dim=(0, 2, 3))
                        elif criterion == "weight":
                            values = torch.sum(
                                module.module.weight.abs(), dim=(1, 2, 3)
                            )

                        if hasattr(module, "output_mask"):
                            values[module.output_mask == 0] = EXTREMELY_HIGH_VALUE
                        self.filter_ranks[name] += (
                            values.cpu() if torch.cuda.is_available() else values
                        )

    def normalize_ranks_per_layer(self):
        for i in self.filter_ranks:
            # average over trials - LRP case (this is not normalization !!)
            v = self.filter_ranks[i]
            v = v / torch.sum(v)  # torch.sum(v) = total number of dataset
            self.filter_ranks[i] = v.cpu()

    def get_pruning_plan(self, pruning_ratio):
        all_indices = []
        for ratio in pruning_ratio:
            ranked_filters = self.lowest_ranking_filters(ratio)

            ranked_filters = [x[:2] for x in ranked_filters]
            layer_to_filter_indices = {}
            for name, value in ranked_filters:
                layer_to_filter_indices.setdefault(name, []).append(value)
            all_indices.append(layer_to_filter_indices)

        return all_indices

    def lowest_ranking_filters(self, pruning_ratio):
        data = []
        num_filters = 0

        # Each layer has a list of filter ranks 
        for i in sorted(self.filter_ranks.keys()):
            num_filters += self.filter_ranks[i].size(0)
            for j in range(self.filter_ranks[i].size(0)):
                data.append((i, j, self.filter_ranks[i][j]))
                # data 변수에 모든 layer의 모든 filter의 값을 쭈욱 나열 시킨다.

        num_filters_to_prune = int(num_filters * pruning_ratio)
        # data = [(layer_name, filter_index, filter_rank), ...]
        # Choose the num smallest filter ranks to prune
        filters_to_prune = nsmallest(
            num_filters_to_prune, data, itemgetter(2)
        ) 

        return filters_to_prune


def lrp(module, R, lrp_var=None, param=None):
    with torch.no_grad():
        if isinstance(module, torch.nn.modules.linear.Linear):  # for Linear
            return Linear(module, R, lrp_var, param)
        elif isinstance(module, torch.nn.modules.conv.Conv2d):  # for Conv
            return Convolution(module, R, lrp_var, param)
        elif isinstance(module, torch.nn.modules.activation.ReLU) or isinstance(
            module, torch.nn.modules.dropout.Dropout
        ):
            return R
        elif isinstance(module, torch.nn.modules.activation.LogSoftmax):
            return module.input * R
        elif (
            isinstance(module, torch.nn.modules.pooling.AvgPool2d)
            or isinstance(module, torch.nn.modules.pooling.MaxPool2d)
            or isinstance(module, torch.nn.modules.pooling.AdaptiveAvgPool2d)
        ):
            return Pooling(module, R, lrp_var, param)
        else:
            NameError("No function")


def gradprop_linear(weight, DY):
    return torch.mm(DY, weight)


def Linear(module, R, lrp_var=None, param=None):
    R_shape = R.shape
    if len(R_shape) != 2:
        output_shape = module.output.shape
        R = torch.reshape(R, output_shape)

    if lrp_var is None or lrp_var.lower() == "none" or lrp_var.lower() == "simple":
        Z = torch.nn.functional.linear(module.input, module.weight) + 1e-9
        S = R / Z
        C = torch.mm(S, module.weight)
        Rn = module.input * C
        return Rn

    elif lrp_var.lower() == "alphabeta" or lrp_var.lower() == "alpha":
        alpha = param
        beta = 1 - alpha

        V = module.weight
        VP = module.weight.clamp(min=0.0)
        VN = module.weight.clamp(max=0.0)

        X = module.input + 1e-9

        ZA = torch.nn.functional.linear(X, VP)
        ZB = torch.nn.functional.linear(X, VN)

        SA = alpha * R / ZA
        SB = beta * R / ZB

        Rn = X * (gradprop_linear(VP, SA) + gradprop_linear(VN, SB))

        return Rn

    elif lrp_var.lower() == "z":
        V = module.weight.clamp(min=0.0)

        Z = torch.nn.functional.linear(module.input, V) + 1e-9
        S = R / Z
        C = torch.mm(S, V)
        Rn = module.input * C
        return Rn

    elif lrp_var.lower() == "w^2" or lrp_var.lower() == "ww":
        return None
        # return _ww_lrp()


def Pooling(module, R, lrp_var=None, param=None):
    R_shape = R.size()
    output_shape = module.output.shape
    if len(R_shape) != 4:
        R = torch.reshape(R, output_shape)

    if isinstance(module, torch.nn.modules.pooling.MaxPool2d):
        pool = nn.MaxPool2d(
            module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            ceil_mode=module.ceil_mode,
            return_indices=True,
        )
        unpool = nn.MaxUnpool2d(
            module.kernel_size, stride=module.stride, padding=module.padding
        )
        Z, indice = pool(module.input)
        S = R / (Z + 1e-9)

        C = unpool(S, indice)
    elif isinstance(module, torch.nn.modules.pooling.AdaptiveAvgPool2d):
        pool = nn.AdaptiveAvgPool2d(module.output_size)
        unpool = nn.Upsample(module.input.shape[2:], mode="bilinear")

        Z = pool(module.input)
        S = R / (Z + 1e-9)

        C = unpool(S)
    return module.input * C


def Convolution(module, R, lrp_var=None, param=None):
    R_shape = R.size()
    output_shape = module.output.shape
    if len(R_shape) != 4:
        R = torch.reshape(R, output_shape)
    N, NF, Hout, Wout = R.size()

    if lrp_var is None or lrp_var.lower() == "none" or lrp_var.lower() == "simple":
        V = module.weight
        Z = (
            torch.nn.functional.conv2d(
                module.input,
                V,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
            )
            + 1e-9
        )
        S = R / Z
        C = torch.nn.functional.conv_transpose2d(
            S,
            V,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
        )
        return module.input * C

    elif lrp_var.lower() == "alphabeta" or lrp_var.lower() == "alpha":
        alpha = param
        beta = 1 - alpha

        VP = module.weight.clamp(min=0.0)
        VN = module.weight.clamp(max=0.0)

        X = module.input + 1e-9

        ZA = torch.nn.functional.conv2d(
            X,
            VP,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
        )
        ZB = torch.nn.functional.conv2d(
            X,
            VN,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
        )

        SA = alpha * R / ZA
        SB = beta * R / ZB

        CP = torch.nn.functional.conv_transpose2d(
            SA,
            VP,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
        )
        CN = torch.nn.functional.conv_transpose2d(
            SB,
            VN,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
        )

        Rn = X * (CP + CN)

        return Rn

    elif lrp_var.lower() == "z":
        V = module.weight.clamp(min=0.0)
        Z = (
            torch.nn.functional.conv2d(
                module.input,
                V,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
            )
            + 1e-9
        )
        S = R / Z
        if module.stride[0] > 1:
            C = torch.nn.functional.conv_transpose2d(
                S,
                V,
                stride=module.stride,
                padding=module.padding,
                output_padding=1,
                dilation=module.dilation,
                groups=module.groups,
            )
        else:
            C = torch.nn.functional.conv_transpose2d(
                S,
                V,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
            )
        return module.input * C

    elif lrp_var.lower() == "w^2" or lrp_var.lower() == "ww":
        return None

    elif lrp_var.lower() == "first":
        lowest = -1.0
        highest = 1.0

        V = module.weight
        VN = module.weight.clamp(max=0.0)
        VP = module.weight.clamp(min=0.0)
        X, L, H = (
            module.input,
            module.input * 0 + lowest,
            module.input * 0 + highest,
        )

        ZX = torch.nn.functional.conv2d(
            X,
            V,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
        )
        ZL = torch.nn.functional.conv2d(
            L,
            VP,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
        )
        ZH = torch.nn.functional.conv2d(
            H,
            VN,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
        )
        Z = ZX - ZL - ZH + 1e-9
        S = R / Z

        C = torch.nn.functional.conv_transpose2d(
            S,
            V,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
        )
        CP = torch.nn.functional.conv_transpose2d(
            S,
            VP,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
        )
        CN = torch.nn.functional.conv_transpose2d(
            S,
            VN,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
        )

        return X * C - L * CP - H * CN
