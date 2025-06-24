import torch
from heapq import nsmallest
from operator import itemgetter
import torch.nn as nn
import copy

PYTORCH_ENABLE_MPS_FALLBACK = 1

# This code is from the authors of the paper "Pruning by Explaining: A Novel Criterion for  Deep Neural Network Pruning"
# with small modifications to fit the current codebase
# https://github.com/seulkiyeom/LRP_Pruning_toy_example


def get_candidates_to_prune(model, num_filters_to_prune, X_test, y_test_true):

    pruner = FilterPruner(model)
    output = pruner.forward_lrp(X_test)
    T = torch.zeros_like(output)
    for ii in range(y_test_true.size(0)):
        T[ii, y_test_true[ii]] = 1.0
    pruner.backward_lrp(output * T)

    pruner.normalize_ranks_per_layer()
    return pruner.get_pruning_plan(num_filters_to_prune)


def fhook(self, input, output):
    self.input = input[0]
    self.output = output.data


class FilterPruner:
    def __init__(self, model):
        self.model = copy.deepcopy(model)
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
        for name, module in self.model.features._modules.items():
            module.register_forward_hook(fhook)
        self.model.avgpool.register_forward_hook(fhook)
        for name, module in self.model.classifier._modules.items():
            module.register_forward_hook(fhook)

    def forward_lrp(self, x):
        self.activation_to_layer = {}
        self.grad_index = 0

        self.activation_index = 0
        for layer, (name, module) in enumerate(self.model.features._modules.items()):
            x = module(x)
            if isinstance(module, torch.nn.modules.conv.Conv2d):
                self.activation_to_layer[self.activation_index] = layer
                self.activation_index += 1

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)  # flatten the tensor
        return self.model.classifier(x)

    def backward_lrp(self, R, relevance_method="z"):
        for name, module in enumerate(self.model.classifier[::-1]):  # 접근 방법
            # print(R[10,:].sum())
            R = lrp(module, R.data, relevance_method, 1)

        R = lrp(self.model.avgpool, R.data, relevance_method, 1)

        for name, module in enumerate(self.model.features[::-1]):  # 접근 방법
            # print(R[10, :].sum())
            if isinstance(module, torch.nn.modules.conv.Conv2d):  # !!!
                activation_index = self.activation_index - self.grad_index - 1

                values = (
                    torch.sum(R, dim=0, keepdim=True)
                    .sum(dim=2, keepdim=True)
                    .sum(dim=3, keepdim=True)[0, :, 0, 0]
                    .data
                )

                if activation_index not in self.filter_ranks:
                    self.filter_ranks[activation_index] = (
                        torch.FloatTensor(R.size(1)).zero_().cuda()
                        if self.cuda
                        else torch.FloatTensor(R.size(1)).zero_()
                    )

                self.filter_ranks[activation_index] += values
                self.grad_index += 1

            R = lrp(module, R.data, relevance_method, 1)

    def forward(self, x):
        self.activations = []  # 전체 conv_layer의 activation map 수
        self.weights = []
        self.gradients = []
        self.grad_index = 0
        self.activation_to_layer = (
            {}
        )  # conv layer의 순서 7: 17의미는 7번째 conv layer가 전체에서 17번째에 있다라는 뜻

        activation_index = 0
        for layer, (name, module) in enumerate(self.model.features._modules.items()):
            x = module(x)  # 일반적인 forward를 수행하면서..
            if isinstance(
                module, torch.nn.modules.conv.Conv2d
            ):  # conv layer 일때 여기를 지나감
                x.register_hook(self.compute_rank)
                if self.method_type == "weight":
                    self.weights.append(module.weight)
                self.activations.append(x)
                self.activation_to_layer[activation_index] = layer
                activation_index += 1

        return self.model.classifier(x.view(x.size(0), -1))

    def compute_rank(self, grad):
        activation_index = (
            len(self.activations) - self.grad_index - 1
        )  # 뒤에서부터 하나씩 끄집어 냄
        activation = self.activations[activation_index]

        if self.method_type == "ICLR":
            values = (
                torch.sum((activation * grad), dim=0, keepdim=True)
                .sum(dim=2, keepdim=True)
                .sum(dim=3, keepdim=True)[0, :, 0, 0]
                .data
            )  # P. Molchanov et al., ICLR 2017
            # Normalize the rank by the filter dimensions
            values = values / (
                activation.size(0) * activation.size(2) * activation.size(3)
            )

        elif self.method_type == "grad":
            values = (
                torch.sum((grad), dim=0, keepdim=True)
                .sum(dim=2, keepdim=True)
                .sum(dim=3, keepdim=True)[0, :, 0, 0]
                .data
            )  # # X. Sun et al., ICML 2017
            # Normalize the rank by the filter dimensions
            values = values / (
                activation.size(0) * activation.size(2) * activation.size(3)
            )

        elif self.method_type == "weight":
            weight = self.weights[activation_index]
            values = (
                torch.sum((weight).abs(), dim=1, keepdim=True)
                .sum(dim=2, keepdim=True)
                .sum(dim=3, keepdim=True)[:, 0, 0, 0]
                .data
            )  # Many publications based on weight and activation(=feature) map

        else:
            values = (
                torch.sum((activation * grad), dim=0, keepdim=True)
                .sum(dim=2, keepdim=True)
                .sum(dim=3, keepdim=True)[0, :, 0, 0]
                .data
            )  # P. Molchanov et al., ICLR 2017
            # Normalize the rank by the filter dimensions
            values = values / (
                activation.size(0) * activation.size(2) * activation.size(3)
            )

        if activation_index not in self.filter_ranks:
            self.filter_ranks[activation_index] = (
                torch.FloatTensor(activation.size(1)).zero_().cuda()
                if self.cuda
                else torch.FloatTensor(activation.size(1)).zero_()
            )

        self.filter_ranks[activation_index] += values
        self.grad_index += 1

    def normalize_ranks_per_layer(self):
        for i in self.filter_ranks:

            if (
                self.relevance
            ):  # average over trials - LRP case (this is not normalization !!)
                v = self.filter_ranks[i]
                v = v / torch.sum(v)  # torch.sum(v) = total number of dataset
                self.filter_ranks[i] = v.cpu()
            else:
                if self.norm:  # L2-norm for global rescaling
                    if (
                        self.method_type == "weight"
                    ):  # weight & L1-norm (Li et al., ICLR 2017)
                        v = self.filter_ranks[i]
                        v = v / torch.sum(v)  # L1
                        # v = v / torch.sqrt(torch.sum(v * v)) #L2
                        self.filter_ranks[i] = v.cpu()
                    elif (
                        self.method_type == "ICLR"
                    ):  # |grad*act| & L2-norm (Molchanov et al., ICLR 2017)
                        v = torch.abs(self.filter_ranks[i])
                        v = v / torch.sqrt(torch.sum(v * v))
                        self.filter_ranks[i] = v.cpu()
                    elif (
                        self.method_type == "grad"
                    ):  # |grad| & L2-norm (Sun et al., ICML 2017)
                        v = torch.abs(self.filter_ranks[i])
                        v = v / torch.sqrt(torch.sum(v * v))
                        self.filter_ranks[i] = v.cpu()
                else:
                    if self.method_type == "weight":  # weight
                        v = self.filter_ranks[i]
                        self.filter_ranks[i] = v.cpu()
                    elif self.method_type == "ICLR":  # |grad*act|
                        v = torch.abs(self.filter_ranks[i])
                        self.filter_ranks[i] = v.cpu()
                    elif self.method_type == "grad":  # |grad|
                        v = torch.abs(self.filter_ranks[i])
                        self.filter_ranks[i] = v.cpu()

    def get_pruning_plan(self, num_filters_to_prune):
        filters_to_prune = self.lowest_ranking_filters(num_filters_to_prune)
        # filters_to_prune: filters to be pruned 1) layer number, 2) filter number, 3) its value

        # After each of the k filters are prunned,
        # the filter index of the next filters change since the model is smaller.
        filters_to_prune_per_layer = {}
        for l, f, _ in filters_to_prune:
            if l not in filters_to_prune_per_layer:
                filters_to_prune_per_layer[l] = []
            filters_to_prune_per_layer[l].append(f)

        filters_to_prune = []
        for l in filters_to_prune_per_layer:
            for i in filters_to_prune_per_layer[l]:
                filters_to_prune.append((l, i))

        layer_names = list(dict(self.model.named_modules()).keys())[2:]
        layer_to_filter_indices = {
            layer_names[index]: indices
            for index, indices in filters_to_prune_per_layer.items()
        }

        return layer_to_filter_indices

    def lowest_ranking_filters(self, num):
        data = []
        for i in sorted(self.filter_ranks.keys()):
            for j in range(self.filter_ranks[i].size(0)):
                data.append((self.activation_to_layer[i], j, self.filter_ranks[i][j]))
                # data 변수에 모든 layer의 모든 filter의 값을 쭈욱 나열 시킨다.

        return nsmallest(
            num, data, itemgetter(2)
        )  # data list 내에서 가장 작은 수를 num(=512개) 만큼 뽑아서 리스트에 저장


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

        Rn = X * (self.gradprop_linear(VP, SA) + self.gradprop_linear(VN, SB))

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
