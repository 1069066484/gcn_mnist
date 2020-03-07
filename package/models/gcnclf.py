import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.utils.checkpoint import checkpoint
import numpy as np
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, layer_szs, act=nn.ReLU(inplace=True), last_act=None, pre_act=None, bias=True, bn=True,
                 last_inplace=None, noise=None):
        super(MLP, self).__init__()
        self.layer_szs = layer_szs
        self.act = act
        self.last_act = last_act
        self.noise = noise
        try:
            if last_inplace is not None:
                self.last_act.inplace = last_inplace
        except:
            pass
        self.pre_act = pre_act
        self.bias = bias
        self.bn = bn
        self.linears = []
        self._make_layers()

    def _make_layers(self):
        modules = []
        if self.pre_act:
            modules.append(self.pre_act)
        for i in range(len(self.layer_szs) - 1):
            modules.append(nn.Linear(self.layer_szs[i], self.layer_szs[i+1], bias=self.bias))
            self.linears.append(modules[-1])
            if self.bn:
                modules.append(nn.BatchNorm1d(self.layer_szs[i+1]))
            if self.noise is not None:
                modules.append(GaussianNoiseLayer(std=self.noise))
            if self.act is not None:
                modules.append(self.act)
        modules.pop()
        if self.last_act is not None:
            modules.append(self.last_act)
        self.features = nn.Sequential(*modules)

    def forward(self, x):
        return self.features(x)


class GaussianNoiseLayer(nn.Module):
    def __init__(self, mean=0.0, std=0.2):
        super(GaussianNoiseLayer, self).__init__()
        self.mean = mean
        self.std = std

    def forward(self, x):
        if self.training:
            noise = x.data.new(x.size()).normal_(self.mean, self.std)
            if x.is_cuda:
                noise = noise.cuda()
            x = x + noise
        return x


class NormLayer(nn.Module):
    def __init__(self):
        super(NormLayer, self).__init__()

    def forward(self, x):
        return F.normalize(x)


class Triplet(nn.Module):
    def __init__(self, margin=10):
        super(Triplet, self).__init__()
        self.margin = margin
        self.cosine = nn.CosineSimilarity()
        self.p_d = nn.PairwiseDistance()

    def forward(self, sketch, image_p, image_n):
        d_sk_imp = self.p_d(sketch, image_p)
        # d_sk_imn = torch.clamp(self.p_d(sketch, image_n), max=self.margin3)
        d_sk_neg = self.p_d(sketch, image_n)
        loss = torch.clamp(d_sk_imp + self.margin - d_sk_neg, min=0)
        return torch.mean(loss)


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, act=nn.ReLU(), bias=False):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)
        self.act = act

    def forward(self, inputs, Ah):
        out = Ah @ self.linear(inputs)
        if self.act is not None:
            return self.act(out)
        return out


def calc_Ahat(A=None, inputs=None, method='eye', retA=False):
    """
    :param A: adjacent matrix.
    :param inputs: inputs matrix
    :param method: 'eye' or 'cos'
    """
    eye = torch.stack([torch.eye(inputs.shape[1]).cuda()] * inputs.shape[0])
    if A is None:
        if inputs is None:
            raise Exception("Either A or inputs should not be None")
        if method == 'eye':
            A = eye
        elif method == 'cos':
            # normed_inputs = F.normalize(inputs, dim=-1)
            A = inputs @ inputs.permute(0, 2, 1) + eye
        else:
            raise Exception("Expect method to be 'eye' or 'cos', but got {}.".format(method))
    D = torch.sum(A, dim=-1)
    D = torch.stack([torch.diag(d).cuda() for d in D]) # torch.diag(D).float()
    D = torch.inverse(D) ** 0.5
    # Ah = torch.stack([D[i] @ A[i] @ D[i] for i in range(len(inputs))])
    Ah = D @ A @ D
    if retA:
        return Ah, A
    return Ah


class GCN(nn.Module):
    """
    Define a 2-layer GCN model.
    """
    def __init__(self, layers, Ah=None, act=nn.ReLU(inplace=True), last_act=nn.Sigmoid(), method='eye'):
        super(GCN, self).__init__()
        in_feats = layers[0]
        gcns = []
        self.Ah = Ah
        self.method = method
        self.act = act
        self.last_act = last_act
        for layer in layers[1:]:
            gcns.append(GCNLayer(in_feats, layer, act=self.last_act if layer==layers[-1] else self.act))
            in_feats = layer
        self.features = nn.Sequential(*gcns)

    def forward(self, inputs, Ah=None):
        if Ah is None:
            if self.Ah is None:
                Ah = calc_Ahat(inputs=inputs, method=self.method)
            else:
                Ah = self.Ah
        for i, layer in enumerate(self.features):
            inputs = layer(inputs, Ah=Ah)
        return inputs


class HPool(nn.Module):
    def __init__(self, in_nodes, out_nodes):
        super(HPool, self).__init__()
        self.gcn_s = GCNLayer(in_nodes, out_nodes)

    def forward(self, inputs, A, Ah):
        s = self.gcn_s(inputs, Ah=Ah)
        sT = s.permute(0, 2, 1)
        x_next = sT @ inputs
        A_new = sT @ A @ s
        return x_next, A_new


class GCNCLF(nn.Module):
    def __init__(self, layers, lr, logger=None, method='eye'):
        super(GCNCLF, self).__init__()
        self.logger = logger
        self.num_cls = layers[-1]
        self.loss_num = 2
        self.method = method
        self.cross_ent = nn.CrossEntropyLoss()
        self.gcn = GCN(layers=layers, last_act=None, method=self.method)
        self.hpool = HPool(in_nodes=layers[-1], out_nodes=1)
        self.opt = Adam(lr=lr, params=sum([list(model.parameters()) for model in [self.gcn, self.hpool]], []))

    def forward(self, inputs):
        Ah, A = calc_Ahat(inputs=inputs, method=self.method, retA=True)
        gcn_o = self.gcn(inputs, Ah=Ah)
        pooled, A = self.hpool(gcn_o, A=A, Ah=Ah)
        out = pooled.reshape([pooled.shape[0], -1])
        # print(pooled.shape)
        return out

    def backward(self, inputs, label):
        out = self.forward(inputs)
        label = label.long()
        loss = self.cross_ent(out, label)
        acc = torch.mean((torch.argmax(out, -1) == label).float())
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return [loss, acc]

    def optimize_params(self, inputs, label):
        losses = self.backward(inputs, label)
        for i in range(len(losses)):
            losses[i] = float(losses[i])
        return losses


def _test_gcn_clf():
    clf = GCNCLF(layers=[28, 36, 10], lr=0.001).cuda()
    xs = torch.rand(3, 28, 28).cuda()
    out = clf(xs)
    print(out.shape)


if __name__=='__main__':
    _test_gcn_clf()


"""
# nohup sh sempcyc.sh > sempcyc_sketchy.out 2>&1
python main_sempcyc.py --gpu 3  --npy_dir 0 --dataset qd --save_dir sempcycs/pcyc_0_sketchy_128 --dim_enc 128 --paired 0
20645


"""