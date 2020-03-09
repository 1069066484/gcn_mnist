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
    :param method: 'eye' or 'cos' or 'dg2
    """
    if A is None:
        eye = torch.stack([torch.eye(inputs.shape[1]).cuda()] * inputs.shape[0])
        if inputs is None:
            raise Exception("Either A or inputs should not be None")
        if method == 'eye':
            A = eye
        elif method == 'cos':
            # normed_inputs = F.normalize(inputs, dim=-1)
            A = inputs @ inputs.permute(0, 2, 1) + eye
        elif method == 'dg2':
            A = torch.eye(inputs.shape[1]).cuda()
            A[1:, :-1] = torch.eye(inputs.shape[1]-1).cuda()
            A += A.T
            A = torch.stack([A] * inputs.shape[0])
        else:
            raise Exception("Expect method to be 'eye' or 'cos', but got {}.".format(method))
    else:
        # A += torch.stack([torch.eye(A.shape[1]).cuda()] * A.shape[0])
        pass
    D = torch.sum(A, dim=-1)
    # print(D[:2,:2])
    D = torch.stack([torch.diag(d).cuda() for d in D]) # torch.diag(D).float()
    D = torch.inverse(D) ** 0.5
    # Ah = torch.stack([D[i] @ A[i] @ D[i] for i in range(len(inputs))])
    Ah = D @ A @ D
    if retA:
        return Ah, A
    return Ah


def calc_lp_le(A, S):
    """
    :param A: adj mat.
    :param S: batch_size * nl * n(l+1). Each row tells the chances the node belongs to the next-layer cluster
    """
    # print(( S @ S.permute(0,2,1)).shape)
    # exit()
    # print(A.shape, S.shape, (A - S @ S.permute(0,2,1)).shape)
    # exit()
    # L_lp = torch.mean(torch.norm(p='fro', input=A - S @ S.permute(0,2,1), dim=[1,2]))
    link_loss = A - S @ S.permute(0,2,1) + 1e-8
    link_loss = torch.norm(link_loss, p=2)
    L_lp = link_loss / A.numel()
    # L_e = -torch.mean(torch.sum(torch.log(S + 1e-8), dim=-1))
    L_e = (-S * torch.log(S + 1e-8)).sum(dim=-1).mean()
    # print(L_lp, L_e); exit()
    return L_lp, L_e


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
    def __init__(self, in_feat_d, out_nodes):
        super(HPool, self).__init__()
        # S: batch_size * nl * n(l+1). Each row tells the chances the node belongs to the next-layer cluster
        self.gcn_s = GCNLayer(in_feat_d, out_nodes, act=nn.Softmax(dim=-1))
        self.Ah = 0
        self.s = 0

    def forward(self, inputs, A, Ah):
        A = Ah
        s = self.gcn_s(inputs, Ah=Ah)
        # print(s.shape); print(s[0]); exit()
        # print(s.shape, inputs.shape, A.shape, Ah.shape) # torch.Size([256, 28, 10]) torch.Size([256, 28, 28]) torch.Size([256, 28, 28])
        # exit()
        sT = s.transpose(2, 1)
        x_next = sT @ inputs
        A_new = sT @ A @ s
        # print(A_new.shape, s.shape) # torch.Size([256, 16, 16]) torch.Size([256, 28, 16])
        # exit()
        self.Ah = Ah # + torch.stack([torch.eye(Ah.shape[1]).cuda()] * Ah.shape[0])
        self.s = s
        return x_next, A_new


class GCNCLF(nn.Module):
    def __init__(self, layers, lr, logger=None, method='eye'):
        super(GCNCLF, self).__init__()
        self.logger = logger
        self.num_cls = layers[-1]
        self.loss_num = 4
        self.method = method
        self.cross_ent = nn.CrossEntropyLoss()
        self.layers = layers
        self.features = nn.Sequential(*self._init_feats())
        self.opt = Adam(lr=lr, params=sum([list(model.parameters()) for model in self.features], []))

    def _init_feats(self):
        curr_feats = []
        features = []
        for layer in self.layers:
            if layer > 0:
                curr_feats.append(layer)
            else:
                features.append(GCN(layers=curr_feats, last_act=None, method=self.method))
                features.append(HPool(in_feat_d=curr_feats[-1], out_nodes=-layer))
                curr_feats = [curr_feats[-1]]
        return features

    def forward(self, inputs, with_loss=False, A=None):
        Ah, A = calc_Ahat(inputs=inputs, method=self.method, retA=True, A=A)
        xs = inputs
        L_lp = 0
        L_e = 0
        for feature in self.features:
            if isinstance(feature, GCN):
                xs = feature(xs, Ah=Ah)
            elif isinstance(feature, HPool):
                xs, Ah = feature(xs, A=Ah, Ah=Ah)
                # print(Ah[0, :4, :4])
                if with_loss:
                    L_lp_, L_e_ = calc_lp_le(A=feature.Ah, S=feature.s)
                    L_lp += L_lp_
                    L_e += L_e_
            else:
                raise Exception("Wrong module: {}".format(feature))
        out = xs.reshape([xs.shape[0], -1])
        if with_loss:
            return out, L_lp, L_e
        return out

    def backward(self, inputs, label):
        out, L_lp, L_e = self.forward(inputs, with_loss=True)
        label = label.long()
        loss = self.cross_ent(out, label)
        acc = torch.mean((torch.argmax(out, -1) == label).float())
        self.opt.zero_grad()
        (loss + L_e + L_lp).backward()
        self.opt.step()
        return [loss, L_lp, L_e, acc]

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