import torch
import torch.nn as nn
from fba import utils
from collections import namedtuple
from torchvision import models as tv


class PerceptualLoss(torch.nn.Module):
    # VGG using our perceptually-learned weights (LPIPS metric)
    def __init__(
        self,
        model="net-lin",
        net="alex",
        spatial=False,
        ):
        super().__init__()
        assert model == "net-lin"

        self.net = PNetLin(
            pnet_rand=False,
            pnet_tune=False,
            pnet_type=net,
            use_dropout=True,
            spatial=spatial,
            version="0.1",
            lpips=True)

        state_dict = torch.hub.load_state_dict_from_url(
            "https://folk.ntnu.no/haakohu/checkpoints/perceptual_similarity/alex.pth", map_location=utils.get_device())
        self.net.load_state_dict(state_dict, strict=False)
        self.net.eval()
        self.net = utils.to_cuda(self.net)

    def forward(self, pred, target, normalize=False):
        """
        Pred and target are Variables.
        If normalize is True, assumes the images are between [0,1] and then scales them between [-1,+1]
        If normalize is False, assumes the images are already between [-1,+1]

        Inputs pred and target are Nx3xHxW
        Output pytorch Variable N long
        """
        if normalize:
            target = 2 * target - 1
            pred = 2 * pred - 1

        return self.net.forward(target, pred, retPerLayer=False)



def spatial_average(in_tens, keepdim=True):
    return in_tens.mean([2, 3], keepdim=keepdim)


def upsample(in_tens, out_H=64):  # assumes scale factor is same for H and W
    in_H = in_tens.shape[2]
    scale_factor = 1. * out_H / in_H

    return nn.Upsample(
        scale_factor=scale_factor, mode='bilinear', align_corners=False)(
        in_tens)


def normalize_tensor(in_feat, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(in_feat**2, dim=1, keepdim=True))
    return in_feat / (norm_factor + eps)


# Learned perceptual metric
class PNetLin(nn.Module):
    def __init__(
            self, pnet_type='vgg', pnet_rand=False, pnet_tune=False,
            use_dropout=True, spatial=False, version='0.1', lpips=True):
        super(PNetLin, self).__init__()

        self.pnet_type = pnet_type
        self.pnet_tune = pnet_tune
        self.pnet_rand = pnet_rand
        self.spatial = spatial
        self.lpips = lpips
        self.version = version
        self.scaling_layer = ScalingLayer()

        net_type = alexnet
        self.chns = [64, 192, 384, 256, 256]
        self.L = len(self.chns)

        self.net = net_type(
            pretrained=not self.pnet_rand,
            requires_grad=self.pnet_tune)

        if(lpips):
            self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
            self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
            self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
            self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
            self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
            self.lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
            if(self.pnet_type == 'squeeze'):  # 7 layers for squeezenet
                self.lin5 = NetLinLayer(self.chns[5], use_dropout=use_dropout)
                self.lin6 = NetLinLayer(self.chns[6], use_dropout=use_dropout)
                self.lins += [self.lin5, self.lin6]

    def forward(self, in0, in1, retPerLayer=False):
        # v0.0 - original release had a bug, where input was not scaled
        in0_input, in1_input = (
            self.scaling_layer(in0),
            self.scaling_layer(in1)) if self.version == '0.1' else(
            in0, in1)
        outs0, outs1 = self.net.forward(in0_input), self.net.forward(in1_input)
        feats0, feats1, diffs = {}, {}, {}

        for kk in range(self.L):
            feats0[kk], feats1[kk] = normalize_tensor(
                outs0[kk]), normalize_tensor(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk])**2

        if(self.lpips):
            if(self.spatial):
                res = [
                    upsample(
                        self.lins[kk].model(
                            diffs[kk]),
                        out_H=in0.shape[2]) for kk in range(
                        self.L)]
            else:
                res = [
                    spatial_average(
                        self.lins[kk].model(
                            diffs[kk]),
                        keepdim=True) for kk in range(
                        self.L)]
        else:
            if(self.spatial):
                res = [
                    upsample(
                        diffs[kk].sum(
                            dim=1,
                            keepdim=True),
                        out_H=in0.shape[2]) for kk in range(
                        self.L)]
            else:
                res = [
                    spatial_average(
                        diffs[kk].sum(
                            dim=1,
                            keepdim=True),
                        keepdim=True) for kk in range(
                        self.L)]

        val = res[0]
        for l in range(1, self.L):
            val += res[l]

        if(retPerLayer):
            return (val, res)
        else:
            return val


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer(
            'shift', torch.Tensor([-.030, -.088, -.188])
            [None, :, None, None])
        self.register_buffer('scale', torch.Tensor(
            [.458, .448, .450])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    ''' A single linear layer which does a 1x1 conv '''

    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()

        layers = [nn.Dropout(), ] if(use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1,
                             padding=0, bias=False), ]
        self.model = nn.Sequential(*layers)


class alexnet(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(alexnet, self).__init__()
        alexnet_pretrained_features = tv.alexnet(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(2):
            self.slice1.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(2, 5):
            self.slice2.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(5, 8):
            self.slice3.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(8, 10):
            self.slice4.add_module(str(x), alexnet_pretrained_features[x])
        for x in range(10, 12):
            self.slice5.add_module(str(x), alexnet_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1 = h
        h = self.slice2(h)
        h_relu2 = h
        h = self.slice3(h)
        h_relu3 = h
        h = self.slice4(h)
        h_relu4 = h
        h = self.slice5(h)
        h_relu5 = h
        alexnet_outputs = namedtuple(
            "AlexnetOutputs", ["relu1", "relu2", "relu3", "relu4", "relu5"]
        )
        out = alexnet_outputs(h_relu1, h_relu2, h_relu3, h_relu4, h_relu5)

        return out
