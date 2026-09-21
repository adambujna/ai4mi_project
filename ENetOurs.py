#!/usr/bin/env python3.10

# MIT License

# Copyright (c) 2025 Hoel Kervadec, Jose Dolz
# Modifications (c) 2026 — incremental U-Net-style changes on top of the
# original ENet baseline: InstanceNorm in place of BatchNorm, a learned
# up-convolution + full-resolution skip replacing the unlearned final
# nearest-neighbor upsample, and an optional deep-supervision head.
# Everything not explicitly touched below is left identical to the
# original ENet.py so the diff against the baseline stays readable.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import torch.nn as nn
from torch import Tensor


def random_weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_normal_(m.weight.data)
    elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)) and m.affine:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def conv_block(in_dim, out_dim, **kwconv):
    return nn.Sequential(nn.Conv2d(in_dim, out_dim, **kwconv),
                         nn.InstanceNorm2d(out_dim, affine=True),
                         nn.PReLU())


def conv_block_asym(in_dim, out_dim, *, kernel_size: int):
    return nn.Sequential(nn.Conv2d(in_dim, out_dim,
                                   kernel_size=(kernel_size, 1),
                                   padding=(2, 0)),
                         nn.Conv2d(out_dim, out_dim,
                                   kernel_size=(1, kernel_size),
                                   padding=(0, 2)),
                         nn.InstanceNorm2d(out_dim, affine=True),
                         nn.PReLU())


class BottleNeck(nn.Module):
    def __init__(self, in_dim, out_dim, projectionFactor,
                 *, dropoutRate=0.01, dilation=1,
                 asym: bool = False, dilate_last: bool = False):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        mid_dim: int = in_dim // projectionFactor

        # Main branch

        # Secondary branch
        self.block0 = conv_block(in_dim, mid_dim, kernel_size=1)

        if not asym:
            self.block1 = conv_block(mid_dim, mid_dim, kernel_size=3, padding=dilation, dilation=dilation)
        else:
            self.block1 = conv_block_asym(mid_dim, mid_dim, kernel_size=5)

        self.block2 = conv_block(mid_dim, out_dim, kernel_size=1)

        self.do = nn.Dropout(p=dropoutRate)
        self.PReLU_out = nn.PReLU()

        if in_dim > out_dim:
            self.conv_out = conv_block(in_dim, out_dim, kernel_size=1)
        elif dilate_last:
            self.conv_out = conv_block(in_dim, out_dim, kernel_size=3, padding=1)
        else:
            self.conv_out = nn.Identity()

    def forward(self, in_) -> Tensor:
        # Main branch
        # Secondary branch
        b0 = self.block0(in_)
        b1 = self.block1(b0)
        b2 = self.block2(b1)
        do = self.do(b2)

        output = self.PReLU_out(self.conv_out(in_) + do)

        return output


class BottleNeckDownSampling(nn.Module):
    def __init__(self, in_dim, out_dim, projectionFactor):
        super().__init__()
        mid_dim: int = in_dim // projectionFactor

        # Main branch
        self.maxpool0 = nn.MaxPool2d(2, return_indices=False)

        # Secondary branch
        self.block0 = conv_block(in_dim, mid_dim, kernel_size=2, padding=0, stride=2)
        self.block1 = conv_block(mid_dim, mid_dim, kernel_size=3, padding=1)
        self.block2 = conv_block(mid_dim, out_dim, kernel_size=1)

        # Regularizer
        self.do = nn.Dropout(p=0.01)
        self.PReLU = nn.PReLU()

        # Out

    def forward(self, in_) -> tuple[Tensor, Tensor]:
        # Main branch
        maxpool_output = self.maxpool0(in_)

        # Secondary branch
        b0 = self.block0(in_)
        b1 = self.block1(b0)
        b2 = self.block2(b1)
        do = self.do(b2)

        _, c, _, _ = maxpool_output.shape
        output = do
        output[:, :c, :, :] += maxpool_output

        final_output = self.PReLU(output)

        return final_output


class BottleNeckUpSampling(nn.Module):
    """Upsampling with T. convolution instead of MaxUnpool"""
    def __init__(self, in_dim, out_dim, projectionFactor):
        super().__init__()
        mid_dim: int = in_dim // projectionFactor
        up_dim: int = in_dim // 2

        # Main branch
        self.up = nn.ConvTranspose2d(up_dim, up_dim, kernel_size=2, stride=2)

        # Secondary branch
        self.block0 = conv_block(in_dim, mid_dim, kernel_size=3, padding=1)
        self.block1 = conv_block(mid_dim, mid_dim, kernel_size=3, padding=1)
        self.block2 = conv_block(mid_dim, out_dim, kernel_size=1)

        # Regularizer
        self.do = nn.Dropout(p=0.01)
        self.PReLU = nn.PReLU()

        # Out

    def forward(self, args) -> Tensor:
        # nn.Sequential cannot handle multiple parameters:
        in_, skip = args

        # Main branch
        up = self.up(in_)

        # Secondary branch
        b0 = self.block0(torch.cat((up, skip), dim=1))
        b1 = self.block1(b0)
        b2 = self.block2(b1)
        do = self.do(b2)

        output = self.PReLU(up + do)

        return output


def _dilated_bottleneck_stack(K8: int, Fp: int, *, out_dim: int = None, dilate_last: bool = False) -> nn.Sequential:
    out_last = out_dim if out_dim is not None else K8
    return nn.Sequential(
            BottleNeck(K8, K8, Fp, dropoutRate=0.1),
            BottleNeck(K8, K8, Fp, dilation=2),
            BottleNeck(K8, K8, Fp, dropoutRate=0.1, asym=True),
            BottleNeck(K8, K8, Fp, dilation=4),
            BottleNeck(K8, K8, Fp, dropoutRate=0.1),
            BottleNeck(K8, K8, Fp, dilation=8),
            BottleNeck(K8, K8, Fp, dropoutRate=0.1, asym=True),
            BottleNeck(K8, out_last, Fp, dilation=16, dilate_last=dilate_last),
    )


class ENetImproved(nn.Module):
    """Baseline with changes."""
    def __init__(self, in_dim: int, out_dim: int, deep_supervision: bool = False, **kwargs):
        super().__init__()
        Fp: int = kwargs["factor"] if "factor" in kwargs else 4  # Projecting factor
        K: int = kwargs["kernels"] if "kernels" in kwargs else 16  # n_kernels
        self.deep_supervision = deep_supervision

        self.emb = nn.Sequential(
                conv_block(in_dim, K, kernel_size=3, padding=1),
                conv_block(K, K, kernel_size=3, padding=1),
        )

        # Initial operations
        self.conv0 = nn.Conv2d(in_dim, K - 1, kernel_size=3, stride=2, padding=1)
        self.maxpool0 = nn.MaxPool2d(2, return_indices=False, ceil_mode=False)

        # Downsampling half
        self.bottleneck1_0 = BottleNeckDownSampling(K, K * 4, Fp)
        self.bottleneck1_1 = nn.Sequential(BottleNeck(K * 4, K * 4, Fp),
                                           BottleNeck(K * 4, K * 4, Fp),
                                           BottleNeck(K * 4, K * 4, Fp),
                                           BottleNeck(K * 4, K * 4, Fp))
        self.bottleneck2_0 = BottleNeckDownSampling(K * 4, K * 8, Fp)
        self.bottleneck2_1 = _dilated_bottleneck_stack(K * 8, Fp)

        # Middle operations
        self.bottleneck3 = _dilated_bottleneck_stack(K * 8, Fp, out_dim=K * 4, dilate_last=True)

        # Upsampling half
        self.bottleneck4 = nn.Sequential(BottleNeckUpSampling(K * 8, K * 4, Fp),
                                         BottleNeck(K * 4, K * 4, Fp, dropoutRate=0.1),
                                         BottleNeck(K * 4, K, Fp, dropoutRate=0.1))
        self.bottleneck5 = nn.Sequential(BottleNeckUpSampling(K * 2, K, Fp),
                                         BottleNeck(K, K, Fp, dropoutRate=0.1))

        self.up_final = nn.ConvTranspose2d(K, K, kernel_size=2, stride=2)

        # Final convolutions
        self.final = nn.Sequential(conv_block(K * 2, K, kernel_size=3, padding=1, bias=False, stride=1),
                                   conv_block(K, K, kernel_size=3, padding=1, bias=False, stride=1),
                                   nn.Conv2d(K, out_dim, kernel_size=1))

        if self.deep_supervision:
                self.aux_head_h4 = nn.Conv2d(K, out_dim, kernel_size=1)
                self.aux_head_h8 = nn.Conv2d(K * 4, out_dim, kernel_size=1)

        print(f"> Initialized {self.__class__.__name__} ({in_dim=}->{out_dim=}, {deep_supervision=}) with {kwargs}")


    def forward(self, input):
        full_res = self.emb(input)

        # Initial operations
        conv_0 = self.conv0(input)
        maxpool_0 = self.maxpool0(input)
        outputInitial = torch.cat((conv_0, maxpool_0), dim=1)

        # Downsampling half
        bn1_0 = self.bottleneck1_0(outputInitial)
        bn1_out = self.bottleneck1_1(bn1_0)
        bn2_0 = self.bottleneck2_0(bn1_out)
        bn2_out = self.bottleneck2_1(bn2_0)

        # Middle operations
        bn3_out = self.bottleneck3(bn2_out)

        # Upsampling half
        bn4_out = self.bottleneck4((bn3_out, bn1_out))
        bn5_out = self.bottleneck5((bn4_out, outputInitial))

        upsampled = self.up_final(bn5_out)
        final_in = torch.cat((upsampled, full_res), dim=1)
        main_out = self.final(final_in)

        if not self.deep_supervision:
                return main_out

        aux_h4 = self.aux_head_h4(bn4_out)
        aux_h8 = self.aux_head_h8(bn3_out)
        return main_out, [aux_h4, aux_h8]

    def init_weights(self, *args, **kwargs):
        self.apply(random_weights_init)
