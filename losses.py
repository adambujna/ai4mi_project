#!/usr/bin/env python3

# MIT License

# Copyright (c) 2025 Hoel Kervadec

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


from torch import einsum, topk

from utils import simplex, sset


class CrossEntropy():
    def __init__(self, **kwargs):
        # Self.idk is used to filter out some classes of the target mask. Use fancy indexing
        self.idk = kwargs['idk']
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def _prepare_inputs(self, pred_softmax, weak_target):
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        pred = pred_softmax[:, self.idk, ...]
        target = weak_target[:, self.idk, ...].float()
        log_p = (pred + 1e-10).log()
        return pred, target, log_p

    def __call__(self, pred_softmax, weak_target):
        _, target, log_p = self._prepare_inputs(pred_softmax, weak_target)

        loss = - einsum("bkwh,bkwh->", target, log_p)
        loss /= target.sum() + 1e-10

        return loss


class TopKCrossEntropy(CrossEntropy):
    def __init__(self, **kwargs):
        self.k = kwargs.pop('topk', 0.2)
        if not 0 < self.k <= 1:
            raise ValueError("topk must be in the interval (0, 1]")
        super().__init__(**kwargs)

    def __call__(self, pred_softmax, weak_target):
        _, target, log_p = self._prepare_inputs(pred_softmax, weak_target)

        pixel_loss = -(target * log_p).sum(dim=1)
        valid_pixels = target.sum(dim=1) > 0
        pixel_loss = pixel_loss[valid_pixels]

        if pixel_loss.numel() == 0:
            return pred_softmax.sum() * 0

        count = max(1, int(self.k * pixel_loss.numel() + 0.999999))
        return topk(pixel_loss, count).values.mean()


TopKLoss = TopKCrossEntropy


class FocalLoss(CrossEntropy):
    def __init__(self, **kwargs):
        self.gamma = kwargs.pop('gamma', 2.0)
        if self.gamma < 0:
            raise ValueError("gamma must be non-negative")
        super().__init__(**kwargs)

    def __call__(self, pred_softmax, weak_target):
        pred, target, log_p = self._prepare_inputs(pred_softmax, weak_target)
        
        focal_factor = (1 - pred).pow(self.gamma)
        loss = -einsum("bkwh,bkwh->", target * focal_factor, log_p)
        loss /= target.sum() + 1e-10

        return loss


class SoftDiceLoss():
    def __init__(self, **kwargs):
        # self.idk is used to filter out some classes of the target mask. Use fancy indexing
        self.idk = kwargs['idk']
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, pred_softmax, weak_target):
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        pred = pred_softmax[:, self.idk, ...]
        target = weak_target[:, self.idk, ...]

        spatial_dims = tuple(range(2, pred.ndim))
        intersection = (pred * target).sum(dim=spatial_dims)
        pred_sq = (pred ** 2).sum(dim=spatial_dims)
        target_sq = (target ** 2).sum(dim=spatial_dims)

        epsilon = 1e-10
        dice_per_class = (2 * intersection + epsilon) / (pred_sq + target_sq + epsilon)

        return 1 - dice_per_class.mean()


class DiceCELoss():
    def __init__(self, **kwargs):
        # self.idk is used to filter out some classes of the target mask. Use fancy indexing
        self.idk = kwargs['idk']
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

        self.ce = CrossEntropy(**kwargs)
        self.dice = SoftDiceLoss(**kwargs)

    def __call__(self, pred_softmax, weak_target):
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        return self.dice(pred_softmax, weak_target) + self.ce(pred_softmax, weak_target)


class DiceTopKLoss(DiceCELoss):
    def __init__(self, **kwargs):
        self.idk = kwargs['idk']
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

        self.topk = TopKLoss(**kwargs)
        self.dice = SoftDiceLoss(**kwargs)

    def __call__(self, pred_softmax, weak_target):
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        return self.dice(pred_softmax, weak_target) + self.topk(pred_softmax, weak_target)


class DiceFocalLoss(DiceCELoss):
    def __init__(self, **kwargs):
        self.idk = kwargs['idk']
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

        self.focal = FocalLoss(**kwargs)
        self.dice = SoftDiceLoss(**kwargs)

    def __call__(self, pred_softmax, weak_target):
        assert pred_softmax.shape == weak_target.shape
        assert simplex(pred_softmax)
        assert sset(weak_target, [0, 1])

        return self.dice(pred_softmax, weak_target) + self.focal(pred_softmax, weak_target)
