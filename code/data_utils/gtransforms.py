# Borrowed from: https://github.com/yjxiong/tsn-pytorch/blob/master/transforms.py

import torchvision
import random
from PIL import Image
import numbers
import torch
import torchvision.transforms.functional as F


class GroupResize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.worker = torchvision.transforms.Resize(size, interpolation)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class GroupRandomCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img_group):

        w, h = img_group[0].size
        th, tw = self.size

        out_images = list()

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        for img in img_group:
            assert (img.size[0] == w and img.size[1] == h)
            if w == tw and h == th:
                out_images.append(img)
            else:
                out_images.append(img.crop((x1, y1, x1 + tw, y1 + th)))

        return out_images


class GroupCenterCrop(object):
    def __init__(self, size):
        self.worker = torchvision.transforms.CenterCrop(size)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class GroupRandomHorizontalFlip(object):
    def __call__(self, img_group):
        if random.random() < 0.5:
            img_group = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
        return img_group


class GroupNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):  # (T, 3, 224, 224)
        for b in range(tensor.size(0)):
            for t, m, s in zip(tensor[b], self.mean, self.std):
                t.sub_(m).div_(s)
        return tensor


class LoopPad(object):

    def __init__(self, max_len):
        self.max_len = max_len

    def __call__(self, tensor):
        length = tensor.size(0)

        if length == self.max_len:
            return tensor

        # repeat the clip as many times as is necessary
        n_pad = self.max_len - length
        pad = [tensor] * (n_pad // length)
        if n_pad % length > 0:
            pad += [tensor[0:n_pad % length]]

        tensor = torch.cat([tensor] + pad, 0)
        return tensor


# NOTE: Returns [0-255] rather than torchvision's [0-1]
class ToTensor(object):
    def __init__(self):
        self.worker = lambda x: F.to_tensor(x) * 255

    def __call__(self, img_group):
        img_group = [self.worker(img) for img in img_group]
        return torch.stack(img_group, 0)

class GroupMultiScaleCrop(object):
    def __init__(self, output_size, scales=None, max_distort=1, fix_crop=True, more_fix_crop=True,
                 center_crop_only=False):
        self.scales = scales if scales is not None else [1, .875, .75, .66]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.center_crop_only = center_crop_only
        assert center_crop_only is False or max_distort == 0 and len(self.scales) == 1, \
            'Center crop should only be performed during testing time.'
        self.output_size = output_size if not isinstance(output_size, int) else [output_size, output_size]
        self.interpolation = Image.BILINEAR

    def __call__(self, img_group):

        im_size = img_group[0].size

        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)
        crop_img_group = [img.crop((offset_w, offset_h, offset_w + crop_w, offset_h + crop_h)) for img in img_group]
        ret_img_group = [img.resize((self.output_size[0], self.output_size[1]), self.interpolation)
                         for img in crop_img_group]
        return ret_img_group, (offset_h, offset_w, crop_h, crop_w)

    def _sample_crop_size(self, im_size):
        image_w, image_h = im_size[0], im_size[1]

        # find a crop size
        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * x) for x in self.scales]
        crop_h = [self.output_size[1] if abs(x - self.output_size[1]) < 3 else x for x in crop_sizes]
        crop_w = [self.output_size[0] if abs(x - self.output_size[0]) < 3 else x for x in crop_sizes]

        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= self.max_distort:
                    pairs.append((w, h))

        crop_pair = random.choice(pairs)
        if not self.fix_crop:
            w_offset = random.randint(0, image_w - crop_pair[0])
            h_offset = random.randint(0, image_h - crop_pair[1])
        else:
            w_offset, h_offset = self._sample_fix_offset(image_w, image_h, crop_pair[0], crop_pair[1])

        return crop_pair[0], crop_pair[1], w_offset, h_offset

    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        offsets = self.fill_fix_offset(self.center_crop_only, self.more_fix_crop, image_w, image_h, crop_w, crop_h)
        return random.choice(offsets)

    @staticmethod
    def fill_fix_offset(center_crop_only, more_fix_crop, image_w, image_h, crop_w, crop_h):
        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        ret = list()
        ret.append((0, 0))  # upper left
        ret.append((2 * w_step, 2 * h_step))  # center
        if center_crop_only:
            return ret
        ret.append((4 * w_step, 0))  # upper right
        ret.append((0, 4 * h_step))  # lower left
        ret.append((4 * w_step, 4 * h_step))  # lower right

        if more_fix_crop:
            ret.append((0, 2 * h_step))  # center left
            ret.append((4 * w_step, 2 * h_step))  # center right
            ret.append((2 * w_step, 4 * h_step))  # lower center
            ret.append((2 * w_step, 0 * h_step))  # upper center

            ret.append((1 * w_step, 1 * h_step))  # upper left quarter
            ret.append((3 * w_step, 1 * h_step))  # upper right quarter
            ret.append((1 * w_step, 3 * h_step))  # lower left quarter
            ret.append((3 * w_step, 3 * h_step))  # lower righ quarter

        return ret
