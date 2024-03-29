# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Modified by Qi Zhu, November 2019
from __future__ import division

import torch


class ImageList(object):
    """
    Structure that holds a list of images (of possibly
    varying sizes) as a single tensor.
    This works by padding the images to the same size,
    and storing in a field the original sizes of each image,
    adding a field that is the bounding box information
    """

    def __init__(self, tensors, image_sizes, image_bboxes):
        """
        Arguments:
            tensors (tensor)
            image_sizes (list[tuple[int, int]])
            image_bboxes (list[bounding box]), where bounding box is 2-dimension(Nx4) array 
        """
        self.tensors = tensors
        self.image_sizes = image_sizes
        self.image_bboxes = image_bboxes # added by zhuqi

    def to(self, *args, **kwargs):
        cast_tensor = self.tensors.to(*args, **kwargs)
        return ImageList(cast_tensor, self.image_sizes, self.image_bboxes) #modified by zhuqi


def to_image_list(tensors, image_bboxes, size_divisible=0):
    """
    tensors can be an ImageList, a torch.Tensor or
    an iterable of Tensors. It can't be a numpy array.
    When tensors is an iterable of Tensors, it pads
    the Tensors with zeros so that they have the same
    shape
    """
    if isinstance(tensors, torch.Tensor) and size_divisible > 0:
        tensors = [tensors]

    if isinstance(tensors, ImageList):
        return tensors
    elif isinstance(tensors, torch.Tensor):
        # single tensor shape can be inferred
        assert tensors.dim() == 4
        image_sizes = [tensor.shape[-2:] for tensor in tensors]
        return ImageList(tensors, image_sizes, image_bboxes)
    elif isinstance(tensors, (tuple, list)):
        max_size = tuple(max(s) for s in zip(*[img.shape for img in tensors]))

        # TODO Ideally, just remove this and let me model handle arbitrary
        # input sizs
        if size_divisible > 0:
            import math

            stride = size_divisible
            max_size = list(max_size)
            max_size[1] = int(math.ceil(max_size[1] / stride) * stride)
            max_size[2] = int(math.ceil(max_size[2] / stride) * stride)
            max_size = tuple(max_size)

        batch_shape = (len(tensors),) + max_size
        batched_imgs = tensors[0].new(*batch_shape).zero_()
        for img, pad_img in zip(tensors, batched_imgs):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

        image_sizes = [im.shape[-2:] for im in batched_imgs]

        
        return ImageList(batched_imgs, image_sizes, image_bboxes)# modified by zhuqi
    else:
        raise TypeError("Unsupported type for to_image_list: {}".format(type(tensors)))
