import random
import cv2
import tempfile
import torch
from argparse import Namespace
import fairseq
from fairseq import checkpoint_utils, options, tasks, utils
import importlib
import sys

sys.path.append("/home/nabizadz/Projects/Mahsa/sources/av_hubert/avhubert")

import hubert_asr, hubert_pretraining, hubert


class Compose(object):
    """Compose several preprocess together.
    Args:
        preprocess (list of ``Preprocess`` objects): list of preprocess to compose.
    """

    def __init__(self, preprocess):
        self.preprocess = preprocess

    def __call__(self, sample):
        for t in self.preprocess:
            sample = t(sample)
        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.preprocess:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class Normalize(object):
    """Normalize a ndarray image with mean and standard deviation.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, frames):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor image.
        """
        frames = (frames - self.mean) / self.std
        return frames

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class CenterCrop(object):
    """Crop the given image at the center
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, frames):
        """
        Args:
            img (numpy.ndarray): Images to be cropped.
        Returns:
            numpy.ndarray: Cropped image.
        """
        t, h, w = frames.shape
        th, tw = self.size
        delta_w = int(round((w - tw)) / 2.)
        delta_h = int(round((h - th)) / 2.)
        frames = frames[:, delta_h:delta_h + th, delta_w:delta_w + tw]
        return frames


class RandomCrop(object):
    """Crop the given image at the center
    """

    def __init__(self, size):
        self.size = size

    def __call__(self, frames):
        """
        Args:
            img (numpy.ndarray): Images to be cropped.
        Returns:
            numpy.ndarray: Cropped image.
        """
        t, h, w = frames.shape
        th, tw = self.size
        delta_w = random.randint(0, w - tw)
        delta_h = random.randint(0, h - th)
        frames = frames[:, delta_h:delta_h + th, delta_w:delta_w + tw]
        return frames

    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)


user_dir = "/home/nabizadz/Projects/Mahsa/sources/av_hubert/avhubert"
ckpt_path = "/home/nabizadz/Projects/Mahsa/sources/av_hubert/finetune-model.pt"


def load_avhubert(ckpt_path=ckpt_path, user_dir=user_dir, is_finetune_ckpt=False):
    # utils.import_user_module(Namespace(user_dir=user_dir))
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
    model = models[0]
    if hasattr(models[0], 'decoder'):
        print(f"Checkpoint: fine-tuned")
        model = models[0].encoder.w2v_model
    else:
        print(f"Checkpoint: pre-trained w/o fine-tuning")

    model.eval()
    model.to("cuda")
    return model, task


def extract_visual_feature(model, task, frames):

    transform = Compose([Normalize(0.0, 255.0),
                         CenterCrop((task.cfg.image_crop_size, task.cfg.image_crop_size)),
                         Normalize(task.cfg.image_mean, task.cfg.image_std)])

    frames = transform(frames)
    frames = torch.FloatTensor(frames).unsqueeze(dim=0).unsqueeze(dim=0).to('cuda')

    with torch.no_grad():
        feature, _ = model.extract_finetune(source={'video': frames, 'audio': None},
                                            padding_mask=None,
                                            output_layer=None)
        feature = feature.squeeze(dim=0)
    return feature.detach().cpu().numpy()
