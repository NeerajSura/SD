import pickle
import glob
import os
import torch


def load_latents(latent_path):
    r"""
    Simple utility to save latents to speed up the ldm training
    :param latent_path:
    :return:
    """
    latent_maps = {}
    for fname in glob.glob(os.path.join(latent_path, "*.pkl")):
        s = pickle.load(open(fname, 'rb'))
        for k, v in s.items():
            latent_maps[k] = v[0]
    return latent_maps


def drop_class_condition(class_condition, class_drop_prob, im):
    if class_drop_prob > 0:
        class_drop_mask = torch.zeros((im.shape[0], 1), device=im.device).float().uniform_(0,1) > class_drop_prob
        return class_condition * class_drop_mask
    else:
        return class_condition