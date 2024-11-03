import glob
import os
import random
import torch
import torchvision
import numpy as np
from PIL import Image
from utils_sd.diffusion_utils import load_latents
from tqdm import tqdm
from torch.utils.data.dataset import Dataset

class CelebDataset(Dataset):
    r"""
    Celeb dataset will by default centre crop and resize the images.
    This can be replaced by any other dataset. As long as all the images
    are under one directory.
    """

    def __init__(self, split, im_path, im_size=256, im_channels=3, im_ext='jpg', use_latents=False, latent_path=None, condition_config=None):
        self.split = split
        self.im_path = im_path
        self.im_size = im_size
        self.im_channels = im_channels
        self.im_ext = im_ext
        self.latent_maps = None
        self.use_latents = False

        self.condition_types = [] if condition_config is None else condition_config['condition_types']

        self.idx_to_cls_map = {}
        self.cls_to_idx_map = {}

        if 'image' in self.condition_types:
            self.mask_channels = condition_config['image_condition_config']['image_condition_input_channels']
            self.mask_h = condition_config['image_condition_config']['image_condition_h']
            self.mask_w = condition_config['image_condition_config']['image_condition_w']

        self.images, self.texts, self.masks = self.load_images(im_path)

        #Whether to load images or to load the latents
        if use_latents and latent_path is not None:
            latent_maps = load_latents(latent_path)
            if len(latent_maps) == len(self.images):
                self.use_latents = True
                self.latent_maps = latent_maps
            else:
                print('latents not found')

    def load_images(self, im_path):
        r'''
        Get all the images from the path and stack them all up
        :return:
        '''
        assert os.path.exists(im_path), f'Images path {im_path} does not exists'
        ims = []
        fnames = glob.glob(os.path.join(im_path, 'CelebA-HQ-img/*.{}'.format('png')))
        fnames += glob.glob(os.path.join(im_path, 'CelebA-HQ-img/*.{}'.format('jpg')))
        fnames += glob.glob(os.path.join(im_path, 'CelebA-HQ-img/*.{}'.format('jpeg')))
        texts = []
        masks = []

        if 'image' in self.condition_types:
            label_list = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth',
                          'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']
            self.idx_to_cls_map = {idx : label_list[idx] for idx in range(len(label_list))}
            self.cls_to_idx_map = {label_list[idx] : idx for idx in range(len(label_list))}

        for fname in tqdm(fnames):
            ims.append(fname)

            if 'image' in self.condition_types:
                im_name = int(os.path.split(fname)[1].split('.')[0])
                masks.append(os.path.join(im_path, 'CelebAMask-HQ-mask', f'{im_name}.png'))

        if 'image' in self.condition_types:
            assert len(masks) == len(ims), 'Condition type is Image but could not find masks for all images'
        print(f'Found {len(ims)} images')
        print(f'Found {len(masks)} masks')

        return ims, texts, masks

    def get_masks(self, index):
        r'''
        Methos to get mask of WxH for given index and convert it to CxWxH mask images
        :param index:
        :return:
        '''
        mask_im = Image.open(self.masks[index])
        mask_im = np.array(mask_im)
        im_base = np.zeros(self.mask_h, self.mask_w, self.mask_channels)
        for orig_idx in range(len(self.idx_to_cls_map)):
            im_base[mask_im == (orig_idx+1), orig_idx] = 1
        mask = torch.from_numpy(im_base).permute(2, 0, 1).float()
        return mask

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # Set conditioning info
        cond_inputs = {}
        if 'image' in self.condition_types:
            mask = self.get_masks(index)
            cond_inputs['image'] = mask

        if self.use_latents:
            latent = self.latent_maps[self.images[index]]
            if len(self.condition_types) == 0:
                return latent
            else:
                return latent, cond_inputs
        else:
            im = Image.open(self.images[index])
            im_tensor = torchvision.transforms.Compose([
                torchvision.transforms.Resize(self.im_size),
                torchvision.transforms.CenterCrop(self.im_size),
                torchvision.transforms.ToTensor(),
            ])(im)
            im.close()

            # Convert input from -1 to 1
            im_tensor = (2 * im_tensor) - 1
            if len(self.condition_types) == 0:
                return im_tensor
            else:
                return im_tensor, cond_inputs
