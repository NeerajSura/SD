import glob
import os
import torchvision
from tqdm import tqdm
from utils_sd.diffusion_utils import load_latents
from torch.utils.data.dataset import Dataset
from PIL import Image

class MnistDataset(Dataset):
    r"""
    MnistDataset class for mnist images
    """
    def __init__(self, split, im_path, im_size, im_channels, use_latents=False, latent_path=None, condition_config=None):
        r"""
        Init method for initializing the dataset attributes
        :param split:
        :param im_path:
        :param im_size:
        :param im_channels:
        :param use_latents:
        :param latent_path:
        :param condition_config:
        """
        self.split = split
        self.im_size = im_size
        self.im_channels = im_channels

        # Should we use the latent or not
        self.use_latents = use_latents
        self.latent_maps = None

        # Conditioning for the dataset
        self.condition_type = [] if condition_config is None else condition_config['condition_types']

        self.images, self.labels = self.load_images(im_path)

        # Whether to load images and call vae or load saved latents
        if use_latents and latent_path is not None:
            latent_maps = load_latents(latent_path)
            if len(latent_maps) == len(self.images):
                self.use_latents = True
                self.latent_maps = latent_maps
                print(f'Found {len(self.latent_maps)} latents')
            else:
                print('Latents not found')


    def load_images(self, im_path):
        r'''
        Gets all images from im_path and stacks them all up
        :param im_path:
        :return:
        '''

        assert os.path.exists(im_path), f'images path {im_path} does not exist'
        ims = []
        labels = []
        for dname in tqdm(os.listdir(im_path)):
            fnames = glob.glob(os.path.join(im_path, dname, '*.jpg'))
            fnames += glob.glob(os.path.join(im_path, dname, '*.jpeg'))
            fnames += glob.glob(os.path.join(im_path, dname, '*.png'))
            for fname in fnames:
                ims.append(fname)
                if 'class' in self.condition_type:
                    labels.append(dname)
        print(f'Found {len(ims)} images for split {self.split}')
        return ims, labels


    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        cond_inputs = {}
        if 'class' in self.condition_type:
            cond_inputs['class'] = self.labels[index]

        if self.use_latents:
            latent = self.latent_maps[self.images[index]]
            if len(self.condition_type) == 0:
                return latent
            else:
                return latent, cond_inputs

        else:
            im = Image.open(self.images[index])
            im_tensor = torchvision.transforms.ToTensor()(im)

            #convert im to -1 to 1 range
            im_tensor = (im_tensor)*2 - 1

            if len(self.condition_type) == 0:
                return im_tensor
            else:
                return im_tensor, cond_inputs