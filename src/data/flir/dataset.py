import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Sampler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from src.data.utils import warp_image
from src.data.utils import four_point_to_homography

from PIL import Image

class Dataset(Dataset):
    def __init__(self, dataset_root, transforms=None, isvalid=False):
        self.dataset_root = dataset_root
        self.transforms = transforms
        self.img_filenames = [f for f in os.listdir(self.dataset_root) if '.jpeg' in f or '.npy' in f]

        self.img_filepaths = [os.path.join(self.dataset_root, f) for f in self.img_filenames]
        self.img_filepaths.sort()
        self.isvalid=isvalid
        self.iter_num = 0

    def __iter__(self):
        self.iterator_n = 0
        return self

    def __next__(self):
        if self.iterator_n < len(self):
            self.iterator_n += 1
            return self[[self.iterator_n - 1]]
        else:
            raise StopIteration

    def __len__(self):
        return len(self.img_filenames)

    def __getitem__(self, indices):

        # Read images
        images = []

        for idx in indices:
            img, rgb, img_aug, rgb_aug = self.load_image(idx)
        
        images.append([img,rgb]) #[0][0], [0][1] == Thermal, RGB
        images.append([img_aug, rgb_aug]) #[1][0], [1][1] == FakeThermal, FakeRGB

        # Transforms
        if self.transforms:

            data = self.transforms((images, None))

        if self.isvalid:
            data['img_path'] = self.img_filepaths[idx]

        
        return data

    def collate_fn(self, batch):

        image_1 = list()
        image_2 = list()
        patch_1 = list()
        patch_2 = list()
        corners = list()
        target = list()
        delta = list()
        homography = list()
        pairs_flag = list()

        nopd = list()
        for b in batch :
            image_1.append(b['image_1'])
            image_2.append(b['image_2'])
            patch_1.append(b['patch_1'])
            patch_2.append(b['patch_2'])
            target.append(b['target'])
            corners.append(b['corners'])
            delta.append(b['delta'])
            homography.append(b['homography'])
            pairs_flag.append(b['pairs_flag'])
            nopd.append(b['nopd_patch_2'])
            
        image_1 = torch.stack(image_1, dim=0)
        image_2 = torch.stack(image_2, dim=0)
        patch_1 = torch.stack(patch_1, dim=0)
        patch_2 = torch.stack(patch_2, dim=0)
        target = torch.stack(target, dim=0)
        corners = torch.stack(corners, dim=0)
        delta = torch.stack(delta, dim=0)
        homography = torch.stack(homography, dim=0)
        pairs_flag = torch.stack(pairs_flag, dim=0)
        nopd = torch.stack(nopd, dim=0)
        
        return {'image_1': image_1, 'image_2': image_2, 'patch_1': patch_1, 'patch_2': patch_2, 'corners': corners,
                'target': target, 'delta': delta, 'homography': homography, 'pairs_flag': pairs_flag, 'nopd_patch_2': nopd}
    
    def load_image(self, idx):
        
        filepath = self.img_filepaths[idx]
        
        if '.jpeg' in filepath:
            
            rgb_filepath = self.img_filepaths[idx].replace("jpeg","jpg").replace("PreviewData","RGB")
            
            img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
            if os.path.exists(rgb_filepath):
                rgb = cv2.cvtColor(cv2.imread(rgb_filepath), cv2.COLOR_BGR2RGB)
            
            else :
                rgb = None
                
        elif '.npy' in filepath:
            
            rgb_filepath = self.img_filepaths[idx].replace("PreviewData","RGB")
            
            img = np.load(filepath, allow_pickle=True)
            if os.path.exists(rgb_filepath):

                rgb = np.load(rgb_filepath, allow_pickle=True)
            
            else :
                rgb = None
            
            if not self.isvalid:

                path, file_name = os.path.split(filepath)
                img_aug = np.load(os.path.join(path+'_StyleAug',file_name))

                path, file_name = os.path.split(rgb_filepath)
                rgb_aug = np.load(os.path.join(path+'_StyleAug',file_name))

                return img, rgb, img_aug, rgb_aug
            else:
                return img, rgb, None , None
        else:
            assert False, 'I dont know this format'
        
        return img, rgb, img_aug, rgb_aug


class DatasetSampler(Sampler):

    def __init__(self, data_source: Dataset, batch_size: int, samples_per_epoch=10000, mode='pair', random_seed=None):
        """
        Sampler constructor.

        There is 77 sequences with RTK data and each sequence has on average about 30k images, which results in about
        2.5 million of images. I've assumed that each epoch will have 10k images (specified with @samples_per_epoch).
        Positive sample will be randomly chosen between +-positive_max_frame_dist, stereo camera frame rate is 16Hz,
        so I would recommend to choose positive_max_frame_dist=16.

        Args:
            data_source (Dataset): Oxford dataset object.
            batch_size (int): Size of the batch
            samples_per_epoch (int): How many images should I produce in each epoch?
            mode (str): Should I sample single image from the dataset, pair of images from the same sequence but distant
                in time specified by @pair_max_frame_dist, or triplet of frames with corresponding pose, but captured in
                different sequences?
            random_seed (int): If passed will be used as a seed for numpy random generator.
        """
        
        self.data_source = data_source
        self.batch_size = batch_size
        self.samples_per_epoch = samples_per_epoch
        self.mode = mode
        self.random_seed = random_seed
        if self.random_seed is not None:
            self.random_state = np.random.RandomState(self.random_seed)

    def generate_batches(self):
        if self.random_seed is not None:
            
            if self.random_seed == 0 :
                self.iterated_idcs = np.arange(len(self.data_source.img_filepaths))
            else :
                self.iterated_idcs = self.random_state.choice(np.arange(len(self.data_source.img_filepaths)),self.samples_per_epoch)
        else:
            self.iterated_idcs = np.random.choice(np.arange(len(self.data_source.img_filepaths)),self.samples_per_epoch)

    def __len__(self):
        return self.samples_per_epoch // self.batch_size

    def __iter__(self):
        # Re-generate batches every epoch
        self.generate_batches()

        self.sampled_batch = []

        for sample_idx, idx in enumerate(self.iterated_idcs):
            self.sampled_batch.append([idx])
            if sample_idx % self.batch_size == self.batch_size - 1:
                yield self.sampled_batch
                self.sampled_batch = []