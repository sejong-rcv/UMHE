import os
import cv2
import yaml
import pickle
import argparse
import importlib
import numpy as np
import matplotlib.pyplot as pltm
import imageio
import json
import glob
import time

from src.data.utils import warp_image
from src.data.utils import four_point_to_homography
from tqdm import tqdm
from PIL import Image

import torch
import torchvision
import src.data.transforms as transform_module
from torch.utils.data import DataLoader
from src.utils.checkpoint import CheckPointer

PATH="dataset/flir_aligned/validation"
parse_txt = os.path.join(PATH,'align_validation.txt')
parse_list = list()
day_parse_txt = os.path.join(PATH,'align_validation_day.txt')
day_parse_list = list() 
night_parse_txt = os.path.join(PATH,'align_validation_night.txt')
night_parse_list = list() 

for line in open(os.path.join(parse_txt)):
    parse_list.append(line.strip()[5:10])
for line in open(os.path.join(day_parse_txt)):
    day_parse_list.append(line.strip()[5:10])
for line in open(os.path.join(night_parse_txt)):
    night_parse_list.append(line.strip()[5:10])

parse_list.sort()
day_parse_list.sort()
night_parse_list.sort()

correspondence_list = glob.glob(os.path.join(PATH,'Coordinates/*.json'))
correspondence_list.sort()

def print_network(net):

    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    # print(net)
    print('Total number of parameters: %d' % num_params)

def geometricDistance(correspondence, h):
    """
    Correspondence err
    :param correspondence: Coordinate
    :param h: Homography
    :return: L2 distance
    """
    
    p1 = np.transpose(np.matrix([correspondence[0][0], correspondence[0][1], 1]))

    estimatep2 = np.dot(h, p1)
    estimatep2 = (1/estimatep2.item(2))*estimatep2

    p2 = np.transpose(np.matrix([correspondence[1][0], correspondence[1][1], 1]))

    error = p2 - estimatep2

    return np.linalg.norm(error)


def make_flir_dataloader(dataset_name: str, dataset_root: str, split: str, transforms: list, batch_size: int,
                              samples_per_epoch: int, mode: str, num_workers: int, random_seed=None,
                              collator_patch_1=None, collator_patch_2=None, collator_blob_porosity=None,
                              collator_blobiness=None, **kwargs):
    """


    Args:
        dataset_name (string): Name of the dataset (name of the folder in src.data dir)
        dataset_root (string): Path to the root of the dataset used.
        camera_models_root (string): Path to the directory with camera models.
        split (string): Path to the file with names of sequences to be loaded.
        transforms (list of callables): What transforms apply to the images?
        batch_size (int): Size of the batch.
        samples_per_epoch (int): How many images should I produce in each epoch?
        mode (str): Should I sample single image from the dataset, pair of images from the same sequence but distant
            in time specified by @pair_max_frame_dist, or triplet of frames with corresponding pose, but captured in
            different sequences?
        pair_max_frame_dist (int): Number of frames we search the positive frame in.
        num_workers: Number of data perp workers.
        random_seed (int): If passed will be used as a seed for numpy random generator.

    Returns:

    """
    
    # Import data class
    dataset_module = importlib.import_module('src.data.{}.dataset'.format(dataset_name))
    dataset_class_to_call = getattr(dataset_module, 'Dataset')
    sampler_class_to_call = getattr(dataset_module, 'DatasetSampler')

    # Compose transforms
    transforms_list = []
    for transform in transforms:
        # Get transform class name and params
        t_name = list(transform.keys())[0]
        t_args = transform[t_name]
        t_class_to_call = getattr(transform_module, t_name)
        transforms_list.append(t_class_to_call(*(t_args + [random_seed])))

    composed_transforms = torchvision.transforms.Compose(transforms_list)

    # Call dataset class
    dataset = dataset_class_to_call(dataset_root=split, transforms=composed_transforms, isvalid=True)

    # Call sampler class
    sampler = sampler_class_to_call(data_source=dataset, batch_size=batch_size, samples_per_epoch=samples_per_epoch,
                                    mode=mode, random_seed=0)

    # Return dataloader
    if (collator_patch_1 is None or collator_patch_2 is None or collator_blob_porosity is None or
            collator_blobiness is None):
        return DataLoader(dataset, batch_sampler=sampler, num_workers=num_workers)
    else:
        collator = transform_module.CollatorWithBlobs(patch_1_key=collator_patch_1, patch_2_key=collator_patch_2,
                                                      blob_porosity=collator_blob_porosity,
                                                      blobiness=collator_blobiness, random_seed=random_seed)
        return DataLoader(dataset, batch_sampler=sampler, num_workers=num_workers, collate_fn=collator)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class ModelWrapper(torch.nn.Sequential):
    def __init__(self, *args):
        super(ModelWrapper, self).__init__(*args)

    def predict_homography(self, data):
        for idx, m in enumerate(self):
            data = m.predict_homography(data)
        return data

def evaluate(model: torch.nn.Module, eval_dataloader: torch.utils.data.DataLoader, loss_fn: torch.nn.modules.loss._Loss,
             device: str, patch_keys: list, self_supervised=False, postprocess=False, log_filepath=None):

    ###########################################################################
    # Device setting
    ###########################################################################

    if device == 'cuda' and torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            print('Multiple GPUs detected. Using DataParallel mode.')
            model = torch.nn.DataParallel(model)
        model.to(device)
    print('Model device: {}'.format(device))

    ###########################################################################
    # Eval
    ###########################################################################

    # Training phase
    model.eval()
    
    model_time = []
    postprocess_start = torch.cuda.Event(enable_timing=True)
    postprocess_end = torch.cuda.Event(enable_timing=True)
    postprocess_time = []
    
    total_err = []
    day_err = []
    night_err = []
    
    homography_list = []
    
    total_err = []
    total_point = []
    gt_H_list = []
    W_list = []
    V_list = []
    pred_H_list = []

    ###Match correspondence Dataset
    color_list = [(255,0,0),(0,255,0),(0,0,255),(255,255,0),(255,0,255),(0,255,255),(128,0,0),(0,128,0), (0, 0, 128), (128,128,0), (128,0,128), (0, 128, 128), (64, 128, 255), (255, 128, 64), (64, 255, 128)]

    with torch.no_grad():
        for iter, data in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):

            img_err = []  

            with open(correspondence_list[iter]) as j :
                correspondence_info = json.load(j)

            match_points = correspondence_info['match_pts']

            # move data to device
            for key in data:
                if not 'path' in key:
                    data[key] = data[key].to(device, dtype=torch.float)

            # Get homography
            model_start = torch.cuda.Event(enable_timing=True)
            model_end = torch.cuda.Event(enable_timing=True)
            model_start.record()

            delta_hat, delta_hat_21 = model.predict_homography(data)
            total_point.append(delta_hat.cpu())
            
            model_end.record()
            torch.cuda.synchronize()
            model_time.append(model_start.elapsed_time(model_end))
            
            batch_size = data['image_1'].shape[0]
            
            for idx in range(batch_size):

                c_h = data['image_1'].shape[2]
                c_w = data['image_1'].shape[3]

                corners = np.expand_dims(np.float32([[0, 0], [c_w, 0], [c_w, c_h], [0, c_h]]), axis=0)
                
                homography = four_point_to_homography(corners=corners, deltas=delta_hat[idx].reshape(1, 4, 2).detach().cpu().numpy(),crop=True)
                pred_H_list.append(np.array(homography))

                source_pts = []
                target_pts = []
                for l, match_pt in enumerate(match_points) :
                    match_pt = np.array(match_pt).astype('int32')
                    source_pts.append([match_pt[0][0], match_pt[0][1]])
                    target_pts.append([match_pt[1][0], match_pt[1][1]])

                    img_err.append(geometricDistance(match_pt, homography))

                value = len(source_pts)
                
                gt_H, _ = cv2.findHomography(np.array(source_pts).reshape(-1,1,2), np.array(target_pts).reshape(-1,1,2), 0)#cv2.RANSAC

                total_err.append(np.array(img_err).mean())
                
                if parse_list[iter] in day_parse_list :
                    day_err.append(np.array(img_err).mean())
                else : 
                    night_err.append(np.array(img_err).mean())

        print("Total Mean Error : {}".format(np.array(total_err).mean()))
        print("Day Mean Error : {}".format(np.array(day_err).mean()))
        print("Night Mean Error : {}".format(np.array(night_err).mean()))

        print("Infer Time(ms) - Mean : {}".format(np.array(model_time).mean()))
        print("Infer Time(ms) - Std : {}".format(np.array(model_time).std()))
        
                
def main(config_file_path: str, ckpt_file_path: str, batch_size: int, log_filepath=None):

    # Load yaml config file
    with open(config_file_path, 'r') as file:
        config = yaml.full_load(file)

    ###########################################################################
    # Make test data loader
    ###########################################################################

    # Fix numpy seed
    np.random.seed(config['DATA']['SAMPLER']['TEST_SEED'])

    # Dataset fn
    if 'flir' in config['DATA']['NAME']:
        make_dataloader_fn = make_flir_dataloader
    else:
        assert False, 'I dont know this dataset yet.'

    # Camera models root
    camera_models_root = (os.path.join(BASE_DIR, config['DATA']['CAMERA_MODELS_ROOT']) if 'CAMERA_MODELS_ROOT' in
                          config['DATA'] is not None else None)

    # Test cache
    test_cache = config['DATA']['DATASET_TEST_CACHE'] if 'DATASET_TEST_CACHE' in config['DATA'] is not None else None

    # Collator
    collator_blob_porosity = config['DATA']['AUGMENT_BLOB_POROSITY'] if 'AUGMENT_BLOB_POROSITY' in config[
        'DATA'] else None
    collator_blobiness = config['DATA']['AUGMENT_BLOBINESS'] if 'AUGMENT_BLOBINESS' in config['DATA'] else None

    # Data sampler mode
    data_sampler_mode = config['DATA']['SAMPLER']['MODE'] if 'MODE' in config['DATA']['SAMPLER'] else None
    data_sampler_frame_dist = config['DATA']['SAMPLER']['PAIR_MAX_FRAME_DIST'] if 'PAIR_MAX_FRAME_DIST'\
                                                                                  in config['DATA']['SAMPLER'] else None

    # Eval dataloader
    eval_dataloader = make_dataloader_fn(dataset_name=config['DATA']['NAME'],
                                         dataset_root=os.path.join(BASE_DIR, config['DATA']['DATASET_ROOT']),
                                         camera_models_root=camera_models_root,
                                         split=os.path.join(config['DATA']['TEST_SPLIT']),
                                         transforms=config['DATA']['TEST_TRANSFORM'],
                                         batch_size=1,
                                         samples_per_epoch=1013,
                                         mode=data_sampler_mode,
                                         pair_max_frame_dist=data_sampler_frame_dist,
                                         num_workers=config['DATA']['NUM_WORKERS'],
                                         random_seed=config['DATA']['SAMPLER']['TEST_SEED'],
                                         cache_path=test_cache,
                                         collator_patch_1=config['MODEL']['BACKBONE']['PATCH_KEYS'][0],
                                         collator_patch_2=config['MODEL']['BACKBONE']['PATCH_KEYS'][1],
                                         collator_blob_porosity=collator_blob_porosity,
                                         collator_blobiness=collator_blobiness)

    ###########################################################################
    # Import and create the model
    ###########################################################################

    # Import model
    backbone_module = importlib.import_module('src.backbones.{}'.format(config['MODEL']['BACKBONE']['NAME']))
    backbone_class_to_call = getattr(backbone_module, 'Model')

    # Create model class
    backbone = backbone_class_to_call(**config['MODEL']['BACKBONE'])

    ###########################################################################
    # Import and create the head
    ###########################################################################

    # Import backbone
    head_module = importlib.import_module('src.heads.{}'.format(config['MODEL']['HEAD']['NAME']))
    head_class_to_call = getattr(head_module, 'Model')

    # Create backbone class
    head = head_class_to_call(backbone, **config['MODEL']['HEAD'])

    ###########################################################################
    # Import and create the head
    ###########################################################################
    print_network(backbone)
    model = ModelWrapper(backbone, head)

    ###########################################################################
    # Create training elements
    ###########################################################################

    # Training elements
    if config['SOLVER']['OPTIMIZER'] == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config['SOLVER']['LR'],
                                     betas=(config['SOLVER']['MOMENTUM_1'], config['SOLVER']['MOMENTUM_2']))
    else:
        assert False, 'I do not have this solver implemented yet.'
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config['SOLVER']['MILESTONES'],
                                                     gamma=config['SOLVER']['LR_DECAY'])

    try:
        loss_fn = getattr(torch.nn, config['SOLVER']['LOSS'])()
    except:
        loss_fn = config['SOLVER']['LOSS']

    ###########################################################################
    # Checkpoint
    ###########################################################################

    arguments = {"step": 0}
    checkpointer = CheckPointer(model, optimizer, scheduler, config['LOGGING']['DIR'], True, None,
                                device=config['SOLVER']['DEVICE'])

    extra_checkpoint_data = checkpointer.load(f=ckpt_file_path)
    arguments.update(extra_checkpoint_data)

    del model[1].auxiliary_net
    ###########################################################################
    # Do evaluate
    ###########################################################################

    evaluate(model=model, eval_dataloader=eval_dataloader, loss_fn=loss_fn, device=config['SOLVER']['DEVICE'],
             patch_keys=config['MODEL']['BACKBONE']['PATCH_KEYS'],
             self_supervised=(data_sampler_mode is None or data_sampler_mode == 'single'),
             postprocess=(config['MODEL']['BACKBONE']['NAME'] == 'Rethinking'),
             log_filepath=log_filepath)
    print('DONE!')


if __name__ == "__main__":

    # params
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, required=True, help='Config file with learning settings')
    parser.add_argument('--ckpt', type=str, required=True, help='Model path')
    parser.add_argument('--batch_size', type=int, required=False, default=1, help='Test batch size')
    parser.add_argument('--log', type=str, required=False, help='log filepath')
   
    args = parser.parse_args()
    # Call main
    main(args.config_file, args.ckpt, args.batch_size, args.log)
