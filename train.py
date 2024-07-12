import os
import yaml
import pickle
import argparse
import importlib
import numpy as np
import glob
import json

from tqdm import tqdm

import torch
import torchvision
import torch.nn as nn 

from src.data.utils import four_point_to_homography
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from src.data.utils import warp_image

import src.data.transforms as transform_module
from src.utils.checkpoint import CheckPointer
import random
import wandb

# fixed randomseed
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PATH="dataset/flir_aligned/validation"
parse_txt = os.path.join(PATH,'align_validation.txt')
parse_list = list() 

for line in open(os.path.join(parse_txt)):
    parse_list.append(line.strip()[5:10])
parse_list.sort()

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

def test_predict_homography(model, data):
    for idx, m in enumerate(model):
        data = m.predict_homography(data)
    return data

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

def make_flir_dataloader(dataset_name: str, dataset_root: str, split: str, isvalid:bool, transforms: list, batch_size: int,
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
    dataset = dataset_class_to_call(dataset_root=split, transforms=composed_transforms, isvalid=isvalid)

    # # Call sampler class
    sampler = sampler_class_to_call(data_source=dataset, batch_size=batch_size, samples_per_epoch=samples_per_epoch,
                                    mode=mode, random_seed=random_seed)

    # Return dataloader
    if (collator_patch_1 is None or collator_patch_2 is None or collator_blob_porosity is None or collator_blobiness is None):
        return DataLoader(dataset, batch_sampler=sampler, num_workers=num_workers, collate_fn=dataset.collate_fn)
    else:
        collator = transform_module.CollatorWithBlobs(patch_1_key=collator_patch_1, patch_2_key=collator_patch_2,
                                                      blob_porosity=collator_blob_porosity,
                                                      blobiness=collator_blobiness, random_seed=random_seed)
        return DataLoader(dataset, batch_sampler=sampler, num_workers=num_workers, collate_fn=collator)


def train_one_epoch(model: torch.nn.Sequential,
                    train_dataloader: torch.utils.data.DataLoader,
                    optimizer: torch.optim.Optimizer,
                    gradient_clip: float,
                    scheduler: torch.optim.lr_scheduler._LRScheduler,
                    loss_fn: torch.nn.modules.loss._Loss,
                    epoch: int, steps_per_epoch: int, batch_size: int, device: str,
                    checkpointer: CheckPointer, checkpoint_arguments: dict, log_step: int,
                    self_supervised=False, log_verbose=False, config=None):

    # Training phase
    model.train()
    
    # Loop for the whole epoch
    avg_loss = 0

    current_lr = scheduler._last_lr[0]
    print(len(train_dataloader))
    for iter_no, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):

        # Global Step
        step = epoch*steps_per_epoch + iter_no + 1

        # zero the parameter gradients
        optimizer.zero_grad()

        for key in data:
            data[key] = data[key].to(device, dtype=torch.float)
        
        #######################################################################
        # Loss is the MSE between predicted 4pDelta and ground truth 4pDelta
        # Loss is L1 loss, in which case we have to do additional postprocessing
        if (type(loss_fn) == torch.nn.MSELoss or type(loss_fn) == torch.nn.L1Loss or
                type(loss_fn) == torch.nn.SmoothL1Loss):
            
            ground_truth, network_output, delta_gt, delta_hat = model(data)
            loss = loss_fn(ground_truth, network_output)

        # Triple loss scenario
        elif type(loss_fn) == str and loss_fn == 'CosineDistance':
            ground_truth, network_output, delta_gt, delta_hat = model(data)
            loss = torch.sum(1 - torch.cosine_similarity(ground_truth, network_output, dim=1))
        # Triple loss scenario
        elif type(loss_fn) == str and (loss_fn == 'TripletLoss' or loss_fn == 'iHomE' or loss_fn == 'biHomE'):
            # Calc loss
            loss, delta_gt, loss_dict = model(data)
        else:
            assert False, "Do not know the loss: " + str(type(loss_fn))
        #######################################################################
        loss.mean().backward()

        # Optimize
        optimizer.step()
        scheduler.step()
        if current_lr != scheduler._last_lr[0]:
            print(f'current_lr : {current_lr}, after_lr : {scheduler._last_lr[0]}, iters_no : {iter_no}')
            current_lr = scheduler._last_lr[0]
            
        avg_loss += loss.mean()
        # Log5

        if step % log_step == 0:
            
            if not config["ISDEBUG"]:
                wandb.log({"loss":loss.mean()})
                for k, v in loss_dict.items():
                    wandb.log({k: v.mean()})

            if log_verbose:
                print('Epoch: {} iter: {}/{} loss: {}'.format(epoch, iter_no+1, steps_per_epoch, loss.mean().item()))
    
    if not config["ISDEBUG"]:
        wandb.log({"avg_loss":avg_loss / len(train_dataloader)})
        wandb.log({"epoch":epoch+1})
    # Save state
    checkpoint_arguments['step'] = step
    checkpointer.save("model_{:02d}".format(epoch+1), **checkpoint_arguments)


def eval_one_epoch(model: torch.nn.Sequential,
                   test_dataloader: torch.utils.data.DataLoader,
                   loss_fn: torch.nn.modules.loss._Loss,
                   epoch: int, steps_per_epoch: int, batch_size: int, device: str,
                   self_supervised=False, log_verbose=False, config=None):
    # Training phase
    model.eval()

    # Time measurement
    model_start = torch.cuda.Event(enable_timing=True)
    model_end = torch.cuda.Event(enable_timing=True)
    model_time = []
    postprocess_start = torch.cuda.Event(enable_timing=True)
    postprocess_end = torch.cuda.Event(enable_timing=True)
    postprocess_time = []
    
    total_err = []
    if isinstance(model, nn.DataParallel):
        model = model.module

    with torch.no_grad():
        for iter, data in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            
            img_err = []  

            with open(correspondence_list[iter]) as j :
                correspondence_info = json.load(j)

            match_points = correspondence_info['match_pts']

            # move data to device
            for key in data:
                if not 'path' in key:
                    data[key] = data[key].to(device, dtype=torch.float)

            # Get homography
            model_start.record()
            delta_hat, _ = test_predict_homography(model, data)
            model_end.record()
            torch.cuda.synchronize()
            model_time.append(model_start.elapsed_time(model_end))

            batch_size = data['patch_1'].shape[0]
            
            
            for idx in range(batch_size):

                c_h = data['patch_1'].shape[2]
                c_w = data['patch_1'].shape[3]

                corners = np.expand_dims(np.float32([[0, 0], [c_w, 0], [c_w, c_h], [0, c_h]]), axis=0)
                if isinstance(delta_hat, list):
                    H = np.eye(3) #requires_grad = False
                    for pts in delta_hat:
                        H_after = four_point_to_homography(corners=corners, deltas=pts.reshape(1,4,2).detach().cpu().numpy(), crop=False)
                        H = np.matmul(H, H_after)
                else:
                    H = four_point_to_homography(corners=corners, deltas=delta_hat[idx].reshape(1, 4, 2).detach().cpu().numpy(),crop=True)

                for match_pt in match_points :
                    img_err.append(geometricDistance(match_pt, H))

                total_err.append(np.array(img_err).mean())
            
        print("Mean Error : {}".format(np.array(total_err).mean()))
        print("Inference_time : {}".format(np.array(model_time).mean()))

    # Save state
    if not config["ISDEBUG"]:
        wandb.log({"total_err":np.array(total_err).mean()})
    

def do_train(model: torch.nn.Sequential,
             train_dataloader: torch.utils.data.DataLoader,
             test_dataloader: torch.utils.data.DataLoader,
             optimizer: torch.optim.Optimizer,
             gradient_clip: float,
             scheduler: torch.optim.lr_scheduler._LRScheduler,
             loss_fn: torch.nn.modules.loss._Loss,
             epochs: int, steps_per_epoch: int, batch_size: int, device: str,
             checkpointer: CheckPointer, checkpoint_arguments: dict, log_dir='logs', log_step=1,
             self_supervised=False, log_verbose=False, config=None):

    ###########################################################################
    # Initialize TensorBoard writer
    ###########################################################################

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
    # Training loop
    ###########################################################################
    start_epoch = checkpoint_arguments['step'] // steps_per_epoch

    for epoch in range(start_epoch, epochs):
        
        # # Train part
        # print('Training epoch: {}'.format(epoch))
        train_one_epoch(model=model, train_dataloader=train_dataloader, optimizer=optimizer,
                        gradient_clip=gradient_clip, scheduler=scheduler, loss_fn=loss_fn, epoch=epoch,
                        steps_per_epoch=steps_per_epoch, batch_size=batch_size, device=device,
                        checkpointer=checkpointer, checkpoint_arguments=checkpoint_arguments, log_step=log_step,
                        self_supervised=self_supervised, log_verbose=log_verbose, config=config)
        # Test part

        if test_dataloader is not None:
            print('Testing epoch: {}'.format(epoch))
            eval_one_epoch(model=model, test_dataloader=test_dataloader, loss_fn=loss_fn, epoch=epoch,
                           steps_per_epoch=steps_per_epoch, batch_size=batch_size, device=device,
                           self_supervised=self_supervised, log_verbose=log_verbose, config=config)


def main(args):

    ###########################################################################
    # Make train/test data loaders
    ###########################################################################

    # Dataset fn
    if 'flir' in config['DATA']['NAME']:
        make_dataloader_fn = make_flir_dataloader
    else:
        assert False, 'I dont know this dataset yet.'

    # Camera models root
    camera_models_root = (os.path.join(BASE_DIR, config['DATA']['CAMERA_MODELS_ROOT']) if 'CAMERA_MODELS_ROOT' in
                          config['DATA'] is not None else None)

    # Train/test cache
    train_cache = config['DATA']['DATASET_TRAIN_CACHE'] if 'DATASET_TRAIN_CACHE' in config['DATA'] is not None else None
    test_cache = config['DATA']['DATASET_TEST_CACHE'] if 'DATASET_TEST_CACHE' in config['DATA'] is not None else None

    # Collator
    collator_blob_porosity = config['DATA']['AUGMENT_BLOB_POROSITY'] if 'AUGMENT_BLOB_POROSITY' in config[
        'DATA'] else None
    collator_blobiness = config['DATA']['AUGMENT_BLOBINESS'] if 'AUGMENT_BLOBINESS' in config['DATA'] else None

    # Data sampler mode
    data_sampler_mode = config['DATA']['SAMPLER']['MODE'] if 'MODE' in config['DATA']['SAMPLER'] else None
    data_sampler_frame_dist = config['DATA']['SAMPLER']['PAIR_MAX_FRAME_DIST'] if 'PAIR_MAX_FRAME_DIST'\
                                                                                  in config['DATA']['SAMPLER'] else None

    # Train dataloader
    train_dataloader = make_dataloader_fn(dataset_name=config['DATA']['NAME'],
                                          dataset_root=os.path.join(BASE_DIR, config['DATA']['DATASET_ROOT']),
                                          camera_models_root=camera_models_root,
                                          split=os.path.join(BASE_DIR, config['DATA']['TRAIN_SPLIT']),
                                          isvalid=False,
                                          transforms=config['DATA']['TRANSFORMS'],
                                          batch_size=config['DATA']['SAMPLER']['BATCH_SIZE'],
                                          samples_per_epoch=config['DATA']['SAMPLER']['TRAIN_SAMPLES_PER_EPOCH'],
                                          mode=data_sampler_mode,
                                          pair_max_frame_dist=data_sampler_frame_dist,
                                          num_workers=config['DATA']['NUM_WORKERS'],
                                          random_seed=config['DATA']['SAMPLER']['TRAIN_SEED'],
                                          cache_path=train_cache,
                                          collator_patch_1=config['MODEL']['BACKBONE']['PATCH_KEYS'][0],
                                          collator_patch_2=config['MODEL']['BACKBONE']['PATCH_KEYS'][1],
                                          collator_blob_porosity=collator_blob_porosity,
                                          collator_blobiness=collator_blobiness)

    # Test dataloader
    test_dataloader = None
    if "TEST_SPLIT" in config['DATA']:
        
        test_dataloader = make_dataloader_fn(dataset_name=config['DATA']['NAME'],
                                             dataset_root=os.path.join(BASE_DIR, config['DATA']['DATASET_ROOT']),
                                             camera_models_root=camera_models_root,
                                             split=os.path.join(BASE_DIR, config['DATA']['TEST_SPLIT']),
                                             isvalid=True,
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
    # Import and create the backbone
    ###########################################################################

    # Import backbone
    backbone_module = importlib.import_module('src.backbones.{}'.format(config['MODEL']['BACKBONE']['NAME']))
    backbone_class_to_call = getattr(backbone_module, 'Model')

    # Create backbone class
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
    model = torch.nn.Sequential(backbone, head)
    print_network(backbone)
    print_network(head)

    ###########################################################################
    # Create training elements
    ###########################################################################

    # Training elements
    if config['SOLVER']['OPTIMIZER'] == 'Adam':
        l2_reg = float(config['SOLVER']['L2_WEIGHT_DECAY']) if 'L2_WEIGHT_DECAY' in config['SOLVER'] is not None else 0
        optimizer = torch.optim.Adam(model.parameters(), lr=config['SOLVER']['LR'],
                                     betas=(config['SOLVER']['MOMENTUM_1'], config['SOLVER']['MOMENTUM_2']),
                                     weight_decay=l2_reg)
    elif config['SOLVER']['OPTIMIZER'] == 'AdamW':
        l2_reg = float(config['SOLVER']['L2_WEIGHT_DECAY']) if 'L2_WEIGHT_DECAY' in config['SOLVER'] is not None else 0
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['SOLVER']['LR'],
                                     betas=(config['SOLVER']['MOMENTUM_1'], config['SOLVER']['MOMENTUM_2']),
                                     weight_decay=l2_reg)
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
    restart_lr = config['SOLVER']['RESTART_LEARNING_RATE'] if 'RESTART_LEARNING_RATE' in config['SOLVER'] is not None else False
    optim_to_load = optimizer

    if restart_lr:
        optim_to_load = None
    checkpointer = CheckPointer(model, optim_to_load, scheduler, config['LOGGING']['DIR'], True, None,
                                device=config['SOLVER']['DEVICE'])
    extra_checkpoint_data = checkpointer.load()
    arguments.update(extra_checkpoint_data)

    ###########################################################################
    # Load pretrained model
    ###########################################################################

    pretrained_model = config['MODEL']['PRETRAINED'] if 'PRETRAINED' in config['MODEL'] is not None else None
    if pretrained_model is not None:
        checkpoint = torch.load(pretrained_model, map_location=torch.device("cpu"))
        model_ = model
        if isinstance(model_, DistributedDataParallel):
            model_ = model.module
        model_.load_state_dict(checkpoint.pop("model"))
        print('Pretrained model loaded!')

    ###########################################################################
    # Do train
    ###########################################################################

    gradient_clip = config['SOLVER']['GRADIENT_CLIP'] if 'GRADIENT_CLIP' in config['SOLVER'] is not None else -1

    do_train(model=model, device=config['SOLVER']['DEVICE'], train_dataloader=train_dataloader,
             test_dataloader=test_dataloader, optimizer=optimizer, gradient_clip=gradient_clip, scheduler=scheduler,
             loss_fn=loss_fn, batch_size=config['DATA']['SAMPLER']['BATCH_SIZE'], epochs=config['SOLVER']['NUM_EPOCHS'],
             steps_per_epoch=(config['DATA']['SAMPLER']['TRAIN_SAMPLES_PER_EPOCH'] //
                              config['DATA']['SAMPLER']['BATCH_SIZE']),
             log_dir=config['LOGGING']['DIR'], log_step=config['LOGGING']['STEP'], checkpointer=checkpointer,
             checkpoint_arguments=arguments, log_verbose=config['LOGGING']['VERBOSE'],
             self_supervised=(data_sampler_mode is None or data_sampler_mode == 'single'), config=config)
    print('DONE!')


if __name__ == "__main__":

    # params
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, required=True, help='Config file with learning settings')
    
    args = parser.parse_args()

    # Load yaml config file
    config_file_path = args.config_file
    with open(config_file_path, 'r') as file:
        config = yaml.full_load(file)

    if not config['ISDEBUG'] :
        wandb.init(project=config['WANDB']['PROJECT_NAME'])
        wandb.run.name = config['WANDB']['NAME']

    source_dir = os.path.join("log",config['WANDB']['NAME'], "source")

    if os.path.isdir(source_dir) is False:
        os.makedirs(source_dir)

    import tarfile
    tar = tarfile.open( os.path.join(source_dir, 'sources.tar'), 'w' )

    tar.add( 'src' )
    tar.add( 'config' )
    tar.add( 'train.py' )
    tar.add( 'scripts' )
    tar.close()
    # Call main
    main(args)
