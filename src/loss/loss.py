import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from src.data.utils import warp_image
from src.data.utils import image_shape_to_corners
from src.data.utils import four_point_to_homography
from src.heads.ransac_utils import DSACSoftmax
from src.utils.phase_congruency import _phase_congruency

import kornia.geometry as kornia

def smoothness_loss(deformation, img=None, alpha=0.0):
    """Calculate the smoothness loss of the given defromation field
    :param deformation: the input deformation
    :param img: the image that the deformation is applied on (will be used for the bilateral filtering).
    :param alpha: the alpha coefficient used in the bilateral filtering.
    :return:
    """
    
    diff_1 = torch.abs(deformation[:, :, 1::, :] - deformation[:, :, 0:-1, :])
    diff_2 = torch.abs((deformation[:, :, :, 1::] - deformation[:, :, :, 0:-1]))
    diff_3 = torch.abs(deformation[:, :, 0:-1, 0:-1] - deformation[:, :, 1::, 1::])
    diff_4 = torch.abs(deformation[:, :, 0:-1, 1::] - deformation[:, :, 1::, 0:-1])
    if img is not None and alpha > 0.0:
        mask = img
        weight_1 = torch.exp(-alpha * torch.abs(mask[:, :, 1::, :] - mask[:, :, 0:-1, :]))
        weight_1 = torch.mean(weight_1, dim=1, keepdim=True).repeat(1, 2, 1, 1)
        weight_2 = torch.exp(- alpha * torch.abs(mask[:, :, :, 1::] - mask[:, :, :, 0:-1]))
        weight_2 = torch.mean(weight_2, dim=1, keepdim=True).repeat(1, 2, 1, 1)
        weight_3 = torch.exp(- alpha * torch.abs(mask[:, :, 0:-1, 0:-1] - mask[:, :, 1::, 1::]))
        weight_3 = torch.mean(weight_3, dim=1, keepdim=True).repeat(1, 2, 1, 1)
        weight_4 = torch.exp(- alpha * torch.abs(mask[:, :, 0:-1, 1::] - mask[:, :, 1::, 0:-1]))
        weight_4 = torch.mean(weight_4, dim=1, keepdim=True).repeat(1, 2, 1, 1)
    else:
        weight_1 = weight_2 = weight_3 = weight_4 = 1.0

    loss = torch.mean(weight_1 * diff_1) + torch.mean(weight_2 * diff_2) \
           + torch.mean(weight_3 * diff_3) + torch.mean(weight_4 * diff_4)
    return loss

def _warp(image, delta_hat, corners=None):
    if corners is None:
        corners = image_shape_to_corners(patch=image)

    if isinstance(delta_hat, list):
        H = torch.eye(3, dtype=delta_hat[0].dtype, device=delta_hat[0].device).unsqueeze(dim=0).repeat(delta_hat[0].shape[0], 1, 1) #requires_grad = False
        for pts in delta_hat:
            H_after = four_point_to_homography(corners=corners, deltas=pts, crop=False)
            H = torch.matmul(H, H_after)
    else:
        H = four_point_to_homography(corners=corners, deltas=delta_hat, crop=False)
    image_warped = warp_image(image, H, target_h=image.shape[-2], target_w=image.shape[-1])

    return image_warped, H

def _warp_with_h(image, homography):
    image_warped = warp_image(image, homography, target_h=image.shape[-2], target_w=image.shape[-1])
    return image_warped
    
    
def __upsample(img, scale_factor):
    return nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)(img)

def get_identity_grid(h, w):
    """Returns a sampling-grid that represents the identity transformation."""

    # x = torch.linspace(-1.0, 1.0, self.ow)
    # y = torch.linspace(-1.0, 1.0, self.oh)
    x = torch.linspace(0, w-1, w)
    y = torch.linspace(0, h-1, h)
    xx, yy = torch.meshgrid([y, x])
    xx = xx.unsqueeze(dim=0)
    yy = yy.unsqueeze(dim=0)
    identity = torch.cat((yy, xx), dim=0).unsqueeze(0)
    ones = torch.ones(1, h, w).unsqueeze(0)

    return torch.cat([identity, ones], dim=1).to('cuda', dtype=torch.float)

def EPE_loss(data, identity_grid, h12, h21, isbihome=True):

    gt_h = data['homography']
    inv_gt_h = torch.linalg.inv(gt_h)
    
    IsSS = ~(data['pairs_flag'] == 0)
    gt_grid = torch.matmul(gt_h, identity_grid)
    gt_grid = gt_grid[:,:2,:] / gt_grid[:,2,:].unsqueeze(1) + 1e-7
    pred_grid = torch.matmul(h12, identity_grid)
    pred_grid = pred_grid[:,:2,:] / pred_grid[:,2,:].unsqueeze(1) + 1e-7

    l1_epe = torch.log(torch.norm(gt_grid - pred_grid, p=2, dim=1)[IsSS]).mean() * 10

    ##########################EPE###################################
    if isbihome:
        gt_grid = torch.matmul(inv_gt_h, identity_grid)
        gt_grid = gt_grid[:,:2,:] / gt_grid[:,2,:].unsqueeze(1) + 1e-7

        pred_grid = torch.matmul(h21, identity_grid)
        pred_grid = pred_grid[:,:2,:] / pred_grid[:,2,:].unsqueeze(1) + 1e-7
        
        l2_epe = torch.log(torch.norm(gt_grid - pred_grid, p=2, dim=1)[IsSS]).mean() * 10

        return l1_epe, l2_epe


def PhaseCongruency_loss(config, pa, pn, warp_mask, target_mask):
    den_s = torch.sum(torch.sum(warp_mask * target_mask, dim=-1), dim=-1)

    if isinstance(config["TRIPLET_MARGIN"], str):
        loss_mat_s = torch.sum(pa - pn, dim=1)
    else:
        loss_mat_s = torch.sum(torch.max(pa - pn + config["TRIPLET_MARGIN"]*pa.shape[1], torch.zeros_like(pa)), dim=1)
    
    ln_s = torch.sum(torch.sum(warp_mask * target_mask * loss_mat_s, dim=-1), dim=-1) / \
        torch.max(den_s, torch.ones_like(den_s))

    return torch.mean(ln_s)

def Perceptual_loss(config, pa, pn, warp_mask, target_mask):
    # First loss elem
    den = torch.sum(torch.sum(warp_mask * target_mask, dim=-1), dim=-1)

    
    if isinstance(config["TRIPLET_MARGIN"], str):
        if config["TRIPLET_AGGREGATION"] == 'channel-aware':
            loss_mat = torch.sum(pa - pn, dim=1)
        elif config["TRIPLET_AGGREGATION"] == 'channel-agnostic':
            loss_mat = torch.sum(pa, dim=1) - torch.sum(pn, dim=1)

        else:
            assert False, 'Do not know this aggregation technique'
    else:
        if config["TRIPLET_AGGREGATION"] == 'channel-aware':
            loss_mat = torch.sum(torch.max(pa - pn + config["TRIPLET_MARGIN"], torch.zeros_like(pa)), dim=1)
        elif config["TRIPLET_AGGREGATION"] == 'channel-agnostic':

            loss_mat = torch.max(torch.sum(pa, dim=1) - torch.sum(pn, dim=1) + config["TRIPLET_MARGIN"]*pa.shape[1], torch.zeros_like(pa)[:,0,:,:])
        else:
            assert False, 'Do not know this aggregation technique'

    ln = torch.sum(torch.sum(warp_mask * target_mask * loss_mat, dim=-1), dim=-1) / \
          torch.max(den, torch.ones_like(den))

    return torch.mean(ln)

def triplet_resnet_loss(data, delta_hats, delta_hats_21=None, scores=None, loss_net = None,
                        patch_keys=None, mask_keys=None, hypothesis_no=1, patch_size=440,
                        sampling_strategy=None, triplet_version='double-line', triplet_distance='l1',
                        config=None):

    assert (delta_hats_21 is not None and scores is None) or (delta_hats_21 is None and scores is not None) or\
            (delta_hats_21 is None and scores is None), \
        'They should not be on at the same time - at least its not implemented yet'

    #######################################################################
    # Fetch keys and data
    #######################################################################

    e1, e2 = patch_keys
    patch_1 = data[e1]
    patch_2 = data['nopd_patch_2']
    
    # patch_2 = data[e2]

    if len(mask_keys):
        m1, m2 = mask_keys
        patch_1_m = data[m1]
        patch_2_m = data[m2]
    else:
        patch_1_m = torch.ones_like(patch_1)
        patch_2_m = torch.ones_like(patch_2)

    sl1 = smoothness_loss(patch_1_m)
    sl2 = smoothness_loss(patch_2_m)
    # mask loss
    log_m1 = torch.mean(-1 * torch.log(torch.clamp(torch.mean(torch.mean(patch_1_m, dim=-1), dim=-1) * (1/config["MASK_MU"]), 0 + 1e-9, 1)))
    log_m2 = torch.mean(-1 * torch.log(torch.clamp(torch.mean(torch.mean(patch_2_m, dim=-1), dim=-1) * (1/config["MASK_MU"]), 0 + 1e-9, 1)))

    #######################################################################
    # Prepare first patch warped
    #######################################################################
        #######################################################################
    # Size mismatch fix strategies
    #######################################################################
    # preprocessing(patch_1= patch_1, patch_2=patch_2, patch_1_m=patch_1_m, patch_2_m=patch_2_m, delta_hats = delta_hats, delta_hats_21=delta_hats_21, loss_net=loss_net, patch_size=patch_size, hypothesis_no=hypothesis_no)
    # for every hypothesis
    b, ch, i, _ = patch_1.shape
    n = hypothesis_no
    # i = patch_size

    # Repeat patch_1 and extract features

    patch_1 = patch_1.reshape(b, ch, i, i).repeat(1, n, 1, 1).reshape(b * n, ch, i, i)
    patch_1_s = _phase_congruency(patch_1)
    patch_1_f_list = loss_net(patch_1)
    patch_1_f = patch_1_f_list[-1]
    
    # Repeat patch 2 and extract features
    patch_2 = patch_2.reshape(b, ch, i, i).repeat(1, n, 1, 1).reshape(b * n, ch, i, i)
    patch_2_s = _phase_congruency(patch_2)
    patch_2_f_list = loss_net(patch_2)
    patch_2_f = patch_2_f_list[-1]

    # Warp patch 1 and extract features
    if isinstance(delta_hats, list):
        delta_hats = [d.reshape(b*n, -1, 2) for d in delta_hats]
    else:
        delta_hats = delta_hats.reshape(b * n, -1, 2)  # [B*N, 4, 2]
    
    patch_1_prime, h1 = _warp(patch_1, delta_hat=delta_hats)
    # patch_1_f_prime = _warp_with_h(patch_1_f, h1)
    patch_1_f_prime = loss_net(patch_1_prime)[-1]  # [B*N, C, H, W]
    patch_1_s_prime = _phase_congruency(patch_1_prime)
    
    # Repeat mask_1 and warp it

    m_i = i // 4
    patch_1_m = patch_1_m.reshape(b, 1, m_i, m_i).repeat(1, n, 1, 1).reshape(b * n, 1, m_i, m_i)
    patch_2_m = patch_2_m.reshape(b, 1, m_i, m_i).repeat(1, n, 1, 1).reshape(b * n, 1, m_i, m_i)
    patch_1_m_prime = _warp_with_h(patch_1_m, h1)
    # Repeat mask_1 and warp it
    # patch_1_m = patch_1_m.reshape(b, 1, i, i).repeat(1, n, 1, 1).reshape(b * n, 1, i, i)
    # patch_2_m = patch_2_m.reshape(b, 1, i, i).repeat(1, n, 1, 1).reshape(b * n, 1, i, i)
    # patch_1_m_prime = _warp_with_h(patch_1_m, h1)
    
    #######################################################################
    # Prepare second patch warped
    #######################################################################

    if 'double-line' in triplet_version:

        # Warp patch 2 and extract features
        if isinstance(delta_hats, list):
            delta_hats_21 = [d.reshape(b*n, -1, 2) for d in delta_hats_21]
        else:
            delta_hats_21 = delta_hats_21.reshape(b * n, -1, 2)                                  # [B*N, 4, 2]
        patch_2_prime, h2 = _warp(patch_2, delta_hat=delta_hats_21)
        patch_2_s_prime = _phase_congruency(patch_2_prime) 
        patch_2_f_prime = loss_net(patch_2_prime)[-1]
        # patch_2_f_prime = _warp_with_h(patch_2_f, h2)

        # Warp mask 2
        patch_2_m_prime = _warp_with_h(patch_2_m, h2)

        # #########################EPE###########################################
        # identity_grid = get_identity_grid(i, i).repeat(b,1,1,1).view(b,3,-1)
        # l1_epe, l2_epe = EPE_loss(data,identity_grid, h1, h2)

    #######################################################################
    # Old loss to be added to AFM
    #######################################################################
    if sampling_strategy == 'downsample-mask' or True:
        _, f_c, f_h, f_w = patch_1_f_prime.shape
        # Downsample mask
        downsample_factor = patch_1_m.shape[-1] // f_h
        downsample_layer = torch.nn.AvgPool2d(kernel_size=downsample_factor, stride=downsample_factor, padding=0)
        # patch_1_m_prime_ori = patch_1_m_prime
        patch_1_m_prime = downsample_layer(patch_1_m_prime)
        # patch_2_m_ori = patch_2_m
        patch_2_m = downsample_layer(patch_2_m)
        
        # Prepare second patch warped
        if 'double-line' in triplet_version:
            # patch_1_m_ori = patch_1_m
            patch_1_m = downsample_layer(patch_1_m)
            # patch_2_m_prime_ori = patch_2_m_prime
            patch_2_m_prime = downsample_layer(patch_2_m_prime)
            
    if sampling_strategy == 'upsample-mask' or True:
        _, c, h, w = patch_1.shape
        # Downsample mask
        upsample_factor = h // patch_1_m.shape[-1]
        
        upsample_layer = torch.nn.Upsample(scale_factor=upsample_factor, mode='bilinear', align_corners=True)
        patch_1_m_prime_ori = patch_1_m_prime
        patch_1_m_prime_ori = upsample_layer(patch_1_m_prime_ori)
        patch_2_m_ori = patch_2_m
        patch_2_m_ori = upsample_layer(patch_2_m_ori)
        
        # Prepare second patch warped
        if 'double-line' in triplet_version:
            patch_1_m_ori = patch_1_m
            patch_1_m_ori = upsample_layer(patch_1_m_ori)
            patch_2_m_prime_ori = patch_2_m_prime
            patch_2_m_prime_ori = upsample_layer(patch_2_m_prime_ori)
    # ######################################################################
    # # Style LOSS
    # #######################################################################
    
    lg = 0

    # f1 = loss_net(data['feature_1'][:,0:1])[-1]
    # f2 = loss_net(data['feature_2'][:,0:1])[-1]

    # lg_mean = 0
    # lg_std = 0

    # # for f1, f2 in zip(feat1, feat2):
        
    # b, ch, _, _ = f1.shape
    # f1 = f1.view(b, ch, -1)
    # f2 = f2.view(b, ch, -1)
    # gm1 = torch.matmul(f1, f1.permute(0,2,1))
    # gm2 = torch.matmul(f2, f2.permute(0,2,1))

    # lg += torch.abs(gm1 - gm2).mean()
    # ###########################################################
    # #Content Loss
    # ###########################################################
    # patch_1_f_flip = loss_net(torch.flip(patch_1, (2,3)))[0]
    # patch_2_f_flip = loss_net(torch.flip(patch_2, (2,3)))[0]
    
    # lg = lg_mean + lg_std
    # c1 = data['content_1']
    # c2 = data['content_2']

    # pa1 = torch.abs(c1 - patch_1_f_list[0])
    # pn1 = torch.abs(c1 - patch_1_f_flip)
    # pa2 = torch.abs(c2 - patch_2_f_list[0])
    # pn2 = torch.abs(c2 - patch_2_f_flip)

    # lc1 = torch.abs(torch.sum(pa1, dim=1) - torch.sum(pn1, dim=1)).mean()
    # lc2 = torch.abs(torch.sum(pa2, dim=1) - torch.sum(pn2, dim=1)).mean()
    # lc = lc1+lc2

    if 'double-line' in triplet_version:

        # Distance L1
        # import pdb;pdb.set_trace()
        if triplet_distance == 'l1':
            l1 = torch.abs(patch_1_f_prime - patch_2_f)
            l2 = torch.abs(patch_2_f_prime - patch_1_f)
            l3 = torch.abs(patch_1_f - patch_2_f)
            
            # l3 = torch.abs(patch_1_f_prime - patch_1_f)
            # l4 = torch.abs(patch_2_f_prime - patch_2_f)


            l1_s = torch.abs(patch_1_s_prime - patch_2_s)
            l2_s = torch.abs(patch_2_s_prime - patch_1_s)
            l3_s = torch.abs(patch_1_s - patch_2_s)

            # l3_s = torch.abs(patch_1_s_prime - patch_1_s)
            # l4_s = torch.abs(patch_2_s_prime - patch_2_s)

        # Distance L2
        elif triplet_distance == 'l2':

            l1 = torch.mean(torch.square(patch_1_f_prime - patch_2_f), axis=1)
            l2 = torch.mean(torch.square(patch_2_f_prime - patch_1_f), axis=1)
            l3 = torch.mean(torch.square(patch_1_f - patch_2_f), axis=1)

        # Distances cosine
        elif triplet_distance == 'cosine':

            l1 = 1 - torch.cosine_similarity(patch_1_f_prime, patch_2_f, dim=1)
            l2 = 1 - torch.cosine_similarity(patch_2_f_prime, patch_1_f, dim=1)
            l3 = 1 - torch.cosine_similarity(patch_1_f, patch_2_f, dim=1)

        else:
            assert False, 'Do not know this distance metric'

        # Prepare masks
        patch_1_m = torch.squeeze(patch_1_m, dim=1)
        patch_2_m = torch.squeeze(patch_2_m, dim=1)
        patch_1_m_prime = torch.squeeze(patch_1_m_prime, dim=1)
        patch_2_m_prime = torch.squeeze(patch_2_m_prime, dim=1)
        
        patch_1_m_ori = torch.squeeze(patch_1_m_ori, dim=1)
        patch_2_m_ori = torch.squeeze(patch_2_m_ori, dim=1)
        patch_1_m_prime_ori = torch.squeeze(patch_1_m_prime_ori, dim=1)
        patch_2_m_prime_ori = torch.squeeze(patch_2_m_prime_ori, dim=1)
        
        # Original BiHome version
        ln1 = Perceptual_loss(config, l1, l3, warp_mask=patch_1_m_prime, target_mask=patch_2_m)
        ln1_s = PhaseCongruency_loss(config, l1_s, l3_s, warp_mask=patch_1_m_prime_ori, target_mask=patch_2_m_ori)
        
        ln2 = Perceptual_loss(config, l2, l3, warp_mask=patch_2_m_prime, target_mask=patch_1_m)
        ln2_s = PhaseCongruency_loss(config, l2_s, l3_s, warp_mask=patch_2_m_prime_ori, target_mask=patch_1_m_ori)

        # # Test version (Real Triplet Loss)
        # ln1 = Perceptual_loss(config, l1, l3, warp_mask=patch_1_m_prime, target_mask=patch_2_m)
        # ln1_s = PhaseCongruency_loss(config, l1_s, l3_s, warp_mask=patch_1_m_prime_ori, target_mask=patch_2_m_ori)
        
        # ln2 = Perceptual_loss(config, l2, l4, warp_mask=patch_2_m_prime, target_mask=patch_1_m)
        # ln2_s = PhaseCongruency_loss(config, l2_s, l4_s, warp_mask=patch_2_m_prime_ori, target_mask=patch_1_m_ori)

        # import pdb;pdb.set_trace()
        # Forth loss elem
        batch_size = data[e1].shape[0]
        eye = torch.eye(3, dtype=h1.dtype, device=h1.device).unsqueeze(dim=0).repeat(batch_size, 1, 1)
        ln3 = torch.mean((torch.matmul(h1, h2) - eye) ** 2)
        
        ssId = data['pairs_flag'] < 2
        if ssId.sum()==0:
            ln4_1=0
        else:
            ln4_1 = torch.linalg.norm((data['delta'] - delta_hats)[ssId], ord=1, dim=-1).mean() + 1e-6 #.mean(dim=1).sum() 
        lg = ln3.clone()
        loss = ln1  + ln2 + ln1_s + ln2_s + config["TRIPLET_MU"] * ln3 + 100 * (log_m1 + log_m2) + 0.1*ln4_1 #+ lg# + lc# + ln4_3

        # print(loss.item(), ln1.item(), ln2.item(), ln1_s.item(), ln2_s.item())
        # print(ln3.item(), log_m1.item(), log_m2.item(), lg.item(), ln4_1.item())
        # import pdb;pdb.set_trace()
        # Final loss

        # print(loss)
    #######################################################################
    # Dual loss needs to be merged
    #######################################################################

    # if 'dual' in self.triplet_version:
    #     loss = loss + loss_dual

    #######################################################################
    # Tensorboard logs
    #######################################################################
    loss_dict={}
    if 'summary_writer' in data:
        step = data['summary_writer_step']

        # Feature space
        
        # loss_dict['l1_epe'] = l1_epe
        # loss_dict['l2_epe'] = l2_epe

        loss_dict['loss_12_pos'] = torch.abs(patch_2_f - patch_1_f_prime)
        loss_dict['loss_12_neg'] = torch.abs(patch_2_f - patch_1_f)
        loss_dict['loss_21_pos'] = torch.abs(patch_1_f - patch_2_f_prime)
        loss_dict['loss_21_neg'] = torch.abs(patch_1_f - patch_2_f)

        loss_dict['pcloss_12_pos'] = torch.abs(patch_2_s - patch_1_s_prime)
        loss_dict['pcloss_12_neg'] = torch.abs(patch_2_s - patch_1_s)
        loss_dict['pcloss_21_pos'] = torch.abs(patch_1_s - patch_2_s_prime)
        loss_dict['pcloss_21_neg'] = torch.abs(patch_1_s - patch_2_s)

        # loss_dict['loss_12_pos'] = torch.abs(patch_2_f - patch_1_f_prime)
        # loss_dict['loss_12_neg'] = torch.abs(patch_1_f_prime - patch_1_f)
        # loss_dict['loss_21_pos'] = torch.abs(patch_1_f - patch_2_f_prime)
        # loss_dict['loss_21_neg'] = torch.abs(patch_2_f_prime - patch_2_f)

        # loss_dict['pcloss_12_pos'] = torch.abs(patch_2_s - patch_1_s_prime)
        # loss_dict['pcloss_12_neg'] = torch.abs(patch_2_s_prime - patch_1_s)
        # loss_dict['pcloss_21_pos'] = torch.abs(patch_1_s - patch_2_s_prime)
        # loss_dict['pcloss_21_neg'] = torch.abs(patch_2_s_prime - patch_2_s)

        loss_dict['smoothnessloss_1'] = sl1
        loss_dict['smoothnessloss_2'] = sl2

        loss_dict['gramloss'] = lg
        # loss_dict['contentloss'] = lc

        loss_dict['non_singular_loss'] = torch.sum((torch.matmul(h1, h2) - eye) ** 2)
        loss_dict['corner_point_loss_12'] = torch.linalg.norm((data['delta'] - delta_hats)[ssId], ord=1, dim=-1).mean(dim=1).sum()#ln4_1
        loss_dict['corner_point_loss_21'] = torch.linalg.norm((data['delta'] + delta_hats_21)[ssId], ord=1, dim=-1).mean(dim=1).sum()#ln4_2
        loss_dict['Identity_corner_point_loss'] = torch.linalg.norm((delta_hats + delta_hats_21), ord=1, dim=-1).mean(dim=1).sum()#ln4_3
        
        data['summary_writer'].add_scalars('feature_space', {'patch_1_f': torch.mean(patch_1_f).item()}, step)
        data['summary_writer'].add_scalars('feature_space', {'patch_2_f': torch.mean(patch_2_f).item()}, step)
        data['summary_writer'].add_scalars('feature_space', {'patch_1_f_prime': torch.mean(patch_1_f_prime).item()},step)

        # Loss componets
        data['summary_writer'].add_scalars('loss_comp', {'l1': torch.mean(torch.abs(patch_2_f - patch_1_f_prime)).item()}, step)
        data['summary_writer'].add_scalars('loss_comp', {'l3': torch.mean(torch.abs(patch_2_f - patch_1_f)).item()}, step)
        
        data['summary_writer'].add_scalars('loss_comp_s', {'l1': torch.mean(torch.abs(patch_2_s - patch_1_s_prime)).item()}, step)
        data['summary_writer'].add_scalars('loss_comp_s', {'l3': torch.mean(torch.abs(patch_2_s - patch_1_s)).item()}, step)
        
        eye = torch.eye(3, dtype=h1.dtype, device=h1.device).unsqueeze(dim=0).repeat(h1.shape[0], 1, 1)
        data['summary_writer'].add_scalars('h', {'h1': (torch.sum((h1 - eye) ** 2)).item()}, step)

        # if 'double-line' in config["TRIPLET_LOSS"]:
        #     data['summary_writer'].add_scalars('loss_den', {'l1_den': torch.min(ln1_den).item()}, step)
        #     data['summary_writer'].add_scalars('loss_den', {'l2_den': torch.min(ln2_den).item()}, step)
            
        #     data['summary_writer'].add_scalars('loss_den', {'l1_s_den': torch.min(ln1_den_s).item()}, step)
        #     data['summary_writer'].add_scalars('loss_den', {'l2_s_den': torch.min(ln2_den_s).item()}, step)

    #######################################################################
    # Delta GT
    #######################################################################
    # delta
    delta_gt, delta_hat = None, None
    if 'delta' in data:
        delta_gt = data['delta']

    # # Calc average of delta_hat
    # if scores is not None:
    #     if isinstance(delta_hats, list):
    #         delta_hats 
    #     delta_hats = delta_hats * scores.reshape(b * n, 1, 1).repeat(1, 4, 2)   # [B*N, 4, 2]
    #     delta_hats = torch.sum(delta_hats.reshape(b, n, 4, 2), dim=1)           # [B, 4, 2]
    
    # Return loss: ground_truth, original_non_patched_image, delta_gt, delta_hat
    return loss, delta_gt, loss_dict




# import numpy as np

# import torch
# import kornia
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision.models as models

# from src.data.utils import warp_image
# from src.data.utils import image_shape_to_corners
# from src.data.utils import four_point_to_homography
# from src.heads.ransac_utils import DSACSoftmax
# from src.utils.phase_congruency import _phase_congruency

# import kornia

# def _warp(image, delta_hat, corners=None):
#     if corners is None:
#         corners = image_shape_to_corners(patch=image)

#     homography = four_point_to_homography(corners=corners, deltas=delta_hat, crop=False)
#     image_warped = warp_image(image, homography, target_h=image.shape[-2], target_w=image.shape[-1])
#     return image_warped, homography

# def _warp_with_h(image, homography):
#     image_warped = warp_image(image, homography, target_h=image.shape[-2], target_w=image.shape[-1])
#     return image_warped
    
    
# def __upsample(img, scale_factor):
#     return nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)(img)

# def get_identity_grid(h, w):
#     """Returns a sampling-grid that represents the identity transformation."""

#     # x = torch.linspace(-1.0, 1.0, self.ow)
#     # y = torch.linspace(-1.0, 1.0, self.oh)
#     x = torch.linspace(0, w-1, w)
#     y = torch.linspace(0, h-1, h)
#     xx, yy = torch.meshgrid([y, x])
#     xx = xx.unsqueeze(dim=0)
#     yy = yy.unsqueeze(dim=0)
#     identity = torch.cat((yy, xx), dim=0).unsqueeze(0)
#     ones = torch.ones(1, h, w).unsqueeze(0)

#     return torch.cat([identity, ones], dim=1).to('cuda', dtype=torch.float)

# def EPE_loss(data, identity_grid, h12, h21, isbihome=True):

#     gt_h = data['homography']
#     inv_gt_h = torch.linalg.inv(gt_h)
    
#     IsSS = ~(data['pairs_flag'] == 0)
#     gt_grid = torch.matmul(gt_h, identity_grid)
#     gt_grid = gt_grid[:,:2,:] / gt_grid[:,2,:].unsqueeze(1) + 1e-7
#     pred_grid = torch.matmul(h12, identity_grid)
#     pred_grid = pred_grid[:,:2,:] / pred_grid[:,2,:].unsqueeze(1) + 1e-7

#     l1_epe = torch.log(torch.norm(gt_grid - pred_grid, p=2, dim=1)[IsSS]).mean() * 10

#     ##########################EPE###################################
#     if isbihome:
#         gt_grid = torch.matmul(inv_gt_h, identity_grid)
#         gt_grid = gt_grid[:,:2,:] / gt_grid[:,2,:].unsqueeze(1) + 1e-7

#         pred_grid = torch.matmul(h21, identity_grid)
#         pred_grid = pred_grid[:,:2,:] / pred_grid[:,2,:].unsqueeze(1) + 1e-7
        
#         l2_epe = torch.log(torch.norm(gt_grid - pred_grid, p=2, dim=1)[IsSS]).mean() * 10

#         return l1_epe, l2_epe


# def PhaseCongruency_loss(config, pa, pn, warp_mask, target_mask):
#     den_s = torch.sum(torch.sum(warp_mask * target_mask, dim=-1), dim=-1)

#     if isinstance(config["TRIPLET_MARGIN"], str):
#         loss_mat_s = torch.sum(pa - pn, dim=1)
#     else:
#         loss_mat_s = torch.sum(torch.max(pa - pn + config["TRIPLET_MARGIN"]*pa.shape[1], torch.zeros_like(pa)), dim=1)
    
#     ln_s = torch.sum(torch.sum(warp_mask * target_mask * loss_mat_s, dim=-1), dim=-1) / \
#         torch.max(den_s, torch.ones_like(den_s))

#     return torch.sum(ln_s)

# def Perceptual_loss(config, pa, pn, warp_mask, target_mask):
#     # First loss elem
#     den = torch.sum(torch.sum(warp_mask * target_mask, dim=-1), dim=-1)
    
#     if isinstance(config["TRIPLET_MARGIN"], str):
#         if config["TRIPLET_AGGREGATION"] == 'channel-aware':
#             loss_mat = torch.sum(pa - pn, dim=1)
#         elif config["TRIPLET_AGGREGATION"] == 'channel-agnostic':
#             loss_mat = torch.sum(pa, dim=1) - torch.sum(pn, dim=1)
#         else:
#             assert False, 'Do not know this aggregation technique'
        
#     else:
#         if config["TRIPLET_AGGREGATION"] == 'channel-aware':
#             loss_mat = torch.sum(torch.max(pa - pn + config["TRIPLET_MARGIN"], torch.zeros_like(pa)), dim=1)
#         elif config["TRIPLET_AGGREGATION"] == 'channel-agnostic':

#             loss_mat = torch.max(torch.sum(pa, dim=1) - torch.sum(pn, dim=1) + config["TRIPLET_MARGIN"]*pa.shape[1], torch.zeros_like(pa)[:,0,:,:])
#         else:
#             assert False, 'Do not know this aggregation technique'
    
#     ln = torch.sum(torch.sum(warp_mask * target_mask * loss_mat, dim=-1), dim=-1) / \
#           torch.max(den, torch.ones_like(den))
    
#     return torch.sum(ln)

# def triplet_resnet_loss(data, delta_hats, delta_hats_21=None, scores=None, loss_net = None,
#                         patch_keys=None, mask_keys=None, hypothesis_no=1, patch_size=440,
#                         sampling_strategy=None, triplet_version='double-line', triplet_distance='l1',
#                         config=None):

#     assert (delta_hats_21 is not None and scores is None) or (delta_hats_21 is None and scores is not None) or\
#             (delta_hats_21 is None and scores is None), \
#         'They should not be on at the same time - at least its not implemented yet'

#     #######################################################################
#     # Fetch keys and data
#     #######################################################################

#     e1, e2 = patch_keys
#     patch_1 = data[e1]
#     patch_2 = data[e2]

#     if len(mask_keys):
#         m1, m2 = mask_keys
#         patch_1_m = data[m1]
#         patch_2_m = data[m2]
#     else:
#         patch_1_m = torch.ones_like(patch_1)
#         patch_2_m = torch.ones_like(patch_2)

#     #######################################################################
#     # Prepare first patch warped
#     #######################################################################

#     # for every hypothesis
#     b = delta_hats.shape[0]
#     n = hypothesis_no
#     i = patch_size
    
#     # Repeat patch_1 and extract features
#     patch_1 = patch_1.reshape(b, 1, i, i).repeat(1, n, 1, 1).reshape(b * n, 1, i, i)
#     patch_1_f = loss_net(patch_1)
    
#     # patch_1_s = self.laplacian_kernel(patch_1)
#     patch_1_s = _phase_congruency(patch_1)
    
#     # Repeat patch 2 and extract features
#     patch_2 = patch_2.reshape(b, 1, i, i).repeat(1, n, 1, 1).reshape(b * n, 1, i, i)
#     patch_2_f = loss_net(patch_2)
    
#     # patch_2_s = self.laplacian_kernel(patch_2)
#     patch_2_s = _phase_congruency(patch_2)

#     # Warp patch 1 and extract features
#     delta_hats = delta_hats.reshape(b * n, 4, 2)  # [B*N, 4, 2]
    
#     patch_1_prime, h1 = _warp(patch_1, delta_hat=delta_hats)
#     patch_1_f_prime = loss_net(patch_1_prime)  # [B*N, C, H, W]
#     patch_1_s_prime = _phase_congruency(patch_1_prime)
    
#     # Repeat mask_1 and warp it
#     patch_1_m = patch_1_m.reshape(b, 1, i, i).repeat(1, n, 1, 1).reshape(b * n, 1, i, i)
#     patch_2_m = patch_2_m.reshape(b, 1, i, i).repeat(1, n, 1, 1).reshape(b * n, 1, i, i)
#     patch_1_m_prime = _warp_with_h(patch_1_m, h1)
    
#     #######################################################################
#     # Prepare second patch warped
#     #######################################################################

#     if 'double-line' in triplet_version:

#         # Warp patch 2 and extract features
#         delta_hats_21 = delta_hats_21.reshape(b * n, 4, 2)                                  # [B*N, 4, 2]
#         patch_2_prime, h2 = _warp(patch_2, delta_hat=delta_hats_21)
#         patch_2_s_prime = _phase_congruency(patch_2_prime) 
#         patch_2_f_prime = loss_net(patch_2_prime)

#         # Warp mask 2
#         patch_2_m_prime = _warp_with_h(patch_2_m, h2)

#         # #########################EPE###########################################
#         # identity_grid = get_identity_grid(i, i).repeat(b,1,1,1).view(b,3,-1)
#         # l1_epe, l2_epe = EPE_loss(data,identity_grid, h1, h2)

#     #######################################################################
#     # Old loss to be added to AFM
#     #######################################################################
    
#     #######################################################################
#     # Size mismatch fix strategies
#     #######################################################################

#     _, f_c, f_h, f_w = patch_1_f_prime.shape
#     if sampling_strategy == 'downsample-mask' or True:
        
#         # Downsample mask
#         downsample_factor = patch_1_m.shape[-1] // f_h
        
#         downsample_layer = torch.nn.AvgPool2d(kernel_size=downsample_factor, stride=downsample_factor, padding=0)
#         patch_1_m_prime_ori = patch_1_m_prime
#         patch_1_m_prime = downsample_layer(patch_1_m_prime)
#         patch_2_m_ori = patch_2_m
#         patch_2_m = downsample_layer(patch_2_m)
        
#         # Prepare second patch warped
#         if 'double-line' in triplet_version:
#             patch_1_m_ori = patch_1_m
#             patch_1_m = downsample_layer(patch_1_m)
#             patch_2_m_prime_ori = patch_2_m_prime
#             patch_2_m_prime = downsample_layer(patch_2_m_prime)

#     #######################################################################
#     # LOSS
#     #######################################################################

#     if 'double-line' in triplet_version:

#         # Distance L1
#         if triplet_distance == 'l1':
#             l1 = torch.abs(patch_1_f_prime - patch_2_f)
#             l2 = torch.abs(patch_2_f_prime - patch_1_f)
#             l3 = torch.abs(patch_1_f - patch_2_f)
            
#             l1_s = torch.abs(patch_1_s_prime - patch_2_s)
#             l2_s = torch.abs(patch_2_s_prime - patch_1_s)
#             l3_s = torch.abs(patch_1_s - patch_2_s)

#         # Distance L2
#         elif triplet_distance == 'l2':

#             l1 = torch.mean(torch.square(patch_1_f_prime - patch_2_f), axis=1)
#             l2 = torch.mean(torch.square(patch_2_f_prime - patch_1_f), axis=1)
#             l3 = torch.mean(torch.square(patch_1_f - patch_2_f), axis=1)

#         # Distances cosine
#         elif triplet_distance == 'cosine':

#             l1 = 1 - torch.cosine_similarity(patch_1_f_prime, patch_2_f, dim=1)
#             l2 = 1 - torch.cosine_similarity(patch_2_f_prime, patch_1_f, dim=1)
#             l3 = 1 - torch.cosine_similarity(patch_1_f, patch_2_f, dim=1)

#         else:
#             assert False, 'Do not know this distance metric'

#         # Prepare masks
#         patch_1_m = torch.squeeze(patch_1_m, dim=1)
#         patch_2_m = torch.squeeze(patch_2_m, dim=1)
#         patch_1_m_prime = torch.squeeze(patch_1_m_prime, dim=1)
#         patch_2_m_prime = torch.squeeze(patch_2_m_prime, dim=1)
        
#         patch_1_m_ori = torch.squeeze(patch_1_m_ori, dim=1)
#         patch_2_m_ori = torch.squeeze(patch_2_m_ori, dim=1)
#         patch_1_m_prime_ori = torch.squeeze(patch_1_m_prime_ori, dim=1)
#         patch_2_m_prime_ori = torch.squeeze(patch_2_m_prime_ori, dim=1)
        
        
#         # double-line loss
#         ln1 = Perceptual_loss(config, l1, l3, warp_mask=patch_1_m_prime, target_mask=patch_2_m)
#         ln1_s = PhaseCongruency_loss(config, l1_s, l3_s, warp_mask=patch_1_m_prime_ori, target_mask=patch_2_m_ori)
        
#         ln2 = Perceptual_loss(config, l2, l3, warp_mask=patch_2_m_prime, target_mask=patch_1_m)
#         ln2_s = PhaseCongruency_loss(config, l2_s, l3_s, warp_mask=patch_2_m_prime_ori, target_mask=patch_1_m_ori)

#         # Forth loss elem
#         batch_size = data[e1].shape[0]
#         eye = torch.eye(3, dtype=h1.dtype, device=h1.device).unsqueeze(dim=0).repeat(batch_size, 1, 1)
#         ln3 = torch.sum((torch.matmul(h1, h2) - eye) ** 2)
        
#         # mask loss
#         log_m1 = torch.sum(-1 * torch.log(torch.clamp(torch.mean(torch.mean(patch_1_m, dim=-1), dim=-1) * (1/config["MASK_MU"]), 0 + 1e-9, 1)))
#         log_m2 = torch.sum(-1 * torch.log(torch.clamp(torch.mean(torch.mean(patch_2_m, dim=-1), dim=-1) * (1/config["MASK_MU"]), 0 + 1e-9, 1)))

#         # Final loss
#         loss = ln1  + ln2 + ln1_s + ln2_s + config["TRIPLET_MU"] * ln3 + 100 * (log_m1 + log_m2)
#     #######################################################################
#     # Dual loss needs to be merged
#     #######################################################################

#     # if 'dual' in self.triplet_version:
#     #     loss = loss + loss_dual

#     #######################################################################
#     # Tensorboard logs
#     #######################################################################
#     loss_dict={}
#     if 'summary_writer' in data:
#         step = data['summary_writer_step']

#         # Feature space
        
#         # loss_dict['l1_epe'] = l1_epe
#         # loss_dict['l2_epe'] = l2_epe

#         loss_dict['loss_12_pos'] = torch.abs(patch_2_f - patch_1_f_prime)
#         loss_dict['loss_12_neg'] = torch.abs(patch_2_f - patch_1_f)
#         loss_dict['loss_21_pos'] = torch.abs(patch_1_f - patch_2_f_prime)
#         loss_dict['loss_21_neg'] = torch.abs(patch_1_f - patch_2_f)

#         loss_dict['pcloss_12_pos'] = torch.abs(patch_2_s - patch_1_s_prime)
#         loss_dict['pcloss_12_neg'] = torch.abs(patch_2_s - patch_1_s)
#         loss_dict['pcloss_21_pos'] = torch.abs(patch_1_s - patch_2_s_prime)
#         loss_dict['pcloss_21_neg'] = torch.abs(patch_1_s - patch_2_s)

#         loss_dict['non_singular_loss'] = torch.sum((torch.matmul(h1, h2) - eye) ** 2)

#         data['summary_writer'].add_scalars('feature_space', {'patch_1_f': torch.mean(patch_1_f).item()}, step)
#         data['summary_writer'].add_scalars('feature_space', {'patch_2_f': torch.mean(patch_2_f).item()}, step)
#         data['summary_writer'].add_scalars('feature_space', {'patch_1_f_prime': torch.mean(patch_1_f_prime).item()},step)

#         # Loss componets
#         data['summary_writer'].add_scalars('loss_comp', {'l1': torch.mean(torch.abs(patch_2_f - patch_1_f_prime)).item()}, step)
#         data['summary_writer'].add_scalars('loss_comp', {'l3': torch.mean(torch.abs(patch_2_f - patch_1_f)).item()}, step)
        
#         data['summary_writer'].add_scalars('loss_comp_s', {'l1': torch.mean(torch.abs(patch_2_s - patch_1_s_prime)).item()}, step)
#         data['summary_writer'].add_scalars('loss_comp_s', {'l3': torch.mean(torch.abs(patch_2_s - patch_1_s)).item()}, step)
        
#         eye = torch.eye(3, dtype=h1.dtype, device=h1.device).unsqueeze(dim=0).repeat(h1.shape[0], 1, 1)
#         data['summary_writer'].add_scalars('h', {'h1': (torch.sum((h1 - eye) ** 2)).item()}, step)

#         # if 'double-line' in config["TRIPLET_LOSS"]:
#         #     data['summary_writer'].add_scalars('loss_den', {'l1_den': torch.min(ln1_den).item()}, step)
#         #     data['summary_writer'].add_scalars('loss_den', {'l2_den': torch.min(ln2_den).item()}, step)
            
#         #     data['summary_writer'].add_scalars('loss_den', {'l1_s_den': torch.min(ln1_den_s).item()}, step)
#         #     data['summary_writer'].add_scalars('loss_den', {'l2_s_den': torch.min(ln2_den_s).item()}, step)

#     #######################################################################
#     # Delta GT
#     #######################################################################
#     # delta
#     delta_gt, delta_hat = None, None
#     if 'delta' in data:
#         delta_gt = data['delta']

#     # Calc average of delta_hat
#     if scores is not None:
#         delta_hats = delta_hats * scores.reshape(b * n, 1, 1).repeat(1, 4, 2)   # [B*N, 4, 2]
#         delta_hats = torch.sum(delta_hats.reshape(b, n, 4, 2), dim=1)           # [B, 4, 2]

#     # Return loss: ground_truth, original_non_patched_image, delta_gt, delta_hat
#     return loss, delta_gt, loss_dict