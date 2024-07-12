import numpy as np

import torch
import kornia
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from src.data.utils import warp_image
from src.data.utils import image_shape_to_corners
from src.data.utils import four_point_to_homography
from src.heads.ransac_utils import DSACSoftmax
from src.utils.phase_congruency import _phase_congruency
from torchvision.transforms.functional import to_pil_image
from src.loss.loss import triplet_resnet_loss
import kornia

class InitLossNetwork():
    def __init__(self, resnet_config):

        model_type = resnet_config['LossNet']

        if 'resnet' in model_type:
            print("LossNet : {}".format(model_type))
            self.model = AuxiliaryResnet(**resnet_config)

class AuxiliaryResnet(nn.Module):

    def __init__(self, **kwargs):
        super(AuxiliaryResnet, self).__init__()

        # Define resnet model
        resnet_fn = getattr(models, kwargs['LossNet'])
        self.resnet = resnet_fn(pretrained=True, progress=True)

        # Clear unnecessary layers
        self.auxiliary_resnet_output_layer = kwargs['AUXILIARY_RESNET_OUTPUT_LAYER']
        if self.auxiliary_resnet_output_layer < 2:
            self.resnet.layer2 = torch.nn.Identity()
        if self.auxiliary_resnet_output_layer < 3:
            self.resnet.layer3 = torch.nn.Identity()
        if self.auxiliary_resnet_output_layer < 4:
            self.resnet.layer4 = torch.nn.Identity()
        self.resnet.avgpool = torch.nn.Identity()
        self.resnet.fc = torch.nn.Identity()

        # Freeze the model
        self.freeze = kwargs['AUXILIARY_RESNET_FREEZE'] if 'AUXILIARY_RESNET_FREEZE' in kwargs else True
        if self.freeze:
            for param in self.resnet.parameters():
                param.requires_grad = False

        # Add projection head
        self.with_projection_head = kwargs['WITH_PROJECTION_HEAD'] if 'WITH_PROJECTION_HEAD' in kwargs else None
        self.projection_head = nn.ModuleList()
        if self.with_projection_head is not None:
            for idx, layer in enumerate(self.with_projection_head):
                self.projection_head.append(torch.nn.Linear(layer[0], layer[1]))
                if idx != len(self.with_projection_head) - 1:
                    self.projection_head.append(torch.nn.ReLU())

    def forward(self, x):

        x_list = []
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x_list.append(x)
        if self.auxiliary_resnet_output_layer > 1:
            x = self.resnet.layer2(x)
            x_list.append(x)
        if self.auxiliary_resnet_output_layer > 2:
            x = self.resnet.layer3(x)
            x_list.append(x)
        if self.auxiliary_resnet_output_layer > 3:
            x = self.resnet.layer4(x)
            x_list.append(x)
        # Projection head
        if self.with_projection_head is not None:
            x = x.permute(0, 2, 3, 1)
            for layer in self.projection_head:
                x = layer(x)
            x = x.permute(0, 3, 1, 2)

        return x_list
        # return x_list


class Model(nn.Module):

    def __init__(self, backbone, **kwargs):
        super(Model, self).__init__()

        # Patch keys
        self.four_points_12 = None
        self.four_points_21 = None
        self.patch_size = kwargs['PATCH_SIZE']
        self.patch_keys = kwargs['PATCH_KEYS']
        self.delta_hat_keys = kwargs['DELTA_HAT_KEYS']

        # No DSAC needed
        if len(self.delta_hat_keys):
            self.hypothesis_no = 1

        # DSAC if required
        else:
            self.coordinate_field_12 = None
            self.coordinate_field_21 = None
            self.pf_keys = kwargs['PF_KEYS']
            self.hypothesis_no = kwargs['RANSAC_HYPOTHESIS_NO']
            self.point_per_hypothesis = kwargs['POINTS_PER_HYPOTHESIS']
            self.dsac = DSACSoftmax(**kwargs)

        # Triplet version
        self.triplet_version = kwargs['TRIPLET_LOSS']
        if self.triplet_version != '':
            self.mask_keys = kwargs['MASK_KEYS']
            self.change_detection_mask = kwargs['MASK_CRD'] if 'MASK_CRD' in kwargs else False
            self.triplet_margin = kwargs['TRIPLET_MARGIN']
            self.triplet_channel_aggregation = kwargs['TRIPLET_AGGREGATION']
            self.sampling_strategy = kwargs['SAMPLING_STRATEGY']
            self.triplet_distance = kwargs['TRIPLET_DISTANCE']
            if 'one-line' in self.triplet_version:
                self.triplet_loss = torch.nn.TripletMarginLoss(margin=self.triplet_margin, p=1, reduction='none')
            elif 'double-line' in self.triplet_version:
                self.triplet_mu = kwargs['TRIPLET_MU']
            self.mask_mu = kwargs['MASK_MU']
        
        self.config = kwargs
        #######################################################################
        # Auxiliary resnet
        #######################################################################
        
        self.auxiliary_net = InitLossNetwork(self.config).model

    def forward_map_field(self, perspective_field, self_coordinate_field, self_four_points):

        #######################################################################
        # Create field of the coordinates if needed
        #######################################################################

        batch_size = perspective_field.shape[0]
        pf_predicted_size = (batch_size, perspective_field.shape[-2] * perspective_field.shape[-1], 2)
        if self_coordinate_field is None or self_coordinate_field.shape != pf_predicted_size:
            y_patch_grid, x_patch_grid = np.mgrid[0:perspective_field.shape[-2], 0:perspective_field.shape[-1]]
            x_patch_grid = np.tile(x_patch_grid.reshape(1, -1), (batch_size, 1))
            y_patch_grid = np.tile(y_patch_grid.reshape(1, -1), (batch_size, 1))
            coordinate_field = np.stack((x_patch_grid, y_patch_grid), axis=1).transpose(0, 2, 1)
            self_coordinate_field = torch.from_numpy(coordinate_field).float().to(perspective_field.device)
            four_points = np.array([[0, 0], [perspective_field.shape[-1], 0],
                                    [perspective_field.shape[-1], perspective_field.shape[-2]],
                                    [0, perspective_field.shape[-2]]])
            four_points = torch.from_numpy(four_points).float().to(perspective_field.device)
            self_four_points = torch.unsqueeze(four_points, dim=0).repeat(batch_size*self.hypothesis_no, 1, 1)

        perspective_field = perspective_field.reshape(batch_size, 2, -1).permute(0, 2, 1)
        return self_coordinate_field + perspective_field, self_coordinate_field, self_four_points

    def forward(self, data):

        #######################################################################
        # DSAC is required
        #######################################################################
        if not len(self.delta_hat_keys):

            #######################################################################
            # Prepare data
            #######################################################################

            # Fetch perspective field data
            perspective_field_12 = data[self.pf_keys[0]]

            # Create coord field
            map_field_12, self.coordinate_field_12, self.four_points_12 = self.forward_map_field(
                perspective_field_12, self.coordinate_field_12, self.four_points_12)

            #######################################################################
            # DSAC with SoftMax
            #######################################################################

            batch_size = perspective_field_12.shape[0]
            homography_hats_12, homography_scores_12 = self.dsac(self.coordinate_field_12, map_field_12,
                                                                 hypothesis_no=self.hypothesis_no,
                                                                 points_per_hypothesis=self.point_per_hypothesis)
            four_points_transformed_12 = kornia.transform_points(homography_hats_12.reshape(-1, 3, 3),
                                                                 self.four_points_12)
            delta_hats_12 = (four_points_transformed_12 - self.four_points_12).reshape(batch_size, self.hypothesis_no,
                                                                                       4, 2)

            #######################################################################
            # Doubleline
            #######################################################################

            if 'double-line' in self.triplet_version:

                # Fetch perspective field data
                perspective_field_21 = data[self.pf_keys[1]]

                # Create coord field
                map_field_21, self.coordinate_field_21, self.four_points_21 = self.forward_map_field(
                    perspective_field_21, self.coordinate_field_21, self.four_points_21)

                #######################################################################
                # DSAC with SoftMax
                #######################################################################

                batch_size = perspective_field_21.shape[0]
                homography_hats_21, homography_scores_21 = self.dsac(self.coordinate_field_21, map_field_21,
                                                                     hypothesis_no=self.hypothesis_no,
                                                                     points_per_hypothesis=self.point_per_hypothesis)
                four_points_transformed_21 = kornia.transform_points(homography_hats_21.reshape(-1, 3, 3),
                                                                     self.four_points_21)
                delta_hats_21 = (four_points_transformed_21 - self.four_points_21).reshape(batch_size,
                                                                                           self.hypothesis_no,
                                                                                           4, 2)

        #######################################################################
        # No DSAC needed
        #######################################################################

        else:
            # Get delta_hats
            delta_hats_12 = data[self.delta_hat_keys[0]]
            homography_scores_12 = None

            if 'double-line' in self.triplet_version:
                delta_hats_21 = data[self.delta_hat_keys[1]]
        
        #######################################################################
        # Triplet loss
        #######################################################################

        # Triplet loss
        if 'one-line' in self.triplet_version:
            return triplet_resnet_loss(data, delta_hats_12, scores=homography_scores_12)
        elif 'double-line' in self.triplet_version:
            return triplet_resnet_loss(data, delta_hats_12, delta_hats_21=delta_hats_21, loss_net=self.auxiliary_net,
                                       patch_keys = self.patch_keys, mask_keys=self.mask_keys, hypothesis_no=self.hypothesis_no, 
                                       patch_size=self.patch_size, sampling_strategy=self.sampling_strategy, config=self.config)

        #######################################################################
        # Multihead loss
        #######################################################################

        else:
            return self.multihead_resnet_loss(data, delta_hats_12, scores=homography_scores_12)

    def predict_homography(self, data):

        #######################################################################
        # No DSAC needed
        #######################################################################

        if len(self.delta_hat_keys):

            # Get delta_hats
            delta_hats = data[self.delta_hat_keys[0]]
            # delta_hats_21 = data[self.delta_hat_keys[1]]
            return delta_hats, None

        #######################################################################
        # DSAC is required
        #######################################################################

        else:

            #######################################################################
            # Prepare data
            #######################################################################

            # Fetch perspective field data
            perspective_field_12 = data[self.pf_keys[0]]

            # Create coord field
            map_field_12, self.coordinate_field_12, self.four_points_12 = self.forward_map_field(
                perspective_field_12, self.coordinate_field_12, self.four_points_12)

            #######################################################################
            # DSAC with SoftMax
            #######################################################################

            batch_size = perspective_field_12.shape[0]
            homography_hats, homography_scores = self.dsac(self.coordinate_field_12, map_field_12,
                                                           hypothesis_no=self.hypothesis_no,
                                                           points_per_hypothesis=self.point_per_hypothesis)

            # Find the best homography
            indices = torch.argmax(homography_scores, dim=-1, keepdim=False)
            indices = indices.reshape(-1, 1, 1, 1).repeat(1, 1, 3, 3)
            homography_hats = torch.gather(homography_hats, dim=1, index=indices)

            # Find delta hats
            four_points = np.array([[0, 0], [perspective_field_12.shape[-1], 0],
                                    [perspective_field_12.shape[-1], perspective_field_12.shape[-2]],
                                    [0, perspective_field_12.shape[-2]]])
            four_points = torch.from_numpy(four_points).float().to(perspective_field_12.device)
            four_points = torch.unsqueeze(four_points, dim=0).repeat(batch_size, 1, 1)
            four_points_transformed = kornia.transform_points(homography_hats.reshape(-1, 3, 3), four_points)
            delta_hats = (four_points_transformed - four_points).reshape(batch_size, 4, 2)
            
            return delta_hats, None
