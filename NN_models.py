from collections import OrderedDict
import torchvision
import numpy as np
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FCNNet(nn.Module):

    def __init__(self, use_cuda):
        super(FCNNet, self).__init__()
        self.use_cuda = use_cuda
        self.num_rotations = 16

        # Initialize network trunks with pre-trained nets on ImageNet
        res_model = torchvision.models.resnet18(pretrained=True)
        # res_model = torchvision.models.resnet50(pretrained=True)
        # res_model = torchvision.models.resnet101(pretrained=True)
        self.color_trunk_features = torch.nn.Sequential(*list(res_model.children())[:-2])

        # swin_model = torchvision.models.swin_v2_t(pretrained=True)
        # self.color_trunk_features = torch.nn.Sequential(*list(swin_model.children())[:-3])

        # convnex_model = torchvision.models.convnext_base(pretrained=True)
        # self.color_trunk_features = torch.nn.Sequential(*list(convnex_model.children())[:-2])

        self.feature_dim = 512  # for ResNet-18
        # self.feature_dim = 768  # for swin transformer v2_t, v2_s, convnext_tiny, convnext_small
        # self.feature_dim = 1024  # for swin transformer v2_b, convnext_base
        # self.feature_dim = 2048  # for ResNet-101, ResNet-50

        # Construct network branches for pushing
        self.push_net = nn.Sequential(OrderedDict([
            ('push-norm0', nn.BatchNorm2d(self.feature_dim)),
            ('push-relu0', nn.ReLU(inplace=True)),
            ('push-conv0', nn.Conv2d(self.feature_dim, 64, kernel_size=1, stride=1, bias=False)),
            ('push-norm1', nn.BatchNorm2d(64)),
            ('push-relu1', nn.ReLU(inplace=True)),
            ('push-conv1', nn.Conv2d(64, 1, kernel_size=1, stride=1, bias=False))
        ]))

        # Initialize network weights
        for m in self.named_modules():
            if 'push-' in m[0]:
                if isinstance(m[1], nn.Conv2d):
                    nn.init.kaiming_normal(m[1].weight.data)
                elif isinstance(m[1], nn.BatchNorm2d):
                    m[1].weight.data.fill_(1)
                    m[1].bias.data.zero_()

    def forward(self, input_color_data, is_volatile=False, specific_rotation=-1):

        if is_volatile:
            with torch.no_grad():
                output_prob = []
                interm_feat = []

                # Apply rotations to images
                for rotate_idx in range(self.num_rotations):
                    rotate_theta = np.radians(rotate_idx * (360 / self.num_rotations))

                    # Compute sample grid for rotation BEFORE neural network
                    affine_mat_before = np.asarray([[np.cos(-rotate_theta), np.sin(-rotate_theta), 0],
                                                    [-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
                    affine_mat_before.shape = (2, 3, 1)
                    affine_mat_before = torch.from_numpy(affine_mat_before).permute(2, 0, 1).float()
                    if self.use_cuda:
                        flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False).cuda(),
                                                         input_color_data.size())
                    else:
                        flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False),
                                                         input_color_data.size())

                    # Rotate images clockwise
                    if self.use_cuda:
                        rotate_color = F.grid_sample(Variable(input_color_data, volatile=True).cuda(), flow_grid_before,
                                                     mode='nearest')

                    else:
                        rotate_color = F.grid_sample(Variable(input_color_data, volatile=True), flow_grid_before,
                                                     mode='nearest')

                    # Compute intermediate features
                    interm_push_color_feat = self.color_trunk_features(rotate_color)
                    interm_feat.append([interm_push_color_feat])

                    # Compute sample grid for rotation AFTER branches
                    affine_mat_after = np.asarray([[np.cos(rotate_theta), np.sin(rotate_theta), 0],
                                                   [-np.sin(rotate_theta), np.cos(rotate_theta), 0]])
                    affine_mat_after.shape = (2, 3, 1)
                    affine_mat_after = torch.from_numpy(affine_mat_after).permute(2, 0, 1).float()
                    if self.use_cuda:
                        flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False).cuda(),
                                                        interm_push_color_feat.data.size())
                    else:
                        flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False),
                                                        interm_push_color_feat.data.size())

                    # Forward pass through branches, undo rotation on output predictions, upsample results
                    output_prob.append([nn.Upsample(scale_factor=16, mode='bilinear').forward(
                        F.grid_sample(self.push_net(interm_push_color_feat), flow_grid_after, mode='nearest'))])

            return output_prob, interm_feat

        else:
            self.output_prob = []
            self.interm_feat = []

            # Apply rotations to intermediate features
            # for rotate_idx in range(self.num_rotations):
            rotate_idx = specific_rotation
            rotate_theta = np.radians(rotate_idx * (360 / self.num_rotations))

            # Compute sample grid for rotation BEFORE branches
            affine_mat_before = np.asarray(
                [[np.cos(-rotate_theta), np.sin(-rotate_theta), 0], [-np.sin(-rotate_theta), np.cos(-rotate_theta), 0]])
            affine_mat_before.shape = (2, 3, 1)
            affine_mat_before = torch.from_numpy(affine_mat_before).permute(2, 0, 1).float()
            if self.use_cuda:
                flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False).cuda(),
                                                 input_color_data.size())
            else:
                flow_grid_before = F.affine_grid(Variable(affine_mat_before, requires_grad=False),
                                                 input_color_data.size())

            # Rotate images clockwise
            if self.use_cuda:
                rotate_color = F.grid_sample(Variable(input_color_data, requires_grad=False).cuda(), flow_grid_before,
                                             mode='nearest')
            else:
                rotate_color = F.grid_sample(Variable(input_color_data, requires_grad=False), flow_grid_before,
                                             mode='nearest')

            # Compute intermediate features
            interm_push_color_feat = self.color_trunk_features(rotate_color)
            self.interm_feat.append([interm_push_color_feat])

            # Compute sample grid for rotation AFTER branches
            affine_mat_after = np.asarray(
                [[np.cos(rotate_theta), np.sin(rotate_theta), 0], [-np.sin(rotate_theta), np.cos(rotate_theta), 0]])
            affine_mat_after.shape = (2, 3, 1)
            affine_mat_after = torch.from_numpy(affine_mat_after).permute(2, 0, 1).float()
            if self.use_cuda:
                flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False).cuda(),
                                                interm_push_color_feat.data.size())
            else:
                flow_grid_after = F.affine_grid(Variable(affine_mat_after, requires_grad=False),
                                                interm_push_color_feat.data.size())

            # Forward pass through branches, undo rotation on output predictions, upsample results
            self.output_prob.append([nn.Upsample(scale_factor=16, mode='bilinear').forward(
                F.grid_sample(self.push_net(interm_push_color_feat), flow_grid_after, mode='nearest'))])

            return self.output_prob, self.interm_feat
