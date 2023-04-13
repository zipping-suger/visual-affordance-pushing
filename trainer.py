import os
import numpy as np
import torch
from torch.autograd import Variable
from NN_models import FCNNet
from scipy import ndimage


# Basic trainer class
class Trainer(object):
    def __init__(self, load_snapshot, snapshot_file, force_cpu):

        # Check if CUDA can be used
        if torch.cuda.is_available() and not force_cpu:
            print("CUDA detected. Running with GPU acceleration.")
            self.use_cuda = True
        elif force_cpu:
            print("CUDA detected, but overriding with option '--cpu'. Running with only CPU.")
            self.use_cuda = False
        else:
            print("CUDA is *NOT* detected. Running with only CPU.")
            self.use_cuda = False

        # Initialize Loss criterion
        self.criterion = torch.nn.SmoothL1Loss(reduce=False)  # Huber loss
        if self.use_cuda:
            self.criterion = self.criterion.cuda()

        self.iteration = 0
        self.future_reward_discount = 0.5

        # Fully convolutional network
        self.model = FCNNet(use_cuda=self.use_cuda)

        # Load pre-trained model
        if load_snapshot:
            self.model.load_state_dict(torch.load(snapshot_file))
            print('Pre-trained model snapshot loaded from: %s' % snapshot_file)

        # Convert model from CPU to GPU
        if self.use_cuda:
            self.model = self.model.cuda()

        # Set model to training mode
        self.model.train()

        # Initialize optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-4, momentum=0.9, weight_decay=2e-5)

        # Initialize lists to save execution info and RL variables
        self.executed_action_log = []
        self.label_value_log = []
        self.reward_value_log = []
        self.predicted_value_log = []

    # Preload execution info and RL variables
    def preload(self, transitions_directory):
        # self.executed_action_log = np.loadtxt(os.path.join(transitions_directory, 'executed-action.log.txt'),
        #                                       delimiter=' ')
        # self.iteration = self.executed_action_log.shape[0] - 2
        # self.executed_action_log = self.executed_action_log[0:self.iteration, :]
        # self.executed_action_log = self.executed_action_log.tolist()
        # self.label_value_log = np.loadtxt(os.path.join(transitions_directory, 'label-value.log.txt'), delimiter=' ')
        # self.label_value_log = self.label_value_log[0:self.iteration]
        # self.label_value_log.shape = (self.iteration, 1)
        # self.label_value_log = self.label_value_log.tolist()
        self.predicted_value_log = np.loadtxt(os.path.join(transitions_directory, 'predicted-value.log.txt'),
                                              delimiter=' ')
        self.iteration = self.predicted_value_log.shape[0]
        self.predicted_value_log = self.predicted_value_log[0:self.iteration]
        # self.predicted_value_log.shape = (self.iteration, 1)
        self.predicted_value_log = self.predicted_value_log.tolist()
        self.reward_value_log = np.loadtxt(os.path.join(transitions_directory, 'reward-value.log.txt'), delimiter=' ')
        self.reward_value_log = self.reward_value_log[0:self.iteration]
        # self.reward_value_log.shape = (self.iteration, 1)
        self.reward_value_log = self.reward_value_log.tolist()

    # Compute forward pass through model to compute affordances/Q
    def forward(self, color_heightmap, is_volatile=False, specific_rotation=-1):

        # Apply 2x scale to input heightmaps
        color_heightmap_2x = ndimage.zoom(color_heightmap, zoom=[2, 2, 1], order=0)
        # Add extra padding (to handle rotations inside network)
        diag_length = float(color_heightmap_2x.shape[0]) * np.sqrt(2)
        diag_length = np.ceil(diag_length / 32) * 32
        padding_width = int((diag_length - color_heightmap_2x.shape[0]) / 2)
        color_heightmap_2x_r = np.pad(color_heightmap_2x[:, :, 0], padding_width, 'constant', constant_values=0)
        color_heightmap_2x_r.shape = (color_heightmap_2x_r.shape[0], color_heightmap_2x_r.shape[1], 1)
        color_heightmap_2x_g = np.pad(color_heightmap_2x[:, :, 1], padding_width, 'constant', constant_values=0)
        color_heightmap_2x_g.shape = (color_heightmap_2x_g.shape[0], color_heightmap_2x_g.shape[1], 1)
        color_heightmap_2x_b = np.pad(color_heightmap_2x[:, :, 2], padding_width, 'constant', constant_values=0)
        color_heightmap_2x_b.shape = (color_heightmap_2x_b.shape[0], color_heightmap_2x_b.shape[1], 1)
        color_heightmap_2x = np.concatenate((color_heightmap_2x_r, color_heightmap_2x_g, color_heightmap_2x_b), axis=2)

        # Pre-process color image (scale and normalize)
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
        input_color_image = color_heightmap_2x.astype(float) / 255
        for c in range(3):
            input_color_image[:, :, c] = (input_color_image[:, :, c] - image_mean[c]) / image_std[c]

        # Construct minibatch of size 1 (b,c,h,w)
        input_color_image.shape = (
            input_color_image.shape[0], input_color_image.shape[1], input_color_image.shape[2], 1)
        input_color_data = torch.from_numpy(input_color_image.astype(np.float32)).permute(3, 2, 0, 1)

        output_prob, state_feat = self.model.forward(input_color_data, is_volatile, specific_rotation)

        # Return Q values (and remove extra padding)
        for rotate_idx in range(len(output_prob)):
            if rotate_idx == 0:
                push_predictions = output_prob[rotate_idx][0].cpu().data.numpy()[:, 0,
                                   int(padding_width / 2):int(color_heightmap_2x.shape[0] / 2 - padding_width / 2),
                                   int(padding_width / 2):int(color_heightmap_2x.shape[0] / 2 - padding_width / 2)]
            else:
                push_predictions = np.concatenate((push_predictions,
                                                   output_prob[rotate_idx][0].cpu().data.numpy()[:, 0,
                                                   int(padding_width / 2):int(
                                                       color_heightmap_2x.shape[0] / 2 - padding_width / 2),
                                                   int(padding_width / 2):int(
                                                       color_heightmap_2x.shape[0] / 2 - padding_width / 2)]),
                                                  axis=0)

        return push_predictions, state_feat

    def backprop(self, color_heightmap, best_pix_ind, label_value):

        # Compute labels for grasp quality
        label = np.zeros((1, 320, 320))
        action_area = np.zeros((224, 224))
        action_area[best_pix_ind[1]][best_pix_ind[2]] = 1
        tmp_label = np.zeros((224, 224))
        tmp_label[action_area > 0] = label_value
        label[0, 48:(320 - 48), 48:(320 - 48)] = tmp_label

        # Compute label mask
        label_weights = np.zeros((1, 320, 320))
        tmp_label_weights = np.zeros((224, 224))
        tmp_label_weights[action_area > 0] = 1
        label_weights[0, 48:(320 - 48), 48:(320 - 48)] = tmp_label_weights

        # Compute loss and backward pass
        self.optimizer.zero_grad()
        loss_value = 0

        # Do forward pass with specified rotation (to save gradients)
        push_predictions, state_feat = self.forward(color_heightmap, is_volatile=False,
                                                    specific_rotation=best_pix_ind[0])

        if self.use_cuda:
            loss = self.criterion(self.model.output_prob[0][0].view(1, 320, 320),
                                  Variable(torch.from_numpy(label).float().cuda())) * Variable(
                torch.from_numpy(label_weights).float().cuda(), requires_grad=False)
        else:
            loss = self.criterion(self.model.output_prob[0][0].view(1, 320, 320),
                                  Variable(torch.from_numpy(label).float())) * Variable(
                torch.from_numpy(label_weights).float(), requires_grad=False)

        loss = loss.sum()
        loss.backward()
        loss_value = loss.cpu().data.numpy()
        print('Training loss: %f' % loss_value)
        self.optimizer.step()
        return loss_value

    def get_label_value(self, change_detected):
        # Compute current reward
        current_reward = 0
        if change_detected:
            current_reward = 1

        print('Current reward: %f' % (current_reward))

        return current_reward

    def get_reward(self, next_color_heightmap, change_detected):

        # Compute current reward
        global future_reward
        reward = 0
        future_reward = 0

        if change_detected:
            reward = 1
            next_push_predictions, next_state_feat = self.forward(next_color_heightmap, is_volatile=True)
            future_reward = np.max(next_push_predictions)

        print('Current reward: %f' % reward)
        print('Future reward: %f' % (future_reward))
        expected_reward = reward + self.future_reward_discount * future_reward
        print('Expected reward: %f + %f x %f = %f' % (
        reward, self.future_reward_discount, future_reward, expected_reward))
        return expected_reward
