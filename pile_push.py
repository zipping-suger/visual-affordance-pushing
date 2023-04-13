#!/usr/bin/env python
import time
import os
import argparse
import numpy as np
import cv2
import torch
from robot import Robot
from trainer import Trainer
from logger import Logger
import utils
import threading
import warnings
from field import field_generator

warnings.filterwarnings('ignore')  # CAREFULL!!! Supress all warnings


def quick_save_model(save_dir, iteration, model, name):
    torch.save(model.state_dict(),
               os.path.join(save_dir, 'snapshot-%06d.%s.pth' % (iteration, name)))


def main(args):
    # --------------- Setup options ---------------
    is_sim = args.is_sim  # Run in simulation?
    tcp_host_ip = args.tcp_host_ip if not is_sim else None  # IP and port to robot arm as TCP client (UR5)
    tcp_port = args.tcp_port if not is_sim else None
    rtc_host_ip = args.rtc_host_ip if not is_sim else None  # IP and port to robot arm as real-time client (UR5)
    rtc_port = args.rtc_port if not is_sim else None
    obj_mesh_dir = os.path.abspath(
        args.obj_mesh_dir) if is_sim else None  # Directory containing 3D mesh files (.obj) of objects to be added to simulation
    num_obj = args.num_obj if is_sim else None  # Number of objects to add to simulation
    if is_sim:
        workspace_limits = np.asarray([[-0.724, -0.276], [-0.224, 0.224], [-0.0001,
                                                                           0.4]])  # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
    else:
        pass  # TODO realworld workspace

    heightmap_resolution = args.heightmap_resolution  # Meters per pixel of heightmap
    random_seed = args.random_seed
    force_cpu = args.force_cpu

    # ------------- Test options -------------
    is_test = args.is_test
    heuristic_bootstrap = args.heuristic_bootstrap
    max_test_trials = args.max_test_trials  # Maximum number of test runs per case/scenario
    test_preset_cases = args.test_preset_cases
    test_preset_file = os.path.abspath(args.test_preset_file) if test_preset_cases else None

    # ------ Pre-loading and logging options ------
    snapshot_file = os.path.abspath(args.snapshot_file)
    continue_logging = args.continue_logging  # Continue logging from previous session
    logging_directory = os.path.abspath(args.logging_directory) if continue_logging else os.path.abspath('logs')
    save_visualizations = args.save_visualizations  # Save visualizations of FCN predictions? It takes some time.

    # Set random seed
    np.random.seed(random_seed)

    # Initialize camera and robot
    robot = Robot(is_sim, obj_mesh_dir, num_obj, workspace_limits,
                  tcp_host_ip, tcp_port, rtc_host_ip, rtc_port,
                  is_test, test_preset_cases, test_preset_file)

    # Initialize trainer
    trainer = Trainer(True, snapshot_file, False)
    # trainer.model.eval()

    # Make sure simulation is still stable (if not, reset simulation)
    if is_sim:
        robot.check_sim()

    # Get latest RGB-D image
    color_img, depth_img = robot.get_camera_data()
    depth_img = depth_img * robot.cam_depth_scale  # Apply depth scale from calibration

    # Get heightmap from RGB-D image (by re-projecting 3D point cloud)
    color_heightmap, depth_heightmap = utils.get_heightmap(color_img, depth_img, robot.cam_intrinsics,
                                                           robot.cam_pose, workspace_limits, heightmap_resolution)
    valid_depth_heightmap = depth_heightmap.copy()
    valid_depth_heightmap[np.isnan(valid_depth_heightmap)] = 0

    # Get Obstacle mask
    obstacle_mask = valid_depth_heightmap.copy()
    obstacle_mask = np.where(obstacle_mask > 0.02, 1, 0)

    # Define Goal
    mask, potential_field, theta_idx = field_generator(obstacle_mask, use_sfpa=True, use_vortex=False)
    normalized_potential_field = (potential_field - np.min(potential_field)) / (np.max(potential_field) - np.min(potential_field))

    robot.add_objects()

    # Initialize change count
    no_change_count = 0

    save_dir = './logs/pile_push_model'

    # Quick hack for nonlocal memory between threads in Python 2
    nonlocal_variables = {'executing_action': False,
                          'best_pix_ind': None}

    def process_actions():
        while True:
            if nonlocal_variables['executing_action']:
                # Determine whether grasping or pushing should be executed based on network predictions
                best_push_conf = np.max(push_predictions)
                print('Primitive confidence scores: %f (push)' % best_push_conf)

                nonlocal_variables['best_pix_ind'] = np.unravel_index(np.argmax(push_predictions),
                                                                      push_predictions.shape)
                predicted_value = np.max(push_predictions)

                # Save predicted confidence value
                trainer.predicted_value_log.append([predicted_value])

                # Compute 3D position of pixel
                print('Push at (%d, %d, %d)' % (nonlocal_variables['best_pix_ind'][0],
                                                nonlocal_variables['best_pix_ind'][1],
                                                nonlocal_variables['best_pix_ind'][2]))

                best_rotation_angle = np.deg2rad(
                    nonlocal_variables['best_pix_ind'][0] * (360.0 / trainer.model.num_rotations))
                best_pix_x = nonlocal_variables['best_pix_ind'][2]
                best_pix_y = nonlocal_variables['best_pix_ind'][1]

                push_len = normalized_potential_field[
                               nonlocal_variables['best_pix_ind'][1], nonlocal_variables['best_pix_ind'][
                                   2]] * 0.08 + 0.02   #0.1 +0.05 for obstacle
                print("push length:{}".format(push_len))
                primitive_position = [best_pix_x * heightmap_resolution + workspace_limits[0][0],
                                      best_pix_y * heightmap_resolution + workspace_limits[1][0],
                                      0.026 + workspace_limits[2][0]]

                # Save executed primitive
                trainer.executed_action_log.append(
                    [0, nonlocal_variables['best_pix_ind'][0], nonlocal_variables['best_pix_ind'][1],
                     nonlocal_variables['best_pix_ind'][2]])  # push

                # Visualize executed primitive, and affordances
                if save_visualizations:
                    push_pred_vis = utils.get_prediction_vis(push_predictions, color_heightmap,
                                                             nonlocal_variables['best_pix_ind'])
                    # push_pred_vis = utils.get_prediction_vis(heuristic_push_predictions, color_heightmap,
                    #                                          nonlocal_variables['best_pix_ind'])

                    cv2.imwrite('visualization.push.png', push_pred_vis)

                # Initialize variables that influence reward
                change_detected = False

                # Execute primitive
                robot.push(primitive_position, best_rotation_angle, workspace_limits, push_len)

                nonlocal_variables['executing_action'] = False
            time.sleep(0.01)

    action_thread = threading.Thread(target=process_actions)
    action_thread.daemon = True
    action_thread.start()
    exit_called = False
    # -------------------------------------------------------------
    # -------------------------------------------------------------

    # Start main training/testing loop
    while True:
        print(trainer.use_cuda)
        print('\n%s iteration: %d' % ('Training', trainer.iteration))
        iteration_time_0 = time.time()

        # Make sure simulation is still stable (if not, reset simulation)
        if is_sim:
            robot.check_sim()

        # Get latest RGB-D image
        color_img, depth_img = robot.get_camera_data()
        depth_img = depth_img * robot.cam_depth_scale  # Apply depth scale from calibration

        # Get heightmap from RGB-D image (by re-projecting 3D point cloud)
        color_heightmap, depth_heightmap = utils.get_heightmap(color_img, depth_img, robot.cam_intrinsics,
                                                               robot.cam_pose, workspace_limits, heightmap_resolution)
        valid_depth_heightmap = depth_heightmap.copy()
        valid_depth_heightmap[np.isnan(valid_depth_heightmap)] = 0

        # Pre-process color_height and the valid_depth_heightmap(mask)
        color_heightmap[mask == 255] = [45, 45, 45]
        valid_depth_heightmap[mask == 255] = 0

        # Reset simulation or pause real-world training if table is empty
        stuff_count = np.zeros(valid_depth_heightmap.shape)
        stuff_count[valid_depth_heightmap > 0.01] = 1
        empty_threshold = 0  # 300 (68 per cube)
        if np.sum(stuff_count) < empty_threshold or (is_sim and no_change_count > 10):  # 8 for 30 cubes
            no_change_count = 0
            if is_sim:
                print('Not enough objects in view (value: %d)! Repositioning objects.' % (np.sum(stuff_count)))
                robot.restart_sim()
                robot.add_objects()
            else:
                # print('Not enough stuff on the table (value: %d)! Pausing for 30 seconds.' % (np.sum(stuff_count)))
                # time.sleep(30)
                print('Not enough stuff on the table (value: %d)! Flipping over bin of objects...' % (
                    np.sum(stuff_count)))
                robot.restart_real()
            continue

        if not exit_called:
            # Run forward pass with network to get affordances
            original_push_predictions, state_feat = trainer.forward(color_heightmap, is_volatile=True)

            # calculated weighted pushing quality
            weighted_push_predictis = np.zeros((16, 224, 224))
            for rot_idx in range(int(16)):
                tmp_rot = np.cos(abs(theta_idx - rot_idx) / 8 * np.pi)
                positive_push_pred = original_push_predictions[rot_idx].copy()
                positive_push_pred[positive_push_pred < 0] = 0
                weighted_push_predictis[rot_idx] = np.multiply(positive_push_pred, tmp_rot)
                weighted_push_predictis[rot_idx][mask == 255] = 0

            if heuristic_bootstrap:  # and no_change_count >= 1:
                print('Change not detected for more than two pushes. Running heuristic pushing.')
                heuristic_push_predictions, nonlocal_variables['best_pix_ind'] = utils.push_heuristic(
                    valid_depth_heightmap)
                weighted_push_predictis = np.zeros((16, 224, 224))
                for rot_idx in range(int(16)):
                    tmp_rot = np.cos(abs(theta_idx - rot_idx) / 8 * np.pi)
                    positive_push_pred = heuristic_push_predictions[rot_idx].copy()
                    positive_push_pred[positive_push_pred < 0] = 0
                    weighted_push_predictis[rot_idx] = np.multiply(positive_push_pred, tmp_rot)
                    weighted_push_predictis[rot_idx] = np.multiply(normalized_potential_field,
                                                                   weighted_push_predictis[rot_idx])
                    weighted_push_predictis[rot_idx][mask == 255] = 0
                push_predictions = weighted_push_predictis

            else:
                weighted_push_predictis = np.zeros((16, 224, 224))
                for rot_idx in range(int(16)):
                    tmp_rot = np.cos(abs(theta_idx - rot_idx) / 8 * np.pi)
                    positive_push_pred = original_push_predictions[rot_idx].copy()
                    positive_push_pred[positive_push_pred < 0] = 0
                    weighted_push_predictis[rot_idx] = np.multiply(positive_push_pred, tmp_rot)
                    weighted_push_predictis[rot_idx] = np.multiply(normalized_potential_field,
                                                                   weighted_push_predictis[rot_idx])
                    weighted_push_predictis[rot_idx][mask == 255] = 0
                push_predictions = weighted_push_predictis

            # Execute the best primitive action on robot in another thread
            nonlocal_variables['executing_action'] = True

        # Run training iteration in current thread (aka training thread)
        if 'prev_color_img' in locals():
            # Detect changes
            depth_diff = abs(depth_heightmap - prev_depth_heightmap)
            depth_diff[np.isnan(depth_diff)] = 0
            depth_diff[depth_diff > 0.3] = 0
            depth_diff[depth_diff < 0.01] = 0
            depth_diff[depth_diff > 0] = 1
            change_threshold = 68
            change_value = np.sum(depth_diff)
            change_detected = change_value > change_threshold
            print('Change detected: %r (value: %d)' % (change_detected, change_value))

            if change_detected:
                no_change_count = 0
            else:
                no_change_count += 1

            # Compute training labels
            reward_value = trainer.get_label_value(change_detected)
            # Backpropagation
            loss_value = trainer.backprop(prev_color_heightmap, prev_best_pix_ind, reward_value)

            # Save model snapshot
            quick_save_model(save_dir, 0, trainer.model, 'backup')
            if trainer.iteration % 100 == 0:
                quick_save_model(save_dir, trainer.iteration, trainer.model, 'push')
                if trainer.use_cuda:
                    trainer.model = trainer.model.cuda()

        # Sync both action thread and training thread
        while nonlocal_variables['executing_action']:
            time.sleep(0.01)

        if exit_called:
            break

        # Save information for next training step
        prev_color_img = color_img.copy()
        prev_depth_img = depth_img.copy()
        prev_color_heightmap = color_heightmap.copy()
        prev_depth_heightmap = depth_heightmap.copy()
        prev_valid_depth_heightmap = valid_depth_heightmap.copy()
        prev_best_pix_ind = nonlocal_variables['best_pix_ind']

        trainer.iteration += 1
        iteration_time_1 = time.time()
        print('Time elapsed: %f' % (iteration_time_1 - iteration_time_0))


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(
        description='Train robotic agents to learn how to plan complementary pushing actions for manipulation with deep reinforcement learning in PyTorch.')

    # --------------- Setup options ---------------
    parser.add_argument('--is_sim', dest='is_sim', action='store_true', default=True, help='run in simulation?')
    parser.add_argument('--obj_mesh_dir', dest='obj_mesh_dir', action='store', default='objects/VPG_0.3',
                        help='directory containing 3D mesh files (.obj) of objects to be added to simulation')
    parser.add_argument('--num_obj', dest='num_obj', type=int, action='store', default=10,  # 30 works
                        help='number of objects to add to simulation')
    parser.add_argument('--tcp_host_ip', dest='tcp_host_ip', action='store', default='100.127.7.223',
                        help='IP address to robot arm as TCP client (UR5)')
    parser.add_argument('--tcp_port', dest='tcp_port', type=int, action='store', default=30002,
                        help='port to robot arm as TCP client (UR5)')
    parser.add_argument('--rtc_host_ip', dest='rtc_host_ip', action='store', default='100.127.7.223',
                        help='IP address to robot arm as real-time client (UR5)')
    parser.add_argument('--rtc_port', dest='rtc_port', type=int, action='store', default=30003,
                        help='port to robot arm as real-time client (UR5)')
    parser.add_argument('--heightmap_resolution', dest='heightmap_resolution', type=float, action='store',
                        default=0.002, help='meters per pixel of heightmap')
    parser.add_argument('--random_seed', dest='random_seed', type=int, action='store', default=1234,
                        help='random seed for simulation and neural net initialization')
    parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False,
                        help='force code to run in CPU mode')

    # -------------- Testing options --------------
    parser.add_argument('--is_test', dest='is_test', action='store_true', default=False)
    parser.add_argument('--use_heuristic', dest='heuristic_bootstrap', action='store_true', default=False)
    parser.add_argument('--max_test_trials', dest='max_test_trials', type=int, action='store', default=100,
                        help='maximum number of test runs per case/scenario')
    parser.add_argument('--test_preset_cases', dest='test_preset_cases', action='store_true', default=False)
    parser.add_argument('--test_preset_file', dest='test_preset_file', action='store', default='test-10-obj-01.txt')

    # ------ Pre-loading and logging options ------
    parser.add_argument('--load_snapshot', dest='load_snapshot', action='store_true', default=False,
                        help='load pre-trained snapshot of model?')
    parser.add_argument('--snapshot_file', dest='snapshot_file', action='store',
                        default='./logs/ResNet18_m2o_mixed/models/snapshot-011000.push.pth')
                        # default='./logs/ResNet101_o2m_mixed/models/snapshot-007400.push.pth')

    parser.add_argument('--continue_logging', dest='continue_logging', action='store_true', default=False,
                        help='continue logging from previous session?')
    parser.add_argument('--logging_directory', dest='logging_directory', action='store')
    parser.add_argument('--save_visualizations', dest='save_visualizations', action='store_true', default=True,
                        help='save visualizations of FCN predictions?')

    # Run main program with specified arguments
    args = parser.parse_args()
    main(args)
