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

warnings.filterwarnings('ignore')  # CAREFULL!!! Supress all warnings


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
    has_obstacle = args.has_obstacle

    # ------------- Test options -------------
    is_test = args.is_test
    max_test_trials = args.max_test_trials  # Maximum number of test runs per case/scenario
    test_preset_cases = args.test_preset_cases
    test_preset_file = os.path.abspath(args.test_preset_file) if test_preset_cases else None

    # ------ Pre-loading and logging options ------
    load_snapshot = args.load_snapshot  # Load pre-trained snapshot of model?
    snapshot_file = os.path.abspath(args.snapshot_file) if load_snapshot else None
    continue_logging = args.continue_logging  # Continue logging from previous session
    logging_directory = os.path.abspath(args.logging_directory) if continue_logging else os.path.abspath('logs')
    save_visualizations = args.save_visualizations  # Save visualizations of FCN predictions? It takes some time.

    # Set random seed
    np.random.seed(random_seed)

    # Initialize pick-up system (camera and robot)
    # Initialize pick-and-place system (camera and robot)
    robot = Robot(is_sim, obj_mesh_dir, num_obj, workspace_limits,
                  tcp_host_ip, tcp_port, rtc_host_ip, rtc_port,
                  is_test, test_preset_cases, test_preset_file)

    if has_obstacle:
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

        # # dilate
        # obstacle_mask = np.uint8(obstacle_mask)
        # kernel = np.ones((3, 3), np.uint8)
        # obstacle_mask = cv2.dilate(obstacle_mask, kernel, iterations=4)

    robot.add_objects()

    # Initialize trainer
    trainer = Trainer(load_snapshot, snapshot_file, force_cpu)

    # Initialize data logger
    logger = Logger(continue_logging, logging_directory)
    logger.save_camera_info(robot.cam_intrinsics, robot.cam_pose,
                            robot.cam_depth_scale)  # Save camera intrinsics and pose
    logger.save_heightmap_info(workspace_limits, heightmap_resolution)  # Save heightmap parameters

    # Find last executed iteration of preloaded log, and load execution info and RL variables
    if continue_logging:
        trainer.iteration = int(input("Trainer iteration:"))
        # trainer.preload(logger.transitions_directory)

    # use heuristic_boostrap
    heuristic_bootstrap = True
    # Initialize change count
    no_change_count = 0

    # Quick hack for nonlocal memory between threads in Python 2
    nonlocal_variables = {'executing_action': False,
                          'best_pix_ind': None}

    # Parallel thread to process network output and execute actions
    # -------------------------------------------------------------
    def process_actions():
        while True:
            if nonlocal_variables['executing_action']:
                # Determine whether grasping or pushing should be executed based on network predictions
                if has_obstacle:
                    masked_prediction = push_predictions.copy()
                    masked_prediction[masked_prediction < 0] = 0
                    for rot_idx in range(int(16)):
                        masked_prediction[rot_idx][obstacle_mask > 0] = 0
                    best_push_conf = np.max(masked_prediction)
                else:
                    best_push_conf = np.max(push_predictions)
                print('Primitive confidence scores: %f (push)' % best_push_conf)

                # If heuristic bootstrapping is enabled: if change has not been detected more than 2 times,
                # execute heuristic algorithm to detect grasps/pushes NOTE: typically not necessary and can reduce
                # final performance.
                if heuristic_bootstrap and no_change_count >= 1:
                    print('Change not detected for more than two pushes. Running heuristic pushing.')
                    heuristic_push_predictions, nonlocal_variables['best_pix_ind'] = utils.push_heuristic(
                        valid_depth_heightmap)
                    predicted_value = push_predictions[nonlocal_variables['best_pix_ind']]

                else:
                    nonlocal_variables['best_pix_ind'] = np.unravel_index(np.argmax(push_predictions),
                                                                          push_predictions.shape)
                    predicted_value = np.max(push_predictions)

                # Save predicted confidence value
                trainer.predicted_value_log.append([predicted_value])
                logger.write_to_log('predicted-value', trainer.predicted_value_log)

                # Compute 3D position of pixel
                print('Push at (%d, %d, %d)' % (nonlocal_variables['best_pix_ind'][0],
                                                nonlocal_variables['best_pix_ind'][1],
                                                nonlocal_variables['best_pix_ind'][2]))
                best_rotation_angle = np.deg2rad(
                    nonlocal_variables['best_pix_ind'][0] * (360.0 / trainer.model.num_rotations))
                best_pix_x = nonlocal_variables['best_pix_ind'][2]
                best_pix_y = nonlocal_variables['best_pix_ind'][1]
                primitive_position = [best_pix_x * heightmap_resolution + workspace_limits[0][0],
                                      best_pix_y * heightmap_resolution + workspace_limits[1][0],
                                      0.026 + workspace_limits[2][0]]

                # # For pushing, adjust start position, and make sure z value is safe and not too low
                # finger_width = 0.02
                # safe_kernel_width = int(np.round((finger_width/2)/heightmap_resolution))
                # local_region = valid_depth_heightmap[max(best_pix_y - safe_kernel_width, 0):min(best_pix_y + safe_kernel_width + 1, valid_depth_heightmap.shape[0]), max(best_pix_x - safe_kernel_width, 0):min(best_pix_x + safe_kernel_width + 1, valid_depth_heightmap.shape[1])]
                # if local_region.size == 0:
                #     safe_z_position = workspace_limits[2][0]
                # else:
                #     safe_z_position = np.max(local_region) + workspace_limits[2][0]
                # primitive_position[2] = safe_z_position

                # Save executed primitive
                trainer.executed_action_log.append(
                    [0, nonlocal_variables['best_pix_ind'][0], nonlocal_variables['best_pix_ind'][1],
                     nonlocal_variables['best_pix_ind'][2]])  # push

                # Visualize executed primitive, and affordances
                if save_visualizations:
                    push_pred_vis = utils.get_prediction_vis(push_predictions, color_heightmap,
                                                             nonlocal_variables['best_pix_ind'])
                    logger.save_visualizations(trainer.iteration, push_pred_vis, 'push')
                    cv2.imwrite('visualization.push.png', push_pred_vis)

                # Initialize variables that influence reward
                change_detected = False

                # Execute primitive
                robot.push(primitive_position, best_rotation_angle, workspace_limits, push_length=0.03)

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

        if has_obstacle:
            # Pre-process color_height (mask)
            color_heightmap[obstacle_mask > 0] = [0, 0, 0]  # [45,45,45] for background
            valid_depth_heightmap[obstacle_mask > 0] = 0

        # Save RGB-D images and RGB-D heightmaps
        logger.save_images(trainer.iteration, color_img, depth_img, '0')
        logger.save_heightmaps(trainer.iteration, color_heightmap, valid_depth_heightmap, '0')

        # Reset simulation or pause real-world training if table is empty
        stuff_count = np.zeros(valid_depth_heightmap.shape)
        stuff_count[valid_depth_heightmap > 0.005] = 1
        empty_threshold = 50  # 300 (68 per cube)
        if np.sum(stuff_count) < empty_threshold or (is_sim and no_change_count > 3):  # 8 for 30 cubes
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
            push_predictions, state_feat = trainer.forward(color_heightmap, is_volatile=True)
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
            # reward_value = trainer.get_label_value(change_detected)  # pure single pushing reward
            reward_value = trainer.get_reward(color_heightmap, change_detected)

            trainer.reward_value_log.append([reward_value])
            logger.write_to_log('reward-value', trainer.reward_value_log)
            if not is_test:
                # Backpropagation
                loss_value = trainer.backprop(prev_color_heightmap, prev_best_pix_ind, reward_value)
                logger.writer.add_scalar('loss', loss_value, trainer.iteration)

            # Save model snapshot
            logger.save_backup_model(trainer.model, 'push')
            if trainer.iteration % 100 == 0:
                logger.save_model(trainer.iteration, trainer.model, 'push')
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
    parser.add_argument('--num_obj', dest='num_obj', type=int, action='store', default=2,  # 30 works
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
    parser.add_argument('--has_obstacle', dest='has_obstacle', action='store_true', default=False)

    # -------------- Testing options --------------
    parser.add_argument('--is_test', dest='is_test', action='store_true', default=False)
    parser.add_argument('--max_test_trials', dest='max_test_trials', type=int, action='store', default=100,
                        help='maximum number of test runs per case/scenario')
    parser.add_argument('--test_preset_cases', dest='test_preset_cases', action='store_true', default=False)
    parser.add_argument('--test_preset_file', dest='test_preset_file', action='store', default='test-10-obj-01.txt')

    # ------ Pre-loading and logging options ------
    parser.add_argument('--load_snapshot', dest='load_snapshot', action='store_true', default=False,
                        help='load pre-trained snapshot of model?')
    parser.add_argument('--snapshot_file', dest='snapshot_file', action='store')
    parser.add_argument('--continue_logging', dest='continue_logging', action='store_true', default=False,
                        help='continue logging from previous session?')
    parser.add_argument('--logging_directory', dest='logging_directory', action='store')
    parser.add_argument('--save_visualizations', dest='save_visualizations', action='store_true', default=True,
                        help='save visualizations of FCN predictions?')

    # Run main program with specified arguments
    args = parser.parse_args()
    main(args)
