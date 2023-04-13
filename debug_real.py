#!/usr/bin/env python

import numpy as np
from utils import rotate_matrix
from robot import Robot
import time

# User options (change me)
# Nominal2Real -45; Real2Nominal +45;
# --------------- Setup options ---------------
tcp_host_ip = '192.168.1.10'  # IP and port to robot arm as TCP client (UR5)
tcp_port = 30002
workspace_limits = np.asarray([[0.3, 0.748], [-0.224, 0.224], [-0.255,
                                                               -0.1]])  # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
# ---------------------------------------------

# Initialize robot and move to home pose
robot = Robot(False, None, None, workspace_limits,
              tcp_host_ip, tcp_port, None, None,
              False, None, None)
# Default joint speed configuration
robot.joint_acc = 1.4  # Safe: 1.4
robot.joint_vel = 1.05  # Safe: 1.05
# Default tool speed configuration
robot.tool_acc = 1.2  # Safe: 0.5
robot.tool_vel = 0.25  # Safe: 0.2

# # # Workspace test
# nominal_home = np.array([0.6, 0, 0.3])
# #
# robot.move_to(rotate_matrix(nominal_home, -45), [0, 0, np.pi/2])
# robot.move_to(rotate_matrix((nominal_home + np.array([0.2, 0.2, -0.1])), -45), [0, 0, 1.5726983572039481])
# robot.move_to(rotate_matrix((nominal_home + np.array([0.2, -0.2, -0.1])), -45), [0, 0, 1.5726983572039481])
# robot.move_to(rotate_matrix((nominal_home + np.array([-0.2, -0.2, -0.1])), -45), [0, 0, 1.5726983572039481])
# robot.move_to(rotate_matrix((nominal_home + np.array([-0.2, 0.2, -0.1])), -45), [0, 0, 1.5726983572039481])
#
# robot.move_to(rotate_matrix(nominal_home, -45), [0, 0, 1.5726983572039481])


workspace_limits = np.asarray([[0.4, 0.8], [-0.2, 0.2], [0.2, 0.25]])  # Cols: min max, Rows: x y z

# # Make robot gripper point upwards
# robot.move_joints(robot.home_joint_config+[np.pi/6, 0, 0, 0, 0, 0])
# time.sleep(4)
# print("here")
# robot.move_joints(robot.home_joint_config+[np.pi/6, np.pi/6, 0, 0, 0, 0])
# time.sleep(4)
# robot.move_joints(robot.home_joint_config+[np.pi/6, np.pi/6, np.pi/6, 0, 0, 0])
# time.sleep(4)

# Calibration points test
calib_grid_step = 0.05
tool_orientation = [-1.7098953869561828, 0.0028747355674945325, 1.2130540341778435]
robot.move_to(rotate_matrix([0.4,0.05,0.2], -45), tool_orientation)
robot.move_to(rotate_matrix([0.4,0.05,0.3], -45), tool_orientation)
robot.go_home()

# Construct 3D calibration grid across workspace
gridspace_x = np.linspace(workspace_limits[0][0], workspace_limits[0][1],
                          1 + int((workspace_limits[0][1] - workspace_limits[0][0]) / calib_grid_step))
gridspace_y = np.linspace(workspace_limits[1][0], workspace_limits[1][1],
                          1 + int((workspace_limits[1][1] - workspace_limits[1][0]) / calib_grid_step))
gridspace_z = np.linspace(workspace_limits[2][0], workspace_limits[2][1],
                          1 + int((workspace_limits[2][1] - workspace_limits[2][0]) / calib_grid_step))
calib_grid_x, calib_grid_y, calib_grid_z = np.meshgrid(gridspace_x, gridspace_y, gridspace_z)
num_calib_grid_pts = calib_grid_x.shape[0] * calib_grid_x.shape[1] * calib_grid_x.shape[2]
calib_grid_x.shape = (num_calib_grid_pts, 1)
calib_grid_y.shape = (num_calib_grid_pts, 1)
calib_grid_z.shape = (num_calib_grid_pts, 1)
calib_grid_pts = np.concatenate((calib_grid_x, calib_grid_y, calib_grid_z), axis=1)

for calib_pt_idx in range(num_calib_grid_pts):
    tool_position = calib_grid_pts[calib_pt_idx, :]
    print(tool_position)
    robot.move_to(rotate_matrix(tool_position, -45), tool_orientation)
    time.sleep(1)

robot.go_home()




