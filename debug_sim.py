#!/usr/bin/env python

import numpy as np
import time
from robot import Robot

# User options (change me)
# --------------- Setup options ---------------

# --------------- Setup options ---------------
obj_mesh_dir = '/home/zippingsugar/Programs/visual-affordance-pushing/objects/cube'
num_obj = 100
workspace_limits = np.asarray([[-0.724, -0.276], [-0.224, 0.224], [-0.0001,
                                                                   0.6]])  # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
heightmap_resolution = 0.002
# Initialize pick-and-place system (camera and robot)
robot = Robot(True, obj_mesh_dir, num_obj, workspace_limits,
              None, None, None, None,
              False, False, None)

nonlocal_variables = {'best_pix_ind': [2, 100, 100]}

# Compute 3D position of pixel
print('Push at (%d, %d, %d)' % (
    nonlocal_variables['best_pix_ind'][0], nonlocal_variables['best_pix_ind'][1],
    nonlocal_variables['best_pix_ind'][2]))
best_rotation_angle = np.deg2rad(nonlocal_variables['best_pix_ind'][0] * (360.0 / 16))
best_pix_x = nonlocal_variables['best_pix_ind'][2]
best_pix_y = nonlocal_variables['best_pix_ind'][1]
primitive_position = [best_pix_x * heightmap_resolution + workspace_limits[0][0],
                      best_pix_y * heightmap_resolution + workspace_limits[1][0], 0.026 + workspace_limits[2][0]]
print(primitive_position)
for i in range(10):
    robot.push(primitive_position, best_rotation_angle, workspace_limits)
    time.sleep(1)

robot.restart_sim()
