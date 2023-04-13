import cv2
import numpy as np
from matplotlib.pyplot import imshow
import scipy
from scipy import spatial
import spfa
import time
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

draw = False
window_name = "Paint Goal Mask"

# Dimensions of the image
img_rows, img_cols = 224, 224

# rotation number
rot_num = 16

img = np.zeros((img_rows, img_cols, 1), np.uint8)


# plot related func
def nothing(x):
    pass


def draw_circle(event, x, y, flags, param):
    global draw, img

    if event == cv2.EVENT_LBUTTONDOWN:
        draw = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if draw:
            cv2.circle(img, (x, y), cv2.getTrackbarPos("Brush Size", window_name),
                       (255, 255, 255),
                       -1)

    elif event == cv2.EVENT_LBUTTONUP:
        draw = False
        cv2.circle(img, (x, y), cv2.getTrackbarPos("Brush Size", window_name),
                   (255, 255, 255),
                   -1)


# Modified by dist and discretized
def find_closest(val, arr):
    idx = np.abs(arr - (val % (2 * np.pi))).argmin()
    return idx, arr[idx]


def field_generator(obstacle_mask, step=8, use_sfpa=True, use_vortex =False):
    global img
    # set up carves
    cv2.namedWindow(window_name)
    cv2.createTrackbar("Brush Size", window_name, 1, 8, nothing)
    cv2.setMouseCallback(window_name, draw_circle)

    img = np.zeros((img_rows, img_cols, 1), np.uint8)

    while True:
        cv2.imshow(window_name, img)
        key = cv2.waitKey(1) & 0xff
        if key == ord('q'):
            mask_img = img  # save the goal mask img
            break
    cv2.destroyAllWindows()

    # Mask invalid actions
    goal_mask = mask_img[:, :, 0]
    obs_mask = obstacle_mask * 255

    # # Increase boundary
    # obs_mask = np.uint8(obs_mask)
    # kernel = np.ones((5, 5), np.uint8)
    # obs_mask = cv2.dilate(obs_mask, kernel, iterations=6)

    mask = goal_mask + obs_mask
    mask[mask > 0] = 255

    # Calculate potential
    goal_coords_list = np.argwhere(mask_img == 255)[:, 0:2]
    obstacle_coords_list = np.argwhere(obstacle_mask == 1)

    # Create grids of row and column indices
    rows, cols = np.meshgrid(np.arange(img_rows), np.arange(img_cols), indexing='ij')

    # Combine the row and column grids into a single array
    pixel_list = np.array([rows.flatten(), cols.flatten()], dtype=int).T
    pixel_coords = pixel_list.reshape(img_rows, img_cols, 2)

    if not use_vortex:

        if use_sfpa:
            # Create map
            free_space_map = np.ones([img_rows, img_cols], dtype=bool)
            free_space_map[obs_mask > 0] = False

            len_goal = len(goal_coords_list)
            sfpa_cdist = np.zeros((img_rows * img_cols, len_goal))
            for index in np.arange(len_goal):
                dists, _ = spfa.spfa(free_space_map, goal_coords_list[index])
                sfpa_cdist[:, index] = dists.reshape(img_rows * img_cols)
            potential_list = np.amin(sfpa_cdist, axis=1)

        else:
            # goal potential (attractive)
            goal_list = np.amin(spatial.distance.cdist(pixel_list, goal_coords_list), axis=1)

            # Log potential for obstacle (repulsive)
            safe_margin = 20
            if len(obstacle_coords_list) == 0:
                print("No obstacle detected")
                potential_list = goal_list
            else:
                obstacle_list = 50 * (
                            -np.log(np.amin(spatial.distance.cdist(pixel_list, obstacle_coords_list), axis=1)) + np.log(
                        safe_margin))
                obstacle_list[obstacle_list < 0] = 0

                # # 1/x potential for obstacle (repulsive)
                # obstacle_list =  5e2/(np.amin(spatial.distance.cdist(pixel_list,obstacle_coords_list),axis=1))

                potential_list = obstacle_list + goal_list
        # Reshape
        potential_field = potential_list.reshape(img_rows, img_cols)

        # Calculate the gradient in the x and ydirection
        img = potential_field
        obs_potential = 4e2
        img[obs_mask > 0] = obs_potential  # Experimental
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    else:
        obs_potential = 4e2
        # goal potential (attractive)
        goal_list = np.amin(spatial.distance.cdist(pixel_list, goal_coords_list), axis=1)

        # Log potential for obstacle (repulsive)
        safe_margin = 20
        obstacle_list = 50 * (
                -np.log(np.amin(spatial.distance.cdist(pixel_list, obstacle_coords_list), axis=1)) + np.log(
            safe_margin))
        obstacle_list[obstacle_list < 0] = 0

        # Reshape
        obstacle_field = obstacle_list.reshape(img_rows, img_cols)
        goal_field = goal_list.reshape(img_rows, img_cols)

        # Calculate the gradient in the x,y direction
        img = obstacle_field
        obs_sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        obs_sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

        # Calculate the gradient in the x,y direction
        img = goal_field
        goal_sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        goal_sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        sobelx = goal_sobelx - obs_sobely
        sobely = goal_sobely + obs_sobelx

        potential_field = goal_list.reshape(img_rows, img_cols)

    discret_theta_array = np.linspace(0, 2 * np.pi - (np.pi / 8), rot_num)

    # Modified by dist and discretized
    theta_org = np.arctan2(sobely, sobelx)
    theta_discret_idx = np.zeros((np.size(theta_org, 0), np.size(theta_org, 1)))
    theta_discret_val = np.zeros((np.size(theta_org, 0), np.size(theta_org, 1)))

    for (x, y), value in np.ndenumerate(theta_org):
        theta_discret_idx[x, y], theta_discret_val[x, y] = find_closest(value, discret_theta_array)

    x_flow = np.multiply(potential_field, np.cos(theta_discret_val))
    y_flow = np.multiply(potential_field, np.sin(theta_discret_val))

    theta_align_idx = (theta_discret_idx + 8) % 16

    # Visualize potential field
    x = np.linspace(0, 223, 224)
    y = np.linspace(0, 223, 224)
    X, Y = np.meshgrid(x, y)
    Z = potential_field
    Z[np.isinf(Z)] = obs_potential
    Z[obs_mask > 0] = obs_potential

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis')
    fig.colorbar(surf, shrink=0.4, aspect=2)

    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    if use_sfpa:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        # ax.set_zlabel('sfpa potential')
        # ax.set_title('sfpa potential visualized')
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        # ax.set_zlabel('Euclidean potential')
        # ax.set_title('Euclidean potential visualized')
    ax.view_init(30, 120)
    plt.show()

    # Plot the gradient field
    plt.quiver(pixel_coords[:, :, 1][::step, ::step], pixel_coords[:, :, 0][::step, ::step], -x_flow[::step, ::step],
               y_flow[::step, ::step])
    # plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.show()
    # plt.savefig('push_field.png')

    return mask, potential_field, theta_align_idx
