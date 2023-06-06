import json
import pprint
from os.path import join
import glob
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.cm as cm
import scipy

from helpers import *

with open(
        '/Users/arthurlamard/Documents/Allemagne/cours/AI-PROJECT-SMART_RECORDING_PIPELINE/PROJET/camera_lidar_semantic_bboxes/cams_lidar.json',
        'r') as f:
    config = json.load(f)

vehicle_view = target_view = config['vehicle']['view']
cam_fc_view = config['cameras']['front_center']['view']
lidar_fc_view = config['lidars']['front_center']['view']

cam_fc_axes = get_axes_of_a_view(cam_fc_view)
lidar_fc_axes = get_axes_of_a_view(lidar_fc_view)

cam_fc_to_lidar_fc = transform_from_to(lidar_fc_view, cam_fc_view)
lidar_fc_to_cam_fc = transform_from_to(cam_fc_view, lidar_fc_view)

##start plotting
# fig, (ax1, ax2, ax3) = plt.subplots(1,3)
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.set_aspect('equal', adjustable='box')
ax1.set_xlabel('x (front) [m]')
ax1.set_ylabel('y (side) [m]')
ax1.set_xlim(0, 60)
ax1.set_ylim(-18, 18)

# ego position
rect = Rectangle((config['vehicle']['ego-dimensions']['x-range'][0],
                  config['vehicle']['ego-dimensions']['y-range'][0]),
                 np.diff(config['vehicle']['ego-dimensions']['x-range']),
                 np.diff(config['vehicle']['ego-dimensions']['y-range']),
                 linewidth=1, edgecolor='b', facecolor='none')

# lidar

file_name_lidar = "/Users/arthurlamard/Documents/Allemagne/cours/AI-PROJECT-SMART_RECORDING_PIPELINE/PROJET/camera_lidar_semantic_bboxes/train/20180807_145028/lidar/cam_front_center/20180807145028_lidar_frontcenter_000000091.npz"

lidar_fc_in_camfc = np.load(file_name_lidar)

# print("file name lidar : ",(list(lidar_fc_in_camfc.keys())))
ax1.add_patch(rect)
lidar_fc_in_global = project_lidar_from_to(lidar_fc_in_camfc, cam_fc_view, vehicle_view)

pc = lidar_fc_in_global['points']
depths = lidar_fc_in_global['depth']
# we say everything lower than half a meter below the lidar is ground.
pseudo_objects = pc[pc[:, 2] > -0.5][:, :2]
pseudo_ground = pc[pc[:, 2] < -0.5][:, :2]
ax1.scatter(*pseudo_ground.T, s=0.1, c='g')
ax1.scatter(*pseudo_objects.T, s=0.1, c='y')
# image
file_name_image = extract_image_file_name_from_lidar_file_name(file_name_lidar)
img = plt.imread(file_name_image)

get_bboxes_coords(file_name_image)

undist_img = undistort_image(img, 'front_center', config)
undistorted_params = np.array([[0., 0., 0., 0., 0.]])
ax2.imshow(img)
pixel_coords = projectPoints(pc, config['cameras']['front_center']['CamMatrix'], undistorted_params)

# with actually distored images use this instead.
# pixel_coords = projectPoints(pc, config['cameras']['front_center']['CamMatrix'],
#                                  config['cameras']['front_center']['Distortion'])


ax2.scatter(*pixel_coords.T, s=0.2, color=cm.rainbow(1 - depths / 40))

# gather 3d position.
coord_list = get_bboxes_coords(file_name_image)
#print("coord_list : ", coord_list[1])
bbx_coord = []
for (i, j) in zip((coord_list[0]), (coord_list[1])):
    # print(i)
    # print("bbx_coord : ", bbx_coord)
    bounding_box_2d = np.array(i)  # top left -> bottom right.
    center = np.mean(bounding_box_2d, axis=0)
    idx_nearest_3d_point = np.argmin(np.sum(np.abs(pixel_coords - center), axis=1))
    truck_position = pc[idx_nearest_3d_point]
    print(f"truck {i} position : {truck_position}| confidence : {j}")
    truck_marker = plt.Circle((truck_position[0], truck_position[1]), 0.5, color='r')
    ax1.add_patch(truck_marker)
    bounding_box = Rectangle(bounding_box_2d[0],
                             np.diff(bounding_box_2d[:, 0]),
                             np.diff(bounding_box_2d[:, 1]),
                             linewidth=1, edgecolor='r', facecolor='none')
    ax2.add_patch(bounding_box)

'''
#road
border_points_road = [[0,1113], [951,753],[1051,798], [1435, 1205]]
ctr = np.array(border_points_road).reshape((-1,1,2)).astype(np.int32)
road_img = cv2.drawContours(img,[ctr],0,(255,255,255),1)
road_point_mask = [cv2.pointPolygonTest(ctr, point, False) == 1.0 for point in pixel_coords]
points_within_road_px = pixel_coords[road_point_mask]
points_within_road_pc = pc[road_point_mask]
center, normal_vec = planeFit(points_within_road_pc)
def generate_optimization_fns(normal_vec, center, target_points):
    v1 = np.cross(normal_vec, (1,0,0))
    v2 = np.cross(normal_vec, v1)
    def project_params(x):
        params = x.reshape(-1,2)
        points= v1*params[:,0:1] + v2*params[:,1:2] + center
        points_px = projectPoints(points, config['cameras']['front_center']['CamMatrix'], undistorted_params)
        return points, points_px
    def loss_fn(x):
        points, points_px = project_params(x)
        return np.linalg.norm(points_px-target_points)
    return project_params, loss_fn
project_fn, loss_fn = generate_optimization_fns(normal_vec, center, border_points_road)
initial_params = np.zeros(len(border_points_road)*2)
res = scipy.optimize.minimize(loss_fn, initial_params)
points3d, points_px = project_fn(res.x)
ax3.scatter(*points_px.T, c='r')


plt.scatter(*points_within_road_px.T, s=0.2)
ax3.imshow(road_img)
'''

plt.show()
