import inspect
import json
import os
import pprint
import shutil
import sys
from os.path import join
import glob
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.cm as cm
import scipy
import subprocess
from helpers import *


from local_test.yolov5 import detect
from YOLOPv2 import demo
from YOLOPv2.utils_ypv2.utils import \
    time_synchronized, select_device, increment_path, \
    scale_coords, xyxy2xywh, non_max_suppression, split_for_trace_model, \
    driving_area_mask, lane_line_mask, plot_one_box, show_seg_result, \
    AverageMeter, \
    LoadImages
"""
TO AVOID PATH PROBLEMS
export PYTHONPATH="${PYTHONPATH}:/home/sa13291/Documents/ARTHUR_LAMARD"

"""


with open(
        '/home/sa13291/Documents/ARTHUR_LAMARD/local_test/camera_lidar_semantic_bboxes/cams_lidar.json',
        'r') as f:
    config = json.load(f)

vehicle_view = target_view = config['vehicle']['view']
cam_fc_view = config['cameras']['front_center']['view']
lidar_fc_view = config['lidars']['front_center']['view']

cam_fc_axes = get_axes_of_a_view(cam_fc_view)
lidar_fc_axes = get_axes_of_a_view(lidar_fc_view)

cam_fc_to_lidar_fc = transform_from_to(lidar_fc_view, cam_fc_view)
lidar_fc_to_cam_fc = transform_from_to(cam_fc_view, lidar_fc_view)




# lidar

file_name_lidar = ["/home/sa13291/Documents/ARTHUR_LAMARD/local_test/camera_lidar_semantic_bboxes/test/20181108_123750/lidar/cam_front_center/20181108123750_lidar_frontcenter_000007332.npz",
                    "/home/sa13291/Documents/ARTHUR_LAMARD/local_test/camera_lidar_semantic_bboxes/test/20181108_123750/lidar/cam_front_center/20181108123750_lidar_frontcenter_000007339.npz",
                   "/home/sa13291/Documents/ARTHUR_LAMARD/local_test/camera_lidar_semantic_bboxes/test/20181108_123750/lidar/cam_front_center/20181108123750_lidar_frontcenter_000007349.npz",
                   "/home/sa13291/Documents/ARTHUR_LAMARD/local_test/camera_lidar_semantic_bboxes/test/20181108_123750/lidar/cam_front_center/20181108123750_lidar_frontcenter_000007350.npz",

                   ]
'''file_name_lidar = ["/home/sa13291/Documents/ARTHUR_LAMARD/local_test/camera_lidar_semantic_bboxes/test/20181204_170238/lidar/cam_front_center/20181204170238_lidar_frontcenter_000114521.npz",
                   "/home/sa13291/Documents/ARTHUR_LAMARD/local_test/camera_lidar_semantic_bboxes/test/20181204_170238/lidar/cam_front_center/20181204170238_lidar_frontcenter_000114542.npz",
                   "/home/sa13291/Documents/ARTHUR_LAMARD/local_test/camera_lidar_semantic_bboxes/test/20181204_170238/lidar/cam_front_center/20181204170238_lidar_frontcenter_000114623.npz",
                   "/home/sa13291/Documents/ARTHUR_LAMARD/local_test/camera_lidar_semantic_bboxes/test/20181204_170238/lidar/cam_front_center/20181204170238_lidar_frontcenter_000114684.npz",
                   "/home/sa13291/Documents/ARTHUR_LAMARD/local_test/camera_lidar_semantic_bboxes/test/20181204_170238/lidar/cam_front_center/20181204170238_lidar_frontcenter_000114950.npz",
                   "/home/sa13291/Documents/ARTHUR_LAMARD/local_test/camera_lidar_semantic_bboxes/test/20181204_170238/lidar/cam_front_center/20181204170238_lidar_frontcenter_000115848.npz",
                   "/home/sa13291/Documents/ARTHUR_LAMARD/local_test/camera_lidar_semantic_bboxes/test/20181204_170238/lidar/cam_front_center/20181204170238_lidar_frontcenter_000116238.npz",
                   "/home/sa13291/Documents/ARTHUR_LAMARD/local_test/camera_lidar_semantic_bboxes/test/20181204_170238/lidar/cam_front_center/20181204170238_lidar_frontcenter_000116402.npz",
                   "/home/sa13291/Documents/ARTHUR_LAMARD/local_test/camera_lidar_semantic_bboxes/test/20181204_170238/lidar/cam_front_center/20181204170238_lidar_frontcenter_000116496.npz",
                   "/home/sa13291/Documents/ARTHUR_LAMARD/local_test/camera_lidar_semantic_bboxes/test/20181204_170238/lidar/cam_front_center/20181204170238_lidar_frontcenter_000116600.npz",
                   ]'''
#file_name_lidar=["/home/sa13291/ai_proj/camera_lidar_semantic/20180810_142822/lidar/cam_front_center/20180810142822_lidar_frontcenter_000006737.npz"]
'''file_name_lidar = ["/home/sa13291/Documents/ARTHUR_LAMARD/local_test/camera_lidar_semantic_bboxes/val/20181108_103155/lidar/cam_front_center/20181108103155_lidar_frontcenter_000177450.npz",
                "/home/sa13291/Documents/ARTHUR_LAMARD/local_test/camera_lidar_semantic_bboxes/val/20181108_103155/lidar/cam_front_center/20181108103155_lidar_frontcenter_000177506.npz",
                "/home/sa13291/Documents/ARTHUR_LAMARD/local_test/camera_lidar_semantic_bboxes/val/20181108_103155/lidar/cam_front_center/20181108103155_lidar_frontcenter_000177683.npz",
                "/home/sa13291/Documents/ARTHUR_LAMARD/local_test/camera_lidar_semantic_bboxes/val/20181108_103155/lidar/cam_front_center/20181108103155_lidar_frontcenter_000177711.npz",
                "/home/sa13291/Documents/ARTHUR_LAMARD/local_test/camera_lidar_semantic_bboxes/val/20181108_103155/lidar/cam_front_center/20181108103155_lidar_frontcenter_000177718.npz",
                "/home/sa13291/Documents/ARTHUR_LAMARD/local_test/camera_lidar_semantic_bboxes/val/20181108_103155/lidar/cam_front_center/20181108103155_lidar_frontcenter_000177762.npz",
                "/home/sa13291/Documents/ARTHUR_LAMARD/local_test/camera_lidar_semantic_bboxes/val/20181108_103155/lidar/cam_front_center/20181108103155_lidar_frontcenter_000177928.npz",
                "/home/sa13291/Documents/ARTHUR_LAMARD/local_test/camera_lidar_semantic_bboxes/val/20181108_103155/lidar/cam_front_center/20181108103155_lidar_frontcenter_000177947.npz",
                ]'''


project_path = "/home/sa13291/Documents/ARTHUR_LAMARD/3d_projection/prediction/"

# Remove the csv file if exists
if os.path.exists(str(project_path)):
    shutil.rmtree(str(project_path))
    print("!!!!!!!!!!!!!!!!results_pipeline deleted!!!!!!!!!!!!!!!!!!!!!")

# set the ploting with 3 axes
for i in range(len(file_name_lidar)):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
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
    lidar_fc_in_camfc = np.load(file_name_lidar[i])

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
    file_name_image = extract_image_file_name_from_lidar_file_name(file_name_lidar[i])
    json_file_name = extract_json_file_name_from_lidar_file_name(file_name_lidar[i])
    img = plt.imread(file_name_image)
    # extract data from each json associated to a png image
    with open(json_file_name) as f:
        json_file = json.load(f)

    undist_img = undistort_image(img, 'front_center', config)
    undistorted_params = np.array([[0., 0., 0., 0., 0.]])
    ax2.imshow(img)
    pixel_coords = projectPoints(pc, config['cameras']['front_center']['CamMatrix'], undistorted_params)




    # detect everything using the models
    traffic_signs = detect.run(weights="/home/sa13291/Documents/ARTHUR_LAMARD/traffic_sign/best.pt",
                                         source=file_name_image,
                                         save_conf=True,
                                         save_txt=True,
                                         project='/home/sa13291/Documents/ARTHUR_LAMARD/3d_projection/prediction/prediction_signs',
                                         nosave=False,
                                        conf_thres=0.20
                                         )
    lane_detect = demo.detect(weights='/home/sa13291/Documents/ARTHUR_LAMARD/YOLOPv2/data/weights/yolopv2.pt',
                              source=file_name_image,
                              save_conf=True,
                              save_txt=True,
                              project='/home/sa13291/Documents/ARTHUR_LAMARD/3d_projection/prediction/prediction_lane',
                              exist_ok=True,
                              conf_thres = 0.25
                              )

    traffic_light_detection = detect.run(weights="/home/sa13291/Documents/ARTHUR_LAMARD/traffic_light/best.pt",
                                         source=file_name_image,
                                         save_conf=True,
                                         save_txt=True,
                                         project='/home/sa13291/Documents/ARTHUR_LAMARD/3d_projection/prediction/prediction_lights',
                                         nosave=False,
                                         conf_thres=0.20
                                         )


    ax2.scatter(*pixel_coords.T, s=0.2, color=cm.rainbow(1 - depths / 40))

    # gather 3d position.
    coord_list = get_bboxes_coords(file_name_image)



    # get the coordinates in the saved files
    for (o,k, j) in zip((coord_list[0]),(coord_list[1]), (coord_list[2])):
        bounding_box_2d = np.array(k)  # top left -> bottom right.
        center = np.mean(bounding_box_2d, axis=0)
        idx_nearest_3d_point = np.argmin(np.sum(np.abs(pixel_coords - center), axis=1))
        truck_position = pc[idx_nearest_3d_point].tolist()
        # class every elements by class and color the bounding boxes and lidar point with the same color
        if int(o) == 1:
            print(f"DATA : Traffic light {k} position : {truck_position}| confidence : {j}")
            truck_marker = plt.Circle((truck_position[0], truck_position[1]), 0.5, color='yellow')
            bounding_box = Rectangle(bounding_box_2d[0],
                                     np.diff(bounding_box_2d[:, 0]),
                                     np.diff(bounding_box_2d[:, 1]),
                                     linewidth=1, edgecolor='yellow', facecolor='none')
        if int(o) == 2:
            print(f"DATA : Sign {k} position : {truck_position}| confidence : {j}")

            truck_marker = plt.Circle((truck_position[0], truck_position[1]), 0.5, color='b')
            bounding_box = Rectangle(bounding_box_2d[0],
                                     np.diff(bounding_box_2d[:, 0]),
                                     np.diff(bounding_box_2d[:, 1]),
                                     linewidth=1, edgecolor='b', facecolor='none')
        if int(o) == 3:
            print(f"DATA : Truck {k} position : {truck_position}| confidence : {j}")
            truck_marker = plt.Circle((truck_position[0], truck_position[1]), 0.5, color='r')
            bounding_box = Rectangle(bounding_box_2d[0],
                                     np.diff(bounding_box_2d[:, 0]),
                                     np.diff(bounding_box_2d[:, 1]),
                                     linewidth=1, edgecolor='r', facecolor='none')

        # Get the values drawing the road mask
        border_points_road = [list(pair) for pair in lane_detect[-1]]
        ctr = np.array(border_points_road).reshape((-1, 1, 2)).astype(np.int32)
        road_img = cv2.drawContours(img, [ctr], 0, (255, 255, 255), 1)
        road_point_mask = [cv2.pointPolygonTest(ctr, point, False) == 1.0 for point in pixel_coords]
        points_within_road_px = pixel_coords[road_point_mask]
        points_within_road_pc = pc[road_point_mask]
        center, normal_vec = planeFit(points_within_road_pc)

        def generate_optimization_fns(normal_vec, center, target_points):
            v1 = np.cross(normal_vec, (1, 0, 0))
            v2 = np.cross(normal_vec, v1)

            def project_params(x):
                params = x.reshape(-1, 2)
                points = v1 * params[:, 0:1] + v2 * params[:, 1:2] + center
                points_px = projectPoints(points, config['cameras']['front_center']['CamMatrix'], undistorted_params)
                return points, points_px

            def loss_fn(x):
                points, points_px = project_params(x)
                return np.linalg.norm(points_px - target_points)

            return project_params, loss_fn


        project_fn, loss_fn = generate_optimization_fns(normal_vec, center, border_points_road)
        initial_params = np.zeros(len(border_points_road) * 2)
        res = scipy.optimize.minimize(loss_fn, initial_params)
        points3d, points_px = project_fn(res.x)
        ax3.scatter(*points_px.T, c='r', s=1)
        plt.scatter(*points_within_road_px.T, s=0.2)
        ax3.imshow(road_img)

        ax1.add_patch(truck_marker)

        ax2.add_patch(bounding_box)
        # write the csv file with every trucks prositons
        class_object = o
        file_writer(class_object,k,truck_position, j, len(lane_detect[-1]),json_file['cam_tstamp'])


    label_path_file = '/home/sa13291/Documents/ARTHUR_LAMARD/3d_projection/prediction/'

    list_of_file = glob.glob(str(label_path_file + '*/'))
    latest_file = max(list_of_file, key=os.path.getctime)
    # save the last png results
    plt.savefig(os.path.join((label_path_file), f'results{i}.png'), dpi = 300)
    fig.show()
    plt.show()
# detect the anomalies
anomaly_detection()


