import csv
import glob
import statistics

import pandas as pd
import numpy as np
import numpy.linalg as la
import os.path
import cv2
from pathlib import Path
import matplotlib.pyplot as plt

# from local_test.yolov5 import detect
# from PROJET.YOLOPv2 import demo
EPSILON = 1e-10


def get_axes_of_a_view(view):
    x_axis = view['x-axis']
    y_axis = view['y-axis']

    x_axis_norm = la.norm(x_axis)
    y_axis_norm = la.norm(y_axis)

    if (x_axis_norm < EPSILON or y_axis_norm < EPSILON):
        raise ValueError("Norm of input vector(s) too small.")

    # normalize the axes
    x_axis = x_axis / x_axis_norm
    y_axis = y_axis / y_axis_norm

    # make a new y-axis which lies in the original x-y plane, but is orthogonal to x-axis
    y_axis = y_axis - x_axis * np.dot(y_axis, x_axis)

    # create orthogonal z-axis
    z_axis = np.cross(x_axis, y_axis)

    # calculate and check y-axis and z-axis norms
    y_axis_norm = la.norm(y_axis)
    z_axis_norm = la.norm(z_axis)

    if (y_axis_norm < EPSILON) or (z_axis_norm < EPSILON):
        raise ValueError("Norm of view axis vector(s) too small.")

    # make x/y/z-axes orthonormal
    y_axis = y_axis / y_axis_norm
    z_axis = z_axis / z_axis_norm

    return x_axis, y_axis, z_axis


def get_origin_of_a_view(view):
    return view['origin']


def get_transform_to_global(view):
    # get axes
    x_axis, y_axis, z_axis = get_axes_of_a_view(view)

    # get origin
    origin = get_origin_of_a_view(view)
    transform_to_global = np.eye(4)

    # rotation
    transform_to_global[0:3, 0] = x_axis
    transform_to_global[0:3, 1] = y_axis
    transform_to_global[0:3, 2] = z_axis

    # origin
    transform_to_global[0:3, 3] = origin

    return transform_to_global


def get_transform_from_global(view):
    # get transform to global
    transform_to_global = get_transform_to_global(view)
    trans = np.eye(4)
    rot = np.transpose(transform_to_global[0:3, 0:3])
    trans[0:3, 0:3] = rot
    trans[0:3, 3] = np.dot(rot, -transform_to_global[0:3, 3])

    return trans


def transform_from_to(src, target):
    transform = np.dot(get_transform_from_global(target), \
                       get_transform_to_global(src))

    return transform


def project_lidar_from_to(lidar, src_view, target_view):
    lidar = dict(lidar)
    trans = transform_from_to(src_view, target_view)
    points = lidar['points']
    points_hom = np.ones((points.shape[0], 4))
    points_hom[:, 0:3] = points
    points_trans = (np.dot(trans, points_hom.T)).T
    lidar['points'] = points_trans[:, 0:3]

    return lidar


def extract_image_file_name_from_lidar_file_name(path_lidar):
    replace_lidar = lambda x: x if x != 'lidar' else 'camera'
    path_lidar = path_lidar.split('/')
    path_lidar = list(map(replace_lidar, path_lidar))
    file_name_image = path_lidar[-1].split('.')[0]
    file_name_image = file_name_image.split('_')
    file_name_image = file_name_image[0] + '_' + \
                      'camera_' + \
                      file_name_image[2] + '_' + \
                      file_name_image[3] + '.png'
    return os.path.join("/".join(path_lidar[:-1] + [file_name_image]))


def extract_semantic_file_name_from_image_file_name(file_name_image):
    file_name_semantic_label = file_name_image.split('/')
    file_name_semantic_label = file_name_semantic_label[-1].split('.')[0]
    file_name_semantic_label = file_name_semantic_label.split('_')
    file_name_semantic_label = file_name_semantic_label[0] + '_' + \
                               'label_' + \
                               file_name_semantic_label[2] + '_' + \
                               file_name_semantic_label[3] + '.png'

    return file_name_semantic_label

def extract_json_file_name_from_lidar_file_name(path_lidar):
    replace_lidar = lambda x: x if x != 'lidar' else 'camera'
    path_lidar = path_lidar.split('/')
    path_lidar = list(map(replace_lidar, path_lidar))
    file_name_json = path_lidar[-1].split('.')[0]
    file_name_json = file_name_json.split('_')
    file_name_json = file_name_json[0] + '_' + \
                      'camera_' + \
                      file_name_json[2] + '_' + \
                      file_name_json[3] + '.json'
    return os.path.join("/".join(path_lidar[:-1] + [file_name_json]))
def undistort_image(image, cam_name, config):
    if cam_name in ['front_left', 'front_center', \
                    'front_right', 'side_left', \
                    'side_right', 'rear_center']:
        # get parameters from config file
        intr_mat_undist = \
            np.asarray(config['cameras'][cam_name]['CamMatrix'])
        intr_mat_dist = \
            np.asarray(config['cameras'][cam_name]['CamMatrixOriginal'])
        dist_parms = \
            np.asarray(config['cameras'][cam_name]['Distortion'])
        lens = config['cameras'][cam_name]['Lens']

        if (lens == 'Fisheye'):
            return cv2.fisheye.undistortImage(image, intr_mat_dist, \
                                              D=dist_parms, Knew=intr_mat_undist)
        elif (lens == 'Telecam'):
            return cv2.undistort(image, intr_mat_dist, \
                                 distCoeffs=dist_parms, newCameraMatrix=intr_mat_undist)
        else:
            return image
    else:
        return image


def map_lidar_points_onto_image(image_orig, lidar, pixel_size=3, pixel_opacity=1):
    image = np.copy(image_orig)

    # get rows and cols
    rows = (lidar['row'] + 0.5).astype(np.int)
    cols = (lidar['col'] + 0.5).astype(np.int)

    # lowest distance values to be accounted for in colour code
    MIN_DISTANCE = np.min(lidar['distance'])
    # largest distance values to be accounted for in colour code
    MAX_DISTANCE = np.max(lidar['distance'])

    # get distances
    distances = lidar['distance']
    # determine point colours from distance
    colours = (distances - MIN_DISTANCE) / (MAX_DISTANCE - MIN_DISTANCE)
    colours = np.asarray([np.asarray(hsv_to_rgb(0.75 * c, \
                                                np.sqrt(pixel_opacity), 1.0)) for c in colours])
    pixel_rowoffs = np.indices([pixel_size, pixel_size])[0] - pixel_size // 2
    pixel_coloffs = np.indices([pixel_size, pixel_size])[1] - pixel_size // 2
    canvas_rows = image.shape[0]
    canvas_cols = image.shape[1]
    for i in range(len(rows)):
        pixel_rows = np.clip(rows[i] + pixel_rowoffs, 0, canvas_rows - 1)
        pixel_cols = np.clip(cols[i] + pixel_coloffs, 0, canvas_cols - 1)
        image[pixel_rows, pixel_cols, :] = \
            (1. - pixel_opacity) * \
            np.multiply(image[pixel_rows, pixel_cols, :], \
                        colours[i]) + pixel_opacity * 255 * colours[i]
    return image.astype(np.uint8)

def planeFit(points):
    import numpy as np
    try:
        points = np.reshape(points, (np.shape(points)[0], -1)).T
    except ValueError:
        raise ValueError("Error: Unable to reshape array.")
        # Handle the error condition appropriately or re-raise the exception

    assert points.shape[0] <= points.shape[1], "There are only {} points in {} dimensions.".format(points.shape[1],
                                                                                                   points.shape[0])
    ctr = points.mean(axis=1)
    x = points - ctr[:, np.newaxis]
    M = np.dot(x, x.T)
    return ctr, np.linalg.svd(M)[0][:, -1]


def projectPoints(points, camMtx, dist):
    pose_to_cam_coord_transform = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
    pixel_coords, _ = cv2.projectPoints(pose_to_cam_coord_transform.dot(points.T).T,
                                        np.asarray([[0, 0, 0]], dtype=np.float64),
                                        np.asarray([[0, 0, 0]], dtype=np.float64),
                                        np.asarray(camMtx),
                                        np.asarray(dist))
    pixel_coords = pixel_coords.squeeze()
    return pixel_coords


def join_txt(list_file: list):
    # create a uniq txt file from a list a txt files
    prediction_file = '/home/sa13291/Documents/ARTHUR_LAMARD/3d_projection/prediction/results_label.txt'
    with open(prediction_file, 'w') as outfile:
        for fname in list_file:
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)
    return prediction_file


def get_bboxes_coords(image):
    # get the normal coordinates of a bbox using the yolo format

    # create the prediction folder if it doesn't exist
    label_path_file = '/home/sa13291/Documents/ARTHUR_LAMARD/3d_projection/prediction'
    if not os.path.exists(label_path_file):
        os.mkdir(label_path_file)

    list_of_file = glob.glob(str(label_path_file + '*/*/*/labels/'))

    '''
        Auto incrementation deactivate for the exp file -> check yolo general.py if you want to reactivate
    '''

    final_file_list = []

    # try to see if any signs/traffic lights or build detected in the images
    # read the txt file created by the yolo model if they exits
    try:
        latest_file_signs = glob.glob(str(list_of_file[0] + '*.txt'))
        latest_file_s = max(latest_file_signs, key=os.path.getctime)
        latest_file_s = change_class(latest_file_s, '2')
        if os.path.exists(str(latest_file_s)):
            file_signs_path = str(latest_file_s)
            final_file_list.append(str(file_signs_path))
    except:
        print("INFORMATION : No traffic sign detected")
        pass

    try:
        latest_file_lane = glob.glob(str(list_of_file[2] + '*.txt'))
        latest_file_la = max(latest_file_lane, key=os.path.getctime)
        if os.path.exists(str(latest_file_la)):
            file_line_path = str(latest_file_la)
            final_file_list.append(file_line_path)
    except:
        print("INFORMATION : No build detected")
        pass
    try:
        latest_file_lights = glob.glob(str(list_of_file[1] + '*.txt'))
        latest_file_l = max(latest_file_lights, key=os.path.getctime)
        latest_file_l = change_class(latest_file_l, '1')
        if os.path.exists(str(latest_file_l)):
            file_line_path = str(latest_file_l)
            final_file_list.append(file_line_path)
    except:
        print("INFORMATION : No traffic lights detected")
        pass

    # create a single txt file from all the detection txt files
    latest_file = join_txt(final_file_list)
    png_name = os.path.basename(os.path.normpath(image))
    txt_path = str(png_name)
    final_extension = txt_path.replace('.png', '.txt')
    final_path = latest_file
    yolo_bbox = final_path

    img = cv2.imread(image)
    dh, dw, _ = img.shape

    fl = open(yolo_bbox, 'r')
    data = fl.readlines()
    fl.close()

    final_coord_list = []
    conf_list = []
    class_list = []

    # convert the yolo coordinates to a normal format
    for dt in data:
        # Split string to float
        class_val, x, y, w, h, conf = map(float, dt.split(' '))
        l = int((x - w / 2) * dw)
        r = int((x + w / 2) * dw)
        t = int((y - h / 2) * dh)
        b = int((y + h / 2) * dh)

        if l < 0:
            l = 0
        if r > dw - 1:
            r = dw - 1
        if t < 0:
            t = 0
        if b > dh - 1:
            b = dh - 1

        cv2.rectangle(img, (l, t), (r, b), (0, 0, 255), 10)
        final_coord = ([l, t], [r, b])
        final_coord_list.append(final_coord)
        conf_list.append(conf)
        class_list.append(class_val)
    return class_list, final_coord_list, conf_list


def file_writer(class_build,truck, coords, conf, corners,cam_stamp):
    # write all the usefull data in a csv file
    projet_path = "/home/sa13291/Documents/ARTHUR_LAMARD/3d_projection/"
    projet_prediction_path = glob.glob(str(projet_path + 'prediction/'))
    file = open(f'{projet_prediction_path[0]}results_pipeline.csv', 'a')
    file.write(str(class_build)+","+str(truck)+","+str(coords)+","+str(conf)+","+str(corners)+","+str(cam_stamp)+'\n')
    file.close()

def change_class(file, replacement_value):
    # function made to change the class of a detetection by the remplacement_value choosen by the user
    # used to have the same class number for each detection
    newdata = []
    fl = open(file, 'r')
    data = fl.readlines()
    fl.close()

    for i in range(len(data)):
        data[i].split(' ')[0] = replacement_value
        Data  = str(replacement_value + ' ' + data[i].split(' ')[1] + ' '+ data[i].split(' ')[2] + ' ' + data[i].split(' ')[3] + ' ' + data[i].split(' ')[4] + ' ' + data[i].split(' ')[5])
        newdata.append(Data)
    f = open(file, 'w')
    for i in range(len(newdata)):
        f.write(newdata[i])
    f.close()
    return file


def anomaly_detection():
    list_class = []
    list_conf = []
    list_coords = []
    list_time = []
    list_build = []
    list_sign = []
    list_light = []
    list_corner  = []
    average = []
    projet_path = "/home/sa13291/Documents/ARTHUR_LAMARD/3d_projection/"
    projet_prediction_path = glob.glob(str(projet_path + 'prediction/'))
    read_file = ((f'{projet_prediction_path[0]}results_pipeline.csv'))
    with open(read_file) as f:
        reader = csv.reader(f)
        for row in reader:
            # appends in list each element to sepearate them from the original list
            list_conf.append(row[len(row)-3])
            list_class.append(row[0])
            list_coords.append([row[5],row[6],row[7]])
            list_time.append(row[-1])
            list_corner.append(row[len(row)-2])

            # get a list of all the specific element
            if row[0] == '1.0':
                list_light.append(row)
            if row[0] == '2.0':
                list_sign.append(row)
            if row[0] == '3.0':
                list_build.append(row)
                paires = list(zip(list_build, list_build[1:] + list_build[:1]))


    #print("paires : ", paires)
    # start recording if the conf is too low
    for i in range(len(list_conf)):
        if list_conf[i]<=str(0.25):
            print("START RECORDING_conf")

    # start recording id the number of point in the road mask is +/-3 points from the average point of the pipeline
    # it indicates a special case
    for i in (list_corner):
        digit = int(i)
        average.append(digit)
        mean = statistics.mean(average)
        if (int(i)) >= mean+3 or (int(i))<=mean-3:
            print("START RECORDING_corners")
    print("list digits : ", average)
    print("average point : ", statistics.mean(average))




if __name__ == "__main__":
    file_name_lidar = "/Users/arthurlamard/Documents/Allemagne/cours/AI-PROJECT-SMART_RECORDING_PIPELINE/PROJET/camera_lidar_semantic_bboxes/test/20181204_170238/lidar/cam_front_center/20181204170238_lidar_frontcenter_000036276.npz"
    file_name_image = extract_image_file_name_from_lidar_file_name(file_name_lidar)
    # print(file_name_image)

    get_bboxes_coords(file_name_image)
