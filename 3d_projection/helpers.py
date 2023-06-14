import glob
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
    print(file_name_image)
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


## taken from https://stackoverflow.com/questions/12299540/plane-fitting-to-4-or-more-xyz-points
def planeFit(points):
    """
    p, n = planeFit(points)

    Given an array, points, of shape (d,...)
    representing points in d-dimensional space,
    fit an d-dimensional plane to the points.
    Return a point, p, on the plane (the point-cloud centroid),
    and the normal, n.
    """
    import numpy as np
    points = np.reshape(points, (np.shape(points)[0], -1)).T  # Collapse trialing dimensions
    assert points.shape[0] <= points.shape[1], "There are only {} points in {} dimensions.".format(points.shape[1],
                                                                                                   points.shape[0])
    ctr = points.mean(axis=1)
    x = points - ctr[:, np.newaxis]
    M = np.dot(x, x.T)  # Could also use np.cov(x) here.
    return ctr, la.svd(M)[0][:, -1]


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
    prediction_file = '/home/sa13291/Documents/ARTHUR_LAMARD/3d_projection/prediction/results_label.txt'
    # prediction_file = '/Users/arthurlamard/Documents/Allemagne/cours/AI-PROJECT-SMART_RECORDING_PIPELINE/3d_projection/prediction/results_label.txt'
    with open(prediction_file, 'w') as outfile:
        for fname in list_file:
            # print("Error code:", e.code)
            # os.remove(f'{fname}.txt')
            with open(fname) as infile:
                for line in infile:
                    outfile.write(line)
    return prediction_file


def get_bboxes_coords(image):
    # label_path_file = '/Users/arthurlamard/Documents/Allemagne/cours/AI-PROJECT-SMART_RECORDING_PIPELINE/3d_projection/prediction/'
    label_path_file = '/home/sa13291/Documents/ARTHUR_LAMARD/3d_projection/prediction'
    if not os.path.exists(label_path_file):
        os.mkdir(label_path_file)

    # list_of_file = glob.glob(str(label_path_file + '*/*/labels/'))

    list_of_file = glob.glob(str(label_path_file + '*/*/*/labels/'))
    # print("list of file == ", list_of_file)
    # print("len(list of file) : ", len(list_of_file))

    '''
        Auto incrementation deactivate for the exp file -> check yolo general.py if you want to reactivate
    '''
    # print("list file 0 : ",os.listdir(os.path.join((list_of_file[0])))[0])
    # print("list_of_file[0]",str(list_of_file[0]))
    # print("list_of_file[1]",str(list_of_file[1]))
    # print("list_of_file[2]", str(list_of_file[2]))

    # print("COUCOU : ",glob.glob(str(list_of_file[0] + '*.txt')))
    #######################################
    # cars = sings and line = cars+line
    #######################################

    '''latest_file_signs = os.listdir(os.path.join((list_of_file[0])))[0]
    latest_file_cars = os.listdir(os.path.join((list_of_file[1])))[0]
    print("latest_file_signs",latest_file_signs)
    print("latest_file_cars",latest_file_cars)'''
    final_file_list = []

    try:
        latest_file_signs = glob.glob(str(list_of_file[0] + '*.txt'))
        # print("latest file signs : ", latest_file_signs)
        if os.path.exists(str(latest_file_signs[0])):
            file_signs_path = str(latest_file_signs[0])
            final_file_list.append(str(file_signs_path))
    except:
        print("INFORMATION : No traffic sign detected")
        pass
    # latest_file_cars = glob.glob(str(list_of_file[0] + '/*.txt'))
    try:
        latest_file_lane = glob.glob(str(list_of_file[2] + '*.txt'))
        # print("latest file lane : ", latest_file_lane)
        if os.path.exists(str(latest_file_lane[0])):
            file_line_path = str(latest_file_lane[0])
            final_file_list.append(file_line_path)
    except:
        print("INFORMATION : No build detected")
        pass
    try:
        latest_file_lights = glob.glob(str(list_of_file[1] + '*.txt'))
        # print("latest file lane : ", latest_file_lights)
        if os.path.exists(str(latest_file_lights[0])):
            file_line_path = str(latest_file_lights[0])
            final_file_list.append(file_line_path)
    except:
        print("INFORMATION : No traffic lights detected")
        pass
    # print("latest_file_signs", latest_file_signs)
    # print("latest_file_cars", latest_file_cars)
    # print("latest_file_line", latest_file_lane)

    # print("final file list :", final_file_list)
    '''latest_file = max(list_of_file, key=os.path.getctime)
    #print("latest file : ", latest_file)
    png_name = os.path.basename(os.path.normpath(image))
    print(png_name)
    txt_path = str(png_name)
    final_extension = txt_path.replace('.png', '.txt')
    print(final_extension)'''

    # file_signs_path = str(latest_file_signs[0])
    # file_cars_path =str(latest_file_cars[0])
    # file_line_path = str(latest_file_lane[0])
    # file_signs_path = str(latest_file_signs[0])
    # print("file_signs_path",latest_file_signs[0])
    # print("file_cars_path",latest_file_lane[0])

    # TODO : Si le fichier n'existe pas, on ne le join pas, ca veut dire que les panneaux ou trafficlights ne sont pas detectés sur l'image
    # latest_file = join_txt([file_signs_path, file_cars_path])
    '''for i in range(len(final_file_list)):
        print("********** i *********** : ",i)
        latest_file = join_txt(final_file_list[i])'''

    # latest_file = join_txt([file_signs_path, file_line_path])
    latest_file = join_txt(final_file_list)
    png_name = os.path.basename(os.path.normpath(image))
    # print(png_name)
    txt_path = str(png_name)
    final_extension = txt_path.replace('.png', '.txt')
    # print(final_extension)

    final_path = latest_file
    # print("&&&&&&&&&&& latest file : ", latest_file)
    # print(final_path)
    yolo_bbox = final_path

    # pbx.convert_bbox(yolo_bbox1, from_type="yolo", to_type="voc", image_size=(W, H))
    img = cv2.imread(image)
    dh, dw, _ = img.shape

    fl = open(yolo_bbox, 'r')
    # print(yolo_bbox)
    data = fl.readlines()
    # print("DATA  :",len(data))
    fl.close()

    '''data = detect.run(weights="/Users/arthurlamard/Documents/Allemagne/cours/AI-PROJECT-SMART_RECORDING_PIPELINE/local_test/yolov5/yolov5s.pt",
                        source=image,
                      save_conf=True,
                      save_txt=True)'''
    # data = demo.detect(weights="/Users/arthurlamard/Documents/Allemagne/cours/AI-PROJECT-SMART_RECORDING_PIPELINE/PROJET/YOLOPv2/data/weights/yolopv2.pt",source=file_name_image,)

    final_coord_list = []
    conf_list = []
    for dt in data:
        # print("data : ", data)
        # print("dt : ",dt)
        # Split string to float
        _, x, y, w, h, conf = map(float, dt.split(' '))

        # Taken from https://github.com/pjreddie/darknet/blob/810d7f797bdb2f021dbe65d2524c2ff6b8ab5c8b/src/image.c#L283-L291
        # via https://stackoverflow.com/questions/44544471/how-to-get-the-coordinates-of-the-bounding-box-in-yolo-object-detection#comment102178409_44592380
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
        # print("final coord : ",type(final_coord))
    # print("final_coord_list : ", final_coord_list)
    # print("confidence : ", conf_list)
    return final_coord_list, conf_list


def file_writer(truck, coords, conf):
    # projet_path = "/Users/arthurlamard/Documents/Allemagne/cours/AI-PROJECT-SMART_RECORDING_PIPELINE/3d_projection/"
    projet_path = "/home/sa13291/Documents/ARTHUR_LAMARD/3d_projection/"
    projet_prediction_path = glob.glob(str(projet_path + 'prediction/'))
    file = open(f'{projet_prediction_path[0]}results_pipeline.csv', 'a')
    file.write(str(truck)+";"+str(coords)+";"+str(conf)+'\n')

    file.close()


    '''print("projet prediction path : ", projet_prediction_path)
    x = coords[0]
    y = coords[1]
    z = coords[2]
    list_line = []
    line = [truck, coords, conf]
    list_line.append(line)
    headerList = ["trucks", "coords", "conf"]
    print("list_line = ",list_line)
    with open(f'{projet_prediction_path[0]}results_pipeline.csv', 'a+') as f:
        f.write(str(list_line[0])+';'+
                str(list_line[1])+';'+
                str(list_line[2])+';')
        f.close()'''


def anomaly_detection():
    projet_path = "/home/sa13291/Documents/ARTHUR_LAMARD/3d_projection/"
    projet_prediction_path = glob.glob(str(projet_path + 'prediction/'))
    read_file = pd.read_csv((f'{projet_prediction_path[0]}results_pipeline.txt'))
    read_file.to_csv((f'{projet_prediction_path[0]}results_pipeline.csv'))
    #file = str(projet_prediction_path[0] + 'results_pipeline.csv')
    #print(pd.read_csv(file))
    '''list_line = []
    with open(file, 'r') as f:
        print("FILE READING ")
        for line in f.readlines():
            list_line.append(line)
            print(list_line)
        f.close()'''


if __name__ == "__main__":
    file_name_lidar = "/Users/arthurlamard/Documents/Allemagne/cours/AI-PROJECT-SMART_RECORDING_PIPELINE/PROJET/camera_lidar_semantic_bboxes/test/20181204_170238/lidar/cam_front_center/20181204170238_lidar_frontcenter_000036276.npz"
    file_name_image = extract_image_file_name_from_lidar_file_name(file_name_lidar)
    # print(file_name_image)

    get_bboxes_coords(file_name_image)
