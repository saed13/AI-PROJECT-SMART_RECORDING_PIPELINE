import numpy as np
import numpy.linalg as la
import os.path
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
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
    lidar['points'] = points_trans[:,0:3]

    return lidar

def extract_image_file_name_from_lidar_file_name(path_lidar):
    replace_lidar = lambda x: x if x!='lidar' else 'camera'
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
            return cv2.fisheye.undistortImage(image, intr_mat_dist,\
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
    points = np.reshape(points, (np.shape(points)[0], -1)).T # Collapse trialing dimensions
    assert points.shape[0] <= points.shape[1], "There are only {} points in {} dimensions.".format(points.shape[1], points.shape[0])
    ctr = points.mean(axis=1)
    x = points - ctr[:,np.newaxis]
    M = np.dot(x, x.T) # Could also use np.cov(x) here.
    return ctr, la.svd(M)[0][:,-1]

def projectPoints(points, camMtx, dist):
    pose_to_cam_coord_transform = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
    pixel_coords, _ = cv2.projectPoints(pose_to_cam_coord_transform.dot(points.T).T,
                      np.asarray([[0,0,0]], dtype=np.float64),
                      np.asarray([[0,0,0]], dtype=np.float64),
                      np.asarray(camMtx),
                      np.asarray(dist))
    pixel_coords = pixel_coords.squeeze()
    return pixel_coords

def get_bboxes_coords(image):
    label_path_file = '/Users/arthurlamard/Documents/Allemagne/cours/AI-PROJECT-SMART_RECORDING_PIPELINE/PROJET/camera_lidar_semantic_bboxes/labels_2/'
    png_name = os.path.basename(os.path.normpath(image))
    #print(png_name)
    txt_path = str(png_name)
    final_extension = txt_path.replace('.png', '.txt')
    #print(final_extension)
    final_path = os.path.join(label_path_file, final_extension)
    #print(final_path)

    yolo_bbox = final_path

    #pbx.convert_bbox(yolo_bbox1, from_type="yolo", to_type="voc", image_size=(W, H))
    img = cv2.imread( image)
    dh, dw, _ = img.shape

    fl = open( yolo_bbox, 'r')
    #print(yolo_bbox)
    data = fl.readlines()
    #print("DATA  :",len(data))
    fl.close()

    final_coord_list = []
    conf_list = []
    for dt in data:
        #print("data : ", data)
        #print("dt : ",dt)
        # Split string to float
        _, x, y, w, h,conf = map(float, dt.split(' '))

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
        final_coord = ([l,t],[r,b])
        final_coord_list.append(final_coord)
        conf_list.append(conf)
        #print("final coord : ",type(final_coord))
    #print("final_coord_list : ", final_coord_list)
    #print("confidence : ", conf_list)
    return final_coord_list, conf_list


    plt.imshow(img)
    plt.show()

if __name__ == "__main__":
    file_name_lidar = "/Users/arthurlamard/Documents/Allemagne/cours/AI-PROJECT-SMART_RECORDING_PIPELINE/PROJET/camera_lidar_semantic_bboxes/test/20181204_170238/lidar/cam_front_center/20181204170238_lidar_frontcenter_000036276.npz"
    file_name_image = extract_image_file_name_from_lidar_file_name(file_name_lidar)
    #print(file_name_image)

    get_bboxes_coords(file_name_image)
