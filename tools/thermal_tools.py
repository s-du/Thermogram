# Standard library imports
import os
import subprocess
from pathlib import Path
from shutil import copyfile, copytree
import time
from concurrent.futures import ThreadPoolExecutor

# Third-party imports
import cv2
import numpy as np
from PIL import Image, ImageOps, ImageFilter
from matplotlib import cm, pyplot as plt
import matplotlib.colors as mcol
from blend_modes import dodge, multiply
from scipy.optimize import differential_evolution

# PySide6 imports
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import QPointF, QRectF
from PySide6.QtWidgets import QGraphicsEllipseItem, QGraphicsTextItem

# custom libraries
import resources as res

# PATHS __________________________________________________
sdk_tool_path = Path(res.find('dji/dji_irp.exe'))
m2t_ir_xml_path = res.find('other/cam_calib_m2t_opencv.xml')
m2t_rgb_xml_path = res.find('other/rgb_cam_calib_m2t_opencv.xml')

m3t_ir_xml_path = res.find('other/cam_calib_m3t_opencv.xml')
m3t_rgb_xml_path = res.find('other/rgb_cam_calib_m3t_opencv.xml')

LIST_CUSTOM_CMAPS = ['Arctic',
                     'Iron',
                     'Rainbow',
                     'Fulgurite',
                     'Iron Red',
                     'Hot Iron',
                     'Medical',
                     'Arctic2',
                     'Rainbow1',
                     'Rainbow2',
                     'Tint',
                     'BlueWhiteRed',
                     'FIJI_Temp']


# USEFUL CLASSES __________________________________________________
class CameraUndistorter:
    def __init__(self, xml_path):
        self.K, self.d = self.read_matrices(xml_path)
        self.newcam = None
        self.roi = None

    def read_matrices(self, xml_path):
        cv_file = cv2.FileStorage(xml_path, cv2.FILE_STORAGE_READ)
        K = cv_file.getNode("Camera_Matrix").mat()
        d = cv_file.getNode("Distortion_Coefficients").mat()
        cv_file.release()

        return K, d

    def undis(self, cv_img):
        h, w = cv_img.shape[:2]

        # Compute new camera matrix only once if it hasn't been computed yet
        if self.newcam is None:
            self.newcam, self.roi = cv2.getOptimalNewCameraMatrix(self.K, self.d, (w, h), 1, (w, h))

        # Undistort image
        dest = cv2.undistort(cv_img, self.K, self.d, None, self.newcam)

        # Crop the image based on the region of interest (ROI)
        x, y, w, h = self.roi
        dest = dest[y:y + h, x:x + w]

        # Get dimensions
        height, width = dest.shape[:2]
        dim = (width, height)

        return dest, dim


class DroneModel():
    def __init__(self, name):
        if name == 'MAVIC2-ENTERPRISE-ADVANCED':
            self.rgb_xml_path = m2t_rgb_xml_path
            self.ir_xml_path = m2t_ir_xml_path
            sample_rgb_path = res.find('img/M2EA_RGB.JPG')
            sample_ir_path = res.find('img/M2EA_IR.JPG')
            cv_rgb = cv_read_all_path(sample_rgb_path)
            cv_ir = cv_read_all_path(sample_ir_path)
            _, self.dim_undis_ir = undis(cv_ir, self.ir_xml_path)
            _, self.dim_undis_rgb = undis(cv_rgb, self.rgb_xml_path)

            self.aspect_factor = (self.dim_undis_rgb[0] / self.dim_undis_rgb[1]) / (
                    self.dim_undis_ir[0] / self.dim_undis_ir[1])

            # read focal parameters
            cv_file = cv2.FileStorage(self.ir_xml_path, cv2.FILE_STORAGE_READ)
            self.K_ir = cv_file.getNode("Camera_Matrix").mat()
            self.d_ir = cv_file.getNode("Distortion_Coefficients").mat()
            self.extend = 0.332
            self.x_offset = 50
            self.y_offset = 35

        elif name == 'M3T':
            self.rgb_xml_path = m3t_rgb_xml_path
            self.ir_xml_path = m3t_ir_xml_path
            sample_rgb_path = res.find('img/M3T_RGB.JPG')
            sample_ir_path = res.find('img/M3T_IR.JPG')
            cv_rgb = cv_read_all_path(sample_rgb_path)
            cv_ir = cv_read_all_path(sample_ir_path)
            _, self.dim_undis_ir = undis(cv_ir, self.ir_xml_path)
            _, self.dim_undis_rgb = undis(cv_rgb, self.rgb_xml_path)
            self.aspect_factor = (self.dim_undis_rgb[0] / self.dim_undis_rgb[1]) / (
                    self.dim_undis_ir[0] / self.dim_undis_ir[1])
            # read focal parameters
            cv_file = cv2.FileStorage(self.ir_xml_path, cv2.FILE_STORAGE_READ)
            self.K_ir = cv_file.getNode("Camera_Matrix").mat()
            self.d_ir = cv_file.getNode("Distortion_Coefficients").mat()
            self.extend = 0.3504
            self.x_offset = 49
            self.y_offset = 53


class ProcessedIm:
    def __init__(self, path, rgb_path, original_path, undistorder_ir, delayed_compute=False):
        # general infos
        self.path = path
        self.rgb_path = rgb_path
        self.rgb_path_original = original_path
        self.preview_path = ''
        self.undistorder_ir = undistorder_ir
        # colormap infos
        self.colormap = 'Greys_r'
        self.n_colors = 256

        self._exif = None

        self.has_data = False

        # ir infos
        self.thermal_param = {'emissivity': 0.95, 'distance': 5, 'humidity': 50, 'reflection': 25}
        if not delayed_compute:
            self.raw_data, self.raw_data_undis = extract_raw_data(self.thermal_param, self.path, self.undistorder_ir)
        self.tmin = np.amin(self.raw_data)
        self.tmax = np.amax(self.raw_data)
        self.tmin_shown = self.tmin
        self.tmax_shown = self.tmax

        self.user_lim_col_low = 'w'
        self.user_lim_col_high = 'w'
        self.post_process = 'none'

        # for annotations
        self.annot_rect_items = []
        self.corresp_cat = []

        # for measurements
        self.nb_meas_rect = 0  # number of rect measurements (classes)
        self.meas_rect_list = []
        self.nb_meas_point = 0  # number of spot measurements (classes)
        self.meas_point_list = []
        self.nb_meas_line = 0  # number of line measurements (classes)
        self.meas_line_list = []

    @property
    def exif(self):
        # Lazy load the EXIF data
        if self._exif is None:
            self._exif = extract_exif(self.path)
        return self._exif

    def update_data(self, new_params):
        self.thermal_param = new_params
        self.raw_data, self.raw_data_undis = extract_raw_data(self.thermal_param, self.path, self.undistorder_ir)
        self.tmin = np.amin(self.raw_data)
        self.tmax = np.amax(self.raw_data)
        self.tmin_shown = self.tmin
        self.tmax_shown = self.tmax

        self.has_data = True

    def update_colormap_data(self, colormap, n_colors, user_lim_col_high, user_lim_col_low, post_process, tmin, tmax):
        self.colormap = colormap
        self.n_colors = n_colors
        self.user_lim_col_high = user_lim_col_high
        self.user_lim_col_low = user_lim_col_low
        self.tmin = tmin
        self.tmax = tmax
        self.post_process = post_process

    def get_colormap_data(self):
        return (self.colormap,
                self.n_colors,
                self.user_lim_col_high,
                self.user_lim_col_low,
                self.tmin,
                self.tmax,
                self.tmin_shown,
                self.tmax_shown,
                self.post_process)


class PointMeas:
    def __init__(self, qpoint):
        self.name = ''
        self.qpoint = qpoint
        self.ellipse_item = None
        self.text_item = None
        self.temp = 0
        self.all_items = []

    def create_items(self):
        self.ellipse_item = QGraphicsEllipseItem()

        p1 = QPointF(self.qpoint.x() - 2, self.qpoint.y() - 2)
        p2 = QPointF(self.qpoint.x() + 2, self.qpoint.y() + 2)

        r = QRectF(p1, p2)
        self.ellipse_item.setRect(r)

        temp = round(self.temp, 2)
        self.text_item = QGraphicsTextItem()
        self.text_item.setPos(self.qpoint)
        self.text_item.setHtml(
            "<div style='background-color:rgba(255, 255, 255, 0.3);'>" + str(temp) + "</div>")


class LineMeas:
    def __init__(self, item):
        self.name = ''
        self.main_item = item
        self.spot_items = []
        self.text_items = []
        self.coords = []
        self.data_roi = []

        self.tmin = 0
        self.tmax = 0
        self.tmean = 0

    def compute_data(self, raw_data):
        start_point = self.main_item.line().p1()
        end_point = self.main_item.line().p2()
        P1 = np.array([start_point.x(), start_point.y()])
        P2 = np.array([end_point.x(), end_point.y()])

        imageH, imageW = raw_data.shape
        P1X, P1Y = P1
        P2X, P2Y = P2

        # Calculate differences
        dX, dY = P2X - P1X, P2Y - P1Y
        dXa, dYa = np.abs(dX), np.abs(dY)

        # Determine the number of steps
        num_steps = int(np.maximum(dXa, dYa))

        # Initialize the buffer
        itbuffer = np.empty((num_steps, 3), dtype=np.float32)
        itbuffer.fill(np.nan)

        # Calculate the points along the line
        if P1X == P2X:  # Vertical line
            itbuffer[:, 0] = P1X
            itbuffer[:, 1] = np.linspace(P1Y, P2Y, num=num_steps, endpoint=False)
        elif P1Y == P2Y:  # Horizontal line
            itbuffer[:, 1] = P1Y
            itbuffer[:, 0] = np.linspace(P1X, P2X, num=num_steps, endpoint=False)
        else:  # Diagonal line
            itbuffer[:, 0] = np.linspace(P1X, P2X, num=num_steps, endpoint=False)
            itbuffer[:, 1] = np.linspace(P1Y, P2Y, num=num_steps, endpoint=False)

        # Remove points outside the image bounds
        valid = (itbuffer[:, 0] >= 0) & (itbuffer[:, 1] >= 0) & (itbuffer[:, 0] < imageW) & (itbuffer[:, 1] < imageH)
        itbuffer = itbuffer[valid]

        # Get intensities from the image
        itbuffer[:, 2] = raw_data[itbuffer[:, 1].astype(int), itbuffer[:, 0].astype(int)]
        self.data_roi = itbuffer[:, 2]

    def compute_highlights(self):
        self.tmax = np.amax(self.data_roi)
        self.tmin = np.amin(self.data_roi)

    def create_items(self):
        pass


class RectMeas:
    def __init__(self, item):
        self.name = ''
        self.main_item = item
        self.ellipse_items = []
        self.text_items = []
        self.coords = []
        self.data_roi = []
        self.th_norm = []

        self.tmin = 0
        self.tmax = 0

    def get_coord_from_item(self, QGraphicsRect):
        rect = QGraphicsRect.rect()
        coord = [rect.topLeft(), rect.bottomRight()]

        return coord

    def compute_data(self, coords, raw_data, rgb_path, ir_path):
        p1 = coords[0]
        p2 = coords[1]

        # crop data to last rectangle
        self.data_roi = raw_data[int(p1.y()):int(p2.y()), int(p1.x()):int(p2.x())]

        cv_rgb = cv_read_all_path(rgb_path)
        h_rgb, w_rgb, _ = cv_rgb.shape
        cv_ir = cv_read_all_path(ir_path)
        h_ir, w_ir, _ = cv_ir.shape
        roi_ir = cv_ir[int(p1.y()):int(p2.y()), int(p1.x()):int(p2.x())]

        p1 = (p1.x(), p1.y())
        p2 = (p2.x(), p2.y())
        scale = w_rgb / w_ir
        crop_tl, crop_tr = get_corresponding_crop_rectangle(p1, p2, scale)

        roi_rgb = cv_rgb[int(crop_tl[1]):int(crop_tr[1]), int(crop_tl[0]):int(crop_tr[0])]

        return roi_ir, roi_rgb

    def compute_highlights(self):
        self.tmax = np.amax(self.data_roi)
        self.tmin = np.amin(self.data_roi)
        self.tmean = np.mean(self.data_roi)

        self.area = self.data_roi.shape[0] * self.data_roi.shape[1]

        # normalized data
        self.th_norm = (self.data_roi - self.tmin) / (self.tmax - self.tmin)

        highlights = [
            ['Size [pxl²]', self.area],
            ['Max. Temp. [°C]', str(self.tmax)],
            ['Min. Temp. [°C]', str(self.tmin)],
            ['Average Temp. [°C]', str(self.tmean)]
        ]
        return highlights

    def create_items(self):
        initial_coords = self.get_coord_from_item(self.main_item)
        top_left = initial_coords[0]
        x_t_l = top_left.x()
        y_t_l = top_left.y()
        # find location of minima
        position_min = np.where(self.data_roi == self.tmin)
        position_max = np.where(self.data_roi == self.tmax)
        # Ensure that there are found positions
        if position_min[0].size > 0 and position_min[1].size > 0:
            # Take the first occurrence
            xmin, ymin = position_min[1][0] + x_t_l, position_min[0][0] + y_t_l
        else:
            xmin, ymin = None, None  # or some default value

        if position_max[0].size > 0 and position_max[1].size > 0:
            # Take the first occurrence
            xmax, ymax = position_max[1][0] + x_t_l, position_max[0][0] + y_t_l
        else:
            xmax, ymax = None, None  # or some default value

        print(f'Minimum position: {xmin, ymin}')
        print(f'Maximum position: {xmax, ymax}')

        temp_min = str(round(self.tmin, 2))
        temp_max = str(round(self.tmax, 2))

        self.create_labelled_point(xmin, ymin, temp_min)
        self.create_labelled_point(xmax, ymax, temp_max)

    def create_labelled_point(self, x, y, string):
        print(x, y)
        ellipse_item = QGraphicsEllipseItem()
        qpoint = QPointF(x, y)

        p1 = QPointF(x - 2, y - 2)
        p2 = QPointF(x + 2, y + 2)

        r = QRectF(p1, p2)
        ellipse_item.setRect(r)

        text_item = QGraphicsTextItem()
        text_item.setPos(qpoint)
        text_item.setHtml(
            "<div style='background-color:rgba(255, 255, 255, 0.3);'>" + string + "</div>")

        self.text_items.append(text_item)
        self.ellipse_items.append(ellipse_item)


# long tasks runner classes
# test with runner
class RunnerSignals(QtCore.QObject):
    progressed = QtCore.Signal(int)
    messaged = QtCore.Signal(str)
    finished = QtCore.Signal()


class RunnerDJI(QtCore.QRunnable):
    def __init__(self, start, stop, out_folder, img_objects, ref_im, edges, edges_params, individual_settings=False,
                 export_tif=False, undis=False):
        super().__init__()
        self.img_objects = img_objects
        self.edges = edges
        self.edges_params = edges_params
        self.export_tif = export_tif

        self.start = start
        self.stop = stop
        self.dest_folder = out_folder

        self.signals = RunnerSignals()
        self.undis = undis

        if not individual_settings:  # if global export from current image settings
            self.custom_params = {
                "tmin": ref_im.tmin_shown,
                "tmax": ref_im.tmax_shown,
                "colormap": ref_im.colormap,
                "n_colors": ref_im.n_colors,
                "col_high": ref_im.user_lim_col_high,
                "col_low": ref_im.user_lim_col_low,
                "post_process": ref_im.post_process
            }

    def run(self):
        # create raw outputs for each image
        nb_im = len(self.img_objects)
        for i, img in enumerate(self.img_objects):
            print(i)
            iter = i * (self.stop - self.start) / nb_im

            self.signals.progressed.emit(self.start + iter)
            self.signals.messaged.emit(f'Processing image {i}/{nb_im} with DJI SDK')

            img_path = img.path

            if i < 9:
                prefix = '000'
            elif 9 < i < 99:
                prefix = '00'
            elif 99 < i < 999:
                prefix = '0'
            _, filename = os.path.split(str(img_path))
            dest_path = os.path.join(self.dest_folder, f'thermal_{prefix}{i}.JPG')

            process_raw_data(img,
                             dest_path,
                             edges=self.edges,
                             edge_params=self.edges_params,
                             custom_params=self.custom_params,
                             export_tif=self.export_tif,
                             undis=self.undis)

            if i == len(self.img_objects) - 1:
                legend_dest_path = os.path.join(self.dest_folder, 'plot_onlycbar_tight.png')
                generate_legend(legend_dest_path, self.custom_params)

        self.signals.finished.emit()


class RunnerMiniature(QtCore.QRunnable):
    def __init__(self, list_rgb_paths, drone_model, scale_percent, dest_crop_folder, start, stop):
        super().__init__()
        self.signals = RunnerSignals()
        self.list_rgb_paths = list_rgb_paths
        self.scale_percent = scale_percent
        self.dest_crop_folder = dest_crop_folder
        self.start = start
        self.stop = stop
        self.drone_model = drone_model

    def run(self):
        nb_im = len(self.list_rgb_paths)
        rgb_xml_path = self.drone_model.rgb_xml_path
        undistorter = CameraUndistorter(rgb_xml_path)

        def process_image(i, rgb_path):
            iter_start_time = time.time()  # Track time for each iteration
            iter = i * (self.stop - self.start) / nb_im

            # update progress
            self.signals.progressed.emit(self.start + iter)
            self.signals.messaged.emit(f'Pre-processing image {i}/{nb_im}')

            # Step 1: Reading the image
            cv_rgb_img = cv_read_all_path(rgb_path)

            # Step 2: Image undistortion
            und, _ = undistorter.undis(cv_rgb_img)

            # Step 3: Crop based on custom parameters
            crop = match_rgb_custom_parameters(und, self.drone_model)

            # Step 4: Resize the image
            width = int(crop.shape[1] * self.scale_percent / 100)
            height = int(crop.shape[0] * self.scale_percent / 100)
            dim = (width, height)
            crop = cv2.resize(crop, dim, interpolation=cv2.INTER_AREA)

            # Step 5: Save the new cropped image
            _, file = os.path.split(rgb_path)
            new_name = file[:-4] + 'crop.JPG'
            dest_path = os.path.join(self.dest_crop_folder, new_name)
            cv_write_all_path(crop, dest_path)

            print(f"Iteration {i} - Total iteration time: {time.time() - iter_start_time} seconds")

        with ThreadPoolExecutor() as executor:
            for i, rgb_path in enumerate(self.list_rgb_paths):
                executor.submit(process_image, i, rgb_path)

        self.signals.finished.emit()


# EXPORT FUNCTIONS __________________________________________________
def re_create_miniature(rgb_path, drone_model, dest_crop_folder, scale_percent=60):
    # read rgb
    cv_rgb_img = cv_read_all_path(rgb_path)

    # undistort rgb
    rgb_xml_path = drone_model.rgb_xml_path
    und, _ = undis(cv_rgb_img, rgb_xml_path)
    crop = match_rgb_custom_parameters(und, drone_model)
    width = int(crop.shape[1] * scale_percent / 100)
    height = int(crop.shape[0] * scale_percent / 100)
    dim = (width, height)

    crop = cv2.resize(crop, dim, interpolation=cv2.INTER_AREA)
    _, file = os.path.split(rgb_path)
    new_name = file[:-4] + 'crop.JPG'

    dest_path = os.path.join(dest_crop_folder, new_name)

    cv_write_all_path(crop, dest_path)


def create_video(image_folder, video_name, fps):
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg") or img.endswith(".JPG")]
    images.sort()  # Sort the images if needed

    # Determine the width and height from the first image
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # For mp4 videos
    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()


#  PATH FUNCTIONS __________________________________________________
def cv_read_all_path(path):
    """
    Allows reading image from any kind of unicode character (useful for french accents, for example)
    """
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    return img


def cv_write_all_path(img, path):
    is_success, im_buf_arr = cv2.imencode(".JPG", img)
    im_buf_arr.tofile(path)


def list_th_rgb_images_from_res(img_folder):
    list_rgb_paths = []
    list_ir_paths = []
    for file in os.listdir(img_folder):
        path = os.path.join(img_folder, file)
        print(path)
        if file.endswith('.jpg') or file.endswith('.JPG'):
            im = Image.open(path)
            w, h = im.size

            if w == 640:
                list_ir_paths.append(path)
            else:
                list_rgb_paths.append(path)

        list_ir_paths.sort()
        list_rgb_paths.sort()

    return list_rgb_paths, list_ir_paths


def list_th_rgb_images(img_folder, string_to_search):
    list_rgb_paths = []
    list_ir_paths = []
    for file in os.listdir(img_folder):
        path = os.path.join(img_folder, file)
        if file.endswith('.jpg') or file.endswith('.JPG'):
            if string_to_search in str(file):
                path = os.path.join(img_folder, file)
                list_ir_paths.append(path)
            else:
                list_rgb_paths.append(path)

    return list_rgb_paths, list_ir_paths


def copy_list_dest(list_paths, dest_folder):
    for path in list_paths:
        _, file = os.path.split(path)
        copyfile(path, os.path.join(dest_folder, file))


# EXIF Readings
def print_exif(img_path):
    img = Image.open(img_path)
    infos = img.getexif()
    print(infos)


def extract_exif(img_path):
    img = Image.open(img_path)
    infos = img.getexif()

    return infos


def get_drone_model(img_path):
    img = Image.open(img_path)
    infos = img.getexif()
    model = infos[272]
    return model


def get_drone_model_from_exif(exifs):
    model = exifs[272]
    return model


def get_resolution(img_path):
    img = Image.open(img_path)
    infos = img.getexif()
    res = infos[256]
    return res


def rename_from_exif(img_folder):
    pass


# LENS RELATED METHODS (GENERAL) __________________________________________________
def undis(cv_img, xml_path):
    def read_matrices(xml_path):
        cv_file = cv2.FileStorage(xml_path, cv2.FILE_STORAGE_READ)
        K = cv_file.getNode("Camera_Matrix").mat()
        d = cv_file.getNode("Distortion_Coefficients").mat()
        cv_file.release()

        return K, d

    h, w = cv_img.shape[:2]
    K, d = read_matrices(xml_path)
    newcam, roi = cv2.getOptimalNewCameraMatrix(K, d, (w, h), 1, (w, h))
    dest = cv2.undistort(cv_img, K, d, None, newcam)
    x, y, w, h = roi
    dest = dest[y:y + h, x:x + w]

    sh = dest.shape
    height = sh[0]
    width = sh[1]
    dim = (width, height)

    return dest, dim


def undis_kd(cv_img, K, d):
    h, w = cv_img.shape[:2]
    newcam, roi = cv2.getOptimalNewCameraMatrix(K, d, (w, h), 1, (w, h))
    dest = cv2.undistort(cv_img, K, d, None, newcam)
    x, y, w, h = roi
    dest = dest[y:y + h, x:x + w]

    return dest


# LENS RELATED METHODS (DRONE SPECIFIC) __________________________________________________
def match_rgb_custom_parameters(cv_img, drone_model, resized=False):
    h2, w2 = cv_img.shape[:2]
    new_h = h2 * drone_model.aspect_factor
    ret_x = int(drone_model.extend * w2)
    ret_y = int(drone_model.extend * new_h)
    rgb_dest = cv_img[int(h2 / 2 + drone_model.y_offset) - ret_y:int(h2 / 2 + drone_model.y_offset) + ret_y,
               int(w2 / 2 + drone_model.x_offset) - ret_x:int(w2 / 2 + drone_model.x_offset) + ret_x]

    if resized:
        rgb_dest = cv2.resize(rgb_dest, drone_model.dim_undis_ir, interpolation=cv2.INTER_AREA)

    return rgb_dest


# CROP OPS __________________________________________________
def get_corresponding_crop_rectangle(p1, p2, scale):
    # Scale the coordinates of p1 and p2 to the larger image size
    p1_large = (int(p1[0] * scale), int(p1[1] * scale))
    p2_large = (int(p2[0] * scale), int(p2[1] * scale))

    # Calculate the top-left and bottom-right coordinates of the crop rectangle
    crop_tl = (min(p1_large[0], p2_large[0]), min(p1_large[1], p2_large[1]))
    crop_br = (max(p1_large[0], p2_large[0]), max(p1_large[1], p2_large[1]))

    print(p1, p2, crop_tl, crop_br)

    return crop_tl, crop_br


# LINES __________________________________________________
def add_lines_from_rgb(path_ir, cv_match_rgb_img, drone_model, dest_path,
                       exif=None, mode=1, color='white', bilateral=True, blur=True, blur_size=3, opacity=0.7):
    cv_ir_img = cv2.imread(path_ir)
    img_gray = cv2.cvtColor(cv_match_rgb_img, cv2.COLOR_BGR2GRAY)

    if blur:
        img_gray = cv2.GaussianBlur(img_gray, (blur_size, blur_size), 0)

    if bilateral:
        img_gray = cv2.bilateralFilter(img_gray, 15, 75, 75)

    if mode == 0:
        scale = 1
        delta = 0
        k_size = 3
        ddepth = cv2.CV_16S

        grad_x = cv2.Sobel(img_gray, ddepth, 1, 0, ksize=k_size, scale=scale, delta=delta,
                           borderType=cv2.BORDER_DEFAULT)
        grad_y = cv2.Sobel(img_gray, ddepth, 0, 1, ksize=k_size, scale=scale, delta=delta,
                           borderType=cv2.BORDER_DEFAULT)

        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)

        edges = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

        pil_edges = Image.fromarray(edges)

    elif mode == 1:
        image_pil = Image.fromarray(img_gray)
        pil_edges = image_pil.filter(ImageFilter.Kernel((3, 3), (-1, -1, -1, -1, 8,
                                                                 -1, -1, -1, -1), 1, 0))

    elif mode == 2:
        image_pil = Image.fromarray(img_gray)
        pil_edges = image_pil.filter(ImageFilter.Kernel((3, 3), (0, -1, 0, -1, 4,
                                                                 -1, 0, -1, 0), 1, 0))

    elif mode == 3:
        edges = cv2.Canny(img_gray, 100, 200)
        pil_edges = Image.fromarray(edges)

    elif mode == 4:
        edges = cv2.Canny(img_gray, 100, 200, L2gradient=True)
        pil_edges = Image.fromarray(edges)

    elif mode == 5:
        edge_detector = cv2.ximgproc.createStructuredEdgeDetection('model.yml/model.yml')
        # detect the edges
        edges = edge_detector.detectEdges(cv_match_rgb_img)

    if color == 'black':
        pil_edges = ImageOps.invert(pil_edges)

    pil_edges = pil_edges.convert('RGB')
    pil_edges_rgba = pil_edges.convert('RGBA')
    foreground = np.array(pil_edges_rgba)

    # resize
    dim = drone_model.dim_undis_ir
    foreground = cv2.resize(foreground, dim, interpolation=cv2.INTER_AREA)
    foreground_float = foreground.astype(float)  # Inputs to blend_modes need to be floats.

    cv_ir_img = cv2.cvtColor(cv_ir_img, cv2.COLOR_BGR2RGB)
    ir_img = Image.fromarray(cv_ir_img)
    ir_img = ir_img.convert('RGBA')
    background = np.array(ir_img)
    background_float = background.astype(float)

    if color != 'black':
        blended = dodge(background_float, foreground_float, opacity)
    else:
        blended = multiply(background_float, foreground_float, opacity)

    blended_img = np.uint8(blended)
    blended_img_raw = Image.fromarray(blended_img)
    blended_img_raw = blended_img_raw.convert('RGB')

    if exif is None:
        blended_img_raw.save(dest_path)
    else:
        blended_img_raw.save(dest_path, exif=exif)


def create_lines(cv_img, bil=True):
    img_gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

    if bil:
        # img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)
        img_gray = cv2.bilateralFilter(img_gray, 7, 75, 75)

    scale = 1
    delta = 0
    ddepth = cv2.CV_16S
    grad_x = cv2.Sobel(img_gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(img_gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    # edges = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    edges = cv2.Canny(image=img_gray, threshold1=50, threshold2=200)

    return edges


# THERMAL PROCESSING __________________________________________________
# custom colormaps
def get_custom_cmaps(colormap_name, n_colors):
    # Artic Color Palette
    colors = [(25, 0, 150), (94, 243, 247), (100, 100, 100), (243, 116, 27), (251, 250, 208)]
    colors_scaled = [np.array(x).astype(np.float32) / 255 for x in colors]
    artic_cmap = mcol.LinearSegmentedColormap.from_list('my_colormap', colors_scaled, N=n_colors)

    # Ironbow Color Palette
    colors = [(0, 0, 0), (144, 15, 170), (230, 88, 65), (248, 205, 35), (255, 255, 255)]
    colors_scaled = [np.array(x).astype(np.float32) / 255 for x in colors]
    ironbow_cmap = mcol.LinearSegmentedColormap.from_list('my_colormap', colors_scaled, N=n_colors)

    # Rainbow Color Palette
    colors = [(8, 0, 75), (43, 80, 203), (119, 185, 31), (240, 205, 35), (245, 121, 47), (236, 64, 100),
              (240, 222, 203)]
    colors_scaled = [np.array(x).astype(np.float32) / 255 for x in colors]
    rainbow_cmap = mcol.LinearSegmentedColormap.from_list('my_colormap', colors_scaled, N=n_colors)

    # FIJI Temp Palette
    colors = [(70, 0, 115), (70, 0, 151), (70, 0, 217), (57, 27, 255),
              (14, 136, 251), (0, 245, 235), (76, 255, 247), (206, 255, 254),
              (251, 254, 243), (178, 255, 163), (57, 255, 51), (37, 255, 1),
              (162, 255, 21), (242, 241, 43), (255, 175, 37), (255, 70, 16), (255, 0, 0)]
    colors_scaled = [np.array(x).astype(np.float32) / 255 for x in colors]
    fiji_cmap = mcol.LinearSegmentedColormap.from_list('my_colormap', colors_scaled, N=256)

    # BlueWhiteRed Palette
    colors = [(0, 0, 255), (255, 255, 255), (255, 0, 0)]
    colors_scaled = [np.array(x).astype(np.float32) / 255 for x in colors]
    bwr_cmap = mcol.LinearSegmentedColormap.from_list('my_colormap', colors_scaled, N=256)

    # New Palette: Fulgurite
    colors = [(77, 0, 0), (204, 0, 0), (255, 153, 0), (255, 255, 0), (255, 255, 255)]
    colors_scaled = [np.array(x).astype(np.float32) / 255 for x in colors]
    fulgurite_cmap = mcol.LinearSegmentedColormap.from_list('my_colormap', colors_scaled, N=n_colors)

    # New Palette: Iron Red
    colors = [(0, 0, 0), (30, 0, 90), (100, 0, 130), (175, 0, 175), (255, 70, 0), (255, 130, 0), (255, 200, 0),
              (255, 255, 0), (255, 255, 200)]
    colors_scaled = [np.array(x).astype(np.float32) / 255 for x in colors]
    iron_red_cmap = mcol.LinearSegmentedColormap.from_list('my_colormap', colors_scaled, N=n_colors)

    # New Palette: Hot Iron
    colors = [(0, 0, 0), (0, 51, 51), (25, 100, 100), (50, 130, 130), (120, 175, 80), (190, 215, 40), (255, 255, 0),
              (255, 102, 0), (255, 0, 0)]
    colors_scaled = [np.array(x).astype(np.float32) / 255 for x in colors]
    hot_iron_cmap = mcol.LinearSegmentedColormap.from_list('my_colormap', colors_scaled, N=n_colors)

    # New Palette: Medical
    colors = [(60, 26, 30), (224, 64, 224), (32, 42, 255), (20, 229, 241), (8, 192, 95), (0, 160, 0), (255, 255, 0),
              (255, 0, 0), (255, 255, 255)]
    colors_scaled = [np.array(x).astype(np.float32) / 255 for x in colors]
    medical_cmap = mcol.LinearSegmentedColormap.from_list('my_colormap', colors_scaled, N=n_colors)

    # New Palette: Arctic (refined)
    colors = [(0, 33, 70), (0, 0, 255), (0, 174, 53), (0, 255, 255), (255, 255, 0), (255, 136, 0), (255, 0, 0),
              (255, 25, 255), (255, 25, 255)]
    colors_scaled = [np.array(x).astype(np.float32) / 255 for x in colors]
    arctic_dji_cmap = mcol.LinearSegmentedColormap.from_list('my_colormap', colors_scaled, N=n_colors)

    # New Palette: Rainbow 1
    colors = [(133, 10, 10), (255, 0, 200), (108, 0, 255), (0, 0, 255), (0, 255, 255), (0, 255, 65), (255, 255, 0),
              (255, 160, 0), (255, 0, 0)]
    colors_scaled = [np.array(x).astype(np.float32) / 255 for x in colors]
    rainbow1_cmap = mcol.LinearSegmentedColormap.from_list('my_colormap', colors_scaled, N=n_colors)

    # New Palette: Rainbow 2
    colors = [(0, 0, 130), (0, 0, 255), (0, 130, 255), (0, 255, 255), (130, 255, 130), (255, 255, 0), (255, 130, 0),
              (255, 0, 0), (130, 0, 0)]
    colors_scaled = [np.array(x).astype(np.float32) / 255 for x in colors]
    rainbow2_cmap = mcol.LinearSegmentedColormap.from_list('my_colormap', colors_scaled, N=n_colors)

    # New Palette: Tint
    colors = [(0, 0, 0), (128, 128, 128), (255, 255, 255), (255, 0, 0)]
    colors_scaled = [np.array(x).astype(np.float32) / 255 for x in colors]
    tint_cmap = mcol.LinearSegmentedColormap.from_list('my_colormap', colors_scaled, N=n_colors)

    # Return the correct colormap based on the input name
    if colormap_name == 'Arctic':
        out_colormap = artic_cmap
    elif colormap_name == 'Iron':
        out_colormap = ironbow_cmap
    elif colormap_name == 'Rainbow':
        out_colormap = rainbow_cmap
    elif colormap_name == 'FIJI_Temp':
        out_colormap = fiji_cmap
    elif colormap_name == 'BlueWhiteRed':
        out_colormap = bwr_cmap
    elif colormap_name == 'Fulgurite':
        out_colormap = fulgurite_cmap
    elif colormap_name == 'Iron Red':
        out_colormap = iron_red_cmap
    elif colormap_name == 'Hot Iron':
        out_colormap = hot_iron_cmap
    elif colormap_name == 'Medical':
        out_colormap = medical_cmap
    elif colormap_name == 'Arctic2':
        out_colormap = arctic_dji_cmap
    elif colormap_name == 'Rainbow1':
        out_colormap = rainbow1_cmap
    elif colormap_name == 'Rainbow2':
        out_colormap = rainbow2_cmap
    elif colormap_name == 'Tint':
        out_colormap = tint_cmap

    return out_colormap


def compute_delta(img_path, thermal_param):
    raw_out = img_path[:-4] + '.raw'
    read_dji_image(img_path, raw_out, thermal_param)

    fd = open(raw_out, 'rb')
    rows = 512
    cols = 640
    f = np.fromfile(fd, dtype='<f4', count=rows * cols)
    im = f.reshape((rows, cols))
    fd.close()

    comp_tmin = np.amin(im)
    comp_tmax = np.amax(im)

    os.remove(raw_out)

    return comp_tmin, comp_tmax


def read_dji_image(img_in, raw_out, param={'emissivity': 0.95, 'distance': 5, 'humidity': 50, 'reflection': 25}):
    dist = param['distance']
    rh = param['humidity']
    refl_temp = param['reflection']
    em = param['emissivity']

    try:
        result = subprocess.run(
            [str(sdk_tool_path), "-s", f"{img_in}", "-a", "measure", "-o", f"{raw_out}", "--measurefmt",
             "float32", "--distance", f"{dist}", "--humidity", f"{rh}", "--reflection", f"{refl_temp}",
             "--emissivity", f"{em}"],
            universal_newlines=True,
            stdout=subprocess.PIPE,  # Capture standard output
            stderr=subprocess.PIPE,  # Capture standard error
            shell=True
        )

        # Always print stdout and stderr for debugging
        print("STDOUT:")
        print(result.stdout)
        print("STDERR:")
        print(result.stderr)

        # Check if the subprocess completed successfully
        if result.returncode != 0:
            print("An error occurred in the subprocess.")
    except Exception as e:
        print(f"An exception occurred while running the subprocess: {e}")

    image = Image.open(img_in)
    exif = image.info['exif']

    return exif


def extract_raw_data(param, ir_img_path, undistorder_ir):
    # Step 1: Create raw file
    _, filename = os.path.split(str(ir_img_path))
    new_raw_path = Path(str(ir_img_path)[:-4] + '.raw')
    _ = read_dji_image(str(ir_img_path), str(new_raw_path), param=param)

    # Step 2: Read raw DJI output
    fd = open(new_raw_path, 'rb')
    rows = 512
    cols = 640
    f = np.fromfile(fd, dtype='<f4', count=rows * cols)
    im = f.reshape((rows, cols))  # notice row, column format
    fd.close()

    # Step 3: Create an undistorted version of the temperature map
    undis_im, _ = undistorder_ir.undis(im)

    # Step 4: Remove raw file
    os.remove(new_raw_path)

    return im, undis_im


def process_raw_data(img_object, dest_path, edges, edge_params, undis=True, custom_params=None, export_tif=False):
    if custom_params is None:
        custom_params = {}
    ed_met, ed_col, ed_bil, ed_blur, ed_bl_sz, ed_op = edge_params

    if not img_object.has_data:
        img_object.update_data(img_object.thermal_param)

    # data
    if undis:
        im = img_object.raw_data_undis
    else:
        im = img_object.raw_data
    exif = img_object.exif

    if custom_params:
        tmin = custom_params["tmin"]
        tmax = custom_params["tmax"]
        colormap = custom_params["colormap"]
        n_colors = custom_params["n_colors"]
        col_high = custom_params["col_high"]
        col_low = custom_params["col_low"]
        post_process = custom_params["post_process"]
    else:
        tmin = img_object.tmin_shown
        tmax = img_object.tmax_shown
        colormap = img_object.colormap
        n_colors = img_object.n_colors
        col_high = img_object.user_lim_col_high
        col_low = img_object.user_lim_col_low
        post_process = img_object.post_process

    # compute new normalized temperature
    thermal_normalized = (im - tmin) / (tmax - tmin)

    # get colormap
    if colormap in LIST_CUSTOM_CMAPS:
        custom_cmap = get_custom_cmaps(colormap, n_colors)
    else:
        custom_cmap = cm.get_cmap(colormap, n_colors)
    if col_high != 'c':
        custom_cmap.set_over(col_high)
    if col_low != 'c':
        custom_cmap.set_under(col_low)

    thermal_cmap = custom_cmap(thermal_normalized)
    thermal_cmap = np.uint8(thermal_cmap * 255)

    img_thermal = Image.fromarray(thermal_cmap[:, :, [0, 1, 2]])
    if export_tif:
        dest_path = dest_path[:-4] + '.tiff'
        img_thermal = Image.fromarray(im)  # export as 32bit array (floating point)
        img_thermal.save(dest_path, exif=exif)

    elif post_process == 'none':
        img_thermal.save(dest_path, exif=exif)
    elif post_process == 'sharpen':
        img_th_sharpened = img_thermal.filter(ImageFilter.SHARPEN)
        img_th_sharpened.save(dest_path, exif=exif)
    elif post_process == 'sharpen strong':
        img_th_sharpened = img_thermal.filter(ImageFilter.SHARPEN)
        img_th_sharpened2 = img_th_sharpened.filter(ImageFilter.SHARPEN)
        img_th_sharpened2.save(dest_path, exif=exif)
    elif post_process == 'edge (simple)':
        img_th_smooth = img_thermal.filter(ImageFilter.SMOOTH)
        img_th_findedge = img_th_smooth.filter(ImageFilter.Kernel((3, 3), (-1, -1, -1, -1, 8,
                                                                           -1, -1, -1, -1), 1, 0))
        img_th_findedge = img_th_findedge.convert('RGBA')
        img_thermal = img_thermal.convert('RGBA')
        foreground = np.array(img_th_findedge)  # Inputs to blend_modes need to be numpy arrays.

        foreground_float = foreground.astype(float)  # Inputs to blend_modes need to be floats.
        background = np.array(img_thermal)
        background_float = background.astype(float)
        blended = dodge(background_float, foreground_float, 0.5)

        blended_img = np.uint8(blended)
        blended_img_raw = Image.fromarray(blended_img)
        blended_img_raw = blended_img_raw.convert('RGB')
        blended_img_raw.save(dest_path, exif=exif)
    elif post_process == 'smooth':
        img_th_smooth = img_thermal.filter(ImageFilter.SMOOTH)
        img_th_smooth.save(dest_path, exif=exif)
    elif post_process == 'contours':
        thermal_normalized_clipped = np.clip(thermal_normalized, 0, 1)

        # Generate a blank canvas to draw contours on, in grayscale for thresholding
        height, width = thermal_normalized.shape[:2]
        # contour_canvas = np.zeros((height, width, 3), dtype=np.uint8)
        contour_canvas = np.full((height, width, 3), 255, dtype=np.uint8)

        # Define specific levels for the contour plot, scaled between 0 and 1
        levels = np.linspace(0, 1, num=img_object.n_colors)  # Adjust number of levels as needed

        # Convert thermal_normalized to a suitable format for finding contours
        thermal_for_contours = np.uint8(255 * thermal_normalized_clipped)  # Use clipped version

        # Iterate over levels to draw contours for each level
        for level in levels:
            # Calculate the corresponding color from the colormap
            color = custom_cmap(level)[:3]  # Get RGB components, ignore alpha if present
            color = tuple(int(c * 255) for c in color)  # Convert to BGR for OpenCV, scale to 0-255

            # Threshold image at the current level
            _, thresh = cv2.threshold(thermal_for_contours, int(level * 255), 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Draw contours on the blank canvas
            cv2.drawContours(contour_canvas, contours, -1, color, 1)  # Change thickness as needed

        # Blend the contour canvas with the original thermal image
        blended_img = cv2.addWeighted(np.array(img_thermal), 0.5, contour_canvas, 0.5, 0)

        # Save the blended image
        img_with_contours = Image.fromarray(contour_canvas)
        img_with_contours.save(dest_path, exif=exif)

    # check if edges need to be added (parameter 'edge' is a boolean)
    if edges:
        drone_model_name = get_drone_model_from_exif(exif)
        drone_model = DroneModel(drone_model_name)
        cv_match_rgb_img = cv_read_all_path(img_object.rgb_path)

        # Use the extracted mode directly
        add_lines_from_rgb(dest_path, cv_match_rgb_img, drone_model, dest_path, exif=exif,
                           mode=ed_met, color=ed_col, bilateral=ed_bil, blur=ed_blur, blur_size=ed_bl_sz, opacity=ed_op)


def process_th_image_with_theta(ir_img_path, rgb_img_path, out_folder, theta, F, CX, CY, d_mat):
    # read images
    cv_rgb = cv2.imread(rgb_img_path)
    h_rgb, w_rgb = cv_rgb.shape[:2]
    cv_ir = cv2.imread(ir_img_path)

    K = np.array([[F, 0, CX],
                  [0, F, CY],
                  [0, 0, 1]])

    d = d_mat

    # undistort image with given parameters
    cv_ir_un = undis_kd(cv_ir, K, d)
    h_ir, w_ir = cv_ir_un.shape[:2]

    dim_undis_ir = (w_ir, h_ir)

    extend = theta[0] / 100
    y_offset = theta[1] * 10
    x_offset = theta[2] * 10

    if h_ir == 0:
        aspect_factor = 1
    else:
        aspect_factor = (w_rgb / h_rgb) / (
                w_ir / h_ir)  # this is necessary to transform the aspect ratio of the rgb image to fit
        # the thermal image. The number represent the resolutions of images (rgb and ir respectively) after undistording
    new_h = h_rgb * aspect_factor

    ret_x = int(extend * w_rgb)
    ret_y = int(extend * new_h)
    rgb_dest = cv_rgb[int(h_rgb / 2 + y_offset) - ret_y:int(h_rgb / 2 + y_offset) + ret_y,
               int(w_rgb / 2 + x_offset) - ret_x:int(w_rgb / 2 + x_offset) + ret_x]

    # resize
    rgb_dest = cv2.resize(rgb_dest, dim_undis_ir, interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(out_folder, 'rescale.JPG'), rgb_dest)

    # create lines
    lines_rgb = create_lines(rgb_dest)
    cv2.imwrite(rgb_img_path[:-4] + '_rgb_lines.JPG', lines_rgb)

    lines_ir = create_lines(cv_ir_un, bil=False)
    cv2.imwrite(ir_img_path[:-4] + '_ir_lines.JPG', lines_ir)

    # compute difference
    diff = cv2.subtract(lines_ir, lines_rgb)
    err = np.sum(diff ** 2)
    mse = err / (float(h_ir * w_ir))

    print(f'mse is {mse}')

    return mse


def generate_legend(legend_dest_path, custom_params):
    tmin = custom_params["tmin"]
    tmax = custom_params["tmax"]
    color_high = custom_params["col_high"]
    color_low = custom_params["col_low"]
    colormap = custom_params["colormap"]
    n_colors = custom_params["n_colors"]

    fig, ax = plt.subplots()
    data = np.clip(np.random.randn(10, 10) * 100, tmin, tmax)
    print(data)

    if colormap in LIST_CUSTOM_CMAPS:
        custom_cmap = get_custom_cmaps(colormap, n_colors)
    else:
        custom_cmap = cm.get_cmap(colormap, n_colors)

    custom_cmap.set_over(color_high)
    custom_cmap.set_under(color_low)

    cax = ax.imshow(data, cmap=custom_cmap)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    if n_colors > 12:
        n_colors = 12
    ticks = np.linspace(tmin, tmax, n_colors + 1, endpoint=True)
    fig.colorbar(cax, ticks=ticks, extend='both')
    ax.remove()

    plt.savefig(legend_dest_path, bbox_inches='tight')


"""
°-°-°-°JUNK°-°-°-°
"""
