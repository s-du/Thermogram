import os
from shutil import copyfile, copytree
import numpy as np
import subprocess
from pathlib import Path
from PIL import Image, ImageOps, ImageFilter
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import QPointF, QRectF
from PySide6.QtWidgets import QGraphicsEllipseItem, QGraphicsTextItem

from matplotlib import cm
from matplotlib import pyplot as plt
import matplotlib.colors as mcol
from blend_modes import dodge, multiply
import fileinput
import shutil
# from skimage.segmentation import felzenszwalb, slic --> Removed superpixel for the moment
# from skimage.color import label2rgb --> Removed superpixel for the moment
# from skimage.color import label2rgb

import cv2

# custom libraries
import resources as res

# PATHS
sdk_tool_path = Path(res.find('dji/dji_irp.exe'))
m2t_ir_xml_path = res.find('other/cam_calib_m2t_opencv.xml')
m2t_rgb_xml_path = res.find('other/rgb_cam_calib_m2t_opencv.xml')

m3t_ir_xml_path = res.find('other/cam_calib_m3t_opencv.xml')
m3t_rgb_xml_path = res.find('other/rgb_cam_calib_m3t_opencv.xml')

LIST_CUSTOM_CMAPS = ['Artic','Iron','Rainbow','FIJI_Temp','BlueWhiteRed']

# USEFUL CLASSES
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
            self.extend = 0.3504
            self.x_offset = 49
            self.y_offset = 53


class ObjectDetectionCategory:
    """
    Class to describe a segmentation category
    """

    def __init__(self):
        self.color = None
        self.name = ''


class ProcessedIm:
    def __init__(self, path, rgb_path, delayed_compute = True):
        # general infos
        self.path = path
        self.rgb_path = rgb_path
        self.preview_path = ''
        self.exif = extract_exif(self.path)
        self.drone_model_name = get_drone_model_from_exif(self.exif)
        # colormap infos
        self.colormap = 'coolwarm'
        self.n_colors = 256

        self.has_data = False

        # ir infos
        self.thermal_param = {'emissivity': 0.95, 'distance': 5, 'humidity': 50, 'reflection': 25}
        if delayed_compute:
            self.raw_data, self.raw_data_undis = extract_raw_data(self.thermal_param, self.path)
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

    def update_data(self, new_params):
        self.thermal_param = new_params
        self.raw_data, self.raw_data_undis = extract_raw_data(self.thermal_param, self.path)
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
    def __init__(self, ir_paths, dest_folder, drone_model, param, tmin, tmax, colormap,
                 color_high, color_low, start, stop, n_colors=256, post_process='none', export_tif = False):
        super().__init__()
        self.ir_paths = ir_paths
        self.dest_folder = dest_folder
        self.post_process = post_process
        self.drone_model = drone_model
        self.param = param
        self.tmin = tmin
        self.tmax = tmax
        self.colormap = colormap
        self.color_high = color_high
        self.color_low = color_low
        self.n_colors = n_colors

        self.export_tif = export_tif

        self.start = start
        self.stop = stop

        self.signals = RunnerSignals()

    def run(self):
        # create raw outputs for each image
        nb_im = len(self.ir_paths)
        for i, img_path in enumerate(self.ir_paths):
            print(i)
            iter = i * (self.stop - self.start) / nb_im

            self.signals.progressed.emit(self.start + iter)
            self.signals.messaged.emit(f'Processing image {i}/{nb_im} with DJI SDK')

            if i < 9:
                prefix = '000'
            elif 9 < i < 99:
                prefix = '00'
            elif 99 < i < 999:
                prefix = '0'
            _, filename = os.path.split(str(img_path))
            dest_path = os.path.join(self.dest_folder, f'thermal_{prefix}{i}.JPG')

            process_one_th_picture(self.param, self.drone_model, img_path, dest_path, self.tmin, self.tmax,
                                           self.colormap, self.color_high, self.color_low, n_colors=self.n_colors,
                                           post_process=self.post_process, export_tif = self.export_tif)

            if i == len(self.ir_paths) - 1:
                legend_dest_path = os.path.join(self.dest_folder, 'plot_onlycbar_tight.png')
                generate_legend(legend_dest_path, self.tmin, self.tmax, self.color_high, self.color_low, self.colormap,
                                self.n_colors)

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
        ir_xml_path = self.drone_model.ir_xml_path
        rgb_xml_path = self.drone_model.rgb_xml_path

        for i, rgb_path in enumerate(self.list_rgb_paths):
            iter = i * (self.stop - self.start) / nb_im

            # update progress
            self.signals.progressed.emit(self.start + iter)
            self.signals.messaged.emit(f'Pre-processing image {i}/{nb_im}')
            cv_rgb_img = cv_read_all_path(rgb_path)

            und, _ = undis(cv_rgb_img, rgb_xml_path)
            crop = match_rgb_custom_parameters(und, self.drone_model)
            width = int(crop.shape[1] * self.scale_percent / 100)
            height = int(crop.shape[0] * self.scale_percent / 100)
            dim = (width, height)

            crop = cv2.resize(crop, dim, interpolation=cv2.INTER_AREA)
            _, file = os.path.split(rgb_path)
            new_name = file[:-4] + 'crop.JPG'

            dest_path = os.path.join(self.dest_crop_folder, new_name)

            cv_write_all_path(crop, dest_path)

        self.signals.finished.emit()


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


def create_undis_folder(list_ir_paths, drone_model, dest_und_folder):
    ir_xml_path = drone_model.ir_xml_path
    for ir_path in list_ir_paths:
        cv_ir_img = cv_read_all_path(ir_path)
        cv_und, _ = undis(cv_ir_img, ir_xml_path)
        _, file = os.path.split(ir_path)
        new_name = file[:-4] + 'undis.JPG'
        dest_path = os.path.join(dest_und_folder, new_name)
        cv_write_all_path(cv_und, dest_path)


def create_rgb_crop_folder(list_rgb_paths, drone_model, scale_percent, dest_crop_folder, progressbar, start, stop):
    nb_im = len(list_rgb_paths)
    print(list_rgb_paths)
    for i, rgb_path in enumerate(list_rgb_paths):
        iter = i * (stop - start) / nb_im
        progressbar.setProperty("value", start + iter)
        cv_rgb_img = cv_read_all_path(rgb_path)

        rgb_xml_path = drone_model.rgb_xml_path
        und = undis(cv_rgb_img, rgb_xml_path)
        crop = match_rgb_custom_parameters(und, drone_model)
        width = int(crop.shape[1] * scale_percent / 100)
        height = int(crop.shape[0] * scale_percent / 100)
        dim = (width, height)

        crop = cv2.resize(crop, dim, interpolation=cv2.INTER_AREA)
        _, file = os.path.split(rgb_path)
        new_name = file[:-4] + 'crop.JPG'

        dest_path = os.path.join(dest_crop_folder, new_name)
        cv2.imwrite(dest_path, crop)


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


# LENS RELATED METHODS (DRONE SPECIFIC)
def match_rgb_custom_parameters(cv_img, drone_model, resized=False):
    print(cv_img)
    h2, w2 = cv_img.shape[:2]
    new_h = h2 * drone_model.aspect_factor
    ret_x = int(drone_model.extend * w2)
    ret_y = int(drone_model.extend * new_h)
    rgb_dest = cv_img[int(h2 / 2 + drone_model.y_offset) - ret_y:int(h2 / 2 + drone_model.y_offset) + ret_y,
               int(w2 / 2 + drone_model.x_offset) - ret_x:int(w2 / 2 + drone_model.x_offset) + ret_x]

    if resized:
        rgb_dest = cv2.resize(rgb_dest, drone_model.dim_undis_ir, interpolation=cv2.INTER_AREA)

    return rgb_dest


def add_lines_from_rgb(path_ir, cv_match_rgb_img, drone_model, dest_path,
                       exif=None, mode=1, color='white', blur=True, blur_size=3, opacity=0.7):


    cv_ir_img = cv2.imread(path_ir)
    img_gray = cv2.cvtColor(cv_match_rgb_img, cv2.COLOR_BGR2GRAY)

    if blur:
        img_gray = cv2.GaussianBlur(img_gray, (blur_size, blur_size), 0)

    if mode==0:
        scale = 1
        delta = 0
        k_size = 3
        ddepth = cv2.CV_16S

        grad_x = cv2.Sobel(img_gray, ddepth, 1, 0, ksize=k_size, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
        grad_y = cv2.Sobel(img_gray, ddepth, 0, 1, ksize=k_size, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)

        edges = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

        pil_edges = Image.fromarray(edges)

    elif mode==1:
        image_pil = Image.fromarray(img_gray)
        pil_edges = image_pil.filter(ImageFilter.Kernel((3, 3), (-1, -1, -1, -1, 8,
                                          -1, -1, -1, -1), 1, 0))

    elif mode==2:
        image_pil = Image.fromarray(img_gray)
        pil_edges = image_pil.filter(ImageFilter.Kernel((3, 3), (0, -1, 0, -1, 4,
                                          -1, 0, -1, 0), 1, 0))

    elif mode==3:
        edges = cv2.Canny(img_gray, 100, 200)
        pil_edges = Image.fromarray(edges)

    elif mode==4:
        edges = cv2.Canny(img_gray, 100, 200, L2gradient = True)
        pil_edges = Image.fromarray(edges)

    elif mode==5:
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


# CROP OPS
def get_corresponding_crop_rectangle(p1, p2, scale):
    # Scale the coordinates of p1 and p2 to the larger image size
    p1_large = (int(p1[0] * scale), int(p1[1] * scale))
    p2_large = (int(p2[0] * scale), int(p2[1] * scale))

    # Calculate the top-left and bottom-right coordinates of the crop rectangle
    crop_tl = (min(p1_large[0], p2_large[0]), min(p1_large[1], p2_large[1]))
    crop_br = (max(p1_large[0], p2_large[0]), max(p1_large[1], p2_large[1]))

    print(p1, p2, crop_tl, crop_br)

    return crop_tl, crop_br


# THERMAL PROCESSING _____________________________________________
# custom colormaps
def get_custom_cmaps(colormap_name, n_colors):
    colors = [(25, 0, 150), (94, 243, 247), (100, 100, 100), (243, 116, 27), (251, 250, 208)]
    colors_scaled = [np.array(x).astype(np.float32) / 255 for x in colors]
    artic_cmap = mcol.LinearSegmentedColormap.from_list('my_colormap', colors_scaled, N=n_colors)

    colors = [(0, 0, 0), (144, 15, 170), (230, 88, 65), (248, 205, 35), (255, 255, 255)]
    colors_scaled = [np.array(x).astype(np.float32) / 255 for x in colors]
    ironbow_cmap = mcol.LinearSegmentedColormap.from_list('my_colormap', colors_scaled, N=n_colors)

    colors = [(8, 0, 75), (43, 80, 203), (119, 185, 31), (240, 205, 35), (245, 121, 47), (236, 64, 100),
              (240, 222, 203)]
    colors_scaled = [np.array(x).astype(np.float32) / 255 for x in colors]
    rainbow_cmap = mcol.LinearSegmentedColormap.from_list('my_colormap', colors_scaled, N=n_colors)

    colors = [(70, 0, 115), (70, 0, 151), (70, 0, 217), (57, 27, 255),
              (14, 136, 251), (0, 245, 235), (76, 255, 247), (206, 255, 254),
              (251, 254, 243), (178, 255, 163), (57, 255, 51), (37, 255, 1),
              (162, 255, 21), (242, 241, 43), (255, 175, 37), (255, 70, 16), (255, 0, 0)]
    colors_scaled = [np.array(x).astype(np.float32) / 255 for x in colors]
    fiji_cmap = mcol.LinearSegmentedColormap.from_list('my_colormap', colors_scaled, N=256)

    colors = [(0, 0, 255), (255, 255, 255), (255, 0, 0)]
    colors_scaled = [np.array(x).astype(np.float32) / 255 for x in colors]
    bwr_cmap = mcol.LinearSegmentedColormap.from_list('my_colormap', colors_scaled, N=256)

    if colormap_name == 'Artic':
        out_colormap = artic_cmap
    elif colormap_name == 'Iron':
        out_colormap = ironbow_cmap
    elif colormap_name == 'Rainbow':
        out_colormap = rainbow_cmap
    elif colormap_name == 'FIJI_Temp':
        out_colormap = fiji_cmap
    elif colormap_name == 'BlueWhiteRed':
        out_colormap = bwr_cmap

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

    subprocess.run(
        [str(sdk_tool_path), "-s", f"{img_in}", "-a", "measure", "-o", f"{raw_out}", "--measurefmt",
         "float32", "--distance", f"{dist}", "--humidity", f"{rh}", "--reflection", f"{refl_temp}",
         "--emissivity", f"{em}"],
        universal_newlines=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        shell=True
    )

    image = Image.open(img_in)
    exif = image.info['exif']

    return exif


def extract_raw_data(param, ir_img_path):
    # Read img infos
    exif = extract_exif(ir_img_path)
    drone_model_name = get_drone_model_from_exif(exif)
    drone_model = DroneModel(drone_model_name)

    # create raw file
    _, filename = os.path.split(str(ir_img_path))
    new_raw_path = Path(str(ir_img_path)[:-4] + '.raw')

    _ = read_dji_image(str(ir_img_path), str(new_raw_path), param=param)
    ir_xml_path = drone_model.ir_xml_path

    # read raw dji output
    fd = open(new_raw_path, 'rb')
    rows = 512
    cols = 640
    f = np.fromfile(fd, dtype='<f4', count=rows * cols)
    im = f.reshape((rows, cols))  # notice row, column format
    fd.close()

    # create an undistort version of the temperature map
    undis_im, _ = undis(im, ir_xml_path)

    # remove raw file
    os.remove(new_raw_path)

    return im, undis_im


def process_raw_data(img_object, dest_path, edges, edge_params):

    ed_met, ed_col, ed_blur, ed_bl_sz, ed_op = edge_params

    if not img_object.has_data:
        img_object.update_data(img_object.thermal_param)
    im = img_object.raw_data_undis
    tmin = img_object.tmin_shown
    tmax = img_object.tmax_shown
    colormap = img_object.colormap
    n_colors = img_object.n_colors
    col_high = img_object.user_lim_col_high
    col_low = img_object.user_lim_col_low
    exif = img_object.exif
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

    if post_process == 'none':
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
            color = tuple(int(c * 255) for c in color)   # Convert to BGR for OpenCV, scale to 0-255

            # Threshold image at the current level
            _, thresh = cv2.threshold(thermal_for_contours, int(level * 255), 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Draw contours on the blank canvas
            cv2.drawContours(contour_canvas, contours, -1, color, 2)  # Change thickness as needed

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
                           mode=ed_met, color=ed_col, blur=ed_blur, blur_size=ed_bl_sz, opacity=ed_op)



def process_one_th_picture(param, drone_model, ir_img_path, dest_path, tmin, tmax, colormap, color_high,
                           color_low, n_colors=256, post_process='none', export_tif = False):
    _, filename = os.path.split(str(ir_img_path))
    new_raw_path = Path(str(ir_img_path)[:-4] + '.raw')

    exif = read_dji_image(str(ir_img_path), str(new_raw_path), param=param)

    # read raw dji output
    fd = open(new_raw_path, 'rb')
    rows = 512
    cols = 640
    f = np.fromfile(fd, dtype='<f4', count=rows * cols)
    im = f.reshape((rows, cols))  # notice row, column format
    fd.close()

    if export_tif:
        dest_path = dest_path[:-4] + '.tiff'
        print(im.dtype)
        img_thermal = Image.fromarray(im)
        img_thermal.save(dest_path, exif=exif)

    else:
        # compute new normalized temperature
        thermal_normalized = (im - tmin) / (tmax - tmin)

        # get colormap
        if colormap in LIST_CUSTOM_CMAPS:
            custom_cmap = get_custom_cmaps(colormap, n_colors)
        else:
            custom_cmap = cm.get_cmap(colormap, n_colors)

        if color_high != 'c':
            custom_cmap.set_over(color_high)
        if color_low != 'c':
            custom_cmap.set_under(color_low)

        thermal_cmap = custom_cmap(thermal_normalized)
        thermal_cmap = np.uint8(thermal_cmap * 255)

        img_thermal = Image.fromarray(thermal_cmap[:, :, [0, 1, 2]])

        if post_process == 'none':
            img_thermal.save(dest_path, exif=exif)
        elif post_process == 'sharpen':
            img_th_sharpened = img_thermal.filter(ImageFilter.SHARPEN)
            img_th_sharpened.save(dest_path, exif=exif)
        elif post_process == 'smooth':
            img_th_smooth = img_thermal.filter(ImageFilter.SMOOTH)
            img_th_smooth.save(dest_path, exif=exif)

    os.remove(new_raw_path)


def generate_legend(legend_dest_path, tmin, tmax, color_high, color_low, colormap, n_colors):
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

def path_info(path):
    """
    Function that reads a path and outputs the folder, the complete filename and the filename without file extension
    @ parameters:
        path -- input path (string)
    """
    folder, file = os.path.split(path)
    extension = file[-4:]
    name = file[:-4]
    return folder, file, name, extension


def find_files_of_type(folder, types=[]):
    files = [os.path.join(folder, file) for file in os.listdir(folder)]
    output = []

    for file in files:
        for type in types:
            if file.endswith(type):
                output.append(file)

    return output


def sort_image_method1(img_folder, dest_rgb_folder, dest_th_folder, string_to_search):
    """
    this function is adapted to sort all images from a folder, where those images are a mix between thermal and corresponding rgb
    """
    # Sorting images in new folders

    count = 0

    for file in os.listdir(img_folder):
        if count < 9:
            prefix = '000'
        elif 9 < count < 99:
            prefix = '00'
        elif 99 < count < 999:
            prefix = '000'
        if file.endswith('.jpg') or file.endswith('.JPG'):
            if string_to_search in str(file):
                new_file = 'image_' + prefix + str(count) + '.jpg'
                copyfile(os.path.join(img_folder, file), os.path.join(dest_th_folder, new_file))
                count += 1
            else:
                if count == 0:
                    count += 1
                new_file = 'image_' + prefix + str(count) + '.jpg'
                copyfile(os.path.join(img_folder, file), os.path.join(dest_rgb_folder, new_file))


def list_th_rgb_images_from_exif(img_folder):
    list_rgb_paths = []
    list_ir_paths = []
    for file in os.listdir(img_folder):

        path = os.path.join(img_folder, file)
        print(path)
        if file.endswith('.jpg') or file.endswith('.JPG'):
            res = get_resolution(path)
            if res == 640:
                list_ir_paths.append(path)
            else:
                list_rgb_paths.append(path)

    return list_rgb_paths, list_ir_paths


def process_all_th_pictures(param, drone_model, ir_paths, dest_folder, tmin, tmax, colormap, color_high, color_low,
                            n_colors=256,
                            post_process='none'):
    """
    this function process all thermal pictures in a folder
    """
    # create raw outputs for each image
    for i, img_path in enumerate(ir_paths):
        print(i)
        if i < 9:
            prefix = '000'
        elif 9 < i < 99:
            prefix = '00'
        elif 99 < i < 999:
            prefix = '0'
        _, filename = os.path.split(str(img_path))
        dest_path = os.path.join(dest_folder, f'thermal_{prefix}{i}.JPG')

        process_one_th_picture(param, drone_model, img_path, dest_path, tmin, tmax, colormap, color_high,
                                   color_low, n_colors=n_colors, post_process=post_process)

        if i == len(ir_paths) - 1:
            legend_dest_path = os.path.join(dest_folder, 'plot_onlycbar_tight.png')
            generate_legend(legend_dest_path, tmin, tmax, color_high, color_low, colormap, n_colors)