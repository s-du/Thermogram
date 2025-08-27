"""Core classes for thermal image processing.

Path processing, raw data extraction, measurements
"""
# Imports
import cv2
import os
import resources as res
from pathlib import Path
import numpy as np
from shutil import copyfile
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtWidgets import QGraphicsEllipseItem, QGraphicsTextItem
from PyQt6.QtGui import QImage, QPainter, QPixmap
import subprocess
from tools import thermal_tools as tt


# Paths
sdk_tool_path = Path(res.find('dji/dji_irp.exe'))
m2t_ir_xml_path = res.find('other/cam_calib_m2t_opencv.xml')
m2t_rgb_xml_path = res.find('other/rgb_cam_calib_m2t_opencv.xml')

m3t_ir_xml_path = res.find('other/cam_calib_m3t_opencv.xml')
m3t_rgb_xml_path = res.find('other/rgb_cam_calib_m3t_opencv.xml')

m30t_ir_xml_path = res.find('other/cam_calib_m30t_opencv.xml')
m30t_rgb_xml_path = res.find('other/rgb_cam_calib_m30t_opencv.xml')

m4t_ir_xml_path = res.find('other/cam_calib_m4t_opencv.xml')
m4t_rgb_xml_path = res.find('other/rgb_cam_calib_m4t_opencv.xml')


#  PATH FUNCTIONS __________________________________________________
def cv_read_all_path(path):
    """
    Allows reading image from any kind of unicode character (useful for french accents, for example)
    """
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    return img


def cv_write_all_path(img, path, extension='PNG'):
    is_success, im_buf_arr = cv2.imencode('.' + extension, img)
    im_buf_arr.tofile(path)


def list_th_rgb_images_from_res(img_folder):
    list_rgb_paths = []
    list_ir_paths = []
    list_z_paths = []
    for file in os.listdir(img_folder):
        path = os.path.join(img_folder, file)

        if file.endswith('.jpg') or file.endswith('.JPG'):
            im = Image.open(path)
            w, h = im.size

            if w == 640 or w == 1280:
                list_ir_paths.append(path)
            else:
                if '_W' or '_V' in file:
                    list_rgb_paths.append(path)
                elif '_Z' in file:
                    list_z_paths.append(path)

        list_ir_paths.sort()
        list_rgb_paths.sort()
        list_z_paths.sort()

    return list_rgb_paths, list_ir_paths, list_z_paths


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


# EXIF FUNCTIONS _____________________________________
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

def get_gps_data(image):
    """Extract EXIF data from an image and return it as a dictionary."""
    exif_data = {}
    try:
        info = image._getexif()
        if info is not None:
            for tag, value in info.items():
                tag_name = TAGS.get(tag, tag)
                if tag_name == "GPSInfo":
                    gps_data = {}
                    for t in value:
                        sub_tag = GPSTAGS.get(t, t)
                        gps_data[sub_tag] = value[t]
                    exif_data[tag_name] = gps_data
                else:
                    exif_data[tag_name] = value

                gps_info = exif_data.get("GPSInfo")
    except AttributeError:
        pass
    return gps_info

def get_coordinates(gps_info):
    """Convert GPS information from EXIF data to latitude and longitude."""
    def convert_to_degrees(value):
        d, m, s = value
        return d + (m / 60.0) + (s / 3600.0)

    if "GPSLatitude" in gps_info and "GPSLongitude" in gps_info:
        lat = convert_to_degrees(gps_info["GPSLatitude"])
        lon = convert_to_degrees(gps_info["GPSLongitude"])

        if gps_info["GPSLatitudeRef"] == "S":
            lat = -lat
        if gps_info["GPSLongitudeRef"] == "W":
            lon = -lon

        return lat, lon
    return None


# BASIC PROCESSING __________________________________
def save_with_white_background(image, path):
    bg_image = QImage(image.size(), QImage.Format.Format_ARGB32_Premultiplied)
    bg_image.fill(Qt.GlobalColor.white)
    painter = QPainter(bg_image)
    painter.drawImage(0, 0, image)
    painter.end()
    bg_image.save(path, 'JPEG', 100)

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
        """print("STDOUT:")
        print(result.stdout)
        print("STDERR:")
        print(result.stderr)"""

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
    # create_vector_plot(im)
    fd.close()

    # Step 3: Create an undistorted version of the temperature map
    undis_im, _ = undistorder_ir.undis(im)

    # Step 4: Remove raw file
    os.remove(new_raw_path)

    return im, undis_im


# CORE CLASSES _____________________________________
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

        # Desired size
        target_w, target_h = 640, 512

        # If larger than target, resize to exactly 640x512
        if w > target_w or h > target_h:
            cv_img = cv2.resize(cv_img, (target_w, target_h), interpolation=cv2.INTER_AREA)
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


class DroneModel:
    def __init__(self, name):
        if name == 'MAVIC2-ENTERPRISE-ADVANCED':
            self.ir_xml_path = m2t_ir_xml_path
            undistorder_ir = CameraUndistorter(self.ir_xml_path)
            sample_ir_path = res.find('img/M2EA_IR.JPG')
            cv_ir = cv_read_all_path(sample_ir_path)
            _, self.dim_undis_ir = undistorder_ir.undis(cv_ir)

            self.aspect_factor = (8000 / 6000) / (
                    self.dim_undis_ir[0] / self.dim_undis_ir[1])

            # read focal parameters
            cv_file = cv2.FileStorage(m2t_ir_xml_path, cv2.FILE_STORAGE_READ)
            self.K_ir = cv_file.getNode("Camera_Matrix").mat()
            self.d_ir = cv_file.getNode("Distortion_Coefficients").mat()
            self.extend = 0.332
            self.x_offset = 50
            self.y_offset = 35
            self.zoom = 1.506

        elif name == 'M3T':
            self.ir_xml_path = m3t_ir_xml_path
            undistorder_ir = CameraUndistorter(self.ir_xml_path)
            sample_ir_path = res.find('img/M3T_IR.JPG')
            cv_ir = cv_read_all_path(sample_ir_path)
            _, self.dim_undis_ir = undistorder_ir.undis(cv_ir)
            self.aspect_factor = (4000 / 3000) / (
                    self.dim_undis_ir[0] / self.dim_undis_ir[1])
            # read focal parameters
            cv_file = cv2.FileStorage(m3t_ir_xml_path, cv2.FILE_STORAGE_READ)
            self.K_ir = cv_file.getNode("Camera_Matrix").mat()
            self.d_ir = cv_file.getNode("Distortion_Coefficients").mat()
            self.extend = 0.3504
            self.x_offset = -19
            self.y_offset = 23
            self.zoom = 1.516

        elif name == 'M30T':
            self.ir_xml_path = m30t_ir_xml_path
            undistorder_ir = CameraUndistorter(self.ir_xml_path)
            sample_ir_path = res.find('img/M30T_IR.JPG')
            cv_ir = cv_read_all_path(sample_ir_path)
            _, self.dim_undis_ir = undistorder_ir.undis(cv_ir)
            self.aspect_factor = (4000 / 3000) / (
                    self.dim_undis_ir[0] / self.dim_undis_ir[1])
            # read focal parameters
            cv_file = cv2.FileStorage(m30t_ir_xml_path, cv2.FILE_STORAGE_READ)
            self.K_ir = cv_file.getNode("Camera_Matrix").mat()
            self.d_ir = cv_file.getNode("Distortion_Coefficients").mat()
            self.extend = 0.3504
            self.x_offset = 48
            self.y_offset = -6.9
            self.zoom = 1.53

        elif name == 'M4T':
            self.ir_xml_path = m4t_ir_xml_path
            undistorder_ir = CameraUndistorter(self.ir_xml_path)
            sample_ir_path = res.find('img/M4T_IR.JPG')
            cv_ir = cv_read_all_path(sample_ir_path)

            # add some control for upscaled IR images


            _, self.dim_undis_ir = undistorder_ir.undis(cv_ir)
            self.aspect_factor = (4000 / 3000) / (
                    self.dim_undis_ir[0] / self.dim_undis_ir[1])
            # read focal parameters
            cv_file = cv2.FileStorage(m4t_ir_xml_path, cv2.FILE_STORAGE_READ)
            self.K_ir = cv_file.getNode("Camera_Matrix").mat()
            self.d_ir = cv_file.getNode("Distortion_Coefficients").mat()
            self.extend = 0.3504
            self.x_offset = 45
            self.y_offset = 83
            self.zoom = 1.12


class ProcessedIm:
    def __init__(self, path, rgb_path, original_path, undistorder_ir, drone_model, delayed_compute=False):
        # General infos
        self.path = path
        self.rgb_path = rgb_path
        self.rgb_path_original = original_path
        self.preview_path = ''
        self.undistorder_ir = undistorder_ir

        # Colormap infos
        self.colormap = 'Greys_r'
        self.n_colors = 256

        # Data
        self._exif = None
        self.has_data = False
        self.drone_model = drone_model

        # IR infos
        self.thermal_param = {'emissivity': 0.95, 'distance': 5, 'humidity': 50, 'reflection': 25}
        if not delayed_compute:
            self.raw_data, self.raw_data_undis = extract_raw_data(self.thermal_param, self.path, self.undistorder_ir)
        self.tmin = np.amin(self.raw_data)
        self.tmax = np.amax(self.raw_data)
        self.tmin_shown = self.tmin
        self.tmax_shown = self.tmax

        self.user_lim_col_low = 'c'
        self.user_lim_col_high = 'c'
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

        # for custom images
        self.nb_custom_imgs = 0
        self.custom_images = []
        
        # for report inclusion
        self.include_in_report = True
        
        # for user remarks
        self.remarks = ""

    @property
    def exif(self):
        # Lazy load the EXIF data
        if self._exif is None:
            self._exif = extract_exif(self.path)
        return self._exif

    def update_data_from_param(self, new_params, reset_shown_values=True):
        # print(new_params)
        self.thermal_param = new_params
        self.raw_data, self.raw_data_undis = extract_raw_data(self.thermal_param, self.path, self.undistorder_ir)
        self.tmin = np.amin(self.raw_data)
        self.tmax = np.amax(self.raw_data)
        if reset_shown_values:
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
                self.post_process)

    def get_temp_data(self):
        return (self.tmin,
                self.tmax,
                self.tmin_shown,
                self.tmax_shown)

    def get_thermal_param(self):
        return self.thermal_param

    def compute_optimal_temp_range(self, lower_percentile=2.5, upper_percentile=97.5):
        temp_flat = self.raw_data.flatten()
        tmin_opt = np.percentile(temp_flat, lower_percentile)
        tmax_opt = np.percentile(temp_flat, upper_percentile)
        return tmin_opt, tmax_opt

    def create_temperature_histogram_canvas(self):
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.figure import Figure

        fig = Figure(figsize=(2.5, 1.5), dpi=100)
        ax = fig.add_subplot(111)
        canvas = FigureCanvas(fig)
        canvas.setFixedSize(250, 150)

        # Attach ax for later use
        canvas.ax = ax

        # Make canvas background transparent
        canvas.setStyleSheet("background-color: transparent;")
        canvas.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        canvas.setAutoFillBackground(False)

        return canvas

    def update_temperature_histogram(self, canvas, bins=50):
        if canvas is None or not hasattr(canvas, 'ax'):
            print("Warning: Tried to update histogram, but canvas is None or incomplete.")
            return

        import numpy as np
        from matplotlib import cm

        ax = canvas.ax
        fig = ax.figure
        ax.clear()  # Clear previous content

        temp_clipped = np.clip(self.raw_data, self.tmin_shown, self.tmax_shown).flatten()

        # Get colormap
        if self.colormap in tt.LIST_CUSTOM_NAMES:
            all_cmaps = tt.get_all_custom_cmaps(self.n_colors)
            cmap = all_cmaps[self.colormap]
        else:
            cmap = cm.get_cmap(self.colormap, self.n_colors)

        # Plot histogram
        counts, bin_edges, patches = ax.hist(temp_clipped, bins=bins, edgecolor='none')

        # Color bars
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        normed = (bin_centers - self.tmin_shown) / (self.tmax_shown - self.tmin_shown)
        normed = np.clip(normed, 0, 1)
        for patch, color_value in zip(patches, normed):
            patch.set_facecolor(cmap(color_value))

        # Transparent background
        ax.set_facecolor("none")
        fig.patch.set_alpha(0.0)

        # Enable x-axis with temperature ticks
        ax.tick_params(axis='x', labelsize=7)
        ax.yaxis.set_visible(False)

        # Light grid if you like
        ax.grid(True, linestyle='--', linewidth=0.3, alpha=0.4)

        fig.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])  # adds breathing room
        canvas.draw()


# MEASUREMENTS CLASSES _____________________________________
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

        maxima_idx, minima_idx = tt.find_local_extrema(self.data_roi, order=10)
        self.local_maxima = maxima_idx
        self.local_minima = minima_idx

    def create_items(self):
        self.spot_items.clear()
        self.text_items.clear()

        start_point = self.main_item.line().p1()
        end_point = self.main_item.line().p2()

        P1 = np.array([start_point.x(), start_point.y()])
        P2 = np.array([end_point.x(), end_point.y()])
        num_points = len(self.data_roi)

        xs = np.linspace(P1[0], P2[0], num_points)
        ys = np.linspace(P1[1], P2[1], num_points)

        for i, idx in enumerate(np.concatenate((self.local_maxima, self.local_minima))):
            x, y = xs[idx], ys[idx]
            temp = self.data_roi[idx]
            temp_str = f"{temp:.2f}"

            # Create ellipse
            ellipse = QGraphicsEllipseItem()
            p1 = QPointF(x - 2, y - 2)
            p2 = QPointF(x + 2, y + 2)
            ellipse.setRect(QRectF(p1, p2))

            # Create text with vertical offset to avoid overlapping
            text = QGraphicsTextItem()
            vertical_offset = (-25 if i % 2 == 0 else 0)
            text.setPos(QPointF(x, y + vertical_offset))
            text.setHtml(
                f"<div style='background-color:rgba(255, 255, 255, 0.3);'>{temp_str}</div>"
            )

            self.spot_items.append(ellipse)
            self.text_items.append(text)


class RectMeas:
    def __init__(self, item):
        self.name = ''
        self.main_item = item
        self.ellipse_items = []
        self.text_items = []
        self.coords = []
        self.data_roi = []
        self.th_norm = []
        self.has_rgb = True

        self.tmin = 0
        self.tmax = 0

    def get_coord_from_item(self, QGraphicsRect):
        self.rect = QGraphicsRect
        rect = QGraphicsRect.rect()
        coord = [rect.topLeft(), rect.bottomRight()]

        return coord

    def compute_temp_data(self, coords, raw_data):
        p1 = coords[0]
        p2 = coords[1]

        # crop data to last rectangle
        self.data_roi = raw_data[int(p1.y()):int(p2.y()), int(p1.x()):int(p2.x())]

    def compute_data(self, coords, raw_data, rgb_path, ir_path):
        p1 = coords[0]
        p2 = coords[1]

        # crop data to last rectangle
        self.data_roi = raw_data[int(p1.y()):int(p2.y()), int(p1.x()):int(p2.x())]

        cv_ir = cv_read_all_path(ir_path)
        h_ir, w_ir, _ = cv_ir.shape
        roi_ir = cv_ir[int(p1.y()):int(p2.y()), int(p1.x()):int(p2.x())]

        p1 = (p1.x(), p1.y())
        p2 = (p2.x(), p2.y())

        if self.has_rgb:
            cv_rgb = cv_read_all_path(rgb_path)
            h_rgb, w_rgb, _ = cv_rgb.shape
            scale = w_rgb / w_ir
            crop_tl, crop_tr = get_corresponding_crop_rectangle(p1, p2, scale)
            roi_rgb = cv_rgb[int(crop_tl[1]):int(crop_tr[1]), int(crop_tl[0]):int(crop_tr[0])]
        else:
            roi_rgb = None

        return roi_ir, roi_rgb

    def compute_highlights(self):
        self.tmax = np.amax(self.data_roi)
        self.tmin = np.amin(self.data_roi)
        self.tmean = np.mean(self.data_roi)
        self.tstd = np.std(self.data_roi)

        self.area = self.data_roi.shape[0] * self.data_roi.shape[1]

        # normalized data
        self.th_norm = (self.data_roi - self.tmin) / (self.tmax - self.tmin)

        highlights = [
            ['Size [pxl²]', self.area],
            ['Max. Temp. [°C]', str(self.tmax)],
            ['Min. Temp. [°C]', str(self.tmin)],
            ['Average Temp. [°C]', str(self.tmean)],
            ['Std. Dev. Temp. [°C]', str(self.tstd)]
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

        # print(f'Minimum position: {xmin, ymin}')
        # print(f'Maximum position: {xmax, ymax}')

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


# CROP OPS __________________________________________________
def get_corresponding_crop_rectangle(p1, p2, scale):
    # Scale the coordinates of p1 and p2 to the larger image size
    p1_large = (int(p1[0] * scale), int(p1[1] * scale))
    p2_large = (int(p2[0] * scale), int(p2[1] * scale))

    # Calculate the top-left and bottom-right coordinates of the crop rectangle
    crop_tl = (min(p1_large[0], p2_large[0]), min(p1_large[1], p2_large[1]))
    crop_br = (max(p1_large[0], p2_large[0]), max(p1_large[1], p2_large[1]))

    # print(p1, p2, crop_tl, crop_br)

    return crop_tl, crop_br
