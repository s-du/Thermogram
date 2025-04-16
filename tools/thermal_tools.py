"""Image processing and colormap management for thermal images."""
# Imports
# Standard library imports
import copy
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image, ImageOps, ImageFilter
from matplotlib import cm, pyplot as plt
import matplotlib.colors as mcol
from scipy.signal import argrelextrema
from blend_modes import dodge, multiply, overlay, normal
from tools.core import *

# PySide6 imports
from PyQt6 import QtCore

# Custom libraries
from tools.core import *
from utils.config import config, thermal_config
from utils.logger import info, error, debug, warning
from utils.exceptions import ThermogramError, FileOperationError

# LISTS __________________________________________________
OUT_LIM = ['continuous', 'black', 'white', 'red']
OUT_LIM_MATPLOT = ['c', 'k', 'w', 'r']
POST_PROCESS = ['none', 'denoise - light', 'denoise - medium', 'denoise - strong', 'smooth', 'sharpen', 'sharpen strong', 'sharpen + denoise', 'edge (simple)', 'contours', 'contours blended']

# Add a mapping for common color names to BGR tuples used by OpenCV
BGR_COLORS = {
    'black': (0, 0, 0),
    'white': (255, 255, 255),
    'red': (0, 0, 255),
    'green': (0, 255, 0),
    'blue': (255, 0, 0),
    'yellow': (0, 255, 255),
    'cyan': (255, 255, 0),
    'magenta': (255, 0, 255)
}

EDGE_METHODS = ['Canny', 'Canny-L2', 'SOBEL', 'Laplacian','Cross']
# Predefined Edge Styles
PREDEFINED_EDGE_STYLES = {
    "Subtle White 1": {"method": 3, "color": "white", "bil": True, "blur": False, "blur_size": 3, "opacity": 0.7},
    "Subtle White 2": {"method": 1, "color": "white", "bil": True, "blur": True, "blur_size": 3, "opacity": 0.6},
    "Crystal Clear White":  {"method": 0, "color": "white", "bil": False, "blur": False, "blur_size": 3, "opacity": 0.4},
    "Subtle Black 1": {"method": 3, "color": "black", "bil": True, "blur": False, "blur_size": 3, "opacity": 0.7},
    "Subtle Black 2": {"method": 1, "color": "black", "bil": True, "blur": True, "blur_size": 3, "opacity": 0.6},
    "Crystal Clear Black": {"method": 0, "color": "black", "bil": False, "blur": False, "blur_size": 3, "opacity": 0.4},
    "Highlight Red": {"method": 3, "color": "red", "bil": False, "blur": False, "blur_size": 3, "opacity": 0.7},
    "Highlight Magenta": {"method": 3, "color": "magenta", "bil": False, "blur": False, "blur_size": 3, "opacity": 0.7},
}

EDGE_STYLE_NAMES = ["Custom"] + list(PREDEFINED_EDGE_STYLES.keys())

# List of palette created for Thermogram
CUSTOM_CMAP_REGISTRY = []

def register_custom_cmap(name, colors):
    """
    Register a new colormap.
    - name: str
    - colors: list of RGB tuples (0–255)
    """
    CUSTOM_CMAP_REGISTRY.append((name, colors))

def get_all_custom_cmaps(n_colors=256):
    """
    Returns a dict {name: cmap} for all registered colormaps.
    Applies n_colors only at runtime.
    """
    colormaps = {}
    for name, colors in CUSTOM_CMAP_REGISTRY:
        # Scale to [0, 1]
        scaled = [tuple(np.array(c).astype(np.float32) / 255) for c in colors]
        cmap = mcol.LinearSegmentedColormap.from_list(name, scaled, N=n_colors)
        colormaps[name] = cmap
    return colormaps

register_custom_cmap('Arctic', [(25, 0, 150), (94, 243, 247), (100, 100, 100), (243, 116, 27), (251, 250, 208)])
register_custom_cmap('Iron', [(0, 0, 0), (144, 15, 170), (230, 88, 65), (248, 205, 35), (255, 255, 255)])
register_custom_cmap('Rainbow', [(8, 0, 75), (43, 80, 203), (119, 185, 31), (240, 205, 35), (245, 121, 47),
                                  (236, 64, 100), (240, 222, 203)])
register_custom_cmap('FIJI_Temp', [(70, 0, 115), (70, 0, 151), (70, 0, 217), (57, 27, 255),
                                   (14, 136, 251), (0, 245, 235), (76, 255, 247), (206, 255, 254),
                                   (251, 254, 243), (178, 255, 163), (57, 255, 51), (37, 255, 1),
                                   (162, 255, 21), (242, 241, 43), (255, 175, 37), (255, 70, 16), (255, 0, 0)])
register_custom_cmap('BlueWhiteRed', [(0, 0, 255), (255, 255, 255), (255, 0, 0)])
register_custom_cmap('Fulgurite', [(77, 0, 0), (204, 0, 0), (255, 153, 0), (255, 255, 0), (255, 255, 255)])
register_custom_cmap('Iron Red', [(0, 0, 0), (30, 0, 90), (100, 0, 130), (175, 0, 175),
                                  (255, 70, 0), (255, 130, 0), (255, 200, 0),
                                  (255, 255, 0), (255, 255, 200)])
register_custom_cmap('Hot Iron', [(0, 0, 0), (0, 51, 51), (25, 100, 100), (50, 130, 130),
                                  (120, 175, 80), (190, 215, 40), (255, 255, 0),
                                  (255, 102, 0), (255, 0, 0)])
register_custom_cmap('Medical', [(60, 26, 30), (224, 64, 224), (32, 42, 255), (20, 229, 241),
                                 (8, 192, 95), (0, 160, 0), (255, 255, 0),
                                 (255, 0, 0), (255, 255, 255)])
register_custom_cmap('Arctic2', [(0, 33, 70), (0, 0, 255), (0, 174, 53), (0, 255, 255),
                                 (255, 255, 0), (255, 136, 0), (255, 0, 0),
                                 (255, 25, 255), (255, 25, 255)])
register_custom_cmap('Rainbow1', [(133, 10, 10), (255, 0, 200), (108, 0, 255), (0, 0, 255),
                                  (0, 255, 255), (0, 255, 65), (255, 255, 0),
                                  (255, 160, 0), (255, 0, 0)])
register_custom_cmap('Rainbow2', [(0, 0, 130), (0, 0, 255), (0, 130, 255), (0, 255, 255),
                                  (130, 255, 130), (255, 255, 0), (255, 130, 0),
                                  (255, 0, 0), (130, 0, 0)])
register_custom_cmap('Tint', [(0, 0, 0), (128, 128, 128), (255, 255, 255), (255, 0, 0)])

LIST_CUSTOM_NAMES = ['Arctic',
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

# List of all available palette (shown names)
COLORMAP_NAMES = ['WhiteHot',
                  'BlackHot',
                  'Arctic',
                  'Iron',
                  'Rainbow',
                  'DJI_Fulgurite',
                  'DJI_Iron Red',
                  'DJI_Hot Iron',
                  'DJI_Medical',
                  'DJI_Arctic',
                  'DJI_Rainbow1',
                  'DJI_Rainbow2',
                  'DJI_Tint',
                  'Pyplot_Hot',
                  'Pyplot_BlueToRed',
                  'Pyplot_BlueWhiteRed',
                  'Pyplot_Plasma',
                  'Pyplot_Inferno',
                  'Pyplot_Jet',
                  'Pyplot_GNUPlot2',
                  'Pyplot_Spectral',
                  'Pyplot_Cividis',
                  'Pyplot_Viridis',
                  'FIJI_Temp']

COLORMAPS = ['Greys_r',
             'Greys',
             'Arctic',
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
             'hot',
             'coolwarm',
             'BlueWhiteRed',
             'plasma',
             'inferno',
             'jet',
             'gnuplot2',
             'Spectral_r',
             'cividis',
             'viridis',
             'FIJI_Temp']

# Runner classes
class RunnerSignals(QtCore.QObject):
    progressed = QtCore.pyqtSignal(int)
    messaged = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal()


class RunnerDJI(QtCore.QRunnable):
    def __init__(self, start, stop, out_folder, img_objects, ref_im, edges, edges_params, individual_settings=False,
                 undis=False, zoom=1, naming_type='rename', file_format='PNG', list_of_ir_export=['IR'],
                 list_of_rgb_export=[]):
        super().__init__()

        self.img_objects = img_objects
        self.edges = edges
        self.edges_params = edges_params
        self.ref_im = ref_im

        self.start = start
        self.stop = stop
        self.dest_folder = out_folder

        self.signals = RunnerSignals()
        self.undis = undis
        self.zoom = zoom
        self.naming_type = naming_type
        self.file_format = file_format
        self.list_of_ir_export = list_of_ir_export
        self.list_of_rgb_export = list_of_rgb_export

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
            # get global radiometric parameters
            self.radio_param = ref_im.thermal_param

    def run(self):
        # Create necessary subfolders upfront to avoid repeated I/O operations
        ir_subfolder = os.path.join(self.dest_folder, 'IR_Images')
        os.makedirs(ir_subfolder, exist_ok=True)

        tiff_subfolder = os.path.join(self.dest_folder, 'TIFF_Images') if 'IR_TIF' in self.list_of_ir_export else None
        if tiff_subfolder:
            os.makedirs(tiff_subfolder, exist_ok=True)

        picpic_subfolder = os.path.join(self.dest_folder,
                                        'Pic-in-Pic_Images') if 'PICPIC' in self.list_of_ir_export else None
        if picpic_subfolder:
            os.makedirs(picpic_subfolder, exist_ok=True)

        rgb_subfolder = os.path.join(self.dest_folder, 'RGB') if 'RGB' in self.list_of_rgb_export else None
        if rgb_subfolder:
            os.makedirs(rgb_subfolder, exist_ok=True)

        rgb_crop_subfolder = os.path.join(self.dest_folder,
                                          'RGB_CROP') if 'RGB_CROP' in self.list_of_rgb_export else None
        if rgb_crop_subfolder:
            os.makedirs(rgb_crop_subfolder, exist_ok=True)

        # Define a worker function for parallel execution
        def process_image(i, img):
            if i < 9:
                prefix = '000'
            elif 9 < i < 99:
                prefix = '00'
            elif 99 < i < 999:
                prefix = '0'

            iter = int(i * (self.stop - self.start) / len(self.img_objects))

            self.signals.progressed.emit(self.start + iter)
            self.signals.messaged.emit(f'Processing image {i}/{len(self.img_objects)} with DJI SDK')

            # Naming the file according to naming convention
            if self.naming_type == 'rename':
                name = f'thermal_{prefix}{i}.' + self.file_format
            elif self.naming_type == 'keep_ir':
                img_path = img.path
                _, filename = os.path.split(str(img_path))
                name = filename[:-3] + self.file_format
            elif self.naming_type == 'match_rgb':
                img_path = img.rgb_path_original
                _, filename = os.path.split(str(img_path))
                name = filename[:-3] + self.file_format

            dest_path = os.path.join(ir_subfolder, name)

            # Process raw data for IR image
            process_raw_data(img,
                             dest_path,
                             edges=self.edges,
                             radio_param=self.radio_param,
                             edge_params=self.edges_params,
                             custom_params=self.custom_params,
                             undis=self.undis,
                             zoom=self.zoom,
                             change_shown=False)

            # Export IR TIFF if needed
            if tiff_subfolder:
                dest_path_tiff = os.path.join(tiff_subfolder, name)
                process_raw_data(img,
                                 dest_path_tiff,
                                 edges=self.edges,
                                 radio_param=self.radio_param,
                                 edge_params=self.edges_params,
                                 custom_params=self.custom_params,
                                 export_tif=True,
                                 undis=self.undis,
                                 zoom=self.zoom,
                                 change_shown=False)

            # Export Pic-in-Pic if needed
            if picpic_subfolder:
                picpic_path = os.path.join(picpic_subfolder, name)
                insert_th_in_rgb(img, dest_path, picpic_path, img.drone_model, self.file_format)

            # Export RGB Image if needed
            if rgb_subfolder:
                _, rgb_name = os.path.split(img.rgb_path_original)
                with Image.open(img.rgb_path_original) as rgb_im:
                    exif = rgb_im.getexif()
                    rgb_path = os.path.join(rgb_subfolder, rgb_name[:-3] + self.file_format)
                    rgb_im.save(rgb_path, exif=exif, compress_level=0)

            # Export RGB Crop if needed
            if rgb_crop_subfolder:
                _, crop_name = os.path.split(img.rgb_path)
                rgb_im = cv_read_all_path(img.rgb_path)
                rgb_crop_path = os.path.join(rgb_crop_subfolder, crop_name[:-3] + self.file_format)
                cv_write_all_path(rgb_im, rgb_crop_path, extension=self.file_format)

        # Use ThreadPoolExecutor to process images in parallel
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_image, i, img) for i, img in enumerate(self.img_objects)]
            for future in as_completed(futures):
                future.result()  # To raise any exceptions that occurred during processing

        # Generate legend in the last step
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

        def process_image(i, rgb_path):
            iter_start_time = time.time()  # Track time for each iteration
            iter = i * (self.stop - self.start) / nb_im

            # update progress
            self.signals.progressed.emit(self.start + iter)
            self.signals.messaged.emit(f'Pre-processing image {i}/{nb_im}')

            # Step 1: Reading the image
            cv_rgb_img = cv_read_all_path(rgb_path)

            # Step 3: Crop based on custom parameters
            crop = match_rgb_custom_parameters_zoom(cv_rgb_img, self.drone_model)

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

            # print(f"Iteration {i} - Total iteration time: {time.time() - iter_start_time} seconds")

        with ThreadPoolExecutor() as executor:
            for i, rgb_path in enumerate(self.list_rgb_paths):
                executor.submit(process_image, i, rgb_path)

        self.signals.finished.emit()


# EXPORT FUNCTIONS __________________________________________________
def create_video(images, video_name, fps):
    images.sort()  # Sort the images if needed

    # Determine the width and height from the first image
    frame = cv_read_all_path(images[0])
    height, width, layers = frame.shape

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # For mp4 videos
    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    for image in images:
        video.write(cv_read_all_path(image))

    video.release()


# LENS RELATED METHODS (DRONE SPECIFIC) __________________________________________________
def match_rgb_custom_parameters_zoom(cv_img, drone_model):
    h2, w2 = cv_img.shape[:2]
    w_ir, h_ir = drone_model.dim_undis_ir

    aspect_factor = (w2 / h2) / (
            w_ir / h_ir)  # Aspect ratio adjustment to fit RGB image to IR

    new_h = h2 * aspect_factor

    # Adjust crop size based on zoom factor
    ret_x = int((w2 / drone_model.zoom) / 2)
    ret_y = int((new_h / drone_model.zoom) / 2)

    # Calculate crop bounds
    x1 = int(w2 / 2 + drone_model.x_offset) - ret_x
    x2 = int(w2 / 2 + drone_model.x_offset) + ret_x
    y1 = int(h2 / 2 + drone_model.y_offset) - ret_y
    y2 = int(h2 / 2 + drone_model.y_offset) + ret_y

    # Check if crop bounds go beyond the image dimensions
    pad_top = max(0, -y1)
    pad_bottom = max(0, y2 - h2)
    pad_left = max(0, -x1)
    pad_right = max(0, x2 - w2)

    # Pad the image if necessary
    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        cv_img = cv2.copyMakeBorder(
            cv_img,
            pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT,
            value=[0, 0, 0]  # Black padding
        )

    # Update bounds after padding
    x1 += pad_left
    x2 += pad_left
    y1 += pad_top
    y2 += pad_top

    rgb_dest = cv_img[y1:y2, x1:x2]

    # Handle zoom < 1: padding is needed to match the final size to the infrared image size
    if drone_model.zoom < 1:
        new_h = int(new_h)

        # Calculate the new dimensions for the frame (hosting image) based on the zoom
        frame_w = int(w2 / drone_model.zoom)
        frame_h = int(new_h / drone_model.zoom)

        # Calculate the padding to center the original image within the new frame
        pad_x = int((frame_w - w2) // 2)
        pad_y = int((frame_h - new_h) // 2)

        # Create the larger frame (canvas) with the desired size, filled with black (or any other color you want)
        frame = cv2.copyMakeBorder(
            rgb_dest,
            pad_y, pad_y, pad_x, pad_x,
            cv2.BORDER_CONSTANT,
            value=[0, 0, 0]  # Black padding
        )

        # Set the new image to the frame
        return frame

    else:
        return rgb_dest


# LINES __________________________________________________
def add_lines_from_rgb(path_ir, cv_match_rgb_img, drone_model, mode=1, color='white', bilateral=True, blur=True,
                       blur_size=3, opacity=0.7):
    """
    Adds edge lines extracted from an RGB image to a thermal IR image.

    Args:
        path_ir (str): Path to the thermal IR image file.
        cv_match_rgb_img (numpy.ndarray): OpenCV image object of the matched RGB image.
        drone_model (str): The model of the drone (used for potential model-specific processing).
        mode (int): Edge detection mode (e.g., 0 for Canny, 1 for custom/laplacian-like). Defaults to 1.
        color (str): Color name for the edge lines (e.g., 'white', 'black', 'red'). Defaults to 'white'.
        bilateral (bool): Apply bilateral filter before edge detection. Defaults to True.
        blur (bool): Apply blur effect to the detected edges. Defaults to True.
        blur_size (int): Kernel size for blurring edges. Defaults to 3.
        opacity (float): Opacity of the overlaid edges (0.0 to 1.0). Defaults to 0.7.

    Returns:
        PIL.Image.Image: Thermal image with edge lines overlaid.

    Raises:
        FileOperationError: If the thermal image cannot be opened.
        ThermogramError: For general processing errors.
    """
    try:
        # 1) load IR image
        try:
            ir_img = Image.open(path_ir).convert('RGB')  # Ensure IR image is RGB for blending
        except FileNotFoundError:
            error(f"Thermal image file not found at {path_ir}")
            raise FileOperationError(f"Thermal image file not found at {path_ir}")
        except Exception as e:
            error(f"Error opening thermal image {path_ir}: {e}")
            raise FileOperationError(f"Could not open thermal image {path_ir}")

        # --- Pre-process RGB image (optional blur) ---
        processed_rgb_img = cv_match_rgb_img.copy()

        # --- Resize RGB image to match thermal dimensions (if needed) ---
        if hasattr(drone_model, 'dim_undis_ir'):
            target_dim = drone_model.dim_undis_ir
            if processed_rgb_img.shape[1] != target_dim[0] or processed_rgb_img.shape[0] != target_dim[1]:
                info(
                    f"Resizing RGB image from {processed_rgb_img.shape[1]}x{processed_rgb_img.shape[0]} to {target_dim[0]}x{target_dim[1]} for edge detection")
                processed_rgb_img = cv2.resize(processed_rgb_img, target_dim, interpolation=cv2.INTER_AREA)
        else:
            warning("drone_model does not have 'dim_undis_ir' attribute. Skipping RGB resize.")

        # 2) get edges from the resized RGB image
        edges_img = create_lines(processed_rgb_img, bil=bilateral, mode=mode)

        if blur:
             # Ensure blur_size is odd
            blur_size = blur_size if blur_size % 2 != 0 else blur_size + 1
            info(f'Applying pre-blur to RGB image with kernel size {blur_size}')
            edges_img = cv2.GaussianBlur(edges_img, (blur_size, blur_size), 0)

        # 3) create color mask for edges
        # Convert edges_img (grayscale) to a 3-channel BGR image
        edges_color_mask_bgr = cv2.cvtColor(edges_img, cv2.COLOR_GRAY2BGR)

        # Get the BGR color tuple from the name
        line_color_bgr = BGR_COLORS.get(color.lower(), (255, 255, 255)) # Default to white if color not found

        # Create a boolean mask where edges are detected (non-zero pixels in original grayscale edges_img)
        edge_mask = edges_img > 0

        # Apply the color: Where edge_mask is True, set the color mask to line_color_bgr
        edges_color_mask_bgr[edge_mask] = line_color_bgr
        # Make non-edge areas black
        edges_color_mask_bgr[~edge_mask] = [0, 0, 0]

        # 4) Blending the layers
        # Convert the thermal image PIL object (RGB) to RGBA NumPy array
        ir_img_rgba = ir_img.convert('RGBA')
        background_float = np.array(ir_img_rgba).astype(float)

        # Convert our BGR color mask to RGBA NumPy array
        # First, convert BGR to RGB for standard RGBA format
        edges_color_mask_rgb = cv2.cvtColor(edges_color_mask_bgr, cv2.COLOR_BGR2RGB)
        # Create an alpha channel based on the original edges_img (edges are opaque, others transparent)
        # Use the blurred edges_img for smoother alpha transition if blur was applied
        alpha_channel = edges_img.astype(np.uint8) # Alpha matches edge intensity (0-255)

        # Merge RGB and alpha channel
        edges_rgba = cv2.merge((edges_color_mask_rgb, alpha_channel))

        # Convert resized edge overlay to float for blending
        foreground_float = edges_rgba.astype(float)

        # --- Blending logic using blend_modes library ---
        blend_opacity = max(0.0, min(1.0, opacity))

        # Dodge works well for adding light details, Multiply for dark.
        # Let's stick to the original logic based on color name.
        if color.lower() != 'black':
             blended_float = normal(background_float, foreground_float, blend_opacity)
        else:
             blended_float = multiply(background_float, foreground_float, blend_opacity)

        # --- Post-blending ---
        # Clip values and convert back to uint8
        blended_float = np.clip(blended_float, 0, 255)
        blended_uint8 = blended_float.astype(np.uint8)

        # Convert final RGBA numpy array back to PIL Image (RGB)
        final_image = Image.fromarray(blended_uint8, 'RGBA').convert('RGB')

        debug(f"Edges overlaid with color {color}, opacity {opacity}, mode {mode}")
        return final_image

    except (FileOperationError, ThermogramError) as e:
        # Re-raise known errors
        raise e
    except Exception as e:
        error(f"Unexpected error in add_lines_from_rgb for {path_ir}: {str(e)}")
        import traceback
        traceback.print_exc() # Print stack trace for debugging
        # Attempt to return original IR image as fallback
        try:
            return Image.open(path_ir).convert('RGB')
        except Exception as fallback_e:
            error(f"Fallback failed: Could not return original image {path_ir}: {fallback_e}")
            raise ThermogramError(f"Failed to process edges or return original image for {path_ir}")


def create_lines(cv_img, bil=True, mode=1): # Added mode parameter
    """
    Creates edge lines from an image using specified method.

    Args:
        cv_img (numpy.ndarray): Input image (OpenCV format, assumed BGR or Grayscale).
        bil (bool): Apply bilateral filter before edge detection. Defaults to True.
        mode (int): Edge detection mode (0 for Canny, 1 for Laplacian). Defaults to 1.

    Returns:
        numpy.ndarray: Grayscale image containing detected edges (0-255).
    """
    # Convert to grayscale if necessary
    if len(cv_img.shape) == 3 and cv_img.shape[2] == 3:
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    elif len(cv_img.shape) == 2:
        gray = cv_img # Assume already grayscale
    else:
         error(f"Unsupported image format for edge detection: shape={cv_img.shape}")
         raise ThermogramError("Unsupported image format for edge detection.")

    # Apply bilateral filter for noise reduction while keeping edges sharp
    if bil:
        # Parameters might need tuning based on image characteristics
        gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    # Edge detection method
    if mode == 0: # Canny Edges
        # Canny thresholds might need adjustment. Lower thresholds detect more edges.
        edges = cv2.Canny(gray, threshold1=100, threshold2=200) # Returns binary image (0 or 255)
    elif mode == 1: # Canny with gradient
        edges = cv2.Canny(gray, 100, 200, L2gradient=True)
    elif mode == 2: # Laplacian Method
        # Using CV_16S to avoid overflow, then convert back
        laplacian = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
        edges = cv2.convertScaleAbs(laplacian) # Converts to CV_8U
        # Laplacian highlights regions of rapid intensity change.
        # It might not produce thin lines like Canny. Optional thresholding:
        # _, edges = cv2.threshold(edges, 30, 255, cv2.THRESH_BINARY)
    elif mode == 3:
        scale = 1
        delta = 0
        k_size = 3
        ddepth = cv2.CV_16S

        grad_x = cv2.Sobel(gray, ddepth, 1, 0, ksize=k_size, scale=scale, delta=delta,
                           borderType=cv2.BORDER_DEFAULT)
        grad_y = cv2.Sobel(gray, ddepth, 0, 1, ksize=k_size, scale=scale, delta=delta,
                           borderType=cv2.BORDER_DEFAULT)

        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)

        edges = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    elif mode == 4:  # Cross Operator
        kernelx = np.array([[1, 0], [0, -1]], dtype=np.float32)
        kernely = np.array([[0, 1], [-1, 0]], dtype=np.float32)
        grad_x = cv2.filter2D(gray, cv2.CV_16S, kernelx)
        grad_y = cv2.filter2D(gray, cv2.CV_16S, kernely)
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        edges = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    else:
        info(f"Unrecognized edge detection mode: {mode}. Defaulting to mode 1 (Laplacian).")
        laplacian = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
        edges = cv2.convertScaleAbs(laplacian)

    # Ensure output is a single channel grayscale image (0-255)
    if len(edges.shape) == 3:
        edges = cv2.cvtColor(edges, cv2.COLOR_BGR2GRAY)

    return edges

# THERMAL PROCESSING __________________________________________________
def find_local_extrema(data, order=20, max_points=4):
    maxima = argrelextrema(data, np.greater, order=order)[0]
    minima = argrelextrema(data, np.less, order=order)[0]

    # Sort by intensity and keep top 2 of each
    sorted_maxima = sorted(maxima, key=lambda i: data[i], reverse=True)[:max_points // 2]
    sorted_minima = sorted(minima, key=lambda i: data[i])[:max_points // 2]

    return np.array(sorted_maxima), np.array(sorted_minima)

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


def process_raw_data(img_object, dest_path, edges=False, edge_params=[], radio_param=None, undis=True,
                     custom_params=None, export_tif=False, zoom=1, change_shown=True):
    if custom_params is None:
        custom_params = {}

    if radio_param:
        img_object.update_data_from_param(radio_param, reset_shown_values=change_shown)

    if not img_object.has_data:
        if radio_param:
            img_object.update_data_from_param(radio_param, reset_shown_values=change_shown)
        else:
            img_object.update_data_from_param(img_object.thermal_param, reset_shown_values=change_shown)
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
    if colormap in LIST_CUSTOM_NAMES:
        all_cmaps = get_all_custom_cmaps(n_colors)
        custom_cmap = all_cmaps[colormap]
    else:
        custom_cmap = cm.get_cmap(colormap, n_colors)
    if col_high != 'c':
        custom_cmap.set_over(col_high)
    if col_low != 'c':
        custom_cmap.set_under(col_low)

    thermal_cmap = custom_cmap(thermal_normalized)
    thermal_cmap = np.uint8(thermal_cmap * 255)

    img_thermal = Image.fromarray(thermal_cmap[:, :, [0, 1, 2]])

    if zoom != 1:
        original_width, original_height = img_thermal.size
        new_width = int(original_width * zoom)
        new_height = int(original_height * zoom)
        img_thermal = img_thermal.resize((new_width, new_height),
                                         Image.LANCZOS)  # Use LANCZOS interpolation for high-quality upscaling

    if export_tif:
        dest_path_tif = dest_path[:-4] + '.tiff'
        img_thermal_tiff = Image.fromarray(im)  # export as 32bit array (floating point)
        img_thermal_tiff.save(dest_path_tif)
        # adding exif
        subprocess.run(["resources/exiftool/exiftool.exe", "-overwrite_original", "-TagsFromFile", img_object.rgb_path_original, dest_path_tif])


    elif post_process == 'none':
        if dest_path.endswith('JPG'):
            img_thermal.save(dest_path, exif=exif, quality=99)
        elif dest_path.endswith('PNG'):
            img_thermal.save(dest_path, exif=exif, compression_level=0)
    elif post_process in ['denoise - light', 'denoise - medium', 'denoise - strong']:
        if post_process == 'denoise - light':
            denoise_factor = 3
        elif post_process == 'denoise - medium':
            denoise_factor = 5
        else:
            denoise_factor = 10
        # Convert PIL image to OpenCV format (NumPy array, BGR)
        img_cv = np.array(img_thermal)[:, :, ::-1]  # RGB to BGR

        # Apply OpenCV's fast non-local means denoising
        denoised = cv2.fastNlMeansDenoisingColored(img_cv, None, h=denoise_factor, hColor=denoise_factor, templateWindowSize=7,
                                                   searchWindowSize=21)

        # Convert back to PIL (RGB)
        denoised_rgb = denoised[:, :, ::-1]  # BGR to RGB
        img_denoised = Image.fromarray(denoised_rgb)

        if dest_path.endswith('JPG'):
            img_denoised.save(dest_path, exif=exif, quality=99)
        elif dest_path.endswith('PNG'):
            img_denoised.save(dest_path, exif=exif, compression_level=0)
    elif post_process == 'sharpen'or post_process == 'sharpen + denoise':
        img_th_sharpened = img_thermal.filter(ImageFilter.SHARPEN)

        if post_process == 'sharpen + denoise':
            img_cv = np.array(img_th_sharpened)
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
            # Apply OpenCV's fast non-local means denoising
            denoised = cv2.fastNlMeansDenoisingColored(img_cv, None, h=5, hColor=5,
                                                       templateWindowSize=7,
                                                       searchWindowSize=21)

            # Convert back to PIL (RGB)
            denoised_rgb = denoised[:, :, ::-1]  # BGR to RGB
            img_th_sharpened = Image.fromarray(denoised_rgb)
        if dest_path.endswith('JPG'):
            img_th_sharpened.save(dest_path, exif=exif, quality=99)
        elif dest_path.endswith('PNG'):
            img_th_sharpened.save(dest_path, exif=exif, compression_level=0)
    elif post_process == 'sharpen strong':
        img_th_sharpened = img_thermal.filter(ImageFilter.SHARPEN)
        img_th_sharpened2 = img_th_sharpened.filter(ImageFilter.SHARPEN)

        if dest_path.endswith('JPG'):
            img_th_sharpened2.save(dest_path, exif=exif, quality=99)
        elif dest_path.endswith('PNG'):
            img_th_sharpened2.save(dest_path, exif=exif, compression_level=0)

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
        if dest_path.endswith('JPG'):
            blended_img_raw.save(dest_path, exif=exif, quality=99)
        elif dest_path.endswith('PNG'):
            blended_img_raw.save(dest_path, exif=exif, compression_level=0)
    elif post_process == 'smooth':
        img_th_smooth = img_thermal.filter(ImageFilter.SMOOTH)
        if dest_path.endswith('JPG'):
            img_th_smooth.save(dest_path, exif=exif, quality=99)
        elif dest_path.endswith('PNG'):
            img_th_smooth.save(dest_path, exif=exif, compression_level=0)
    elif post_process == 'contours' or post_process == 'contours blended':
        print('go contours')
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
            cv2.drawContours(contour_canvas, contours, -1, color, 2)  # Change thickness as needed

        # Blend the contour canvas with the original thermal image
        blended_img = cv2.addWeighted(np.array(img_thermal), 0.3, contour_canvas, 0.7, 0)

        # Save the blended image
        if post_process == 'contours':
            img_with_contours = Image.fromarray(contour_canvas)
        else:
            img_with_contours = Image.fromarray(blended_img)
        if dest_path.endswith('JPG'):
            img_with_contours.save(dest_path, exif=exif, quality=99)
        elif dest_path.endswith('PNG'):
            img_with_contours.save(dest_path, exif=exif, compression_level=0)

    # check if edges need to be added (parameter 'edge' is a boolean)
    if edges:
        ed_met, ed_col, ed_bil, ed_blur, ed_bl_sz, ed_op = edge_params
        drone_model_name = get_drone_model_from_exif(exif)
        drone_model = DroneModel(drone_model_name)
        cv_match_rgb_img = cv_read_all_path(img_object.rgb_path)

        # Use the extracted mode directly
        edged_image = add_lines_from_rgb(dest_path, cv_match_rgb_img, drone_model, mode=ed_met, color=ed_col,
                                         bilateral=ed_bil, blur=ed_blur, blur_size=ed_bl_sz, opacity=ed_op)

        if exif is None:
            if dest_path.endswith('JPG'):
                edged_image.save(dest_path, exif=exif, quality=99)
            elif dest_path.endswith('PNG'):
                edged_image.save(dest_path, exif=exif, compression_level=0)
        else:
            if dest_path.endswith('JPG'):
                edged_image.save(dest_path, exif=exif, quality=99)
            elif dest_path.endswith('PNG'):
                edged_image.save(dest_path, exif=exif, compression_level=0)


def resize_and_pad(img, target_size):
    target_w, target_h = target_size
    h, w = img.shape[:2]

    # Calculate scaling factor to maintain aspect ratio
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize image
    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Pad image to match target size
    delta_w = target_w - new_w
    delta_h = target_h - new_h
    top = delta_h // 2
    bottom = delta_h - top
    left = delta_w // 2
    right = delta_w - left

    padded_img = cv2.copyMakeBorder(resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return padded_img


def process_th_image_with_zoom(img_obj, out_folder, theta, replace_rgb_with_th=False):
    # read images
    cv_rgb = cv_read_all_path(img_obj.rgb_path_original)  # read rgb image
    h_rgb, w_rgb = cv_rgb.shape[:2]
    process_raw_data(img_obj, os.path.join(out_folder, 'IR.JPG'), edges=False)  # read infrared image (undistorded)
    cv_ir = cv_read_all_path(os.path.join(out_folder, 'IR.JPG'))

    h_ir, w_ir = cv_ir.shape[:2]  # get resulting dimension of thermal image
    dim_undis_ir = (w_ir, h_ir)  # The size we need to match

    zoom = theta[0]  # zoom factor
    y_offset = theta[1]  # vertical shift
    x_offset = theta[2]  # horizontal shift

    if h_ir == 0:
        aspect_factor = 1
    else:
        aspect_factor = (w_rgb / h_rgb) / (
                w_ir / h_ir)  # Aspect ratio adjustment to fit RGB image to IR

    new_h = h_rgb * aspect_factor

    # Adjust crop size based on zoom factor
    ret_x = int((w_rgb / zoom) / 2)
    ret_y = int((new_h / zoom) / 2)

    # Calculate crop bounds
    x1 = int(w_rgb / 2 + x_offset) - ret_x
    x2 = int(w_rgb / 2 + x_offset) + ret_x
    y1 = int(h_rgb / 2 + y_offset) - ret_y
    y2 = int(h_rgb / 2 + y_offset) + ret_y

    # Check if crop bounds go beyond the image dimensions
    pad_top = max(0, -y1)
    pad_bottom = max(0, y2 - h_rgb)
    pad_left = max(0, -x1)
    pad_right = max(0, x2 - w_rgb)

    # Pad the image if necessary
    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        cv_rgb = cv2.copyMakeBorder(
            cv_rgb,
            pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT,
            value=[0, 0, 0]  # Black padding
        )

    # Update bounds after padding
    x1 += pad_left
    x2 += pad_left
    y1 += pad_top
    y2 += pad_top

    rgb_crop = cv_rgb[y1:y2, x1:x2]

    if replace_rgb_with_th:
        cv_rgb_copy = copy.deepcopy(cv_rgb)
        # Get the dimensions of the RGB cropped region
        target_h, target_w = rgb_crop.shape[:2]

        # Resize the infrared image to match the cropped RGB region dimensions
        cv_ir_resized = cv2.resize(cv_ir, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        # Replace the RGB crop with the resized infrared image
        cv_rgb_copy[y1:y2, x1:x2] = cv_ir_resized

    # Handle zoom < 1: padding is needed to match the final size to the infrared image size
    if zoom < 1:
        # Resize cropped region according to zoom
        rgb_crop_resized = cv2.resize(rgb_crop, (int(w_ir * zoom), int(h_ir * zoom)))

        # Calculate padding required to match IR image size
        pad_x = max(0, (dim_undis_ir[0] - rgb_crop_resized.shape[1]) // 2)
        pad_y = max(0, (dim_undis_ir[1] - rgb_crop_resized.shape[0]) // 2)

        # Pad the resized image to fit the infrared image size
        rgb_dest = cv2.copyMakeBorder(rgb_crop_resized, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT,
                                      value=[0, 0, 0])

        # Ensure the final padded image is exactly the size of the IR image
        rgb_dest = cv2.resize(rgb_dest, (w_ir, h_ir))  # Trim any excess padding to fit IR dimensions

    else:
        # Resize the cropped image directly to the IR dimensions if zoom >= 1 (no padding needed)
        rgb_dest = cv2.resize(rgb_crop, dim_undis_ir, interpolation=cv2.INTER_AREA)

    # Save resized or padded image
    cv_write_all_path(rgb_dest, os.path.join(out_folder, 'rescale.JPG'))

    # Create lines and save both the RGB and IR versions
    lines_rgb = create_lines(rgb_dest)
    cv_write_all_path(lines_rgb, os.path.join(out_folder, 'rgb_lines.JPG'))

    lines_ir = create_lines(cv_ir, bil=False)
    cv_write_all_path(lines_ir, os.path.join(out_folder, 'ir_lines.JPG'))

    # Compute difference
    diff = cv2.subtract(lines_ir, lines_rgb)
    err = np.sum(diff ** 2)
    mse = err / (float(h_ir * w_ir))

    print(f'mse is {mse}')

    return mse

def insert_th_in_rgb_fast(img_obj, ir_path, dest_path, drone_model, extension):
    # Read RGB and IR images
    cv_rgb = cv_read_all_path(img_obj.rgb_path_original)
    cv_ir = cv_read_all_path(ir_path)

    h_rgb, w_rgb = cv_rgb.shape[:2]
    h_ir, w_ir = cv_ir.shape[:2]

    if h_ir == 0:
        aspect_factor = 1
    else:
        aspect_factor = (w_rgb / h_rgb) / (w_ir / h_ir)

    new_h = h_rgb * aspect_factor

    # Zoom and offset params
    zoom = drone_model.zoom
    x_offset = drone_model.x_offset
    y_offset = drone_model.y_offset

    # Compute crop center and size
    ret_x = int((w_rgb / zoom) / 2)
    ret_y = int((new_h / zoom) / 2)
    center_x = int(w_rgb / 2 + x_offset)
    center_y = int(h_rgb / 2 + y_offset)

    # Determine crop bounds
    x1, x2 = center_x - ret_x, center_x + ret_x
    y1, y2 = center_y - ret_y, center_y + ret_y

    # Calculate required padding
    pad_left = max(0, -x1)
    pad_right = max(0, x2 - w_rgb)
    pad_top = max(0, -y1)
    pad_bottom = max(0, y2 - h_rgb)

    if pad_top or pad_bottom or pad_left or pad_right:
        cv_rgb = cv2.copyMakeBorder(cv_rgb, pad_top, pad_bottom, pad_left, pad_right,
                                    cv2.BORDER_CONSTANT, value=[0, 0, 0])
        center_x += pad_left
        center_y += pad_top
        x1, x2 = center_x - ret_x, center_x + ret_x
        y1, y2 = center_y - ret_y, center_y + ret_y

    # Crop region
    rgb_crop = cv_rgb[y1:y2, x1:x2]

    # Resize IR image to match the crop
    target_h, target_w = rgb_crop.shape[:2]
    cv_ir_resized = cv2.resize(cv_ir, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    # Replace region in RGB image
    cv_rgb[y1:y2, x1:x2] = cv_ir_resized

    # Create and apply mask
    mask = np.zeros((cv_rgb.shape[0], cv_rgb.shape[1]), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255

    # Write outputs
    cv_write_all_path(cv_rgb, dest_path)
    mask_path = os.path.splitext(dest_path)[0] + "_mask." + extension
    cv_write_all_path(mask, mask_path)
def insert_th_in_rgb(img_obj, ir_path, dest_path, drone_model, extension):
    # read images
    cv_rgb = cv_read_all_path(img_obj.rgb_path_original)  # read rgb image
    h_rgb, w_rgb = cv_rgb.shape[:2]
    cv_ir = cv_read_all_path(ir_path)

    h_ir, w_ir = cv_ir.shape[:2]  # get resulting dimension of thermal image

    zoom = drone_model.zoom  # zoom factor
    y_offset = drone_model.y_offset  # vertical shift
    x_offset = drone_model.x_offset  # horizontal shift

    if h_ir == 0:
        aspect_factor = 1
    else:
        aspect_factor = (w_rgb / h_rgb) / (
                w_ir / h_ir)  # Aspect ratio adjustment to fit RGB image to IR

    new_h = h_rgb * aspect_factor

    # Adjust crop size based on zoom factor
    ret_x = int((w_rgb / zoom) / 2)
    ret_y = int((new_h / zoom) / 2)

    # Calculate crop bounds
    x1 = int(w_rgb / 2 + x_offset) - ret_x
    x2 = int(w_rgb / 2 + x_offset) + ret_x
    y1 = int(h_rgb / 2 + y_offset) - ret_y
    y2 = int(h_rgb / 2 + y_offset) + ret_y

    mask = np.zeros((h_rgb, w_rgb), dtype=np.uint8)

    # Check if crop bounds go beyond the image dimensions
    pad_top = max(0, -y1)
    pad_bottom = max(0, y2 - h_rgb)
    pad_left = max(0, -x1)
    pad_right = max(0, x2 - w_rgb)

    # Pad the image if necessary
    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        cv_rgb = cv2.copyMakeBorder(
            cv_rgb,
            pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT,
            value=[0, 0, 0]  # Black padding
        )
        mask = cv2.copyMakeBorder(
            mask,
            pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT,
            value=0  # Black padding
        )

    # Update bounds after padding
    x1 += pad_left
    x2 += pad_left
    y1 += pad_top
    y2 += pad_top

    rgb_crop = cv_rgb[y1:y2, x1:x2]

    # Get the dimensions of the RGB cropped region
    target_h, target_w = rgb_crop.shape[:2]

    # Resize the infrared image to match the cropped RGB region dimensions
    cv_ir_resized = cv2.resize(cv_ir, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    # Replace the RGB crop with the resized infrared image
    cv_rgb[y1:y2, x1:x2] = cv_ir_resized

    # Update the mask: Set the inserted region to white (255)
    mask[y1:y2, x1:x2] = 255

    # Save resized or padded image
    cv_write_all_path(cv_rgb, dest_path)

    # Save the mask with a `_mask` suffix
    mask_path = os.path.splitext(dest_path)[0] + "_mask." + extension
    cv_write_all_path(mask, mask_path)


def generate_legend(legend_dest_path, custom_params):
    tmin = custom_params["tmin"]
    tmax = custom_params["tmax"]
    color_high = custom_params["col_high"]
    color_low = custom_params["col_low"]
    colormap = custom_params["colormap"]
    n_colors = custom_params["n_colors"]

    fig, ax = plt.subplots()
    data = np.clip(np.random.randn(10, 10) * 100, tmin, tmax)
    # print(data)

    if colormap in LIST_CUSTOM_NAMES:
        all_cmaps = get_all_custom_cmaps(n_colors)
        custom_cmap = all_cmaps[colormap]
    else:
        custom_cmap = cm.get_cmap(colormap, n_colors)

    if color_high != 'c':
        custom_cmap.set_over(color_high)
    if color_low != 'c':
        custom_cmap.set_under(color_low)

    cax = ax.imshow(data, cmap=custom_cmap)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    if n_colors > 12:
        n_colors = 12
    ticks = np.linspace(tmin, tmax, n_colors + 1, endpoint=True)
    fig.colorbar(cax, ticks=ticks, extend='both')
    ax.remove()

    plt.savefig(legend_dest_path, bbox_inches='tight')


def create_vector_plot(img_obj):
    thermal_image = img_obj.raw_data_undis
    n_colors = img_obj.n_colors
    tmin = img_obj.tmin_shown
    tmax = img_obj.tmax_shown
    custom_cmap = cm.get_cmap('Greys_r', n_colors)

    """
    if colormap in LIST_CUSTOM_CMAPS:
        custom_cmap = get_custom_cmaps(colormap, n_colors)
    else:
        custom_cmap = cm.get_cmap(colormap, n_colors)
    """
    # Step 2: Calculate the gradients using Sobel filters.
    # Use OpenCV's Sobel function to get the x and y gradients.
    grad_x = cv2.Sobel(thermal_image, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(thermal_image, cv2.CV_64F, 0, 1, ksize=5)

    # Step 3: Normalize the gradients to keep the arrows at a consistent length.
    # Normalize grad_x and grad_y to a range between -1 and 1.
    grad_x_normalized = -grad_x / (np.max(np.abs(grad_x)) + 1e-5)
    grad_y_normalized = -grad_y / (np.max(np.abs(grad_y)) + 1e-5)

    # Step 4: Calculate the magnitude and direction of the gradients.
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

    # Step 5: Downsample the gradients for easier visualization.
    # This helps reduce the number of arrows, making the visualization clearer.
    step = 5  # You can adjust this value to change arrow density
    X, Y = np.meshgrid(np.arange(0, thermal_image.shape[1], step),
                       np.arange(0, thermal_image.shape[0], step))

    U = grad_x_normalized[::step, ::step]  # Normalized gradient in x-direction
    V = grad_y_normalized[::step, ::step]  # Normalized gradient in y-direction
    scaling_factor = 30  # Adjust this factor to make arrows larger or smaller as needed
    U *= scaling_factor
    V *= scaling_factor

    M = magnitude[::step, ::step]  # Magnitude of gradients for visualization

    # Step 6: Plot the quiver plot.

    plt.imshow(thermal_image, cmap=custom_cmap, origin='upper', vmin=tmin, vmax=tmax)  # Display thermal image
    plt.colorbar(label='Temperature (°C)')  # Add a colorbar to indicate temperature

    # Plot the quiver plot (vector arrows representing gradient)
    # Use 'M' (magnitude) to color-code the arrows for better visualization of gradient strength.
    # Adjust the 'scale' parameter to make arrows larger.
    plt.quiver(X, Y, U, V, M, angles='xy', scale_units='xy', scale=1, cmap='cool', headwidth=3, headlength=5)

    plt.title('Temperature Gradient Analysis')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    # plt.grid()
    plt.show()
