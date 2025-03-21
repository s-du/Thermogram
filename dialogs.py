# Standard library imports
import logging
import os
import sys
import traceback
from shutil import copyfile

# Third-party imports
import cv2
from matplotlib import cm
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT
from matplotlib.figure import Figure
import numpy as np
from PIL import Image
import qimage2ndarray

# PyQt6 imports
from PyQt6 import QtCore, QtGui, QtWidgets  # Changed from PySide6
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *
from PyQt6.uic import loadUi  # PyQt6 has uic to load UI files

# Custom library imports
import widgets as wid
import resources as res
from tools import thermal_tools as tt
from tools import ai_tools as ai
from scipy.ndimage import maximum_filter, minimum_filter, zoom

# basic logger functionality
log = logging.getLogger(__name__)
handler = logging.StreamHandler(stream=sys.stdout)
log.addHandler(handler)

CLASS_COLORS = {
    0: QColor(255, 0, 0),   # Red for class 0
    1: QColor(0, 255, 0),   # Green for class 1
    2: QColor(0, 0, 255),   # Blue for class 2
    3: QColor(255, 255, 0), # Yellow for class 3
    4: QColor(255, 0, 255), # Magenta for class 4
    # Add more colors as needed for additional classes
}


def show_exception_box(log_msg):
    """Checks if a QApplication instance is available and shows a messagebox with the exception message.
    If unavailable (non-console application), log an additional notice.
    """
    if QtWidgets.QApplication.instance() is not None:
        errorbox = QtWidgets.QMessageBox()
        errorbox.setText("Oops. An unexpected error occured:\n{0}".format(log_msg))
        errorbox.exec()
    else:
        log.debug("No QApplication instance available.")


class UncaughtHook(QtCore.QObject):
    _exception_caught = QtCore.pyqtSignal(object)

    def __init__(self, *args, **kwargs):
        super(UncaughtHook, self).__init__(*args, **kwargs)

        # this registers the exception_hook() function as hook with the Python interpreter
        sys.excepthook = self.exception_hook

        # connect signal to execute the message box function always on main thread
        self._exception_caught.connect(show_exception_box)

    def exception_hook(self, exc_type, exc_value, exc_traceback):
        """Function handling uncaught exceptions.
        It is triggered each time an uncaught exception occurs.
        """
        if issubclass(exc_type, KeyboardInterrupt):
            # ignore keyboard interrupt to support console applications
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
        else:
            exc_info = (exc_type, exc_value, exc_traceback)
            log_msg = '\n'.join([''.join(traceback.format_tb(exc_traceback)),
                                 '{0}: {1}'.format(exc_type.__name__, exc_value)])
            log.critical("Uncaught exception:\n {0}".format(log_msg), exc_info=exc_info)

            # trigger message box show
            self._exception_caught.emit(log_msg)


# create a global instance of our class to register the hook
qt_exception_hook = UncaughtHook()


class AboutDialog(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('What is this app about?')
        self.setFixedSize(300, 300)
        self.layout = QtWidgets.QVBoxLayout()

        about_text = QtWidgets.QLabel(
            'The Thermogram app was made to simplify the analysis of thermal images. Any question/remark: samuel.dubois@buildwise.be')
        about_text.setWordWrap(True)

        logos1 = QtWidgets.QLabel()
        pixmap = QtGui.QPixmap(res.find('img/logo_buildwise2.png'))
        w = self.width()
        pixmap = pixmap.scaledToWidth(100, QtCore.Qt.TransformationMode.SmoothTransformation)
        logos1.setPixmap(pixmap)

        self.layout.addWidget(about_text)
        self.layout.addWidget(logos1, alignment=QtCore.Qt.AlignmentFlag.AlignCenter)

        self.setLayout(self.layout)


class CustomGraphicsItem(QGraphicsItem):
    def __init__(self, pixmap, target_size=None, parent=None):
        super().__init__(parent)
        self.pixmap = pixmap
        self.opacity = 1.0
        self.composition_mode = QtGui.QPainter.CompositionMode.CompositionMode_SourceOver
        self.target_size = target_size

        if self.target_size:
            self.pixmap = self.pixmap.scaled(self.target_size, QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                                             QtCore.Qt.TransformationMode.SmoothTransformation)

    def boundingRect(self):
        return QRectF(self.pixmap.rect())

    def paint(self, painter, option, widget=None):
        painter.setOpacity(self.opacity)
        painter.setCompositionMode(self.composition_mode)
        painter.drawPixmap(0, 0, self.pixmap)

    def setPixmap(self, pixmap, rescale=True):
        self.pixmap = pixmap
        if rescale:
            self.pixmap = self.pixmap.scaled(self.target_size, QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                                             QtCore.Qt.TransformationMode.SmoothTransformation)
        self.update()


class HotSpotDialog(QDialog):
    def __init__(self, thermal_image, raw_data, parent=None):
        super().__init__(parent)
        self.raw_data = raw_data  # Store the raw temperature data (NumPy array)
        self.thermal_image = thermal_image  # Path or QImage to the thermal JPEG
        self.normalized_data = self.normalize_data(self.raw_data)  # Normalize raw data to [0, 255]
        self.prominence = 80  # Default prominence factor (relative to 0-255 scale)
        self.structure_size = 35  # Default structure size
        self.show_labels = True  # Show temperature labels by default
        self.exclude_edges = False  # Whether to exclude maxima and minima at the edges
        self.selection_mode = 'both'  # Default is to show both maxima and minima

        # List to keep track of spot measurements (both maxima and minima)
        self.spot_measurements = []

        # pen
        self.pen = QPen()
        # self.pen.setStyle(Qt.DashDotLine)
        self.pen.setWidth(2)

        # Layout setup
        self.setWindowTitle("Detect Temperature Hot Spots")
        self.layout = QVBoxLayout()

        # Create a QGraphicsView to show the thermal image
        self.graphics_view = QGraphicsView()
        self.scene = QGraphicsScene(self)
        self.graphics_view.setScene(self.scene)
        self.layout.addWidget(self.graphics_view)

        # Display the image
        self.display_image()

        # Slider to adjust the prominence factor
        self.slider_label = QLabel("Prominence Factor: 80")
        self.layout.addWidget(self.slider_label)

        self.prominence_slider = QSlider(Qt.Orientation.Horizontal)
        self.prominence_slider.setMinimum(1)
        self.prominence_slider.setMaximum(255)  # Adjust as needed
        self.prominence_slider.setValue(80)
        self.prominence_slider.valueChanged.connect(self.update_prominence)
        self.layout.addWidget(self.prominence_slider)

        # Slider to adjust the structure size
        self.structure_slider_label = QLabel("Structure Size: 35")
        self.layout.addWidget(self.structure_slider_label)

        self.structure_slider = QSlider(Qt.Orientation.Horizontal)
        self.structure_slider.setMinimum(5)
        self.structure_slider.setMaximum(100)  # Structure size from 1 to 10
        self.structure_slider.setValue(35)
        self.structure_slider.valueChanged.connect(self.update_structure_size)
        self.layout.addWidget(self.structure_slider)

        # ComboBox to choose between maxima, minima, or both
        self.combo_box_label = QLabel("Show:")
        self.layout.addWidget(self.combo_box_label)

        self.combo_box = QComboBox()
        self.combo_box.addItem("Both")
        self.combo_box.addItem("Only Maxima")
        self.combo_box.addItem("Only Minima")
        self.combo_box.currentTextChanged.connect(self.update_selection_mode)
        self.layout.addWidget(self.combo_box)

        # Checkbox to toggle temperature labels
        self.show_labels_checkbox = QCheckBox("Show Temperature Labels")
        self.show_labels_checkbox.setChecked(True)  # Default is checked
        self.show_labels_checkbox.stateChanged.connect(self.toggle_labels)
        self.layout.addWidget(self.show_labels_checkbox)

        # Checkbox for excluding edge maxima and minima
        self.exclude_edges_checkbox = QCheckBox("Exclude Edge Maxima and Minima")
        self.exclude_edges_checkbox.setChecked(False)
        self.exclude_edges_checkbox.stateChanged.connect(self.toggle_edge_exclusion)
        self.layout.addWidget(self.exclude_edges_checkbox)

        self.setLayout(self.layout)

    def normalize_data(self, raw_data):
        """Normalize the raw temperature data to the 0-255 range."""
        min_val = np.min(raw_data)
        max_val = np.max(raw_data)
        normalized = 255 * (raw_data - min_val) / (max_val - min_val)
        return normalized

    def display_image(self):
        # Convert thermal_image to a QPixmap and display it in the QGraphicsView
        if isinstance(self.thermal_image, str):  # If it's a path, load the image
            image = QImage(self.thermal_image)
        elif isinstance(self.thermal_image, QImage):
            image = self.thermal_image
        else:
            raise ValueError("Invalid thermal image format")

        pixmap = QPixmap.fromImage(image)
        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(self.pixmap_item)  # Add the pixmap to the scene

        # Detect local maxima and minima and display them on the image
        self.detect_hotspots()

    def update_prominence(self):
        # Update prominence factor and refresh hot spot detection
        self.prominence = self.prominence_slider.value()
        self.slider_label.setText(f"Prominence Factor: {self.prominence}")
        self.detect_hotspots()

    def update_structure_size(self):
        # Update structure size and refresh hot spot detection
        self.structure_size = self.structure_slider.value()
        self.structure_slider_label.setText(f"Structure Size: {self.structure_size}")
        self.detect_hotspots()

    def update_selection_mode(self, mode):
        """Update whether to show maxima, minima, or both."""
        if mode == "Only Maxima":
            self.selection_mode = 'maxima'
        elif mode == "Only Minima":
            self.selection_mode = 'minima'
        else:
            self.selection_mode = 'both'
        self.detect_hotspots()

    def toggle_labels(self):
        # Toggle visibility of temperature labels
        self.show_labels = self.show_labels_checkbox.isChecked()
        for spot in self.spot_measurements:
            if spot.text_item:
                spot.text_item.setVisible(self.show_labels)

    def toggle_edge_exclusion(self):
        # Toggle exclusion of edge maxima and minima
        self.exclude_edges = self.exclude_edges_checkbox.isChecked()
        self.detect_hotspots()

    def detect_hotspots(self):
        # Clear previous spots from the scene
        self.clear_spot_measurements()

        # Create a square structure based on the current structure size
        structure = np.ones((self.structure_size, self.structure_size), dtype=bool)

        # Detect local maxima
        local_max = (self.normalized_data == maximum_filter(self.normalized_data, footprint=structure))

        # Detect local minima (inverted maxima detection)
        local_min = (self.normalized_data == minimum_filter(self.normalized_data, footprint=structure))

        # Exclude maxima and minima near the edges if requested
        if self.exclude_edges:
            local_max[0, :] = False
            local_max[-1, :] = False
            local_max[:, 0] = False
            local_max[:, -1] = False

            local_min[0, :] = False
            local_min[-1, :] = False
            local_min[:, 0] = False
            local_min[:, -1] = False

        # Get indices of the detected local maxima
        maxima_indices = np.argwhere(local_max)

        # Get indices of the detected local minima
        minima_indices = np.argwhere(local_min)

        # Filter maxima and minima based on prominence
        filtered_maxima = self.filter_by_prominence(maxima_indices, is_maxima=True)
        filtered_minima = self.filter_by_prominence(minima_indices, is_maxima=False)

        # Based on the selection mode, display only maxima, only minima, or both
        if self.selection_mode in ['maxima', 'both']:
            # Add markers for the filtered maxima (red)
            for maximum in filtered_maxima:
                x, y = maximum
                self.add_spot(x, y, color=Qt.GlobalColor.red)

        if self.selection_mode in ['minima', 'both']:
            # Add markers for the filtered minima (blue)
            for minimum in filtered_minima:
                x, y = minimum
                self.add_spot(x, y, color=Qt.GlobalColor.blue)

    def filter_by_prominence(self, extrema_indices, is_maxima=True):
        """Filter detected maxima or minima based on the prominence value."""
        filtered_extrema = []
        for point in extrema_indices:
            x, y = point
            point_value = self.normalized_data[x, y]

            # Get the neighborhood around the point based on structure size
            min_surrounding_value, max_surrounding_value = self.get_surrounding_extrema(x, y, self.structure_size)

            # Calculate prominence
            if is_maxima:
                prominence = point_value - min_surrounding_value  # Prominence for maxima
            else:
                prominence = max_surrounding_value - point_value  # Prominence for minima

            if prominence >= self.prominence:
                filtered_extrema.append(point)

        return filtered_extrema

    def get_surrounding_extrema(self, x, y, structure_size):
        """Calculate the minimum and maximum values around a point based on the structure size."""
        neighbors = []

        # Adjust the neighborhood size based on structure size
        for i in range(max(0, x - structure_size), min(self.raw_data.shape[0], x + structure_size + 1)):
            for j in range(max(0, y - structure_size), min(self.raw_data.shape[1], y + structure_size + 1)):
                if (i, j) != (x, y):  # Exclude the point itself
                    neighbors.append(self.normalized_data[i, j])

        if neighbors:
            return min(neighbors), max(neighbors)
        else:
            return self.normalized_data[x, y], self.normalized_data[x, y]

    def add_spot(self, x, y, color):
        # Create a QPointF for the spot location
        qpoint = QPointF(y, x)

        # Create a PointMeas object
        spot = tt.PointMeas(qpoint)
        spot.temp = self.raw_data[x, y]  # Assign the temperature value

        # Create the ellipse and text items
        spot.create_items()

        # Set ellipse color
        self.pen.setColor(color)
        spot.ellipse_item.setPen(self.pen)

        # Add ellipse and text to the scene
        self.scene.addItem(spot.ellipse_item)
        if self.show_labels:
            self.scene.addItem(spot.text_item)

        # Add all items to the scene for future reference
        spot.all_items.append(spot.ellipse_item)
        spot.all_items.append(spot.text_item)

        # Store the measurement in the list
        self.spot_measurements.append(spot)

    def clear_spot_measurements(self):
        # Remove all previously added items from the scene
        for spot in self.spot_measurements:
            for item in spot.all_items:
                self.scene.removeItem(item)
        self.spot_measurements.clear()



class ImageFusionDialog(QtWidgets.QDialog):
    """
    Dialog that allows the user to choose advances thermography options
    """

    def __init__(self, img_object, ir_img_path, dest_path_preview):
        super().__init__()
        basepath = os.path.dirname(__file__)
        basename = 'fusion'
        uifile = os.path.join(basepath, 'ui/%s.ui' % basename)
        loadUi(uifile, self)

        self.setWindowTitle("Create a custom overlay")

        self.ir_path = ir_img_path
        self.img_object = img_object
        self.colormap_name = img_object.colormap
        self.n_colors = img_object.n_colors
        self.temperatures = img_object.raw_data_undis

        self.dest_path_preview = dest_path_preview
        self.scene = QGraphicsScene()
        self.view.setScene(self.scene)
        self.image_to_export = None

        min_temp_scaled = self.scale_temperature(img_object.tmin)
        max_temp_scaled = self.scale_temperature(img_object.tmax)
        min_shown_temp_scaled = self.scale_temperature(img_object.tmin_shown)
        max_shown_temp_scaled = self.scale_temperature(img_object.tmax_shown)
        if min_shown_temp_scaled < min_temp_scaled:
            min_shown_temp_scaled = min_temp_scaled
        if max_shown_temp_scaled > max_temp_scaled:
            max_shown_temp_scaled = max_temp_scaled

        # range slider for shown data
        self.range_slider_shown = wid.QRangeSlider()
        self.range_slider_shown.handleWidth = 15
        self.range_slider_shown.handleHeight = 20
        self.range_slider_shown.trackHeight = 20
        self.range_slider_shown.back_color = QColor(220, 220, 220)
        self.range_slider_shown.setLowerValue(min_shown_temp_scaled)
        self.range_slider_shown.setUpperValue(max_shown_temp_scaled)
        self.range_slider_shown.setMinimum(min_temp_scaled)
        self.range_slider_shown.setMaximum(max_temp_scaled)

        # range slider for palette
        self.range_slider_map = wid.QRangeSlider(self.colormap_name)
        self.range_slider_map.setLowerValue(min_shown_temp_scaled)
        self.range_slider_map.setUpperValue(max_shown_temp_scaled)
        self.range_slider_map.setMinimum(min_temp_scaled)
        self.range_slider_map.setMaximum(max_temp_scaled)
        self.range_slider_map.setFixedHeight(50)

        self.range_slider_shown.lowerValueChanged.connect(self.updateIRImage)
        self.range_slider_shown.upperValueChanged.connect(self.updateIRImage)

        self.range_slider_map.lowerValueChanged.connect(self.recolorIRImage)
        self.range_slider_map.upperValueChanged.connect(self.recolorIRImage)

        # edit labels
        self.label_max.setText(f'{str(min_shown_temp_scaled / 100)} °C')
        self.label_min.setText(f'{str(max_shown_temp_scaled / 100)} °C')

        # edit labels
        self.label_max_2.setText(f'{str(max_shown_temp_scaled / 100)} °C')
        self.label_min_2.setText(f'{str(min_shown_temp_scaled / 100)} °C')

        # new label
        self.verticalLayout_2.addWidget(self.range_slider_shown)
        self.verticalLayout_2.addWidget(self.range_slider_map)

        # Load images
        colorPixmap = QPixmap(img_object.rgb_path)
        thermalPixmap = QPixmap(self.ir_path)

        # Create custom graphics items
        self.colorImageItem = CustomGraphicsItem(colorPixmap)
        self.thermalImageItem = CustomGraphicsItem(thermalPixmap, target_size=colorPixmap.size())

        # Add items to scene
        self.scene.addItem(self.colorImageItem)
        self.scene.addItem(self.thermalImageItem)

        # Fusion Mode ComboBox
        modes = ['SourceOver',
                 'Screen',
                 'Multiply',
                 'Overlay',
                 'Darken',
                 'Lighten',
                 'ColorBurn',
                 'HardLight',
                 'SoftLight',
                 ]
        for mode in modes:
            self.modeComboBox.addItem(mode)
        self.modeComboBox.currentTextChanged.connect(self.changeFusionMode)

        self.opacitySlider.setRange(0, 100)
        self.opacitySlider.setValue(100)
        self.opacitySlider.valueChanged.connect(self.changeOpacity)

        self.changeFusionMode(self.modeComboBox.currentText())
        self.changeOpacity(self.opacitySlider.value())

        # button actions
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.grayscaleCheckbox.stateChanged.connect(self.toggleGrayscale)

        self.updateIRImage(0)

    def changeFusionMode(self, mode):
        mode_complete = 'CompositionMode_' + mode
        blend_mode = QPainter.CompositionMode.__members__[mode_complete]
        self.thermalImageItem.composition_mode = blend_mode
        self.thermalImageItem.update()

    def changeOpacity(self, value):
        self.thermalImageItem.opacity = value / 100.0
        self.thermalImageItem.update()

    def showEvent(self, event):
        super().showEvent(event)
        self.fitItemsInView()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fitItemsInView()

    def fitItemsInView(self):
        # Disable scrollbars
        self.view.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # Get the size of the view
        view_rect = self.view.viewport().rect()
        view_width = view_rect.width()
        view_height = view_rect.height()

        # Assuming colorPixmap is the pixmap of the color image item
        pixmap_size = self.colorImageItem.pixmap.size()
        pixmap_width = pixmap_size.width()
        pixmap_height = pixmap_size.height()

        # Calculate scale factors for width and height
        scale_factor_width = view_width / pixmap_width
        scale_factor_height = view_height / pixmap_height

        # Use the smaller scale factor to keep the aspect ratio
        scale_factor = min(scale_factor_width, scale_factor_height)

        # Apply the scale factor to both pixmap items
        self.colorImageItem.setScale(scale_factor)
        self.thermalImageItem.setScale(scale_factor)

        # Adjust the scene size to the scaled pixmap size
        scaled_width = pixmap_width * scale_factor
        scaled_height = pixmap_height * scale_factor
        self.scene.setSceneRect(0, 0, scaled_width, scaled_height)

        # Center the items in the view
        self.colorImageItem.setPos((view_width - scaled_width) / 2, (view_height - scaled_height) / 2)
        self.thermalImageItem.setPos((view_width - scaled_width) / 2, (view_height - scaled_height) / 2)

    def recolorIRImage(self, value):
        min_temp = self.range_slider_map.lowerValue() / 100.0  # Adjust if you used scaling
        max_temp = self.range_slider_map.upperValue() / 100.0

        # edit labels
        self.label_max_2.setText(f'{str(max_temp)} °C')
        self.label_min_2.setText(f'{str(min_temp)} °C')

        # compute new normalized temperature
        thermal_normalized = (self.temperatures - min_temp) / (max_temp - min_temp)

        # get colormap
        if self.colormap_name in tt.LIST_CUSTOM_CMAPS:
            custom_cmap = tt.get_custom_cmaps(self.colormap_name, self.n_colors)
        else:
            custom_cmap = cm.get_cmap(self.colormap_name, self.n_colors)

        thermal_cmap = custom_cmap(thermal_normalized)
        thermal_cmap = np.uint8(thermal_cmap * 255)

        img_thermal = Image.fromarray(thermal_cmap[:, :, [0, 1, 2]])
        img_thermal.save(self.dest_path_preview)
        self.ir_path = self.dest_path_preview

        self.updateIRImage(0)

    def updateIRImage(self, value):
        min_temp = self.range_slider_shown.lowerValue() / 100.0  # Adjust if you used scaling
        max_temp = self.range_slider_shown.upperValue() / 100.0

        # edit labels
        self.label_max.setText(f'{str(max_temp)} °C')
        self.label_min.setText(f'{str(min_temp)} °C')

        # Load the IR image and convert to NumPy array
        ir_image = QImage(self.ir_path)
        ir_image = ir_image.convertToFormat(QImage.Format.Format_ARGB32)
        # Convert QImage to a 4-channel NumPy array (RGBA)
        ir_array = qimage2ndarray.byte_view(ir_image)

        # Ensure that the array has 4 channels (RGBA)
        if ir_array.shape[2] == 3:
            # Add an alpha channel if it's missing
            alpha_channel = np.full((ir_array.shape[0], ir_array.shape[1], 1), 255, dtype=np.uint8)
            ir_array = np.concatenate((ir_array, alpha_channel), axis=2)

        elif ir_array.shape[2] == 4:
            # Swap red and blue channels
            ir_array = ir_array[..., [2, 1, 0, 3]]  # BGR to RGB

        # Create a mask where temperatures are outside the range
        mask = ~((self.temperatures >= min_temp) & (self.temperatures <= max_temp))

        # Apply the mask to the alpha channel, setting those pixels to transparent
        ir_array[..., 3] = np.where(mask, 0, ir_array[..., 3])

        # Convert the modified array back to QImage and then to QPixmap
        updated_image = qimage2ndarray.array2qimage(ir_array, normalize=False)
        updated_pixmap = QPixmap.fromImage(updated_image)

        # Update the item
        self.thermalImageItem.setPixmap(updated_pixmap)
        self.thermalImageItem.update()

    def scale_temperature(self, temp):
        return int(temp * 100)  # Scaling factor of 100

    def toggleGrayscale(self):
        if self.grayscaleCheckbox.isChecked():
            # Convert to grayscale
            gray_pixmap = self.colorImageItem.pixmap.toImage().convertToFormat(QImage.Format.Format_Grayscale8)
            self.colorImageItem.setPixmap(QPixmap.fromImage(gray_pixmap), rescale=False)
        else:
            # Restore original color image
            colorPixmap = QPixmap(self.img_object.rgb_path)  # Assuming img_object is accessible
            self.colorImageItem.setPixmap(colorPixmap, rescale=False)
        self.colorImageItem.update()

    def exportComposedImage(self, output_path):
        """
        Export the composed image (fused color and infrared images) to the specified output path.
        """
        # Determine the size of the composed image
        width = self.colorImageItem.pixmap.width()
        height = self.colorImageItem.pixmap.height()

        # Create an empty QImage with the size of the final composed image
        composed_image = QImage(width, height, QImage.Format.Format_ARGB32)
        composed_image.fill(QtCore.Qt.GlobalColor.transparent)  # Start with a transparent image

        # Initialize a QPainter to draw on the composed QImage
        painter = QPainter(composed_image)

        # Draw the color image first
        painter.drawPixmap(0, 0, self.colorImageItem.pixmap)

        # Set the composition mode (e.g., Overlay, Multiply, etc.) and opacity
        painter.setCompositionMode(self.thermalImageItem.composition_mode)
        painter.setOpacity(self.thermalImageItem.opacity)

        # Draw the infrared image with the appropriate blending mode
        painter.drawPixmap(0, 0, self.thermalImageItem.pixmap)

        # End the painter
        painter.end()

        if output_path.lower().endswith('.jpg') or output_path.lower().endswith('.jpeg'):
            composed_image.save(output_path, 'JPEG', 100)
        else:
            composed_image.save(output_path)  # PNG is lossless by default
        composed_image.save(output_path)


class AlignmentDialog(QDialog):
    def __init__(self, img_object, temp_folder, theta=[1.1, 0, 0]):
        super().__init__()

        # images to process
        # copy images to temp folder
        self.img_object = img_object
        self.temp_folder = temp_folder
        self.ir_path = os.path.join(temp_folder, 'IR.JPG')
        self.rgb_path = os.path.join(temp_folder, 'RGB.JPG')
        copyfile(img_object.path, self.ir_path)
        copyfile(img_object.rgb_path_original, self.rgb_path)

        # get focal and center points
        """
        self.F = F
        self.CX = 320
        self.CY = 256
        self.d_mat = d_mat
        """

        # Initial theta values
        # Parameters are [zoom, y-offset, x-offset]
        self.theta = theta

        # Setup UI
        self.setWindowTitle("Thermal Image Processor")
        self.setGeometry(100, 100, 800, 800)

        # Main layout
        layout = QVBoxLayout(self)

        # Graphics view
        self.graphics_view = QGraphicsView()
        layout.addWidget(self.graphics_view)
        self.scene = QGraphicsScene()
        self.graphics_view.setScene(self.scene)

        # Sliders and Labels for parameters
        self.sliders = []
        self.labels = []
        param_names = ["zoom", "y-offset", "x-offset"]
        param_ranges = [(0.5, 2), (-100, 100), (-100, 100)]

        for i, (param_name, param_range) in enumerate(zip(param_names, param_ranges)):
            label = QLabel(f"{param_name}: {self.theta[i]}")
            layout.addWidget(label)
            self.labels.append(label)

            slider = QSlider(Qt.Orientation.Horizontal)
            if i != 0:
                slider.setMinimum(int(param_range[0] * 10))
                slider.setMaximum(int(param_range[1] * 10))
                slider.setValue(int(self.theta[i] * 10))
            else:
                slider.setMinimum(int(param_range[0] * 500))
                slider.setMaximum(int(param_range[1] * 500))
                slider.setValue(int(self.theta[i] * 500))
            slider.valueChanged.connect(lambda value, x=i: self.update_parameter(x, value))
            self.sliders.append(slider)
            layout.addWidget(slider)

        # Optimize button
        self.optimize_button = QPushButton('Optimize')
        self.optimize_button.clicked.connect(self.optimize)
        layout.addWidget(self.optimize_button)

        # Button box for OK and Cancel
        button_layout = QHBoxLayout()
        self.ok_button = QPushButton("OK")
        self.cancel_button = QPushButton("Cancel")
        button_layout.addWidget(self.ok_button)
        button_layout.addWidget(self.cancel_button)
        layout.addLayout(button_layout)

        # Connect buttons to their actions
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button.clicked.connect(self.reject)

        # Load and process initial images
        self.process_images()

    def update_parameter(self, index, value):
        if index != 0:
            self.theta[index] = value / 10
            self.labels[index].setText(f"{self.labels[index].text().split(':')[0]}: {value / 10}")
        else:
            self.theta[index] = value / 500
            self.labels[index].setText(f"{self.labels[index].text().split(':')[0]}: {value / 500}")
        self.process_images()

    def optimize(self):
        # Placeholder for optimize function
        print("Optimization started...")

    def convert_np_img_to_qimage(self, img):
        """Converts a numpy array image to QImage."""
        if img.ndim == 3:  # Color image
            h, w, ch = img.shape
            bytes_per_line = ch * w
            qimage = QImage(img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        else:  # Grayscale image
            h, w = img.shape
            qimage = QImage(img.data, w, h, QImage.Format.Format_Grayscale8)
        return qimage

    def process_images(self):
        # Image processing placeholder
        print(f"Processing with theta: {self.theta}")
        # This should call your actual image processing code
        tt.process_th_image_with_zoom(self.img_object, self.temp_folder, self.theta)

        # Read the original RGB image
        rgb_image = cv2.imread(os.path.join(self.temp_folder, 'rescale.JPG'))

        # Convert RGB image to grayscale for background
        grayscale_rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)

        # Read or process the IR lines image
        # Assuming 'lines_ir' is a binary image with lines as white (255) and background as black (0)
        # For demonstration, replace this with an actual call to your image processing function
        lines_ir = cv2.imread(os.path.join(self.temp_folder, 'ir_lines.JPG'), cv2.IMREAD_GRAYSCALE)  # Placeholder

        # Convert the grayscale image to a 3-channel image to overlay colors
        grayscale_rgb_image_colored = cv2.cvtColor(grayscale_rgb_image, cv2.COLOR_GRAY2BGR)

        # Overlay IR lines: Set them to blue (or any color you choose) on the grayscale image
        grayscale_rgb_image_colored[lines_ir >= 100] = [255, 0, 0]  # Blue: [B, G, R] - Change this color if needed

        # Convert the overlay image to QImage for display
        qimage = self.convert_np_img_to_qimage(grayscale_rgb_image_colored)

        # Update QGraphicsScene
        self.scene.clear()
        pixmap_item = QGraphicsPixmapItem(QPixmap.fromImage(qimage))
        self.scene.addItem(pixmap_item)


class DialogBatchExport(QtWidgets.QDialog):
    """
    Dialog that allows the user to choose batch export options.
    """

    def __init__(self, img_list, default_folder, parameters_style, parameters_temp, parameters_radiometric, parent=None):
        super().__init__(parent)
        basepath = os.path.dirname(__file__)
        basename = 'export_dialog'
        uifile = os.path.join(basepath, 'ui/%s.ui' % basename)
        loadUi(uifile, self)

        self.comboBox_naming.addItems(['Rename files', 'Keep IR names', 'Match IR with RGB names'])
        self.comboBox_img_format.addItems(['PNG', 'JPG'])

        # ListView Setup
        self.img_list = img_list
        self.list_model = QtGui.QStandardItemModel()
        self.listView.setModel(self.list_model)

        # Enable multi-selection mode (Shift+Click, Ctrl+Click supported)
        self.listView.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)

        self.populate_list_view()

        # Button actions
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        # Set initial parameters
        desc = (f'Palette parameters: \n'
                f'- palette: {parameters_style[0]}')
        self.label_current_settings.setText(desc)
        self.lineEdit.setText(default_folder)

        # Browse folder button
        self.pushButton.clicked.connect(self.browse_folder)

    def populate_list_view(self):
        """
        Populate the ListView with the image list and pre-select all items.
        """
        for img_name in self.img_list:
            item = QtGui.QStandardItem(img_name)
            self.list_model.appendRow(item)

        # Pre-select all items by default
        self.listView.selectAll()

    def get_selected_indices(self):
        """
        Get indices of selected images.
        """
        selected_indexes = self.listView.selectedIndexes()
        return [index.row() for index in selected_indexes]

    def browse_folder(self):
        """
        Open a QFileDialog to select a folder.
        """
        folder_path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder_path:
            self.lineEdit.setText(folder_path)

class DialogEdgeOptions(QtWidgets.QDialog):
    """
    Dialog that allows the user to choose advances thermography options
    """

    def __init__(self, parameters, parent=None):
        QtWidgets.QDialog.__init__(self)
        basepath = os.path.dirname(__file__)
        basename = 'edge_options'
        uifile = os.path.join(basepath, 'ui/%s.ui' % basename)
        loadUi(uifile, self)
        self.comboBox_blur_size.addItems([str(3), str(5), str(7)])
        self.comboBox_color.addItems(['white', 'black'])
        self.comboBox_method.addItems(['Sobel A', 'Kernel 1', 'Kernel 2', 'Canny', 'Canny-L2', 'ML'])

        # button actions
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.checkBox.stateChanged.connect(self.toggle_combo)

        # set initial parameters
        self.parameters = parameters
        self.set_ini_param()

    def set_ini_param(self):
        # Set initial values for combo boxes if parameters are provided
        self.comboBox_method.setCurrentIndex(self.parameters[0])

        index = self.comboBox_color.findText(self.parameters[1], QtCore.Qt.MatchFlag.MatchFixedString)
        if index >= 0:
            self.comboBox_color.setCurrentIndex(index)

        if self.parameters[2]:
            self.checkBox_bil.setChecked(True)
        else:
            self.checkBox_bil.setChecked(False)

        if self.parameters[3]:
            self.checkBox.setChecked(True)
        else:
            self.checkBox.setChecked(False)

        index = self.comboBox_blur_size.findText(str(self.parameters[4]), QtCore.Qt.MatchFlag.MatchFixedString)
        if index >= 0:
            self.comboBox_blur_size.setCurrentIndex(index)

        op_value = self.parameters[5]
        self.horizontalSlider.setValue(int(op_value * 100))

    def toggle_combo(self):
        if self.checkBox.isChecked():
            self.comboBox_blur_size.setEnabled(True)
        else:
            self.comboBox_blur_size.setEnabled(False)


class DialogThParams(QtWidgets.QDialog):
    """
    Dialog that allows the user to choose advances thermography options
    """

    def __init__(self, param, parent=None):
        QtWidgets.QDialog.__init__(self)
        basepath = os.path.dirname(__file__)
        basename = 'dialog_options'
        uifile = os.path.join(basepath, 'ui/%s.ui' % basename)
        loadUi(uifile, self)
        self.lineEdit_em.setText(str(param['emissivity']))
        self.lineEdit_dist.setText(str(param['distance']))
        self.lineEdit_rh.setText(str(param['humidity']))
        self.lineEdit_temp.setText(str(param['reflection']))

        # define constraints on lineEdit

        # button actions
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)


class MeasLineDialog(QtWidgets.QDialog):
    def __init__(self, data):
        QtWidgets.QDialog.__init__(self)
        basepath = os.path.dirname(__file__)
        basename = 'meas_dialog_2d'
        uifile = os.path.join(basepath, 'ui/%s.ui' % basename)
        loadUi(uifile, self)

        self.data = data
        self.y_values = data
        self.x_values = np.arange(len(self.y_values))

        # Assuming a DPI of 100
        dpi = 100
        width, height = 600 / dpi, 400 / dpi  # Convert pixel dimensions to inches

        # Plotting Area
        self.figure = Figure(figsize=(width, height), dpi=dpi)
        self.canvas = FigureCanvas(self.figure)
        self.verticalLayout_2.addWidget(self.canvas)

        ax = self.figure.add_subplot()  # 1 row, 2 columns, second plot
        ax.plot(self.x_values, self.y_values, 'b-')  # Assuming 'r' is accessible and correctly sized
        ax.set_xlabel("length [pixels]")
        ax.set_ylabel("Temperature [°C]")

        # Improve the axis
        ax.spines['top'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.spines['bottom'].set_position(('data', 0))
        ax.spines['left'].set_position(('data', 0))
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        ax.grid(True, linestyle='--', which='both', zorder=0)

        self.highlights = self.create_highlights()

        # add table model for data
        self.model = wid.TableModel(self.highlights)
        self.tableView.setModel(self.model)

        # add matplotlib toolbar
        toolbar = NavigationToolbar2QT(self.canvas, self)
        self.verticalLayout_2.addWidget(toolbar)

        # button actions
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.create_connections()

    def create_connections(self):
        pass

    def create_highlights(self):
        # extrema
        self.tmax = np.amax(self.y_values)
        self.tmin = np.amin(self.y_values)
        self.tmean = np.mean(self.y_values)

        # normalized data
        self.th_norm = (self.y_values - self.tmin) / (self.tmax - self.tmin)

        highlights = [
            ['Max. Temp. [°C]', str(self.tmax)],
            ['Min. Temp. [°C]', str(self.tmin)],
            ['Average Temp. [°C]', str(self.tmean)]
        ]
        return highlights


class Meas3dDialog_simple(QtWidgets.QDialog):
    def __init__(self, rect_annot):
        QtWidgets.QDialog.__init__(self)
        basepath = os.path.dirname(__file__)
        basename = 'meas_dialog_3d_simple'
        uifile = os.path.join(basepath, 'ui/%s.ui' % basename)
        loadUi(uifile, self)

        self.setWindowTitle('Area Measurement')
        self.matplot_c = wid.MplCanvas_project3d(self)
        self.ax = self.matplot_c.figure.add_subplot(projection='3d')  # add subplot, retrieve axis object
        self.ax.view_init(elev=70, azim=-45, roll=0)
        self.ax.set_zlabel('Temperature [°C]')

        self.data = rect_annot.data_roi
        self.highlights = rect_annot.compute_highlights()

        # add table model for data
        self.model = wid.TableModel(self.highlights)
        self.tableView.setModel(self.model)

        # add matplotlib toolbar
        toolbar = NavigationToolbar2QT(self.matplot_c, self)
        self.verticalLayout.addWidget(toolbar)
        self.verticalLayout.addWidget(self.matplot_c)

        # button actions
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.create_connections()

    def create_connections(self):
        pass

    def surface_from_image_matplot(self, colormap, n_colors, col_low, col_high):
        # Step 1: Resize data if any dimension is greater than 100
        max_dim = 100  # Set the maximum dimension size

        # Determine the resize factor based on the larger scaling required
        if self.data.shape[0] > max_dim or self.data.shape[1] > max_dim:
            resize_factor = min(1.0, max_dim / max(self.data.shape[0], self.data.shape[1]))
            # Apply the same resize factor to both dimensions to maintain aspect ratio
            self.data = zoom(self.data, (resize_factor, resize_factor), order=1)

        # Step 2: Colormap operation
        if colormap in tt.LIST_CUSTOM_CMAPS:
            custom_cmap = tt.get_custom_cmaps(colormap, n_colors)
        else:
            custom_cmap = cm.get_cmap(colormap, n_colors)

        custom_cmap.set_over(col_high)
        custom_cmap.set_under(col_low)

        # Step 3: Create meshgrid for plotting
        xx, yy = np.mgrid[0:self.data.shape[0], 0:self.data.shape[1]]

        # Clear the existing plot before plotting new data to avoid memory issues
        self.ax.clear()

        # Step 4: Plot surface
        self.ax.plot_surface(xx, yy, self.data, rstride=1, cstride=1, linewidth=0, cmap=custom_cmap)

        # Step 5: Set aspect ratio correction
        # Adjust the axis limits to match the aspect ratio of the data
        self.ax.set_xlim(0, self.data.shape[0])
        self.ax.set_ylim(0, self.data.shape[1])
        self.ax.set_zlim(np.min(self.data), np.max(self.data))

        self.ax.set_xticks([])  # Remove X-axis ticks
        self.ax.set_yticks([])  # Remove Y-axis ticks
        self.ax.set_zlabel('Temperature [°C]')

        # Calculate the Z scaling factor to match the maximum extent of the X or Y axis
        max_data_dim = max(self.data.shape[0], self.data.shape[1])
        z_extent = np.max(self.data) - np.min(self.data)

        if z_extent != 0:  # To prevent division by zero if data is constant
            z_scaling_factor = max_data_dim / z_extent
        else:
            z_scaling_factor = 1  # Default to 1 if no variation in data

        # Set the box aspect ratio to ensure Z-axis matches the X and Y axes proportionally
        self.ax.set_box_aspect((self.data.shape[0], self.data.shape[1], z_scaling_factor*z_extent))

        # Step 6: Draw the figure
        self.matplot_c.figure.canvas.draw_idle()


class Meas3dDialog(QtWidgets.QDialog):
    def __init__(self, rect_annot):
        QtWidgets.QDialog.__init__(self)
        basepath = os.path.dirname(__file__)
        basename = 'meas_dialog_3d'
        uifile = os.path.join(basepath, 'ui/%s.ui' % basename)
        loadUi(uifile, self)

        self.setWindowTitle('Area Measurement')
        self.matplot_c = wid.MplCanvas_project3d(self)
        self.ax = self.matplot_c.figure.add_subplot(projection='3d')  # add subplot, retrieve axis object
        self.ax.view_init(elev=70, azim=-45, roll=0)
        self.ax.set_zlabel('Temperature [°C]')

        self.data = rect_annot.data_roi
        self.highlights = rect_annot.compute_highlights()

        # create dualviewer
        self.dual_view = wid.DualViewer()
        self.verticalLayout.addWidget(self.dual_view)

        # add table model for data
        self.model = wid.TableModel(self.highlights)
        self.tableView.setModel(self.model)

        # add matplotlib toolbar
        toolbar = NavigationToolbar2QT(self.matplot_c, self)
        self.verticalLayout_3.addWidget(toolbar)
        self.verticalLayout_3.addWidget(self.matplot_c)

        # button actions
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.create_connections()

    def create_connections(self):
        pass

    def surface_from_image_matplot(self, colormap, n_colors, col_low, col_high):
        # Step 1: Resize data if any dimension is greater than 100
        max_dim = 100  # Set the maximum dimension size

        # Determine the resize factor based on the larger scaling required
        if self.data.shape[0] > max_dim or self.data.shape[1] > max_dim:
            resize_factor = min(1.0, max_dim / max(self.data.shape[0], self.data.shape[1]))
            # Apply the same resize factor to both dimensions to maintain aspect ratio
            self.data = zoom(self.data, (resize_factor, resize_factor), order=1)

        # Step 2: Colormap operation
        if colormap in tt.LIST_CUSTOM_CMAPS:
            custom_cmap = tt.get_custom_cmaps(colormap, n_colors)
        else:
            custom_cmap = cm.get_cmap(colormap, n_colors)

        custom_cmap.set_over(col_high)
        custom_cmap.set_under(col_low)

        # Step 3: Create meshgrid for plotting
        xx, yy = np.mgrid[0:self.data.shape[0], 0:self.data.shape[1]]

        # Clear the existing plot before plotting new data to avoid memory issues
        self.ax.clear()

        # Step 4: Plot surface
        self.ax.plot_surface(xx, yy, self.data, rstride=1, cstride=1, linewidth=0, cmap=custom_cmap)

        # Step 5: Set aspect ratio correction
        # Adjust the axis limits to match the aspect ratio of the data
        self.ax.set_xlim(0, self.data.shape[0])
        self.ax.set_ylim(0, self.data.shape[1])
        self.ax.set_zlim(np.min(self.data), np.max(self.data))

        self.ax.set_xticks([])  # Remove X-axis ticks
        self.ax.set_yticks([])  # Remove Y-axis ticks
        self.ax.set_zlabel('Temperature [°C]')

        # Calculate the Z scaling factor to match the maximum extent of the X or Y axis
        max_data_dim = max(self.data.shape[0], self.data.shape[1])
        z_extent = np.max(self.data) - np.min(self.data)

        if z_extent != 0:  # To prevent division by zero if data is constant
            z_scaling_factor = max_data_dim / z_extent
        else:
            z_scaling_factor = 1  # Default to 1 if no variation in data

        # Set the box aspect ratio to ensure Z-axis matches the X and Y axes proportionally
        self.ax.set_box_aspect((self.data.shape[0], self.data.shape[1], z_scaling_factor*z_extent))

        # Step 6: Draw the figure
        self.matplot_c.figure.canvas.draw_idle()

class ZoomableGraphicsView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._zoom_factor = 1.25  # Factor for zooming in/out
        self._current_zoom = 0     # Keep track of current zoom level
        self._zoom_clamp_min = -10  # Min zoom level
        self._zoom_clamp_max = 10   # Max zoom level

        # Enable antialiasing and smooth pixmap transformation
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

    def wheelEvent(self, event):
        """
        Override the wheel event to zoom in or out on scroll.
        """
        if event.angleDelta().y() > 0:
            self.zoom_in()
        else:
            self.zoom_out()

    def zoom_in(self):
        if self._current_zoom < self._zoom_clamp_max:
            self.scale(self._zoom_factor, self._zoom_factor)
            self._current_zoom += 1

    def zoom_out(self):
        if self._current_zoom > self._zoom_clamp_min:
            self.scale(1 / self._zoom_factor, 1 / self._zoom_factor)
            self._current_zoom -= 1
class DetectionDialog(QDialog):
    box_exported = pyqtSignal(QRectF)  # Signal to return the bounding box coordinates

    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.ontology = {}
        self.rect_items = []  # To store the QGraphicsRectItems
        self.class_denominations = []
        self.setWindowTitle("Object Detection Control")

        # QGraphicsView and Scene for displaying the image
        self.graphics_view = ZoomableGraphicsView()
        self.scene = QGraphicsScene()
        self.graphics_view.setScene(self.scene)

        # Load the image into the graphics view
        self.load_image(self.image_path)

        # List widget to display detection classes
        self.class_list = QListWidget()

        # TreeView to display detections grouped by class (on the right side)
        self.tree_view = QTreeView()
        self.model = QStandardItemModel()
        self.model.setHorizontalHeaderLabels(["Detections"])
        self.tree_view.setModel(self.model)

        # Enable custom context menu for the TreeView
        self.tree_view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree_view.customContextMenuRequested.connect(self.open_context_menu)

        # Connect selection change to a method to highlight the corresponding rectangle
        self.tree_view.selectionModel().selectionChanged.connect(self.on_tree_selection_changed)

        # LineEdit for class input
        self.class_input = QLineEdit()
        self.class_input.setPlaceholderText("Enter class name")

        # Button to add class
        self.add_class_button = QPushButton("Add Class")
        self.add_class_button.clicked.connect(self.add_class)

        # Button to run segmentation
        self.run_button = QPushButton("Run Detection")
        self.run_button.clicked.connect(self.run_segmentation)

        # Layouts: Split left and right sections
        left_layout = QVBoxLayout()
        class_layout = QHBoxLayout()
        class_layout.addWidget(self.class_input)
        class_layout.addWidget(self.add_class_button)
        left_layout.addWidget(self.graphics_view)
        left_layout.addWidget(self.class_list)
        left_layout.addLayout(class_layout)
        left_layout.addWidget(self.run_button)

        main_layout = QHBoxLayout()
        main_layout.addLayout(left_layout)  # All widgets on the left side
        main_layout.addWidget(self.tree_view)  # TreeView on the right side

        self.setLayout(main_layout)

    def load_image(self, image_path):
        # Load the image into the QGraphicsScene
        pixmap = QPixmap(image_path)
        self.image_item = QGraphicsPixmapItem(pixmap)
        self.scene.clear()  # Clear any previous items
        self.scene.addItem(self.image_item)

        # Convert QRect to QRectF for setSceneRect
        self.graphics_view.setSceneRect(QRectF(pixmap.rect()))
        self.graphics_view.fitInView(self.image_item, Qt.AspectRatioMode.KeepAspectRatio)

    def add_class(self):
        class_name = self.class_input.text().strip()
        if class_name and class_name not in self.ontology:
            self.ontology[class_name] = class_name  # Add class to ontology dict
            self.class_list.addItem(class_name)
            self.class_input.clear()
            self.class_denominations.append(class_name)

    def run_segmentation(self):
        if not self.ontology:
            print("No classes to detect!")
            return

        # Assuming GroundingDINO and CaptionOntology are set up
        results = ai.run_g_dino(self.image_path, self.ontology)

        # Plot the results on the image using the GroundingDINO results and your plotting function
        self.plot_results(results)

    def plot_results(self, detections):
        """
        Plot bounding boxes and populate the TreeView with the results.
        Group detections by class.
        """
        self.rect_items.clear()  # Clear previous detection rectangles
        self.model.clear()  # Clear the tree view model

        pixmap = self.image_item.pixmap()
        image_width = pixmap.width()
        image_height = pixmap.height()

        # Dictionary to store the class items in the tree
        class_items = {}

        global_index_counter = 0

        # Loop over detections and group by class
        for idx, detection in enumerate(detections):
            box = detection[0]  # Bounding box coordinates
            confidence = detection[2]  # Confidence score
            class_number = detection[3]  # Class number

            # Create the bounding box rectangle
            x_min = box[0]
            y_min = box[1]
            x_max = box[2]
            y_max = box[3]
            rect = QRectF(x_min, y_min, x_max - x_min, y_max - y_min)

            # Set the color based on the class number
            pen_color = CLASS_COLORS.get(class_number, QColor(0, 0, 0))  # Default to black
            pen = QPen(pen_color)
            pen.setWidth(5)

            rect_item = QGraphicsRectItem(rect)
            rect_item.setPen(pen)

            self.scene.addItem(rect_item)
            self.rect_items.append(rect_item)  # Store the rectangle for later highlighting

            # Get class name for the current detection (you can customize this as needed)
            class_to_be_used = self.class_denominations[class_number]
            class_name = f"{class_to_be_used}"  # Example class name

            # Create the class item if it doesn't already exist
            if class_name not in class_items:
                class_item = QStandardItem(class_name)
                class_items[class_name] = class_item
                self.model.appendRow(class_item)

            # Add detection under the corresponding class
            detection_item = QStandardItem(f"Detection {idx + 1}")
            confidence_item = QStandardItem(f"Confidence: {confidence:.2f}")

            # Store the global index in the detection item using UserRole
            detection_item.setData(global_index_counter, Qt.ItemDataRole.UserRole)

            # Add children (confidence) to detection item
            detection_item.appendRow(confidence_item)

            # Add the detection item to the class item
            class_items[class_name].appendRow(detection_item)

            # Increment the global index counter
            global_index_counter += 1

        # Refit the view to account for the bounding boxes
        self.graphics_view.fitInView(self.scene.itemsBoundingRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self.model.setHorizontalHeaderLabels(["Detections"])

    def open_context_menu(self, position: QPoint):
        """
        Open context menu when right-clicking on a detection in the tree view.
        """
        # Get the index of the selected item
        index = self.tree_view.indexAt(position)
        if not index.isValid():
            return

        # Create a context menu
        context_menu = QMenu(self)

        # Add the "Export this box" option
        export_action = QAction("Export this box", self)
        export_action.triggered.connect(lambda: self.export_box(index))
        context_menu.addAction(export_action)

        # Open the menu at the cursor position
        context_menu.exec(self.tree_view.viewport().mapToGlobal(position))

    def export_box(self, index):
        """
        Export the bounding box coordinates for the selected detection.
        """
        # Find the detection index (the grandchild of the class item)
        if index.parent().isValid():  # Ensure it's a child under a detection item
            detection_index = index.parent().row()

            if 0 <= detection_index < len(self.rect_items):
                # Get the corresponding QRectF for the selected detection
                selected_rect = self.rect_items[detection_index].rect()

                # Emit the signal with the exported rectangle coordinates
                self.box_exported.emit(selected_rect)

                # Close the dialog
                self.accept()

    def on_tree_selection_changed(self, selected, deselected):
        """
        Highlight the corresponding rectangle when a detection or class is selected in the tree view.
        """
        # Clear previous bold rectangles
        for rect_item in self.rect_items:
            rect_item.setPen(QPen(rect_item.pen().color(), 5))  # Reset to normal width

        # Get the index of the selected item
        selected_index = selected.indexes()[0] if selected.indexes() else None
        print(selected_index)
        if not selected_index:
            return

        # Get the QStandardItem from the selected index
        selected_item = self.model.itemFromIndex(selected_index)

        # Check if the selected item has the global index stored (valid only for detection items)
        global_index = selected_item.data(Qt.ItemDataRole.UserRole)
        print(global_index)
        if global_index is not None and isinstance(global_index, int):
            # If the selected item is a detection, highlight its rectangle
            if 0 <= global_index < len(self.rect_items):
                rect_item = self.rect_items[global_index]
                pen = QPen(rect_item.pen().color())
                pen.setWidth(20)  # Make the rectangle outline bold
                rect_item.setPen(pen)
        elif not selected_index.parent().isValid():  # This is a top-level (class) item
            class_item = self.model.itemFromIndex(selected_index)
            # Iterate through all children (detections) of the class and highlight their rectangles
            for row in range(class_item.rowCount()):
                detection_item = class_item.child(row)  # Get the detection item
                global_index = detection_item.data(Qt.ItemDataRole.UserRole)  # Get the stored global index

                if global_index is not None and 0 <= global_index < len(self.rect_items):
                    rect_item = self.rect_items[global_index]
                    pen = QPen(rect_item.pen().color())
                    pen.setWidth(20)  # Make the rectangle outline bold
                    rect_item.setPen(pen)