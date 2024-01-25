import os
from PySide6 import QtCore, QtGui, QtWidgets
import logging
import sys
import traceback

from PySide6.QtCore import QRectF
from PySide6.QtGui import QPixmap, QPainter, QImage, QColor
from PySide6.QtWidgets import QDialog, QVBoxLayout, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QComboBox, \
    QSlider, QGraphicsItem
from matplotlib import cm
from matplotlib.backends.backend_qt5agg import (
   FigureCanvasQTAgg as FigureCanvas,
   NavigationToolbar2QT)
from matplotlib.figure import Figure

import qimage2ndarray

from PIL import Image
from matplotlib import cm



import numpy as np

# custom libraries
import widgets as wid
import resources as res
from tools import thermal_tools as tt

# basic logger functionality
log = logging.getLogger(__name__)
handler = logging.StreamHandler(stream=sys.stdout)
log.addHandler(handler)

def show_exception_box(log_msg):
    """Checks if a QApplication instance is available and shows a messagebox with the exception message.
    If unavailable (non-console application), log an additional notice.
    """
    if QtWidgets.QApplication.instance() is not None:
            errorbox = QtWidgets.QMessageBox()
            errorbox.setText("Oops. An unexpected error occured:\n{0}".format(log_msg))
            errorbox.exec_()
    else:
        log.debug("No QApplication instance available.")

class UncaughtHook(QtCore.QObject):
    _exception_caught = QtCore.Signal(object)

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
        self.setFixedSize(300,300)
        self.layout = QtWidgets.QVBoxLayout()

        about_text = QtWidgets.QLabel('This app was made by Buildwise, to simplify the analysis of thermal images.')
        about_text.setWordWrap(True)

        logos1 = QtWidgets.QLabel()
        pixmap = QtGui.QPixmap(res.find('img/logo_buildwise2.png'))
        w = self.width()
        pixmap = pixmap.scaledToWidth(100, QtCore.Qt.SmoothTransformation)
        logos1.setPixmap(pixmap)

        logos2 = QtWidgets.QLabel()
        pixmap = QtGui.QPixmap(res.find('img/logo_pointify.png'))
        pixmap = pixmap.scaledToWidth(100, QtCore.Qt.SmoothTransformation)
        logos2.setPixmap(pixmap)

        self.layout.addWidget(about_text)
        self.layout.addWidget(logos1, alignment=QtCore.Qt.AlignCenter)
        self.layout.addWidget(logos2, alignment=QtCore.Qt.AlignCenter)

        self.setLayout(self.layout)

class CustomGraphicsItem(QGraphicsItem):
    def __init__(self, pixmap, target_size=None, parent=None):
        super().__init__(parent)
        self.pixmap = pixmap
        self.opacity = 1.0
        self.composition_mode = QPainter.CompositionMode_SourceOver
        self.target_size = target_size

        if self.target_size:
            self.pixmap = self.pixmap.scaled(self.target_size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)

    def boundingRect(self):
        return QRectF(self.pixmap.rect())

    def paint(self, painter, option, widget=None):
        painter.setOpacity(self.opacity)
        painter.setCompositionMode(self.composition_mode)
        painter.drawPixmap(0, 0, self.pixmap)

    def setPixmap(self, pixmap):
        self.pixmap = pixmap
        self.pixmap = self.pixmap.scaled(self.target_size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.update()

class ImageFusionDialog(QtWidgets.QDialog):
    """
    Dialog that allows the user to choose advances thermography options
    """

    def __init__(self, img_object, ir_img_path, dest_path_preview):
        super().__init__()
        basepath = os.path.dirname(__file__)
        basename = 'fusion'
        uifile = os.path.join(basepath, 'ui/%s.ui' % basename)
        wid.loadUi(uifile, self)

        self.ir_path = ir_img_path
        self.colormap_name = img_object.colormap
        self.n_colors = img_object.n_colors
        self.temperatures = img_object.raw_data_undis

        self.dest_path_preview = dest_path_preview
        self.scene = QGraphicsScene()
        self.view.setScene(self.scene)

        min_temp_scaled = self.scale_temperature(img_object.tmin)
        max_temp_scaled = self.scale_temperature(img_object.tmax)
        min_shown_temp_scaled = self.scale_temperature(img_object.tmin_shown)
        max_shown_temp_scaled = self.scale_temperature(img_object.tmax_shown)

        # range slider for shown data
        self.range_slider_shown = wid.QRangeSlider()
        self.range_slider_shown.handleSize = 10
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

        self.range_slider_shown.lowerValueChanged.connect(self.updateIRImage)
        self.range_slider_shown.upperValueChanged.connect(self.updateIRImage)

        self.range_slider_map.lowerValueChanged.connect(self.recolorIRImage)
        self.range_slider_map.upperValueChanged.connect(self.recolorIRImage)

        # edit labels
        self.label_max.setText(f'{str(min_shown_temp_scaled/100)} °C')
        self.label_min.setText(f'{str(max_shown_temp_scaled/100)} °C')

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
        modes = ['Screen',
                 'SourceOver',
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
        self.view.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

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
        ir_image = ir_image.convertToFormat(QImage.Format_ARGB32)
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



class DialogThParams(QtWidgets.QDialog):
    """
    Dialog that allows the user to choose advances thermography options
    """

    def __init__(self, param, parent=None):
        QtWidgets.QDialog.__init__(self)
        basepath = os.path.dirname(__file__)
        basename = 'dialog_options'
        uifile = os.path.join(basepath, 'ui/%s.ui' % basename)
        wid.loadUi(uifile, self)
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
        wid.loadUi(uifile, self)


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


class Meas3dDialog(QtWidgets.QDialog):
    def __init__(self, rect_annot):
        QtWidgets.QDialog.__init__(self)
        basepath = os.path.dirname(__file__)
        basename = 'meas_dialog_3d'
        uifile = os.path.join(basepath, 'ui/%s.ui' % basename)
        wid.loadUi(uifile, self)

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
        # colormap operation
        if colormap in tt.LIST_CUSTOM_CMAPS:
            custom_cmap = tt.get_custom_cmaps(colormap, n_colors)
        else:
            custom_cmap = cm.get_cmap(colormap, n_colors)

        custom_cmap.set_over(col_high)
        custom_cmap.set_under(col_low)

        xx, yy = np.mgrid[0:self.data.shape[0], 0:self.data.shape[1]]
        self.ax.plot_surface(xx, yy,self.data, rstride=1, cstride=1, linewidth=0, cmap=custom_cmap)
        self.matplot_c.figure.canvas.draw_idle()


