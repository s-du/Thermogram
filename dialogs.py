import os
from PySide6 import QtCore, QtGui, QtWidgets
import logging
import sys
import traceback
from matplotlib import cm
from matplotlib.backends.backend_qt5agg import (
   FigureCanvasQTAgg as FigureCanvas,
   NavigationToolbar2QT)
from matplotlib.figure import Figure

import numpy as np
import cv2

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
        if colormap == 'Artic' or colormap == 'Iron' or colormap == 'Rainbow':
            custom_cmap = tt.get_custom_cmaps(colormap, n_colors)
        else:
            custom_cmap = cm.get_cmap(colormap, n_colors)

        custom_cmap.set_over(col_high)
        custom_cmap.set_under(col_low)

        xx, yy = np.mgrid[0:self.data.shape[0], 0:self.data.shape[1]]
        self.ax.plot_surface(xx, yy,self.data, rstride=1, cstride=1, linewidth=0, cmap=custom_cmap)
        self.matplot_c.figure.canvas.draw_idle()


