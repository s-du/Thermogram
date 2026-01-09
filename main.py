"""
Thermogram - Thermal Image Processing Application

A comprehensive application for processing and analyzing thermal images captured by DJI drones.
Provides tools for visualization, measurement, and analysis of thermal data.

Author: sdu@bbri.be

TODO:
- Allow the user to define a custom folder - OK
- Allow to import only thermal pictures - OK
- Implement 'context' dialog --> Drone info + location
- Implement save/load project folder

"""

# Qt imports
from PyQt6.QtGui import *  # modified from PySide6.QtGui to PyQt6.QtGui
from PyQt6.QtWidgets import *  # modified from PySide6.QtWidgets to PyQt6.QtWidgets
from PyQt6.QtCore import *  # modified from PySide6.QtCore to PyQt6.QtCore
from PyQt6 import uic

# Standard library imports
import os
import json
import copy
import time
from pathlib import Path
from multiprocessing import freeze_support

# Custom libraries
import widgets as wid
import resources as res
import dialogs as dia

from tools import thermal_tools as tt
from tools.report_tools import create_word_report
from utils.config import config, thermal_config
from utils.logger import info, error, debug, warning
from utils.exceptions import ThermogramError, FileOperationError


_TIMING_ENABLED = str(os.environ.get("THERMOGRAM_TIMING", "")).strip().lower() in {"1", "true", "yes", "on"}


def _tlog(message: str):
    if _TIMING_ENABLED:
        print(f"[timing] {message}")


class _PerfTimer:
    def __init__(self, label: str):
        self.label = label
        self.t0 = time.perf_counter()

    def stop(self, extra: str = ""):
        dt_ms = (time.perf_counter() - self.t0) * 1000
        if extra:
            _tlog(f"{self.label}: {dt_ms:.1f} ms ({extra})")
        else:
            _tlog(f"{self.label}: {dt_ms:.1f} ms")

# PARAMETERS
# Application constants
APP_VERSION = config.APP_VERSION
APP_FOLDER = config.APP_FOLDER

# Names
ORIGIN_THERMAL_IMAGES_NAME = config.ORIGIN_THERMAL_IMAGES_NAME
RGB_ORIGINAL_NAME = config.RGB_ORIGINAL_NAME
RGB_CROPPED_NAME = config.RGB_CROPPED_NAME
ORIGIN_TH_FOLDER = config.ORIGIN_TH_FOLDER
RGB_CROPPED_FOLDER = config.RGB_CROPPED_FOLDER
PROC_TH_FOLDER = config.PROC_TH_FOLDER
CUSTOM_IMAGES_FOLDER = config.CUSTOM_IMAGES_FOLDER

RECT_MEAS_NAME = config.RECT_MEAS_NAME
POINT_MEAS_NAME = config.POINT_MEAS_NAME
LINE_MEAS_NAME = config.LINE_MEAS_NAME

VIEWS = config.VIEWS
LEGEND = config.LEGEND

EDGE_COLOR = thermal_config.EDGE_COLOR
EDGE_BLUR_SIZE = thermal_config.EDGE_BLUR_SIZE
EDGE_METHOD = thermal_config.EDGE_METHOD
EDGE_OPACITY = thermal_config.EDGE_OPACITY
N_COLORS = thermal_config.N_COLORS


# FUNCTIONS
def get_next_available_folder(base_folder, app_folder_base_name=APP_FOLDER):
    # Start with the initial folder name (i.e., 'IRLab_1')
    folder_number = 1
    while True:
        # Construct the folder name with the current number
        app_folder = os.path.join(base_folder, f"{app_folder_base_name}{folder_number}")

        # Check if the folder already exists
        if not os.path.exists(app_folder):
            # If it doesn't exist, return the folder name
            return app_folder

        # Increment the folder number and try again
        folder_number += 1


def add_item_in_tree(parent, line):
    item = QStandardItem(line)
    parent.appendRow(item)


def add_icon(img_source, pushButton_object):
    """
    Function to add an icon to a pushButton
    """
    pushButton_object.setIcon(QIcon(img_source))


# CLASSES
class SplashScreen(QSplashScreen):
    """A splash screen displayed during application startup."""

    def __init__(self) -> None:
        """Initialize the splash screen with the application splash image."""
        try:
            splash_path = res.find('img/splash_thermogram.png')
            if not os.path.exists(splash_path):
                error("Splash image not found")
                raise FileOperationError(f"Splash image not found at: {splash_path}")

            pixmap = QPixmap(splash_path)

            # Create splash screen first (super init)
            super().__init__(pixmap)

            # Now safe to access device pixel ratio
            dpr = self.devicePixelRatioF()
            self.setFixedSize(pixmap.size() / dpr)  # Scale the window size appropriately
            self.setWindowFlag(Qt.WindowType.FramelessWindowHint)

            debug("Splash screen initialized successfully")
        except Exception as e:
            error(f"Failed to initialize splash screen: {str(e)}")
            raise ThermogramError("Could not create splash screen") from e


class DroneIrWindow(QMainWindow):
    """Main application window for the IR Lab application.

    Handles the primary user interface and functionality for thermal image processing,
    including image loading, visualization, measurements, and analysis tools.

    Attributes:
        ...
    """

    def __init__(self, parent=None):
        """Initialize the main window and set up the user interface.

        Args:
            parent: Optional parent widget

        Raises:
            ThermogramError: If initialization fails
            FileOperationError: If required UI files are not found
        """
        super(DroneIrWindow, self).__init__(parent)

        # Load the UI file
        ui_file = Path(config.UI_DIR) / 'main_window.ui'
        if not ui_file.exists():
            error(f"UI file not found: {ui_file}")
            raise FileOperationError(f"UI file not found: {ui_file}")

        info(f"Loading UI file: {ui_file}")
        uic.loadUi(str(ui_file), self)

        # Boolean flag to track the stylesheet state
        self.style_active = False

        # Initialize status
        self.update_progress(nb=100, text="Status: Choose image folder")

        # Set up thread pool for background tasks
        self.__pool = QThreadPool()
        self.__pool.setMaxThreadCount(3)
        debug(f"Thread pool initialized with {self.__pool.maxThreadCount()} threads")

        # Initialize core components
        self.initialize_variables()
        self.initialize_tree_view()

        # Edge detection settings
        self.edges = False
        self.edge_color = EDGE_COLOR
        self.edge_blur = False
        self.edge_bil = True
        self.edge_blur_size = EDGE_BLUR_SIZE
        self.edge_method = EDGE_METHOD
        self.edge_opacity = EDGE_OPACITY

        # Other options
        self.skip_update = False

        # Initialize combo box content
        self._setup_combo_boxes()

        # Set up UI components
        self._setup_ui_components()

        # create connections (signals)
        self.create_connections()

        # Add icons to buttons
        self.add_all_icons()

    # BASIC SETUP _________________________________________________________________
    def _setup_combo_boxes(self) -> None:
        """Initialize and populate combo boxes with their respective items."""
        try:
            # Initialize color limit options
            self._out_of_lim = tt.OUT_LIM
            self._out_of_matp = tt.OUT_LIM_MATPLOT

            self._img_post = tt.POST_PROCESS
            self._colormap_list = tt.COLORMAPS
            self._view_list = copy.deepcopy(VIEWS)

            # Set up legend combobox
            self.comboBox_legend_type.clear()
            self.comboBox_legend_type.addItems(LEGEND)

            # Set up colormap combobox
            self.comboBox_palette.clear()
            self.comboBox_palette.addItems(tt.COLORMAP_NAMES)
            self.comboBox_palette.setCurrentIndex(0)

            # Set up edge style combobox
            self.comboBox_edge_overlay_selection.clear()
            self.comboBox_edge_overlay_selection.addItems(tt.EDGE_STYLE_NAMES)
            self.comboBox_edge_overlay_selection.setCurrentIndex(0)

            # Set up color limit combo boxes
            self.comboBox_colors_low.clear()
            self.comboBox_colors_low.addItems(self._out_of_lim)
            self.comboBox_colors_low.setCurrentIndex(0)
            self.comboBox_colors_high.clear()
            self.comboBox_colors_high.addItems(self._out_of_lim)
            self.comboBox_colors_high.setCurrentIndex(0)

            self.comboBox_view.addItems(self._view_list)
            self.comboBox_post.addItems(self._img_post)

            debug("Combo boxes initialized successfully")

        except Exception as e:
            error(f"Failed to initialize combo boxes: {str(e)}")
            raise ThermogramError("Failed to initialize combo boxes") from e

    def _setup_ui_components(self) -> None:
        """Initialize and set up UI components."""
        try:
            # Group dock widgets

            self.tabifyDockWidget(self.dockWidget, self.dockWidget_2)
            self.dockWidget.raise_()  # Make the first dock widget visible by default

            # Set up range slider
            self.range_slider = wid.QRangeSlider(tt.COLORMAPS[0])
            self.range_slider.setEnabled(False)
            self.range_slider.setLowerValue(0)
            self.range_slider.setUpperValue(20)
            self.range_slider.setMinimum(0)
            self.range_slider.setMaximum(20)
            self.range_slider.setFixedHeight(45)

            # Add range slider to layout
            self.horizontalLayout_slider.addWidget(self.range_slider)

            # Sliders options
            self.slider_sensitive = True

            # Create validator for qlineedit
            onlyInt = QIntValidator()
            onlyInt.setRange(0, 999)
            self.lineEdit_colors.setValidator(onlyInt)
            self.n_colors = N_COLORS  # default number of colors
            self.lineEdit_colors.setText(str(N_COLORS))

            # Add actions to action group (mutually exclusive functions)
            ag = QActionGroup(self)
            ag.setExclusive(True)
            ag.addAction(self.actionRectangle_meas)
            ag.addAction(self.actionHand_selector)
            ag.addAction(self.actionSpot_meas)
            ag.addAction(self.actionLine_meas)

            # Set up photo viewers
            self.viewer = wid.PhotoViewer(self)
            self.verticalLayout_8.addWidget(self.viewer)
            self.dual_viewer = wid.DualViewer()
            self.verticalLayout_10.addWidget(self.dual_viewer)

        except Exception as e:
            error(f"Failed to initialize UI components: {str(e)}")
            raise ThermogramError("Failed to initialize UI components") from e

    def initialize_variables(self):
        """Initialize instance variables with default values.

        Sets up various flags and variables used throughout the application for
        tracking state, image properties, and user settings.
        """
        try:
            # Set bool variables
            self.has_rgb = True  # does the dataset have RGB image
            self.rgb_shown = False
            self.save_colormap_info = True  # TODO make use of it... if True, the colormap and temperature options will be stored for each picture

            # Set path variables
            self.custom_images = []
            self.list_rgb_paths = []
            self.list_ir_paths = []
            self.list_z_paths = []
            self.ir_folder = ''
            self.rgb_folder = ''
            self.preview_folder = ''

            # Set other variables
            self.colormap = None
            self.number_custom_pic = 0

            # Image lists
            self.ir_imgs = ''
            self.rgb_imgs = ''
            self.n_imgs = len(self.ir_imgs)
            self.nb_sets = 0

            # list images classes (where to store all measurements and annotations)
            self.images = []
            self.work_image = None

            # histogram
            self.hist_canvas = None

            self.current_view = 0

            # Default thermal options:
            self.thermal_param = {'emissivity': thermal_config.DEFAULT_EMISSIVITY,
                                  'distance': thermal_config.DEFAULT_DISTANCE,
                                  'humidity': thermal_config.DEFAULT_HUMIDITY,
                                  'reflection': thermal_config.DEFAULT_REFLECTION}

            self.lineEdit_emissivity.setText(str(round(self.thermal_param['emissivity'], 2)))
            self.lineEdit_distance.setText(str(round(self.thermal_param['distance'], 2)))
            self.lineEdit_refl_temp.setText(str(round(self.thermal_param['reflection'], 2)))

            # Image iterator to know which image is active
            self.active_image = 0

            debug("Variables initialized successfully")

        except Exception as e:
            error(f"Failed to initialize variables: {str(e)}")
            raise ThermogramError(f"Failed to initialize application: {str(e)}") from e

    def store_variables(self):
        pass

    def initialize_tree_view(self):
        """Initialize the tree view for displaying image measurements and annotations.

        Sets up the tree view model and configures its appearance and behavior.
        The tree view is used to display hierarchical data about measurements
        and annotations made on the thermal images.
        """
        try:
            # Create model (for the tree structure)
            self.model = QStandardItemModel()
            self.treeView.setModel(self.model)
            self.treeView.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            self.treeView.customContextMenuRequested.connect(self.onContextMenu)

            # Add measurement and annotations categories to tree view
            add_item_in_tree(self.model, RECT_MEAS_NAME)
            add_item_in_tree(self.model, POINT_MEAS_NAME)
            add_item_in_tree(self.model, LINE_MEAS_NAME)
            self.model.setHeaderData(0, Qt.Orientation.Horizontal, 'Added Data')
        except Exception as e:
            error(f"Failed to initialize tree view: {str(e)}")
            raise ThermogramError("Failed to initialize tree view") from e

    def create_connections(self):
        """Create signal-slot connections for UI elements.

        Connects various UI elements (buttons, menus, etc.) to their corresponding
        handler methods. This sets up the interactive behavior of the application.
        """
        try:
            # IO actions
            self.actionLoad_folder.triggered.connect(self.load_folder_phase1)
            self.actionReset_all.triggered.connect(self.full_reset)
            self.actionToggle_stylesheet.triggered.connect(self.toggle_stylesheet)

            # Processing actions
            self.actionRectangle_meas.triggered.connect(self.rectangle_meas)
            self.actionSpot_meas.triggered.connect(self.point_meas)
            self.actionLine_meas.triggered.connect(self.line_meas)
            self.action3D_temperature.triggered.connect(self.show_viz_threed)
            self.actionFind_maxima.triggered.connect(self.find_maxima)
            self.actionDetect_object.triggered.connect(self.detect_object)
            self.actionCompose.triggered.connect(self.compose_pic)

            # Export action
            self.actionSave_Image.triggered.connect(self.save_image)
            self.actionProcess_all.triggered.connect(self.batch_export)
            self.actionCreate_anim.triggered.connect(self.export_anim)
            self.actionCreate_Report.triggered.connect(self.create_report)

            # Other actions
            self.actionInfo.triggered.connect(self.show_info)
            self.actionRadiometric_parameters.triggered.connect(self.show_radio_dock)
            self.actionEdge_Mix.triggered.connect(self.show_edge_dock)

            # Viewers
            self.viewer.endDrawing_rect_meas.connect(self.add_rect_meas)
            self.viewer.endDrawing_point_meas.connect(self.add_point_meas)
            self.viewer.endDrawing_line_meas.connect(self.add_line_meas)

            # PushButtons
            self.pushButton_left.clicked.connect(lambda: self.update_img_to_preview('minus'))
            self.pushButton_right.clicked.connect(lambda: self.update_img_to_preview('plus'))
            self.pushButton_estimate.clicked.connect(self.estimate_temp)
            self.pushButton_meas_color.clicked.connect(self.change_meas_color)
            self.pushButton_match.clicked.connect(self.image_matching)
            self.pushButton_edge_options.clicked.connect(self.edge_options)
            self.pushButton_delete_points.clicked.connect(lambda: self.remove_annotations('point'))
            self.pushButton_delete_lines.clicked.connect(lambda: self.remove_annotations('line'))
            self.pushButton_delete_area.clicked.connect(lambda: self.remove_annotations('area'))
            self.pushButton_reset_range.clicked.connect(self.reset_temp_range)
            self.pushButton_heatflow.clicked.connect(self.viz_heatflow)
            self.pushButton_optimhisto.clicked.connect(self.optimal_range)
            self.pushButton_add_custom_palette.clicked.connect(self.add_palette)

            # Dropdowns Comboboxes
            self.comboBox_palette.currentIndexChanged.connect(self.update_img_preview)
            self.comboBox_colors_low.currentIndexChanged.connect(self.update_img_preview)
            self.comboBox_colors_high.currentIndexChanged.connect(self.update_img_preview)
            self.comboBox_post.currentIndexChanged.connect(self.update_img_preview)
            self.comboBox_img.currentIndexChanged.connect(lambda: self.update_img_to_preview('other'))
            self.comboBox_view.currentIndexChanged.connect(self.update_img_preview)
            self.comboBox_edge_overlay_selection.currentIndexChanged.connect(self.change_edge_style)
            self.comboBox_legend_type.currentIndexChanged.connect(self.update_img_preview)

            # Line edits
            self.lineEdit_min_temp.editingFinished.connect(self.adapt_slider_values)
            self.lineEdit_max_temp.editingFinished.connect(self.adapt_slider_values)
            self.lineEdit_colors.editingFinished.connect(self.update_img_preview)
            self.lineEdit_emissivity.editingFinished.connect(self.define_thermal_parameters)
            self.lineEdit_distance.editingFinished.connect(self.define_thermal_parameters)
            self.lineEdit_refl_temp.editingFinished.connect(self.define_thermal_parameters)

            # Double slider
            self.range_slider.lowerValueChanged.connect(self.change_line_edits)
            self.range_slider.upperValueChanged.connect(self.change_line_edits)

            # Checkboxes
            self.checkBox_legend.stateChanged.connect(self.toggle_legend)
            self.checkBox_edges.stateChanged.connect(self.activate_edges)
            if hasattr(self, 'checkBox_report'):
                self.checkBox_report.stateChanged.connect(self.toggle_report_inclusion)
                
            # Remarks text edit
            if hasattr(self, 'plainTextEdit_remarks'):
                self.plainTextEdit_remarks.textChanged.connect(self.save_remarks)

            # tab widget
            self.tabWidget.currentChanged.connect(self.on_tab_change)

        except Exception as e:
            error(f"Failed to create UI connections: {str(e)}")
            raise ThermogramError("Failed to create UI connections") from e

    # THERMAL PARAMETERS FUNCTIONS __________________________________________________________________________
    def define_thermal_parameters(self):
        try:
            # Store original values to restore if validation fails
            original_thermal_param = copy.deepcopy(self.thermal_param)

            em = float(self.lineEdit_emissivity.text())
            # check if value is acceptable
            if em < 0.1 or em > 1:
                self.lineEdit_emissivity.setText(str(round(self.thermal_param['emissivity'], 2)))
                raise ValueError
            else:
                self.thermal_param['emissivity'] = em

            dist = float(self.lineEdit_distance.text())
            if dist < 1 or dist > 25:
                self.lineEdit_distance.setText(str(round(self.thermal_param['distance'], 1)))
                raise ValueError
            else:
                self.thermal_param['distance'] = dist

            refl_temp = float(self.lineEdit_refl_temp.text())
            if refl_temp < -40 or refl_temp > 500:
                self.lineEdit_refl_temp.setText(str(round(self.thermal_param['reflection'], 1)))
                raise ValueError
            else:
                self.thermal_param['reflection'] = refl_temp

            # Now we can safely switch image data and update preview
            self.switch_image_data()

            self.retrace_items()
            self.update_img_preview()

        except ValueError:
            QMessageBox.warning(self, "Warning",
                                "Oops! Some of the values are not valid!")
            self.thermal_param = original_thermal_param

    # TEMPERATURE RANGE _________________________________________________________________
    def adapt_slider_values(self):
        """Update temperature range based on slider values.

        Updates the temperature range display and image visualization based on
        the current values of the LineEdits. Includes validation to ensure
        values stay within valid bounds.
        """
        # Temporarily disable slider sensitivity to avoid feedback loops
        try:
            # Temporarily disable slider sensitivity to avoid loops
            self.slider_sensitive = False

            # Get current temperature values from Linedits
            tmin = float(self.lineEdit_min_temp.text())
            tmax = float(self.lineEdit_max_temp.text())

            # Check boundaries validity
            if not self.skip_update:
                if tmax <= tmin:
                    QMessageBox.warning(self, "Warning",
                                        "Oops! A least one of the temperatures is not valid.  Try again...")

                    self.lineEdit_min_temp.setText(str(round(self.work_image.tmin_shown, 2)))
                    self.lineEdit_max_temp.setText(str(round(self.work_image.tmax_shown, 2)))
                    return

            # Adapt sliders
            self.range_slider.setLowerValue(tmin * 100)
            self.range_slider.setUpperValue(tmax * 100)
            debug(f"Temperature range updated: {tmin:.1f} - {tmax:.1f}")

        except Exception as e:
            error(f"Error updating temperature range: {str(e)}")
            raise ThermogramError("Failed to update temperature range") from e
            self.lineEdit_min_temp.setText(str(round(self.work_image.tmin_shown, 2)))
            self.lineEdit_max_temp.setText(str(round(self.work_image.tmax_shown, 2)))
        finally:
            # Re-enable slider sensitivity
            self.slider_sensitive = True
            self.update_img_preview()

    def change_line_edits(self, value):
        if self.slider_sensitive:
            tmin = self.range_slider.lowerValue() / 100.0  # Adjust if you used scaling
            tmax = self.range_slider.upperValue() / 100.0

            self.lineEdit_min_temp.setText(str(round(tmin, 2)))
            self.lineEdit_max_temp.setText(str(round(tmax, 2)))

            self.update_img_preview()

    def optimal_range(self):
        tmin_shown, tmax_shown = self.work_image.compute_optimal_temp_range()
        self.lineEdit_min_temp.setText(str(round(tmin_shown, 2)))
        self.lineEdit_max_temp.setText(str(round(tmax_shown, 2)))
        self.range_slider.setLowerValue(tmin_shown * 100)
        self.range_slider.setUpperValue(tmax_shown * 100)

    def reset_temp_range(self):
        """Reset the temperature range to the full range of the current image.

        Updates the temperature range slider and display to show the full
        temperature range of the current thermal image. This resets any user-defined
        temperature range limits.
        """
        try:
            self.work_image.update_data_from_param(copy.deepcopy(self.work_image.thermal_param))
            tmin, tmax, _, _ = self.work_image.get_temp_data()
            # Fill values lineedits
            self.lineEdit_min_temp.setText(str(round(tmin, 2)))
            self.lineEdit_max_temp.setText(str(round(tmax, 2)))
            self.range_slider.setLowerValue(tmin * 100)
            self.range_slider.setUpperValue(tmax * 100)
            self.range_slider.setMinimum(int(tmin * 100))
            self.range_slider.setMaximum(int(tmax * 100))

            debug(f"Temperature range reset to {tmin:.1f} - {tmax:.1f}")

        except Exception as e:
            error(f"Failed to reset temperature range: {str(e)}")
            raise ThermogramError("Failed to reset temperature range") from e

    def estimate_temp(self):
        ref_pic_name = QFileDialog.getOpenFileName(self, 'Open file',
                                                   self.ir_folder, "Image files (*.jpg *.JPG *.gif)")
        img_path = ref_pic_name[0]
        if img_path != '':
            tmin, tmax = tt.compute_delta(img_path, self.thermal_param)
            self.lineEdit_min_temp.setText(str(round(tmin, 2)))
            self.lineEdit_max_temp.setText(str(round(tmax, 2)))

        self.update_img_preview()

    # PALETTE _________________________________________________________________
    def add_palette(self):
        """
        Opens a dialog to create a custom color palette and adds it to the available palettes.
        The user can define color steps and preview the palette before saving it.
        """
        from dialogs import CustomPaletteDialog
        import matplotlib.colors as mcol
        import numpy as np

        # Create and show the custom palette dialog
        dialog = CustomPaletteDialog(self)
        if dialog.exec():
            # Get the palette data from the dialog
            palette_data = dialog.get_palette_data()
            if not palette_data:
                self.statusbar.showMessage("Invalid palette data", 3000)
                return

            palette_name = palette_data['name']
            colors = palette_data['colors']

            # Check if the name already exists
            if palette_name in tt.LIST_CUSTOM_NAMES or palette_name in tt.COLORMAP_NAMES:
                # Add a suffix to make it unique
                i = 1
                original_name = palette_name
                while palette_name in tt.LIST_CUSTOM_NAMES or palette_name in tt.COLORMAP_NAMES:
                    palette_name = f"{original_name}_{i}"
                    i += 1

                info(f"Renamed palette to '{palette_name}' to avoid name collision")

            # Add the new palette to the global lists
            tt.LIST_CUSTOM_NAMES.append(palette_name)
            tt.COLORMAP_NAMES.append(palette_name)
            tt.COLORMAPS.append(palette_name)

            # Create and register the new colormap
            tt.register_custom_cmap(palette_name, colors)

            # Update the combobox with the new palette
            current_index = self.comboBox_palette.currentIndex()
            self.comboBox_palette.clear()
            self.comboBox_palette.addItems(tt.COLORMAP_NAMES)

            # Find the index of the new palette and select it
            new_index = tt.COLORMAP_NAMES.index(palette_name)
            self.comboBox_palette.setCurrentIndex(new_index)

            # Show success message
            self.statusbar.showMessage(f"Custom palette '{palette_name}' created successfully", 3000)
            info(f"Created custom palette: {palette_name}")

    # ANNOTATIONS _________________________________________________________________
    def viz_heatflow(self):
        tt.create_vector_plot(self.work_image)

    def add_rect_meas(self, rect_item):
        """
        Add a region of interest coming from the rectangle tool
        :param rect_item: a rectangle item from the viewer
        """
        # create annotation (object)
        new_rect_annot = tt.RectMeas(rect_item)

        # get image data
        if self.has_rgb:
            rgb_path = os.path.join(self.rgb_folder, self.rgb_imgs[self.active_image])
        else:
            rgb_path = ''
            new_rect_annot.has_rgb = False

        ir_path = self.dest_path_post
        coords = new_rect_annot.get_coord_from_item(rect_item)
        roi_ir, roi_rgb = new_rect_annot.compute_data(coords, self.work_image.raw_data_undis, rgb_path, ir_path)

        roi_ir_path = os.path.join(self.preview_folder, 'roi_ir.JPG')
        tt.cv_write_all_path(roi_ir, roi_ir_path)

        if self.has_rgb:
            roi_rgb_path = os.path.join(self.preview_folder, 'roi_rgb.JPG')
            tt.cv_write_all_path(roi_rgb, roi_rgb_path)
        else:
            roi_rgb_path = ''

        # add interesting data to viewer
        new_rect_annot.compute_highlights()
        new_rect_annot.create_items()
        for item in new_rect_annot.ellipse_items:
            self.viewer.add_item_from_annot(item)
        for item in new_rect_annot.text_items:
            self.viewer.add_item_from_annot(item)

        # create description name
        self.work_image.nb_meas_rect += 1
        desc = 'rect_measure_' + str(self.work_image.nb_meas_rect)
        new_rect_annot.name = desc

        # add annotation to the image annotation list
        self.work_image.meas_rect_list.append(new_rect_annot)

        rect_cat = self.model.findItems(RECT_MEAS_NAME)
        add_item_in_tree(rect_cat[0], desc)
        self.treeView.expandAll()

        # bring data 3d figure
        if self.has_rgb:
            dialog = dia.Meas3dDialog(new_rect_annot)
            dialog.dual_view.load_images_from_path(roi_rgb_path, roi_ir_path)
        else:
            dialog = dia.Meas3dDialog_simple(new_rect_annot)

        dialog.surface_from_image_matplot(self.work_image.colormap, self.work_image.n_colors,
                                          self.work_image.user_lim_col_low,
                                          self.work_image.user_lim_col_high)
        if dialog.exec():
            pass

        # switch back to hand tool
        self.hand_pan()

    def add_line_meas(self, line_item):
        # create annotation (object)
        new_line_annot = tt.LineMeas(line_item)

        # compute stuff
        new_line_annot.compute_data(self.work_image.raw_data_undis)
        new_line_annot.compute_highlights()
        new_line_annot.create_items()
        for item in new_line_annot.spot_items + new_line_annot.text_items:
            self.viewer.add_item_from_annot(item)

        self.work_image.nb_meas_line += 1
        desc = 'line_measure_' + str(self.work_image.nb_meas_line)
        new_line_annot.name = desc

        # add annotation to the image annotation list
        self.work_image.meas_line_list.append(new_line_annot)

        line_cat = self.model.findItems(LINE_MEAS_NAME)
        add_item_in_tree(line_cat[0], desc)

        # bring data figure
        dialog = dia.MeasLineDialog(new_line_annot.data_roi)
        if dialog.exec():
            pass

        self.hand_pan()

    def add_point_meas(self, qpointf):
        # create annotation (object)
        new_pt_annot = tt.PointMeas(qpointf)
        new_pt_annot.temp = self.work_image.raw_data_undis[int(qpointf.y()), int(qpointf.x())]
        new_pt_annot.create_items()
        self.viewer.add_item_from_annot(new_pt_annot.ellipse_item)
        self.viewer.add_item_from_annot(new_pt_annot.text_item)

        # create description name
        self.work_image.nb_meas_point += 1
        desc = 'spot_measure_' + str(self.work_image.nb_meas_point)
        new_pt_annot.name = desc

        # add annotation to the image annotation list
        self.work_image.meas_point_list.append(new_pt_annot)

        point_cat = self.model.findItems(POINT_MEAS_NAME)
        add_item_in_tree(point_cat[0], desc)
        self.hand_pan()

    # measurements methods
    def rectangle_meas(self):
        """Activate rectangle measurement mode."""
        try:
            if self.actionRectangle_meas.isChecked():
                # Activate drawing tool
                self.viewer.rect_meas = True
                self.viewer.toggleDragMode()
        except Exception as e:
            error(f"Failed to activate rectangle measurement: {str(e)}")

    def point_meas(self):
        """Activate point measurement mode."""
        try:
            if self.actionSpot_meas.isChecked():
                # Activate drawing tool
                self.viewer.point_meas = True
                self.viewer.toggleDragMode()
        except Exception as e:
            error(f"Failed to activate point measurement: {str(e)}")

    def line_meas(self):
        """Activate line measurement mode."""
        try:
            if self.actionLine_meas.isChecked():
                # Activate drawing tool
                self.viewer.line_meas = True
                self.viewer.toggleDragMode()
        except Exception as e:
            error(f"Failed to activate line measurement: {str(e)}")

    def remove_annotations(self, type):
        if type == 'point':
            self.images[self.active_image].meas_point_list = []
        elif type == 'line':
            self.images[self.active_image].meas_line_list = []
        elif type == 'area':
            self.images[self.active_image].meas_rect_list = []

        self.retrace_items()

    def retrace_items(self):
        self.viewer.clean_scene()

        # Tree operations
        self.model = QStandardItemModel()
        self.treeView.setModel(self.model)

        add_item_in_tree(self.model, RECT_MEAS_NAME)
        add_item_in_tree(self.model, POINT_MEAS_NAME)
        add_item_in_tree(self.model, LINE_MEAS_NAME)

        self.model.setHeaderData(0, Qt.Orientation.Horizontal, 'Meas. Data')

        point_cat = self.model.findItems(POINT_MEAS_NAME)
        rect_cat = self.model.findItems(RECT_MEAS_NAME)
        line_cat = self.model.findItems(LINE_MEAS_NAME)

        for i, point in enumerate(self.work_image.meas_point_list):
            desc = point.name
            add_item_in_tree(point_cat[0], desc)

            self.viewer.add_item_from_annot(point.ellipse_item)
            self.viewer.add_item_from_annot(point.text_item)

        for i, rect in enumerate(self.work_image.meas_rect_list):
            desc = rect.name
            add_item_in_tree(rect_cat[0], desc)

            self.viewer.add_item_from_annot(rect.main_item)
            for item in rect.ellipse_items:
                self.viewer.add_item_from_annot(item)
            for item in rect.text_items:
                self.viewer.add_item_from_annot(item)

        for i, line in enumerate(self.work_image.meas_line_list):
            desc = line.name
            add_item_in_tree(line_cat[0], desc)

            for item in line.spot_items + line.text_items:
                self.viewer.add_item_from_annot(item)

            self.viewer.add_item_from_annot(line.main_item)

    def change_meas_color(self):
        self.viewer.change_meas_color()
        self.switch_image_data()

    # ANNOTATION CONTEXT MENU IN TREEVIEW _________________________________________________
    def onContextMenu(self, point):
        print("Context menu requested at:", point)  # Debug print
        # Get the index of the item that was clicked
        index = self.treeView.indexAt(point)
        if not index.isValid():
            return

        item = self.model.itemFromIndex(index)

        # Check if the clicked item is a second level item (annotation object)
        if item and item.parent() and item.parent().text() in [RECT_MEAS_NAME, POINT_MEAS_NAME, LINE_MEAS_NAME]:
            # Create Context Menu
            contextMenu = QMenu(self.treeView)  # Use None as parent if self.treeView causes issues

            deleteAction = contextMenu.addAction("Delete Annotation")
            showAction = contextMenu.addAction("Show Annotation")

            # Connect actions to methods
            deleteAction.triggered.connect(lambda: self.deleteAnnotation(item))
            showAction.triggered.connect(lambda: self.showAnnotation(item))

            # Display the menu
            global_point = self.treeView.viewport().mapToGlobal(point)
            contextMenu.exec(global_point)

    def deleteAnnotation(self, item):
        # Implement the logic to delete the annotation
        lookup_text = item.text()
        print(f"Deleting annotation: {lookup_text}")
        if 'line' in lookup_text:
            for i, annot in enumerate(self.images[self.active_image].meas_line_list):
                if annot.name == lookup_text:
                    remove_index = i
                    break

            if remove_index != -1:
                self.images[self.active_image].meas_line_list.pop(remove_index)

        if 'rect' in lookup_text:
            for i, annot in enumerate(self.images[self.active_image].meas_rect_list):
                if annot.name == lookup_text:
                    remove_index = i
                    break

            if remove_index != -1:
                self.images[self.active_image].meas_rect_list.pop(remove_index)

        if 'spot' in lookup_text:
            for i, annot in enumerate(self.images[self.active_image].meas_point_list):
                if annot.name == lookup_text:
                    remove_index = i
                    break

            if remove_index != -1:
                self.images[self.active_image].meas_point_list.pop(remove_index)

        # retrace items
        self.retrace_items()

    def showAnnotation(self, item):
        # Implement the logic to show the annotation
        lookup_text = item.text()
        print(f"Deleting annotation: {lookup_text}")

        if 'line' in lookup_text:
            for i, annot in enumerate(self.images[self.active_image].meas_line_list):
                if annot.name == lookup_text:
                    interest = self.images[self.active_image].meas_line_list[i]

                    # bring data 2d figure
                    dialog = dia.MeasLineDialog(interest.data_roi)
                    if dialog.exec():
                        pass
        if 'rect' in lookup_text:
            for i, annot in enumerate(self.images[self.active_image].meas_rect_list):
                if annot.name == lookup_text:
                    interest = self.images[self.active_image].meas_rect_list[i]

                    # bring data 3d figure
                    if self.has_rgb:
                        rgb_path = os.path.join(self.rgb_folder, self.rgb_imgs[self.active_image])
                    else:
                        rgb_path = ''

                    ir_path = self.dest_path_post
                    coords = interest.get_coord_from_item(interest.main_item)
                    roi_ir, roi_rgb = interest.compute_data(coords, self.work_image.raw_data_undis, rgb_path,
                                                            ir_path)
                    roi_ir_path = os.path.join(self.preview_folder, 'roi_ir.JPG')
                    tt.cv_write_all_path(roi_ir, roi_ir_path)

                    if self.has_rgb:
                        roi_rgb_path = os.path.join(self.preview_folder, 'roi_rgb.JPG')
                        tt.cv_write_all_path(roi_rgb, roi_rgb_path)

                        dialog = dia.Meas3dDialog(interest)
                        dialog.dual_view.load_images_from_path(roi_rgb_path, roi_ir_path)

                    else:
                        dialog = dia.Meas3dDialog_simple(interest)

                    dialog.surface_from_image_matplot(self.work_image.colormap, self.work_image.n_colors,
                                                      self.work_image.user_lim_col_low,
                                                      self.work_image.user_lim_col_high)
                    if dialog.exec():
                        pass

        if 'spot' in lookup_text:
            for i, annot in enumerate(self.images[self.active_image].meas_point_list):
                if annot.name == lookup_text:
                    interest = self.images[self.active_image].meas_point_list[i]

        # show dialog:

    # LOAD AND SAVE ACTIONS ______________________________________________________________________________
    def toggle_report_inclusion(self, state):
        """Update the current image's include_in_report property based on checkbox state"""
        if hasattr(self, 'work_image'):
            self.work_image.include_in_report = (state == Qt.CheckState.Checked)
            
    def save_remarks(self):
        """Save the remarks text to the current image"""
        if hasattr(self, 'work_image') and hasattr(self, 'plainTextEdit_remarks'):
            self.work_image.remarks = self.plainTextEdit_remarks.toPlainText()
            
    def update_image_info_label(self):
        """Update the image info label with EXIF data from the current image"""
        if not hasattr(self, 'work_image') or not hasattr(self, 'label_img_info'):
            return
            
        # Get EXIF data
        exif_data = self.work_image.exif
        
        # Format the information to display
        info_text = "<b>Image Information:</b><br>"
        
        # Add filename
        if hasattr(self.work_image, 'path') and self.work_image.path:
            _, filename = os.path.split(self.work_image.path)
            info_text += f"<b>File:</b> {filename}<br>"
        
        # Try to extract common EXIF tags
        try:
            # Common EXIF tags and their IDs
            tags_to_display = {
                271: "Make",            # Camera manufacturer
                272: "Model",           # Camera model
                306: "DateTime",        # Date and time
                36867: "DateTimeOriginal",  # Original date and time
                37377: "ShutterSpeed",  # Shutter speed
                37378: "Aperture",      # Aperture
                37379: "BrightnessValue", # Brightness
                37380: "ExposureCompensation", # Exposure bias
                37383: "MeteringMode",  # Metering mode
                37385: "Flash",         # Flash
                37386: "FocalLength",   # Focal length
                41728: "FileSource",    # File source
                41729: "SceneType"      # Scene type
            }
            
            for tag_id, tag_name in tags_to_display.items():
                if tag_id in exif_data:
                    value = exif_data[tag_id]
                    # Format certain values
                    if tag_id == 37377:  # ShutterSpeed
                        if value > 0:
                            value = f"1/{int(2**value)}"
                    elif tag_id == 37378:  # Aperture
                        value = f"f/{round(2**(value/2), 1)}"
                    elif tag_id == 37386:  # FocalLength
                        value = f"{value}mm"
                        
                    info_text += f"<b>{tag_name}:</b> {value}<br>"
        except Exception as e:
            info_text += f"<i>Error reading EXIF data: {str(e)}</i><br>"
        
        # Add thermal information
        info_text += f"<b>Temperature Range:</b> {self.work_image.tmin:.1f}°C to {self.work_image.tmax:.1f}°C<br>"
        
        # Set the text to the label
        self.label_img_info.setText(info_text)
        self.label_img_info.setTextFormat(Qt.TextFormat.RichText)
    
    def create_report(self):
        # Check if there are images to include in the report
        if not self.images:
            QMessageBox.warning(self, "No Images", "There are no images to include in the report.")
            return
            
        # Open the report configuration dialog
        from dialogs import ReportConfigDialog
        dialog = ReportConfigDialog(self, images=self.images)
        
        # If the dialog is accepted, create the report
        if dialog.exec():
            # Get the configuration from the dialog
            config = dialog.get_report_config()
            
            # Create a folder for annotated images
            import os
            import tempfile
            annotated_images_folder = os.path.join(self.custom_images_folder, "report_images")
            if not os.path.exists(annotated_images_folder):
                os.makedirs(annotated_images_folder)
            
            # Save the current state to restore later
            current_image_index = self.active_image
            
            # Get the selected images
            images_to_include = config.get('images_to_include', [])
            if not images_to_include:
                QMessageBox.warning(self, "No Images Selected", "Please select at least one image to include in the report.")
                return
                
            try:
                # Dictionary to store annotated image paths
                annotated_image_paths = [None] * len(self.images)
                
                # For each selected image, switch to it, save the annotated view, and store the path
                for idx in images_to_include:
                    # Switch to this image
                    self.active_image = idx
                    self.work_image = self.images[idx]
                    self.update_img_preview(refresh_dual=True)
                    
                    # Save the current view with annotations
                    annotated_filename = f"annotated_image_{idx}.png"
                    annotated_path = os.path.join(annotated_images_folder, annotated_filename)
                    self.save_image(folder=annotated_images_folder, filename=annotated_filename)
                    
                    # Store the path for the report
                    annotated_image_paths[idx] = annotated_path
                
                # Restore the original image
                self.active_image = current_image_index
                self.work_image = self.images[current_image_index]
                self.update_img_preview(refresh_dual=True)
                
                # Create the report with the configured settings and annotated images
                from tools.report_tools import create_word_report
                create_word_report(
                    output_path=config['output_path'],
                    objectives_text=config['objectives_text'],
                    site_conditions_text=config['site_conditions_text'],
                    flight_details_text=config['flight_details_text'],
                    processed_images=self.images,
                    images_to_include=images_to_include,
                    style_template=config['style_template'],
                    include_summary=config['include_summary'],
                    annotated_image_paths=annotated_image_paths,
                    report_title=config.get('report_title', 'Infrared Survey Report'),
                    report_subtitle=config.get('report_subtitle')
                )
                
                # Show success message
                QMessageBox.information(
                    self, 
                    "Report Created", 
                    f"Report successfully created at:\n{config['output_path']}"
                )
                
                # Open the folder containing the report
                import os
                import subprocess
                folder_path = os.path.dirname(os.path.abspath(config['output_path']))
                if os.path.exists(folder_path):
                    subprocess.Popen(f'explorer "{folder_path}"')
                    
            except Exception as e:
                # Show error message if report creation fails
                QMessageBox.critical(
                    self, 
                    "Error Creating Report", 
                    f"An error occurred while creating the report:\n{str(e)}"
                )

    def export_anim(self):
        # select folder
        # Define the starting directory
        starting_directory = self.app_folder  # Change this to the desired starting path

        # Open the file selection dialog for multiple images
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Images",
            starting_directory,
            "Images (*.png *.xpm *.jpg *.jpeg *.bmp *.tiff *.gif)"  # File filter to select image types
        )

        # If the user selects files, update the label
        if files:
            print(files)
            self.update_progress(nb=20, text="Status: Video creation!")
            video_dir = self.app_folder  # Adjust the path to your video's folder
            video_file = "animation_thermal.mp4"  # Adjust to your video file name if needed
            video_path = os.path.join(video_dir, video_file)

            tt.create_video(files, video_path, 3)

            self.update_progress(nb=100, text="Continue analyses!")

            msg_box = QMessageBox()
            msg_box.setWindowTitle("Video Location")
            msg_box.setText(f"Your video is located here:\n{video_path}")

            msg_box.exec()

    def load_folder_phase1(self):
        """Open a folder dialog and initiate the folder loading process.

        This is phase 1 of the folder loading process, which handles:
        1. Opening a folder selection dialog
        2. Validating the selected folder
        3. Setting up project directories
        4. Initiating phase 2 of the loading process
        """
        # Warning message (new project)
        if self.list_rgb_paths != []:
            qm = QMessageBox
            reply = qm.question(self, '', "Are you sure ? It will create a new project",
                                qm.StandardButton.Yes | qm.StandardButton.No)

            if reply == qm.StandardButton.Yes:
                # reset all data
                self.full_reset()

            else:
                return

        # Get the image folder from the user
        folder = str(QFileDialog.getExistingDirectory(self, "Select Directory"))

        # Detect and organize images
        if not folder == '':  # if user cancel selection, stop function
            t_total = _PerfTimer("load_folder_phase1")
            self.main_folder = folder
            self.app_folder = get_next_available_folder(self.main_folder)

            # Update json path
            self.json_file = os.path.join(self.app_folder, 'data.json')

            # Update status
            text_status = 'loading images...'
            self.update_progress(nb=0, text=text_status)

            # Identify content of the folder
            t_scan = _PerfTimer("scan folder")
            self.list_rgb_paths, self.list_ir_paths, self.list_z_paths = tt.list_th_rgb_images_from_res(self.main_folder)
            t_scan.stop(extra=f"ir={len(self.list_ir_paths)} rgb={len(self.list_rgb_paths)}")

            # Create some sub folders for storing images
            self.original_th_img_folder = os.path.join(self.app_folder, ORIGIN_TH_FOLDER)
            self.rgb_crop_img_folder = os.path.join(self.app_folder, RGB_CROPPED_FOLDER)
            self.custom_images_folder = os.path.join(self.app_folder, CUSTOM_IMAGES_FOLDER)

            # If the sub folders do not exist, create them
            if not os.path.exists(self.app_folder):
                os.mkdir(self.app_folder)
            if not os.path.exists(self.original_th_img_folder):
                os.mkdir(self.original_th_img_folder)
            if not os.path.exists(self.rgb_crop_img_folder):
                os.mkdir(self.rgb_crop_img_folder)
            if not os.path.exists(self.custom_images_folder):
                os.mkdir(self.custom_images_folder)

            # Get drone model
            t_model = _PerfTimer("init drone model")
            drone_name = tt.get_drone_model(self.list_ir_paths[0])
            self.drone_model = tt.DroneModel(drone_name)
            t_model.stop(extra=str(drone_name))

            # Create 'undistorder' based on drone model
            t_undis = _PerfTimer("init undistorter")
            self.ir_undistorder = tt.CameraUndistorter(self.drone_model.ir_xml_path)
            t_undis.stop()

            # Create dictionary with main information
            dictionary = {
                "Drone model": drone_name,
                "Number of image pairs": str(len(self.list_ir_paths)),
                "rgb_paths": self.list_rgb_paths,
                "ir_paths": self.list_ir_paths,
                "zoom_paths": self.list_z_paths
            }
            self.write_json(dictionary)  # store original images paths in a JSON

            text_status = 'preparing project...'
            self.update_progress(nb=10, text=text_status)

            # does it have RGB?
            if not self.list_rgb_paths:
                print('No RGB here!')
                self.has_rgb = False
                self.load_folder_phase2()
                t_total.stop(extra="no rgb")

            else:
                    text_status = 'creating rgb miniatures...'
                    self.update_progress(nb=20, text=text_status)

                    self._timing_rgb_mini_start = time.perf_counter()

                    worker_1 = tt.RunnerMiniature(self.list_rgb_paths, self.drone_model, 60, self.rgb_crop_img_folder,
                                                  20,
                                                  100)
                    worker_1.signals.progressed.connect(lambda value: self.update_progress(value))
                    worker_1.signals.messaged.connect(lambda string: self.update_progress(text=string))

                    self.__pool.start(worker_1)
                    def _on_mini_done():
                        if hasattr(self, "_timing_rgb_mini_start"):
                            dt_ms = (time.perf_counter() - self._timing_rgb_mini_start) * 1000
                            _tlog(f"rgb miniatures: {dt_ms:.1f} ms")
                        self.load_folder_phase2()
                        t_total.stop(extra="with rgb")

                    worker_1.signals.finished.connect(_on_mini_done)

    def load_folder_phase2(self):
        """Execute phase 2 of folder loading process for thermal images.

        Handles the second phase of loading a new folder of thermal images,
        including image list updates, UI initialization, and enabling relevant
        controls.
        """

        t_phase2 = _PerfTimer("load_folder_phase2")
        # Get list to main window
        t_list = _PerfTimer("update_img_list")
        self.update_img_list()
        t_list.stop()
        self.viewer.fitInView()
        t_phase2.stop()

        # Activate buttons and options

        #   dock widgets
        self.lineEdit_max_temp.setEnabled(True)
        self.lineEdit_min_temp.setEnabled(True)
        self.lineEdit_distance.setEnabled(True)
        self.lineEdit_emissivity.setEnabled(True)
        self.lineEdit_refl_temp.setEnabled(True)
        self.pushButton_estimate.setEnabled(True)
        self.pushButton_reset_range.setEnabled(True)
        self.pushButton_optimhisto.setEnabled(True)

        self.comboBox_palette.setEnabled(True)
        self.lineEdit_colors.setEnabled(True)
        self.comboBox_colors_low.setEnabled(True)
        self.comboBox_colors_high.setEnabled(True)
        self.comboBox_post.setEnabled(True)
        self.range_slider.setEnabled(True)

        #   image navigation
        self.dockWidget.setEnabled(True)
        self.pushButton_right.setEnabled(True)
        self.comboBox_view.setEnabled(True)
        self.comboBox_img.setEnabled(True)

        #   other buttons
        self.pushButton_heatflow.setEnabled(True)
        self.pushButton_delete_area.setEnabled(True)
        self.pushButton_delete_points.setEnabled(True)
        self.pushButton_delete_lines.setEnabled(True)

        # Enable action
        self.actionHand_selector.setEnabled(True)
        self.actionHand_selector.setChecked(True)

        self.actionRectangle_meas.setEnabled(True)
        self.actionSpot_meas.setEnabled(True)
        self.actionLine_meas.setEnabled(True)
        self.action3D_temperature.setEnabled(True)
        self.actionFind_maxima.setEnabled(True)
        self.actionProcess_all.setEnabled(True)
        self.actionCreate_anim.setEnabled(True)
        self.actionSave_Image.setEnabled(True)

        # RGB condition
        if self.has_rgb:
            self.checkBox_edges.setEnabled(True)
            self.pushButton_edge_options.setEnabled(True)
            self.actionCompose.setEnabled(True)
            self.pushButton_match.setEnabled(True)
            self.tab_2.setEnabled(True)
            self.actionDetect_object.setEnabled(True)

        print('all action enabled!')

    def toggle_thermal_actions(self, enable=True):
        if not enable:
            self.actionRectangle_meas.setEnabled(False)
            self.actionSpot_meas.setEnabled(False)
            self.actionLine_meas.setEnabled(False)
            self.action3D_temperature.setEnabled(False)
            self.actionFind_maxima.setEnabled(False)
            self.actionCompose.setEnabled(False)
            self.dockWidget.setEnabled(False)
        else:
            self.actionRectangle_meas.setEnabled(True)
            self.actionSpot_meas.setEnabled(True)
            self.actionLine_meas.setEnabled(True)
            self.action3D_temperature.setEnabled(True)
            self.actionFind_maxima.setEnabled(True)
            self.actionCompose.setEnabled(True)
            self.dockWidget.setEnabled(True)

    def update_img_list(self):
        """Update the list of images and initialize image processing classes.

        Updates the internal list of thermal and RGB images, and initializes
        the necessary image processing classes for each image pair. This method
        is called after loading a new folder of images.

        Raises:
            ThermogramError: If there's an error updating the image list
            FileOperationError: If required image files are not found
        """
        try:
            self.ir_folder = self.main_folder
            if self.has_rgb:
                self.rgb_folder = self.rgb_crop_img_folder
                self.rgb_imgs = []
                for p in self.list_rgb_paths:
                    base = os.path.basename(p)
                    self.rgb_imgs.append(base[:-4] + 'crop.JPG')

            # list thermal images (no physical copies; keep original files)
            self.ir_imgs = [os.path.basename(p) for p in self.list_ir_paths]
            self.n_imgs = len(self.ir_imgs)

            if self.n_imgs > 1:
                self.pushButton_right.setEnabled(True)

            # update progress
            self.update_progress(nb=5, text='Creating image objects....')

            # Create an image object for each picture
            for i, im in enumerate(self.ir_imgs):
                if self.n_imgs == 1:
                    progress = 100  # or any other value that makes sense in your context
                else:
                    progress = 5 + (95 * i) / (self.n_imgs - 1)
                self.update_progress(nb=progress, text=f'Creating image object {i}/{self.n_imgs}')

                if self.has_rgb:
                    print(f'image {i}: Has rgb!')
                    image = tt.ProcessedIm(self.list_ir_paths[i],
                                           os.path.join(self.rgb_folder, self.rgb_imgs[i]),
                                           self.list_rgb_paths[i], self.ir_undistorder, self.drone_model,
                                           delayed_compute=True)
                else:
                    image = tt.ProcessedIm(self.list_ir_paths[i], '', '', self.ir_undistorder,
                                           self.drone_model, delayed_compute=True)
                self.images.append(image)

            self.update_progress(nb=100, text=f'finalizing')

            # Define active image
            self.active_image = 0
            self.work_image = self.images[self.active_image]

            # create histogram
            if self.hist_canvas is None:
                self.hist_canvas = self.work_image.create_temperature_histogram_canvas()
                self.layout_histo.addWidget(self.hist_canvas)

            # Create temporary folder
            self.preview_folder = os.path.join(self.app_folder, 'preview')
            if not os.path.exists(self.preview_folder):
                os.mkdir(self.preview_folder)

            # Quickly compute temperature delta on first image
            tmin, tmax, tmin_shown, tmax_shown = self.work_image.get_temp_data()

            self.lineEdit_min_temp.setText(str(round(tmin_shown, 2)))
            self.lineEdit_max_temp.setText(str(round(tmax_shown, 2)))

            self.update_img_preview()
            self.comboBox_img.clear()
            self.comboBox_img.addItems(self.ir_imgs)

            # Final progress
            self.update_progress(nb=100, text="Status: You can now process thermal images!")

        except Exception as e:
            error(f"Failed to update image list: {str(e)}")
            raise ThermogramError(f"Failed to update image list: {str(e)}") from e

    def save_image(self, folder='', filename='current_view.jpg'):
        """Save the current thermal image with measurements and legend (PyQt6 compatible)."""
        try:
            from PyQt6.QtCore import QRectF, QSize
            from PyQt6.QtGui import QImage, QPainter
            from PyQt6.QtWidgets import QFileDialog
            from PyQt6.QtCore import Qt
            import os

            def save_with_white_background(image: QImage, path: str):
                bg_image = QImage(image.size(), QImage.Format.Format_ARGB32_Premultiplied)
                bg_image.fill(Qt.GlobalColor.white)
                painter = QPainter(bg_image)
                painter.drawImage(0, 0, image)
                painter.end()
                bg_image.save(path, 'JPEG', 100)

            photo_pixmap = self.viewer._photo.pixmap() if hasattr(self.viewer, "_photo") else None
            if photo_pixmap is not None and not photo_pixmap.isNull():
                scene_rect = QRectF(photo_pixmap.rect())
                self.viewer._scene.setSceneRect(scene_rect)
            else:
                scene_rect = self.viewer._scene.sceneRect()
            content_width = int(scene_rect.width())
            content_height = int(scene_rect.height())

            # Handle legend
            legend_label = getattr(self.viewer, 'legendLabel', None)
            legend_pixmap = None
            legend_width = 0

            if legend_label and legend_label.isVisible() and legend_label.pixmap() is not None:
                original_pixmap = legend_label.pixmap()

                # Scale legend to 2/3 of image height
                target_height = int(content_height * (2 / 3))
                scaled_pixmap = original_pixmap.scaledToHeight(
                    target_height,
                    Qt.TransformationMode.SmoothTransformation
                )
                legend_pixmap = scaled_pixmap
                legend_width = scaled_pixmap.width()
                legend_height = scaled_pixmap.height()

            # Calculate total canvas size
            margin = 20 if legend_pixmap else 0
            total_width = content_width + legend_width + margin
            total_height = content_height

            image = QImage(QSize(total_width, total_height), QImage.Format.Format_ARGB32_Premultiplied)
            image.fill(Qt.GlobalColor.transparent)

            # Paint scene
            painter = QPainter(image)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

            self.viewer._scene.render(painter, QRectF(0, 0, content_width, content_height), scene_rect)

            if legend_pixmap:
                painter.drawPixmap(content_width + margin, 0, legend_pixmap)

            painter.end()

            # Save
            if not folder:
                file_path, _ = QFileDialog.getSaveFileName(
                    None, "Save Image", "",
                    "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg *.JPEG)"
                )
            else:
                file_path = os.path.join(folder, filename)

            if file_path:
                if file_path.lower().endswith(('.jpg', '.jpeg')):
                    save_with_white_background(image, file_path)
                else:
                    image.save(file_path, 'PNG')
                return file_path

        except Exception as e:
            error(f"Failed to save image: {str(e)}")
            raise ThermogramError(f"Failed to save image: {str(e)}") from e

    def batch_export(self):
        """Batch export all thermal images with current settings."""
        try:
            # Get all processing parameters from current image
            self.colormap, self.n_colors, self.user_lim_col_high, self.user_lim_col_low, self.post_process = self.work_image.get_colormap_data()
            tmin, tmax, tmin_shown, tmax_shown = self.work_image.get_temp_data()
            parameters_style = [self.colormap, self.n_colors, self.user_lim_col_high, self.user_lim_col_low,
                                self.post_process]
            parameters_temp = [tmin, tmax, tmin_shown, tmax_shown]
            parameters_radio = self.work_image.thermal_param

            # Launch dialog
            desc = f'{PROC_TH_FOLDER}_{self.colormap}_{str(round(tmin_shown, 0))}_{str(round(tmax_shown, 0))}_{self.post_process}_image-set_{self.nb_sets}'

            # Create output folder
            self.last_out_folder = os.path.join(self.app_folder, desc)
            if not os.path.exists(self.last_out_folder):
                os.mkdir(self.last_out_folder)

            dialog = dia.DialogBatchExport(self.ir_imgs, self.last_out_folder, parameters_style, parameters_temp,
                                           parameters_radio)

            if dialog.exec():
                # Get selected images indices
                selected_indices = dialog.get_selected_indices()
                selected_images = [self.images[i] for i in selected_indices]  # Retrieve selected images

                # Get user options
                list_ir_export = []
                list_rgb_export = []
                undis = dialog.checkBox_undis.isChecked()
                zoom = dialog.spinBox.value()
                if dialog.checkBox_exp_ir.isChecked():
                    list_ir_export.append('IR')
                if dialog.checkBox_exp_tif.isChecked():
                    list_ir_export.append('IR_TIF')
                if dialog.checkBox_exp_picpic.isChecked():
                    list_ir_export.append('PICPIC')
                if dialog.checkBox_exp_rgb.isChecked():
                    list_rgb_export.append('RGB')
                if dialog.checkBox_exp_crop.isChecked():
                    list_rgb_export.append('RGB_CROP')

                # Determining file format
                format_idx = dialog.comboBox_img_format.currentIndex()
                if format_idx == 0:
                    format = 'PNG'
                elif format_idx == 1:
                    format = 'JPG'

                # Determine naming_type based on the selected option
                selected_option = dialog.comboBox_naming.currentIndex()
                if selected_option == 0:
                    naming_type = 'rename'
                elif selected_option == 1:
                    naming_type = 'keep_ir'
                elif selected_option == 2:
                    naming_type = 'match_rgb'

                out_folder = dialog.lineEdit.text()

                worker_1 = tt.RunnerDJI(5, 100, out_folder, selected_images, self.work_image, self.edges,
                                        self.edge_params, undis=undis, zoom=zoom, naming_type=naming_type,
                                        file_format=format,
                                        list_of_ir_export=list_ir_export, list_of_rgb_export=list_rgb_export)
                worker_1.signals.progressed.connect(lambda value: self.update_progress(value))
                worker_1.signals.messaged.connect(lambda string: self.update_progress(text=string))

                self.__pool.start(worker_1)
                worker_1.signals.finished.connect(self.process_all_phase2)

        except Exception as e:
            error(f"Batch export failed: {str(e)}")
            raise ThermogramError(f"Batch export failed: {str(e)}") from e

    def process_all_phase2(self):
        self.update_progress(nb=100, text="Status: Continue analyses!")

    def full_reset(self):
        """
        Reset all model parameters (image and categories)
        """
        if hasattr(self, 'hist_canvas') and self.hist_canvas is not None:
            self.layout_histo.removeWidget(self.hist_canvas)

        self.initialize_variables()
        self.initialize_tree_view()

        # clean graphicscene
        self.viewer.clean_complete()

    # IMAGE ALIGNMENT __________________________________________________________
    def image_matching(self):
        temp_folder = os.path.join(self.app_folder, 'temp')
        if not os.path.exists(temp_folder):
            os.mkdir(temp_folder)

        # get work images
        rgb_path = self.work_image.rgb_path_original
        ir_path = self.work_image.path

        zoom = self.work_image.zoom
        y_off = self.work_image.y_offset
        x_off = self.work_image.x_offset

        print(f'Here are the work values! zoom:{zoom}, y offset:{y_off}, x offset: {x_off}')

        dialog = dia.AlignmentDialog(self.work_image, temp_folder, theta=[zoom, y_off, x_off])
        if dialog.exec():
            zoom, y_off, x_off = dialog.theta

            img_names = []
            for im in self.images:
                try:
                    img_names.append(os.path.basename(im.path))
                except Exception:
                    img_names.append(str(im.path))

            selector = dia.DialogSelectImages(img_names, preselect_indices=[self.active_image], parent=self)
            if not selector.exec():
                self.update_img_preview(refresh_dual=True)
                return

            selected = selector.get_selected_indices()
            if not selected:
                self.update_img_preview(refresh_dual=True)
                return

            for idx in selected:
                if idx < 0 or idx >= len(self.images):
                    continue
                self.images[idx].zoom = zoom
                self.images[idx].y_offset = y_off
                self.images[idx].x_offset = x_off

            print(f'Re-creating RGB crop with zoom {zoom} on {len(selected)} selected images')

            text_status = 'creating rgb miniatures...'
            self.update_progress(nb=20, text=text_status)

            img_objects = [self.images[i] for i in selected if 0 <= i < len(self.images)]
            worker_1 = tt.RunnerMiniature(self.list_rgb_paths, self.drone_model, 60,
                                          self.rgb_crop_img_folder, 20,
                                          100,
                                          img_objects=img_objects)
            worker_1.signals.progressed.connect(lambda value: self.update_progress(value))
            worker_1.signals.messaged.connect(lambda string: self.update_progress(text=string))

            self.__pool.start(worker_1)
            worker_1.signals.finished.connect(self.miniat_finish)

    def miniat_finish(self):
        self.update_progress(nb=100, text='Ready...')
        self.update_img_preview(refresh_dual=True)

    def compose_pic(self):
        self.work_image.nb_custom_imgs += 1
        _, img_name = os.path.split(self.work_image.path)
        
        dest_path_temp = self.dest_path_post[:-4] + img_name[:-4] + f'_custom{self.work_image.nb_custom_imgs}.PNG'
        dialog = dia.ImageFusionDialog(self.work_image, self.dest_path_post, dest_path_temp)
        if dialog.exec():
            qm = QMessageBox
            reply = qm.question(self, '', "Do you want to save the picture on the hard drive?",
                                qm.StandardButton.Yes | qm.StandardButton.No)

            if reply == qm.StandardButton.Yes:

                file_path, _ = QFileDialog.getSaveFileName(
                    None, "Save Image", "", "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg *.JPEG)"
                )

                # Save the image if a file path was provided, using high-quality settings for JPEG
                if file_path:
                    dialog.exportComposedImage(file_path)

            # add custom image to list
            dialog.exportComposedImage(dest_path_temp)
            self.custom_images.append(dest_path_temp)
            self.work_image.custom_images.append(dest_path_temp)

            self.update_combo_view()
        else:
            self.work_image.nb_custom_imgs -= 1

    # EDGE OVERLAY _______________________________________________________________
    def activate_edges(self):
        if self.checkBox_edges.isChecked():
            self.edges = True
        else:
            self.edges = False

        self.update_img_preview()

    def edge_options(self):
        """
        # edge options
        self.edge_color = 'white'
        self.edge_blur = True
        self.edge_blur_size = 3
        self.edge_method = 1
        self.edge_opacity = 0.7

        """

        # lauch dialog
        edge_params = [self.edge_method, self.edge_color, self.edge_bil, self.edge_blur, self.edge_blur_size,
                       self.edge_opacity]
        parameters = {
            'method': edge_params[0],
            'color': edge_params[1],
            'blur_size': edge_params[3],
            # Assuming edge_blur simply enables/disables the blur size combo box and not a value
            'slider_value': edge_params[4]  # Assuming you have a slider for opacity or similar
        }

        dialog = dia.DialogEdgeOptions(edge_params)
        if dialog.exec():

            self.edge_color = dialog.comboBox_color.currentText()
            self.edge_method = dialog.comboBox_method.currentIndex()
            self.edge_bil = dialog.checkBox_bil.isChecked()
            self.edge_blur = dialog.checkBox.isChecked()
            self.edge_blur_size = int(dialog.comboBox_blur_size.currentText())
            self.edge_opacity = dialog.horizontalSlider.value() / 100

            print(
                f'new edge parameters \n {self.edge_color} \n {self.edge_method} \n {self.edge_blur} \n {self.edge_blur_size}')

            # update preview
            if self.checkBox_edges.isChecked():
                self.update_img_preview()

    def change_edge_style(self):
        """Handles changes in the edge overlay style selection combobox."""
        selected_style = self.comboBox_edge_overlay_selection.currentText()

        if selected_style == "Custom":
            self.pushButton_edge_options.setEnabled(True)
            # Optional: Maybe revert to last custom settings if needed, or do nothing
            # until the user clicks the options button. For now, we do nothing.
        else:
            self.pushButton_edge_options.setEnabled(False)
            if selected_style in tt.PREDEFINED_EDGE_STYLES:
                style_params = tt.PREDEFINED_EDGE_STYLES[selected_style]
                self.edge_method = style_params["method"]
                self.edge_color = style_params["color"]
                self.edge_bil = style_params["bil"]
                self.edge_blur = style_params["blur"]
                self.edge_blur_size = style_params["blur_size"]
                self.edge_opacity = style_params["opacity"]

                info(f"Applied predefined edge style: {selected_style}")
                # Update preview if edges are active
                if self.checkBox_edges.isChecked():
                    self.update_img_preview()
            else:
                error(f"Selected style '{selected_style}' not found in predefined styles.")

    # VISUALIZE __________________________________________________________________
    def toggle_legend(self):
        self.viewer.toggleLegendVisibility()

    def show_viz_threed(self):
        from tools import thermal_3d as t3d
        t3d.run_viz_app(self.work_image.raw_data_undis, self.work_image.colormap,
                        self.work_image.user_lim_col_high, self.work_image.user_lim_col_low, self.work_image.n_colors,
                        self.work_image.tmin_shown, self.work_image.tmax_shown)

    def find_maxima(self):
        dialog = dia.HotSpotDialog(self.dest_path_post, self.work_image.raw_data_undis)
        dialog.spot_exported.connect(self.receive_spot_measurements)
        dialog.exec()

    def receive_spot_measurements(self, points: list):
        for point in points:
            self.add_point_meas(point)

    def update_combo_view(self):
        self._view_list = copy.deepcopy(VIEWS)
        print(self._view_list)
        for i in range(self.work_image.nb_custom_imgs):
            self._view_list.append(f'Custom_img_{i + 1}')

        self.comboBox_view.clear()
        self.comboBox_view.addItems(self._view_list)

    # CHANGE AND VIEW IMAGE ___________________________________________________
    def compile_user_temps_values(self):
        tmin = float(self.lineEdit_min_temp.text())
        tmax = float(self.lineEdit_max_temp.text())

        self.work_image.tmin_shown = tmin
        self.work_image.tmax_shown = tmax

    def compile_user_values(self):
        # colormap
        i = self.comboBox_palette.currentIndex()
        self.work_image.colormap = self._colormap_list[i]

        try:
            self.work_image.n_colors = int(self.lineEdit_colors.text())
        except:
            self.work_image.n_colors = 256

        #   temp limits
        try:
            tmin = float(self.lineEdit_min_temp.text())
            tmax = float(self.lineEdit_max_temp.text())

            # Update the image with validated temperature range
            self.work_image.tmin_shown = tmin
            self.work_image.tmax_shown = tmax

        except ValueError as e:
            if not self.skip_update:  # Only show warning if not in the middle of an update
                QMessageBox.warning(self, "Warning",
                                    "Oops! A least one of the temperatures is not valid. Try again...")
                self.lineEdit_min_temp.setText(str(round(self.work_image.tmin_shown, 2)))
                self.lineEdit_max_temp.setText(str(round(self.work_image.tmax_shown, 2)))

        #   out of limits color
        i = self.comboBox_colors_low.currentIndex()
        self.work_image.user_lim_col_low = self._out_of_matp[i]

        i = self.comboBox_colors_high.currentIndex()
        self.work_image.user_lim_col_high = self._out_of_matp[i]

        #   post process operation
        k = self.comboBox_post.currentIndex()
        self.work_image.post_process = self._img_post[k]

    def switch_image_data(self):
        """
        When the shown picture is changed (adapt measurements and user colormap for this picture)
        order:
        - Parameters
        - Temp
        - Palette
        """
        self.skip_update = True
        # load stored data
        self.work_image = self.images[self.active_image]

        # Lazy thermal data loading: only load when the image is visited/used
        self.work_image.ensure_data_loaded(reset_shown_values=True)

        if not self.checkBox_keep_radiometric.isChecked():
            # load radiometric parameters from image and set the lineedit texts
            self.thermal_param = copy.deepcopy(self.work_image.get_thermal_param())

            self.lineEdit_emissivity.setText(str(round(self.thermal_param['emissivity'], 2)))
            self.lineEdit_distance.setText(str(round(self.thermal_param['distance'], 2)))
            self.lineEdit_refl_temp.setText(str(round(self.thermal_param['reflection'], 2)))
        else:
            # compare new to old param
            old_param = copy.deepcopy(self.work_image.get_thermal_param())
            print(f'checking if new thermal parameters {self.thermal_param} = {old_param}')
            for key in old_param:
                if old_param[key] != self.thermal_param.get(key):
                    # assign a **copy** of the current radiometric parameters to the image
                    self.work_image.update_data_from_param(copy.deepcopy(self.thermal_param))

                    # update measurements with new radiometric params
                    for i, point in enumerate(self.work_image.meas_point_list):
                        # update temperatures with new radiometric parameters
                        point.temp = self.work_image.raw_data_undis[int(point.qpoint.y()), int(point.qpoint.x())]

                        point.text_item.clear()
                        point.ellipse_item.clear()

                        # recreate all graphical items
                        point.create_items()

                    for i, rect in enumerate(self.work_image.meas_rect_list):
                        # update temperatures with new radiometric parameters
                        coords = rect.get_coord_from_item(rect.rect)
                        rect.compute_temp_data(coords, self.work_image.raw_data_undis)
                        rect.compute_highlights()

                        rect.text_items.clear()
                        rect.ellipse_items.clear()

                        # recreate all graphical items
                        rect.create_items()

                    for i, line in enumerate(self.work_image.meas_line_list):
                        # update temperatures with new radiometric parameters
                        line.compute_temp_data(self.work_image.raw_data_undis)
                        line.compute_highlights()

                        # recreate all graphical items
                        line.create_items()

                else:
                    print('No change in radiometric parameters')

        if not self.checkBox_keep_temp.isChecked():
            tmin, tmax, tmin_shown, tmax_shown = self.work_image.get_temp_data()

            # fill values lineedits
            self.lineEdit_min_temp.setText(str(round(tmin_shown, 2)))
            self.lineEdit_max_temp.setText(str(round(tmax_shown, 2)))
            self.range_slider.setLowerValue(tmin_shown * 100)
            self.range_slider.setUpperValue(tmax_shown * 100)
            self.range_slider.setMinimum(int(tmin * 100))
            self.range_slider.setMaximum(int(tmax * 100))

        else:
            # set tmin and tmax shown on pictures:
            self.compile_user_temps_values()

        # if 'keep parameters' is not checked, change all parameters according to stored data
        if not self.checkBox_keep_palette.isChecked():
            self.colormap, self.n_colors, self.user_lim_col_high, self.user_lim_col_low, self.post_process = self.work_image.get_colormap_data()

            # find correspondances in comboboxes
            a = self._colormap_list.index(self.colormap)
            b = self._out_of_matp.index(self.user_lim_col_high)
            c = self._out_of_matp.index(self.user_lim_col_low)
            d = self._img_post.index(self.post_process)

            # adapt combos
            self.comboBox_palette.setCurrentIndex(a)
            self.comboBox_colors_high.setCurrentIndex(b)
            self.comboBox_colors_low.setCurrentIndex(c)
            self.comboBox_post.setCurrentIndex(d)
            self.lineEdit_colors.setText(str(self.n_colors))
            self.range_slider.setHandleColorsFromColormap(self.colormap)

        # load custom views for the selected image
        self.update_combo_view()
        
        # Update report inclusion checkbox if it exists
        if hasattr(self, 'checkBox_report'):
            self.checkBox_report.blockSignals(True)
            self.checkBox_report.setChecked(getattr(self.work_image, 'include_in_report', True))
            self.checkBox_report.blockSignals(False)
            
        # Update remarks field if it exists
        if hasattr(self, 'plainTextEdit_remarks'):
            self.plainTextEdit_remarks.blockSignals(True)
            self.plainTextEdit_remarks.setPlainText(getattr(self.work_image, 'remarks', ""))
            self.plainTextEdit_remarks.blockSignals(False)
            
        # Update image info label if it exists
        if hasattr(self, 'label_img_info'):
            self.update_image_info_label()

        # clean measurements and annotations
        self.retrace_items()
        self.skip_update = False

    def update_img_to_preview(self, direction):
        """
        Iterate through images when user change the image
        """

        if direction == 'minus':
            self.active_image -= 1
            self.comboBox_img.setCurrentIndex(self.active_image)
        elif direction == 'plus':
            self.active_image += 1
            self.comboBox_img.setCurrentIndex(self.active_image)
        else:
            self.active_image = self.comboBox_img.currentIndex()

        # remove all measurements and annotations and get new data
        self.switch_image_data()

        self.update_img_preview()

        # change buttons
        if self.active_image == self.n_imgs - 1:
            self.pushButton_right.setEnabled(False)
        else:
            self.pushButton_right.setEnabled(True)

        if self.active_image == 0:
            self.pushButton_left.setEnabled(False)
        else:
            self.pushButton_left.setEnabled(True)

    def create_th_img_preview(self, refresh_dual=False):
        self.compile_user_values()  # store combobox choices in img data
        dest_path_post = os.path.join(self.preview_folder, 'preview_post.PNG')
        img = self.work_image

        # get edge detection parameters
        self.edge_params = [self.edge_method, self.edge_color, self.edge_bil, self.edge_blur, self.edge_blur_size,
                            self.edge_opacity]

        tt.process_raw_data(img, dest_path_post, edges=self.edges, edge_params=self.edge_params)
        self.range_slider.setHandleColorsFromColormap(self.work_image.colormap)

        # set photo
        self.viewer.setPhoto(QPixmap(dest_path_post))
        if self.fit_in_view:
            self.viewer.fitInView()

        # pass thermal data to the viewer
        self.viewer.set_thermal_data(img.raw_data_undis)

        # add legend
        idx = self.comboBox_legend_type.currentIndex()
        if idx == 0:
            self.viewer.setupLegendLabel(self.work_image, legend_type='colorbar')
        elif idx == 1:
            self.viewer.setupLegendLabel(self.work_image, legend_type='histo')

        # set left and right views (in dual viewer)
        if self.has_rgb:
            if refresh_dual:
                # reset the dual viewer
                self.dual_viewer.refresh()
            self.dual_viewer.load_images_from_path(self.work_image.rgb_path, dest_path_post)

        self.dest_path_post = dest_path_post

        # check if legend is needed
        if not self.checkBox_legend.isChecked():
            self.viewer.toggleLegendVisibility()

        # add histogram in label_summary
        self.work_image.update_temperature_histogram(self.hist_canvas)

       

    def update_img_preview(self, refresh_dual=False):
        self.viewer.set_thermal_data([])
        if self.skip_update:  # allows to skip image update
            return

        """
        Update what is shown in the viewer
        """
        # fetch user choices
        v = self.comboBox_view.currentIndex()
        if v == 0 and self.current_view == 0:
            self.fit_in_view = False
        else:
            self.fit_in_view = True
        self.current_view = copy.deepcopy(v)

        if v == 1:  # if rgb view
            self.viewer.setPhoto(QPixmap(self.work_image.rgb_path))
            # scale view
            self.viewer.fitInView()
            self.viewer.clean_scene()

            if self.viewer.legendLabel.isVisible():
                self.viewer.toggleLegendVisibility()

            self.toggle_thermal_actions(enable=False)

        elif v == 2:  # picture-in-picture
            self.create_th_img_preview(refresh_dual=refresh_dual)
            dest_path_post = os.path.join(self.preview_folder, 'preview_picpic.PNG')
            img = self.work_image
            original_th_img = copy.deepcopy(self.dest_path_post)

            tt.insert_th_in_rgb_fast(img, original_th_img, dest_path_post, img, extension='JPG')
            self.viewer.set_thermal_data([]) # avoid getting temperatures within the magnifier

            self.viewer.setPhoto(QPixmap(dest_path_post))
            self.viewer.clean_scene()
            self.viewer.fitInView()

            self.toggle_thermal_actions(enable=False)


        elif v > 2:  # custom compositions
            self.viewer.setPhoto(QPixmap(self.custom_images[v - 3]))
            # scale view
            self.viewer.fitInView()

            self.viewer.clean_scene()
            if self.viewer.legendLabel.isVisible():
                self.viewer.toggleLegendVisibility()

            # add annotations
            self.retrace_items()

            self.toggle_thermal_actions(enable=False)

        else:  # IR picture
            self.create_th_img_preview(refresh_dual=refresh_dual)
            # add annotations
            self.retrace_items()

            self.toggle_thermal_actions(enable=True)

    # AI METHODS __________________________________________________________________________
    def detect_object(self):
        if self.work_image:
            dialog = dia.DetectionDialog(self.work_image.rgb_path, self)
            dialog.box_exported.connect(self.prepare_box_measurements)  # Connect the signal to your method
            dialog.exec()

    def prepare_box_measurements(self, rgb_rect):
        # Step 1: Get drone model info and IR image dimensions
        drone_model = self.work_image.drone_model
        target_dim = drone_model.dim_undis_ir  # (width, height) of IR image

        # Step 2: Get the original RGB image dimensions
        rgb_image = QImage(self.work_image.rgb_path)
        input_dim = (rgb_image.width(), rgb_image.height())

        # Step 3: Map the QRectF from RGB to IR coordinates
        rect = rgb_rect.rect()
        x_scale = target_dim[0] / input_dim[0]
        y_scale = target_dim[1] / input_dim[1]

        ir_rect_f = QRectF(
            rect.x() * x_scale,
            rect.y() * y_scale,
            rect.width() * x_scale,
            rect.height() * y_scale
        )

        # Step 4: Create a new QGraphicsRectItem in IR coordinates
        ir_rect_item = QGraphicsRectItem(ir_rect_f)
        ir_rect_item.setPen(rgb_rect.pen())  # preserve color/style

        # Step 5: Call your existing measurement method
        self.add_rect_meas(ir_rect_item)
        self.retrace_items()


    # GENERAL GUI METHODS __________________________________________________________________________
    def show_radio_dock(self):
        """
        Show the radiometric parameters dock widget if it's hidden or closed.
        """
        if not self.dockWidget_radio.isVisible():
            self.dockWidget_radio.show()

    def show_edge_dock(self):
        """
        Show the edge mix dock widget if it's hidden or closed.
        """
        if not self.dockWidget_edge.isVisible():
            self.dockWidget_edge.show()

    def on_tab_change(self, index):
        """Handle tab widget changes.

        Enables or disables certain UI elements based on the selected tab.

        Args:
            index: Index of the newly selected tab
        """
        if index == 1:
            self.actionRectangle_meas.setDisabled(True)
            self.actionSpot_meas.setDisabled(True)
            self.actionLine_meas.setDisabled(True)
        else:
            self.actionRectangle_meas.setDisabled(False)
            self.actionSpot_meas.setDisabled(False)
            self.actionLine_meas.setDisabled(False)

    def hand_pan(self):
        # switch back to hand tool
        self.actionHand_selector.setChecked(True)

    def write_json(self, dictionary):
        # Serializing json
        json_object = json.dumps(dictionary, indent=4)
        with open(self.json_file, "w") as f:
            f.write(json_object)

    def get_json(self):
        with open(self.json_file, 'r') as f:
            json_object = json.load(f)

        return json_object

    def update_progress(self, nb=None, text=''):
        self.label_status.setText(text)
        if nb is not None:
            self.progressBar.setProperty("value", nb)

            # hide progress bar when 100%
            if nb >= 100:
                self.progressBar.setVisible(False)
            elif self.progressBar.isHidden():
                self.progressBar.setVisible(True)

    def add_all_icons(self):
        # self.add_icon(res.find('img/label.png'), self.pushButton_addCat)
        add_icon(res.find('img/folder.png'), self.actionLoad_folder)
        add_icon(res.find('img/rectangle.png'), self.actionRectangle_meas)
        add_icon(res.find('img/hand.png'), self.actionHand_selector)
        # self.add_icon(res.find('img/forest.png'), self.actionRun)
        add_icon(res.find('img/reset.png'), self.actionReset_all)
        # self.add_icon(res.find('img/settings.png'), self.actionParameters)
        add_icon(res.find('img/info.png'), self.actionInfo)
        add_icon(res.find('img/point.png'), self.actionSpot_meas)
        add_icon(res.find('img/line.png'), self.actionLine_meas)
        add_icon(res.find('img/compare.png'), self.actionCompose)
        add_icon(res.find('img/3d.png'), self.action3D_temperature)
        add_icon(res.find('img/maxima.png'), self.actionFind_maxima)
        add_icon(res.find('img/robot.png'), self.actionDetect_object)
        add_icon(res.find('img/layers.png'), self.actionProcess_all)
        add_icon(res.find('img/save_image.png'), self.actionSave_Image)
        add_icon(res.find('img/report.png'), self.actionCreate_Report)

        add_icon(res.find('img/reset_range.png'), self.pushButton_reset_range)
        add_icon(res.find('img/from_img.png'), self.pushButton_estimate)
        add_icon(res.find('img/histo.png'), self.pushButton_optimhisto)

        add_icon(res.find('img/color_meas.png'), self.pushButton_meas_color)
        add_icon(res.find('img/del_spot.png'), self.pushButton_delete_points)
        add_icon(res.find('img/del_line.png'), self.pushButton_delete_lines)
        add_icon(res.find('img/del_rect.png'), self.pushButton_delete_area)

    def toggle_stylesheet(self):
        # Toggle the application stylesheet on and off
        if self.style_active:
            # Deactivate stylesheet
            QApplication.instance().setStyleSheet("")
            QApplication.instance().setStyle('Fusion')
            self.style_active = False
        else:
            # Test if dark theme is used
            palette = QApplication.instance().palette()
            bg_color = palette.color(QPalette.ColorRole.Window)

            is_dark_theme = bg_color.lightness() < 128
            print(f'Windows dark theme: {is_dark_theme}')
            stylesheet_file = "dark_theme.qss" if is_dark_theme else "light_theme.qss"
            QApplication.instance().setStyleSheet(load_stylesheet(stylesheet_file))
            self.style_active = True

    def show_info(self):
        """Show application information dialog."""
        try:
            info_text = f"""
            IR-Lab v{config.APP_VERSION}

            A comprehensive thermal image processing application
            for DJI drone thermal imagery.

            Features:
            - Thermal image visualization
            - Temperature analysis
            - Measurement tools
            - Batch processing
            
            Contact: sdu@bbri.be
            """

            QMessageBox.information(self, "About IR-Lab", info_text)

        except Exception as e:
            error(f"Failed to show info dialog: {str(e)}")


def load_stylesheet(filename):
    """Load QSS stylesheet from a file."""
    file_path = res.find(f'styles/{filename}')
    with open(file_path, "r") as file:
        return file.read()


def main(argv=None):
    """
    Creates the main window for the application and begins the QApplication if necessary.
    """
    import time  # Simulate loading delay for demo purposes

    # Create the application if necessary
    app = QApplication(argv)
    app.setStyle('Fusion')

    # Test if dark theme is used
    palette = app.palette()
    bg_color = palette.color(QPalette.ColorRole.Window)

    is_dark_theme = bg_color.lightness() < 128
    print(f'Windows dark theme: {is_dark_theme}')

    # Show the splash screen
    splash = SplashScreen()
    splash.show()
    app.processEvents()  # Ensure the splash screen is shown immediately

    # Simulate heavy initializations (remove or replace with real initializations)
    time.sleep(3)  # Replace this with actual loading tasks

    # Create and show the main window
    window = DroneIrWindow()
    window.setWindowIcon(QIcon(res.find('img/ico_512.png')))
    window.showMaximized()

    # Close the splash screen once the main window is ready
    splash.finish(window)

    # Run the application
    return app.exec()


if __name__ == '__main__':
    freeze_support()
    import sys

    sys.exit(main(sys.argv))
