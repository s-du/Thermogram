# imports
from PyQt6.QtGui import *  # modified from PySide6.QtGui to PyQt6.QtGui
from PyQt6.QtWidgets import *  # modified from PySide6.QtWidgets to PyQt6.QtWidgets
from PyQt6.QtCore import *  # modified from PySide6.QtCore to PyQt6.QtCore
from PyQt6 import uic

import os
import json

# custom libraries
import widgets as wid
import resources as res
import dialogs as dia
from tools import thermal_tools as tt
from tools import thermal_3d as t3d

"""
Any remarks, questions: sdu@bbri.be
TODO:
- Allow to import only thermal pictures - OK
- Implement 'context' dialog --> Drone info + location
- Allow the user to define a custom folder
- Implement save/load project folder
"""

# PARAMETERS
APP_FOLDER = 'ThermogramApp'
ORIGIN_THERMAL_IMAGES_NAME = 'Original Thermal Images'
RGB_ORIGINAL_NAME = 'Original RGB Images'
RGB_CROPPED_NAME = 'Cropped RGB Images'
ORIGIN_TH_FOLDER = 'img_th_original'
RGB_CROPPED_FOLDER = 'img_rgb'
PROC_TH_FOLDER = 'img_th_processed'

RECT_MEAS_NAME = 'Rectangle measurements'
POINT_MEAS_NAME = 'Spot measurements'
LINE_MEAS_NAME = 'Line measurements'

VIEWS = ['th. undistorted', 'RGB crop']


# USEFUL CLASSES
class DroneIrWindow(QMainWindow):
    """
    Main Window class for Thermogram
    """

    def __init__(self, parent=None):
        """
        Function to initialize the class
        :param parent:
        """
        super(DroneIrWindow, self).__init__(parent)

        # load the ui
        basepath = os.path.dirname(__file__)
        basename = 'main_window'
        uifile = os.path.join(basepath, 'ui/%s.ui' % basename)
        print(uifile)
        uic.loadUi(uifile, self)

        # initialize status
        self.update_progress(nb=100, text="Status: Choose image folder")

        # threadin
        self.__pool = QThreadPool()
        self.__pool.setMaxThreadCount(3)

        self.initialize_variables()
        self.initialize_tree_view()

        # edge options
        self.edges = False
        self.edge_color = 'white'
        self.edge_blur = False
        self.edge_bil = True
        self.edge_blur_size = 3
        self.edge_method = 0
        self.edge_opacity = 0.7

        # combo boxes content
        self._out_of_lim = tt.OUT_LIM
        self._out_of_matp = tt.OUT_LIM_MATPLOT
        self._img_post = tt.POST_PROCESS
        self._colormap_names = tt.COLORMAP_NAMES
        self._colormap_list = tt.COLORMAPS
        self._view_list = VIEWS

        # add content to comboboxes
        self.comboBox.addItems(self._colormap_names)
        self.comboBox.setCurrentIndex(0)

        self.comboBox_colors_low.addItems(self._out_of_lim)
        self.comboBox_colors_low.setCurrentIndex(0)
        self.comboBox_colors_high.addItems(self._out_of_lim)
        self.comboBox_colors_high.setCurrentIndex(0)

        self.comboBox_view.addItems(self._view_list)
        self.comboBox_post.addItems(self._img_post)

        self.advanced_options = False  # TODO: see if used
        self.skip_update = False

        # create double slider
        self.range_slider = wid.QRangeSlider(tt.COLORMAPS[0])
        self.range_slider.setEnabled(False)
        self.range_slider.setLowerValue(0)
        self.range_slider.setUpperValue(20)
        self.range_slider.setMinimum(0)
        self.range_slider.setMaximum(20)

        self.slider_sensitive = True

        # add double slider to layout
        self.horizontalLayout_slider.addWidget(self.range_slider)

        self.range_slider.lowerValueChanged.connect(self.change_line_edits)
        self.range_slider.upperValueChanged.connect(self.change_line_edits)

        # create validator for qlineedit
        onlyInt = QIntValidator()
        onlyInt.setRange(0, 999)
        self.lineEdit_colors.setValidator(onlyInt)
        self.n_colors = 256  # default number of colors
        self.lineEdit_colors.setText(str(256))

        # add actions to action group (mutually exclusive functions)
        ag = QActionGroup(self)
        ag.setExclusive(True)
        ag.addAction(self.actionRectangle_meas)
        ag.addAction(self.actionHand_selector)
        ag.addAction(self.actionSpot_meas)
        ag.addAction(self.actionLine_meas)

        # Add icons to buttons
        self.add_all_icons()

        # prepare viewer
        self.viewer = wid.PhotoViewer(self)
        self.verticalLayout_8.addWidget(self.viewer)

        # add dual viewer
        self.dual_viewer = wid.DualViewer()
        self.verticalLayout_10.addWidget(self.dual_viewer)

        # create connections (signals)
        self.create_connections()

    def initialize_variables(self):
        # set bool variables
        self.has_rgb = True  # does the dataset have RGB image
        self.rgb_shown = False
        self.save_colormap_info = True  # TODO make use of it... if True, the colormap and temperature options will be stored for each picture

        # set path variables
        self.custom_images = []
        self.list_rgb_paths = []
        self.list_ir_paths = []
        self.list_z_paths = []
        self.ir_folder = ''
        self.rgb_folder = ''
        self.preview_folder = ''

        # set other variables
        self.colormap = None
        self.number_custom_pic = 0

        # list thermal images
        self.ir_imgs = ''
        self.rgb_imgs = ''
        self.n_imgs = len(self.ir_imgs)
        self.nb_sets = 0

        # list images classes (where to store all measurements and annotations)
        self.images = []
        self.work_image = None

        # default thermal options:
        self.thermal_param = {'emissivity': 0.95, 'distance': 5, 'humidity': 50, 'reflection': 25}

        # image iterator to know which image is active
        self.active_image = 0

    def initialize_tree_view(self):
        # Create model (for the tree structure)
        self.model = QStandardItemModel()
        self.treeView.setModel(self.model)
        self.treeView.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.treeView.customContextMenuRequested.connect(self.onContextMenu)

        # add measurement and annotations categories to tree view
        self.add_item_in_tree(self.model, RECT_MEAS_NAME)
        self.add_item_in_tree(self.model, POINT_MEAS_NAME)
        self.add_item_in_tree(self.model, LINE_MEAS_NAME)
        self.model.setHeaderData(0, Qt.Orientation.Horizontal, 'Added Data')

    def create_connections(self):
        """
        Link signals to slots
        """
        # self.pushButton_addCat.clicked.connect(self.add_cat)
        self.actionLoad_folder.triggered.connect(self.load_folder_phase1)
        self.actionRectangle_meas.triggered.connect(self.rectangle_meas)
        self.actionSpot_meas.triggered.connect(self.point_meas)
        self.actionLine_meas.triggered.connect(self.line_meas)
        self.actionReset_all.triggered.connect(self.full_reset)
        self.actionInfo.triggered.connect(self.show_info)
        self.actionSave_Image.triggered.connect(self.save_image)
        self.actionProcess_all.triggered.connect(self.process_all_images)
        self.actionConvert_to_RAW_TIFF.triggered.connect(self.generate_raw_tiff)
        self.action3D_temperature.triggered.connect(self.show_viz_threed)
        self.actionCompose.triggered.connect(self.compose_pic)
        self.actionCreate_anim.triggered.connect(self.export_anim)
        self.actionFind_maxima.triggered.connect(self.find_maxima)
        self.actionConvert_FLIR.triggered.connect(self.convert_flir)
        self.actionDetect_object.triggered.connect(self.detect_object)

        self.viewer.endDrawing_rect_meas.connect(self.add_rect_meas)
        self.viewer.endDrawing_point_meas.connect(self.add_point_meas)
        self.viewer.endDrawing_line_meas.connect(self.add_line_meas)

        # PushButtons
        self.pushButton_left.clicked.connect(lambda: self.update_img_to_preview('minus'))
        self.pushButton_right.clicked.connect(lambda: self.update_img_to_preview('plus'))
        self.pushButton_estimate.clicked.connect(self.estimate_temp)
        # self.pushButton_advanced.clicked.connect(self.define_options)
        self.pushButton_meas_color.clicked.connect(self.change_meas_color)
        self.pushButton_match.clicked.connect(self.image_matching)
        self.pushButton_edge_options.clicked.connect(self.edge_options)
        self.pushButton_delete_points.clicked.connect(lambda: self.remove_annotations('point'))
        self.pushButton_delete_lines.clicked.connect(lambda: self.remove_annotations('line'))
        self.pushButton_delete_area.clicked.connect(lambda: self.remove_annotations('area'))

        # Dropdowns
        self.comboBox.currentIndexChanged.connect(self.update_img_preview)
        self.comboBox_colors_low.currentIndexChanged.connect(self.update_img_preview)
        self.comboBox_colors_high.currentIndexChanged.connect(self.update_img_preview)
        self.comboBox_post.currentIndexChanged.connect(self.update_img_preview)
        self.comboBox_img.currentIndexChanged.connect(lambda: self.update_img_to_preview('other'))
        self.comboBox_view.currentIndexChanged.connect(self.update_img_preview)

        # Line edits
        self.lineEdit_min_temp.editingFinished.connect(self.change_slider_values)
        self.lineEdit_max_temp.editingFinished.connect(self.change_slider_values)
        self.lineEdit_colors.editingFinished.connect(self.update_img_preview)

        self.lineEdit_emissivity.editingFinished.connect(self.define_options)
        self.lineEdit_distance.editingFinished.connect(self.define_options)
        self.lineEdit_refl_temp.editingFinished.connect(self.define_options)

        # Checkboxes
        self.checkBox_legend.stateChanged.connect(self.toggle_legend)
        self.checkBox_edges.stateChanged.connect(self.activate_edges)

        # tab widget
        self.tabWidget.currentChanged.connect(self.on_tab_change)

    def on_tab_change(self, index):
        if index == 1:
            self.actionRectangle_meas.setDisabled(True)
            self.actionSpot_meas.setDisabled(True)
            self.actionLine_meas.setDisabled(True)
        else:
            self.actionRectangle_meas.setDisabled(False)
            self.actionSpot_meas.setDisabled(False)
            self.actionLine_meas.setDisabled(False)

    def change_slider_values(self):
        self.slider_sensitive = False  # to avoid chain reaction

        try:
            tmin = float(self.lineEdit_min_temp.text())
            tmax = float(self.lineEdit_max_temp.text())

            if tmax > tmin:
                if tmin < self.work_image.tmin:  # the encoded value is lower than the minimal value on the image
                    tmin = self.work_image.tmin
                    self.work_image.tmin_shown = tmin
                    self.lineEdit_min_temp.setText(str(round(self.work_image.tmin_shown, 2)))
                else:
                    self.work_image.tmin_shown = tmin

                if tmax > self.work_image.tmax:
                    tmax = self.work_image.tmax
                    self.work_image.tmax_shown = tmax
                    self.lineEdit_max_temp.setText(str(round(self.work_image.tmax_shown, 2)))
                else:
                    self.work_image.tmax_shown = tmax
            else:
                raise ValueError

        except ValueError:
            QMessageBox.warning(self, "Warning",
                                "Oops! A least one of the temperatures is not valid.  Try again...")

            self.lineEdit_min_temp.setText(str(round(self.work_image.tmin_shown, 2)))
            self.lineEdit_max_temp.setText(str(round(self.work_image.tmax_shown, 2)))

        self.range_slider.setLowerValue(tmin * 100)
        self.range_slider.setUpperValue(tmax * 100)

        self.update_img_preview()
        self.slider_sensitive = True

    def update_img_list(self):
        """
        Save information about the number of images and add processed images classes
        """
        self.ir_folder = self.original_th_img_folder
        if self.has_rgb:
            self.rgb_folder = self.rgb_crop_img_folder
            self.rgb_imgs = os.listdir(self.rgb_folder)

        # list thermal images
        self.ir_imgs = os.listdir(self.ir_folder)
        self.n_imgs = len(self.ir_imgs)

        if self.n_imgs > 1:
            self.pushButton_right.setEnabled(True)

        # update progress
        self.update_progress(nb=5, text='Creating image objects....')

        # add classes
        for i, im in enumerate(self.ir_imgs):
            if self.n_imgs == 1:
                progress = 100  # or any other value that makes sense in your context
            else:
                progress = 5 + (95 * i) / (self.n_imgs - 1)
            self.update_progress(nb=progress, text=f'Creating image object {i}/{self.n_imgs}')

            if self.has_rgb:
                print('Has rgb!')
                image = tt.ProcessedIm(os.path.join(self.ir_folder, im),
                                       os.path.join(self.rgb_folder, self.rgb_imgs[i]),
                                       self.list_rgb_paths[i], self.ir_undistorder)
            else:
                image = tt.ProcessedIm(os.path.join(self.ir_folder, im), '', '', self.ir_undistorder)
            self.images.append(image)

        self.active_image = 0
        self.work_image = self.images[self.active_image]

        # create temporary folder
        self.preview_folder = os.path.join(self.ir_folder, 'preview')
        if not os.path.exists(self.preview_folder):
            os.mkdir(self.preview_folder)

        # quickly compute temperature delta on first image
        self.tmin = self.work_image.tmin
        self.tmax = self.work_image.tmax

        self.lineEdit_min_temp.setText(str(round(self.tmin, 2)))
        self.lineEdit_max_temp.setText(str(round(self.tmax, 2)))

        self.update_img_preview()
        self.comboBox_img.clear()
        self.comboBox_img.addItems(self.ir_imgs)

        # final progress
        self.update_progress(nb=100, text="Status: You can now process thermal images!")

    def show_info(self):
        dialog = dia.AboutDialog()
        if dialog.exec():
            pass

    def image_matching(self):
        temp_folder = os.path.join(self.app_folder, 'temp')
        if not os.path.exists(temp_folder):
            os.mkdir(temp_folder)

        # get work images
        rgb_path = self.work_image.rgb_path_original
        ir_path = self.work_image.path

        # get initial distortion parameters from the drone model (stored in the image)
        # [k1, extend, y - offset, x - offset]
        F = self.drone_model.K_ir[0][0]
        d_mat = self.drone_model.d_ir

        zoom = self.drone_model.zoom
        y_off = self.drone_model.y_offset
        x_off = self.drone_model.x_offset

        print(f'Here are the work values! zoom:{zoom}, y offset:{y_off}, x offset: {x_off}')

        dialog = dia.AlignmentDialog(self.work_image, temp_folder, theta=[zoom, y_off, x_off])
        if dialog.exec():

            # ask options
            # Create a QMessageBox
            qm = QMessageBox
            reply = qm.question(self, '', "Do you want to process all pictures with those new parameters",
                                qm.StandardButton.Yes | qm.StandardButton.No)

            if reply == qm.StandardButton.Yes:
                print('Good choice')
                # update values
                zoom, y_off, x_off = dialog.theta
                self.drone_model.zoom = zoom
                self.drone_model.y_offset = y_off
                self.drone_model.x_offset = x_off

                print(f'Re-creating RGB crop with zoom {zoom}')

                # re-run all miniatures
                text_status = 'creating rgb miniatures...'
                self.update_progress(nb=20, text=text_status)

                worker_1 = tt.RunnerMiniature(self.list_rgb_paths, self.drone_model, 60,
                                              self.rgb_crop_img_folder, 20,
                                              100)
                worker_1.signals.progressed.connect(lambda value: self.update_progress(value))
                worker_1.signals.messaged.connect(lambda string: self.update_progress(text=string))

                self.__pool.start(worker_1)
                worker_1.signals.finished.connect(self.miniat_finish)

            else:
                pass

                # re-print-image
                self.update_img_preview(refresh_dual=True)

    def miniat_finish(self):
        self.update_progress(nb=100, text='Ready...')
        self.update_img_preview(refresh_dual=True)

    # ANNOTATIONS _________________________________________________________________
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
        self.add_item_in_tree(rect_cat[0], desc)
        self.treeView.expandAll()

        # bring data 3d figure
        if self.has_rgb:
            dialog = dia.Meas3dDialog(new_rect_annot)
            dialog.dual_view.load_images_from_path(roi_rgb_path, roi_ir_path)
        else:
            dialog = dia.Meas3dDialog_simple(new_rect_annot)

        dialog.surface_from_image_matplot(self.work_image.colormap, self.n_colors, self.user_lim_col_low,
                                          self.user_lim_col_high)
        if dialog.exec():
            pass

        # switch back to hand tool
        self.hand_pan()

    def add_line_meas(self, line_item):
        # create annotation (object)
        new_line_annot = tt.LineMeas(line_item)

        # compute stuff
        new_line_annot.compute_data(self.work_image.raw_data_undis)
        self.work_image.nb_meas_line += 1
        desc = 'line_measure_' + str(self.work_image.nb_meas_line)
        new_line_annot.name = desc

        # add annotation to the image annotation list
        self.work_image.meas_line_list.append(new_line_annot)

        line_cat = self.model.findItems(LINE_MEAS_NAME)
        self.add_item_in_tree(line_cat[0], desc)

        # bring data 3d figure
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
        self.add_item_in_tree(point_cat[0], desc)
        self.hand_pan()

    # measurements methods
    def rectangle_meas(self):
        if self.actionRectangle_meas.isChecked():
            # activate drawing tool
            self.viewer.rect_meas = True
            self.viewer.toggleDragMode()

    def point_meas(self):
        if self.actionSpot_meas.isChecked():
            # activate drawing tool
            self.viewer.point_meas = True
            self.viewer.toggleDragMode()

    def line_meas(self):
        if self.actionLine_meas.isChecked():
            # activate drawing tool
            self.viewer.line_meas = True
            self.viewer.toggleDragMode()

    def remove_annotations(self, type):
        if type == 'point':
            self.images[self.active_image].meas_point_list = []
        elif type == 'line':
            self.images[self.active_image].meas_line_list = []
        elif type == 'area':
            self.images[self.active_image].meas_rect_list = []

        self.retrace_items()

    # THERMAL-RELATED FUNCTIONS __________________________________________________________________________
    def define_options_old(self):
        dialog = dia.DialogThParams(self.thermal_param)
        dialog.setWindowTitle("Choose advanced thermal options")

        if dialog.exec():
            try:
                self.advanced_options = True
                em = float(dialog.lineEdit_em.text())
                if em < 0.1 or em > 1:
                    raise ValueError
                else:
                    self.thermal_param['emissivity'] = em
                dist = float(dialog.lineEdit_dist.text())
                if dist < 1 or dist > 25:
                    raise ValueError
                else:
                    self.thermal_param['distance'] = dist
                rh = float(dialog.lineEdit_rh.text())
                if rh < 20 or rh > 100:
                    raise ValueError
                else:
                    self.thermal_param['humidity'] = rh
                refl_temp = float(dialog.lineEdit_temp.text())
                if refl_temp < -40 or refl_temp > 500:
                    raise ValueError
                else:
                    self.thermal_param['reflection'] = refl_temp

                # update image data
                self.work_image.update_data(self.thermal_param)
                self.switch_image_data()
                self.update_img_preview()


            except ValueError:
                QMessageBox.warning(self, "Warning",
                                    "Oops! Some of the values are not valid!")
                self.define_options()

    def define_options(self):
        try:
            self.advanced_options = True
            em = float(self.lineEdit_emissivity.text())
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

            # update image data
            self.work_image.update_data(self.thermal_param)
            self.switch_image_data()
            self.update_img_preview()


        except ValueError:
            QMessageBox.warning(self, "Warning",
                                "Oops! Some of the values are not valid!")

    def estimate_temp(self):
        ref_pic_name = QFileDialog.getOpenFileName(self, 'Open file',
                                                   self.ir_folder, "Image files (*.jpg *.JPG *.gif)")
        img_path = ref_pic_name[0]
        if img_path != '':
            tmin, tmax = tt.compute_delta(img_path, self.thermal_param)
            self.lineEdit_min_temp.setText(str(round(tmin, 2)))
            self.lineEdit_max_temp.setText(str(round(tmax, 2)))

        self.update_img_preview()

    def change_line_edits(self, value):
        if self.slider_sensitive:
            tmin = self.range_slider.lowerValue() / 100.0  # Adjust if you used scaling
            tmax = self.range_slider.upperValue() / 100.0

            self.lineEdit_min_temp.setText(str(round(tmin, 2)))
            self.lineEdit_max_temp.setText(str(round(tmax, 2)))

            self.update_img_preview()

    # LOAD AND SAVE ACTIONS ______________________________________________________________________________
    def export_anim(self):
        # export all images
        # get parameters
        self.colormap, self.n_colors, self.user_lim_col_high, self.user_lim_col_low, self.tmin, self.tmax, self.tmin_shown, self.tmax_shown, self.post_process = self.work_image.get_colormap_data()

        self.nb_sets += 1

        desc = f'{PROC_TH_FOLDER}_{self.colormap}_{str(round(self.tmin_shown, 0))}_{str(round(self.tmax_shown, 0))}_{self.post_process}_image-set_{self.nb_sets}_anim'

        # create output folder
        self.last_out_folder = os.path.join(self.app_folder, desc)
        if not os.path.exists(self.last_out_folder):
            os.mkdir(self.last_out_folder)

        worker_1 = tt.RunnerDJI(5, 50, self.last_out_folder, self.images, self.work_image, self.edges,
                                self.edge_params, undis=True)
        worker_1.signals.progressed.connect(lambda value: self.update_progress(value))
        worker_1.signals.messaged.connect(lambda string: self.update_progress(text=string))

        self.__pool.start(worker_1)
        worker_1.signals.finished.connect(self.export_video_phase2)

    def export_video_phase2(self):
        self.update_progress(nb=60, text="Status: Video creation!")
        video_dir = self.last_out_folder  # Adjust the path to your video's folder
        video_file = "animation_thermal.mp4"  # Adjust to your video file name if needed
        video_path = os.path.join(video_dir, video_file)

        tt.create_video(self.last_out_folder, video_path, 3)

        self.update_progress(nb=100, text="Continue analyses!")

        msg_box = QMessageBox()
        msg_box.setWindowTitle("Video Location")
        msg_box.setText(f"Your video is located here:\n{video_path}")
        msg_box.setInformativeText("Click 'Open Folder' to view it.")

        # Adding Open Folder button
        open_button = msg_box.addButton("Open Folder", QMessageBox.ButtonRole.ActionRole)

        msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)

        # Display the message box and wait for a user action
        msg_box.exec()

        # Check if the Open Folder button was clicked
        if msg_box.clickedButton() == open_button:
            QDesktopServices.openUrl(QUrl.fromLocalFile(video_path))

        # Display the message box
        msg_box.exec()

    def load_folder_phase1(self):
        # warning message (new project)
        if self.list_rgb_paths != []:
            qm = QMessageBox
            reply = qm.question(self, '', "Are you sure ? It will create a new project", qm.StandardButton.Yes | qm.StandardButton.No)

            if reply == qm.StandardButton.Yes:
                # reset all data
                self.full_reset()

            else:
                return

        folder = str(QFileDialog.getExistingDirectory(self, "Select Directory"))

        # sort images
        if not folder == '':  # if user cancel selection, stop function

            self.main_folder = folder
            self.app_folder = os.path.join(folder, APP_FOLDER)

            # update json path
            self.json_file = os.path.join(self.app_folder, 'data.json')

            # update status
            text_status = 'loading images...'
            self.update_progress(nb=0, text=text_status)

            # Identify content of the folder
            self.list_rgb_paths, self.list_ir_paths, self.list_z_paths = tt.list_th_rgb_images_from_res(self.main_folder)

            # create some sub folders for storing images
            self.original_th_img_folder = os.path.join(self.app_folder, ORIGIN_TH_FOLDER)

            # if the sub folders do not exist, create them
            if not os.path.exists(self.app_folder):
                os.mkdir(self.app_folder)
            if not os.path.exists(self.original_th_img_folder):
                os.mkdir(self.original_th_img_folder)

            # get drone model
            drone_name = tt.get_drone_model(self.list_ir_paths[0])
            self.drone_model = tt.DroneModel(drone_name)

            # create undistorder based on drone model
            self.ir_undistorder = tt.CameraUndistorter(self.drone_model.ir_xml_path)

            print(f'Drone model : {drone_name}')

            dictionary = {
                "Drone model": drone_name,
                "Number of image pairs": str(len(self.list_ir_paths)),
                "rgb_paths": self.list_rgb_paths,
                "ir_paths": self.list_ir_paths,
                "zoom_paths": self.list_z_paths
            }
            self.write_json(dictionary)  # store original images paths in a JSON

            text_status = 'copying thermal images...'
            self.update_progress(nb=10, text=text_status)

            # duplicate thermal images
            tt.copy_list_dest(self.list_ir_paths, self.original_th_img_folder)

            # does it have RGB?
            if not self.list_rgb_paths:
                print('No RGB here!')
                self.has_rgb = False
                self.load_folder_phase2()

            else:
                self.rgb_crop_img_folder = os.path.join(self.app_folder, RGB_CROPPED_FOLDER)
                if not os.path.exists(self.rgb_crop_img_folder):
                    os.mkdir(self.rgb_crop_img_folder)

                    text_status = 'creating rgb miniatures...'
                    self.update_progress(nb=20, text=text_status)

                    worker_1 = tt.RunnerMiniature(self.list_rgb_paths, self.drone_model, 60, self.rgb_crop_img_folder,
                                                  20,
                                                  100)
                    worker_1.signals.progressed.connect(lambda value: self.update_progress(value))
                    worker_1.signals.messaged.connect(lambda string: self.update_progress(text=string))

                    self.__pool.start(worker_1)
                    worker_1.signals.finished.connect(self.load_folder_phase2)

    def load_folder_phase2(self):
        # get list to main window
        self.update_img_list()
        # activate buttons and options

        #   dock widgets
        self.lineEdit_max_temp.setEnabled(True)
        self.lineEdit_min_temp.setEnabled(True)
        self.lineEdit_distance.setEnabled(True)
        self.lineEdit_emissivity.setEnabled(True)
        self.lineEdit_refl_temp.setEnabled(True)
        self.pushButton_estimate.setEnabled(True)

        self.comboBox.setEnabled(True)
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

        # enable action
        self.actionHand_selector.setEnabled(True)
        self.actionHand_selector.setChecked(True)

        self.actionRectangle_meas.setEnabled(True)
        self.actionSpot_meas.setEnabled(True)
        self.actionLine_meas.setEnabled(True)
        self.action3D_temperature.setEnabled(True)
        self.actionFind_maxima.setEnabled(True)

        # RGB condition
        if self.has_rgb:
            self.checkBox_edges.setEnabled(True)
            self.pushButton_edge_options.setEnabled(True)
            self.actionCompose.setEnabled(True)
            self.pushButton_match.setEnabled(True)
            self.tab_2.setEnabled(True)
            self.actionDetect_object.setEnabled(True)

        print('all action enabled!')

    def save_image(self):
        # Create a QImage with the size of the viewport
        image = QImage(self.viewer.viewport().size(), QImage.Format.Format_ARGB32_Premultiplied)
        image.fill(Qt.GlobalColor.transparent)

        # Paint the QGraphicsView's viewport onto the QImage
        painter = QPainter(image)
        self.viewer.render(painter)
        painter.end()

        # Open 'Save As' dialog
        file_path, _ = QFileDialog.getSaveFileName(
            None, "Save Image", "", "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg *.JPEG)"
        )

        # Save the image if a file path was provided, using high-quality settings for JPEG
        if file_path:
            if file_path.lower().endswith('.jpg') or file_path.lower().endswith('.jpeg'):
                image.save(file_path, 'JPEG', 100)
            else:
                image.save(file_path)  # PNG is lossless by default

    def convert_flir(self):
        tt.convert_dji_to_flir_format(self.work_image.raw_data_undis, 'test.tiff')
    def generate_raw_tiff(self):
        qm = QMessageBox
        reply = qm.question(self, '', "Do you want to output TIFF files (Pix4D processing)?",
                            qm.StandardButton.Yes | qm.StandardButton.No)

        if reply == qm.StandardButton.Yes:
            desc = f'{PROC_TH_FOLDER}_tiff_{str(round(self.tmin_shown, 0))}_{str(round(self.tmax_shown, 0))}_image-set_{self.nb_sets}'

            # create output folder
            out_folder = os.path.join(self.app_folder, desc)
            if not os.path.exists(out_folder):
                os.mkdir(out_folder)
            reply = qm.question(self, '', "Do you want to correct deformation?",
                                qm.StandardButton.Yes | qm.StandardButton.No)
            if reply == qm.StandardButton.Yes:
                undis = True
            else:
                undis = False

            worker_1 = tt.RunnerDJI(5, 100, out_folder, self.images, self.work_image, self.edges, self.edge_params,
                                    export_tif=True, undis=undis)
            worker_1.signals.progressed.connect(lambda value: self.update_progress(value))
            worker_1.signals.messaged.connect(lambda string: self.update_progress(text=string))

            self.__pool.start(worker_1)
            worker_1.signals.finished.connect(self.process_all_phase2)

    def process_all_images(self):
        qm = QMessageBox
        reply = qm.question(self, '', "Are you sure to process all pictures with the current parameters?",
                            qm.StandardButton.Yes | qm.StandardButton.No)

        # get parameters
        self.colormap, self.n_colors, self.user_lim_col_high, self.user_lim_col_low, self.tmin, self.tmax, self.tmin_shown, self.tmax_shown, self.post_process = self.work_image.get_colormap_data()

        self.nb_sets += 1

        if reply == qm.StandardButton.Yes:
            desc = f'{PROC_TH_FOLDER}_{self.colormap}_{str(round(self.tmin_shown, 0))}_{str(round(self.tmax_shown, 0))}_{self.post_process}_image-set_{self.nb_sets}'

            # create output folder
            self.last_out_folder = os.path.join(self.app_folder, desc)
            if not os.path.exists(self.last_out_folder):
                os.mkdir(self.last_out_folder)

            reply = qm.question(self, '', "Do you want to correct deformation?",
                                qm.StandardButton.Yes | qm.StandardButton.No)
            if reply == qm.StandardButton.Yes:
                undis = True
            else:
                undis = False

            reply = qm.question(self, '', "Do you want to upscale images (3x)?",
                                qm.StandardButton.Yes | qm.StandardButton.No)

            if reply == qm.StandardButton.Yes:
                zoom = 3
            else:
                zoom = 1

            worker_1 = tt.RunnerDJI(5, 100, self.last_out_folder, self.images, self.work_image, self.edges,
                                    self.edge_params, undis=undis, zoom=zoom)
            worker_1.signals.progressed.connect(lambda value: self.update_progress(value))
            worker_1.signals.messaged.connect(lambda string: self.update_progress(text=string))

            self.__pool.start(worker_1)
            worker_1.signals.finished.connect(self.process_all_phase2)

    def process_all_phase2(self):
        self.update_progress(nb=100, text="Status: Continue analyses!")

    # VISUALIZE __________________________________________________________________
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

    def toggle_legend(self):
        self.viewer.toggleLegendVisibility()

    def retrace_items(self):
        self.viewer.clean_scene()

        # Tree operations
        self.model = QStandardItemModel()
        self.treeView.setModel(self.model)

        self.add_item_in_tree(self.model, RECT_MEAS_NAME)
        self.add_item_in_tree(self.model, POINT_MEAS_NAME)
        self.add_item_in_tree(self.model, LINE_MEAS_NAME)

        self.model.setHeaderData(0, Qt.Orientation.Horizontal, 'Meas. Data')

        point_cat = self.model.findItems(POINT_MEAS_NAME)
        rect_cat = self.model.findItems(RECT_MEAS_NAME)
        line_cat = self.model.findItems(LINE_MEAS_NAME)

        for i, point in enumerate(self.work_image.meas_point_list):
            desc = point.name
            self.add_item_in_tree(point_cat[0], desc)

            self.viewer.add_item_from_annot(point.ellipse_item)
            self.viewer.add_item_from_annot(point.text_item)

        for i, rect in enumerate(self.work_image.meas_rect_list):
            desc = rect.name
            self.add_item_in_tree(rect_cat[0], desc)

            self.viewer.add_item_from_annot(rect.main_item)
            for item in rect.ellipse_items:
                self.viewer.add_item_from_annot(item)
            for item in rect.text_items:
                self.viewer.add_item_from_annot(item)

        for i, line in enumerate(self.work_image.meas_line_list):
            desc = line.name
            self.add_item_in_tree(line_cat[0], desc)

            self.viewer.add_item_from_annot(line.main_item)

    def show_viz_threed(self):
        t3d.run_viz_app(self.work_image.raw_data_undis, self.work_image.colormap,
                        self.work_image.user_lim_col_high, self.work_image.user_lim_col_low, self.work_image.n_colors,
                        self.work_image.tmin_shown, self.work_image.tmax_shown)

    def change_meas_color(self):
        self.viewer.change_meas_color()
        self.switch_image_data()

    def find_maxima(self):
        dialog = dia.HotSpotDialog(self.dest_path_post, self.work_image.raw_data_undis)
        if dialog.exec():
            pass

    def compose_pic(self):
        self.number_custom_pic += 1
        dest_path_temp = self.dest_path_post[:-4] + f'_custom{self.number_custom_pic}.png'
        print(dest_path_temp)
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
            self._view_list.append(f'Custom_img{self.number_custom_pic}')
            self.update_combo_view()
        else:
            self.number_custom_pic -= 1

    def update_combo_view(self):
        self.comboBox_view.clear()
        self.comboBox_view.addItems(self._view_list)

    def compile_user_values(self):
        # colormap
        i = self.comboBox.currentIndex()
        self.work_image.colormap = self._colormap_list[i]

        try:
            self.work_image.n_colors = int(self.lineEdit_colors.text())
        except:
            self.work_image.n_colors = 256

        #   temp limits
        try:
            tmin = float(self.lineEdit_min_temp.text())
            tmax = float(self.lineEdit_max_temp.text())

            if tmax > tmin:
                self.work_image.tmin_shown = tmin
                self.work_image.tmax_shown = tmax
            else:
                raise ValueError

        except ValueError:
            QMessageBox.warning(self, "Warning",
                                "Oops! A least one of the temperatures is not valid.  Try again...")
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
        """
        # load stored data
        self.work_image = self.images[self.active_image]

        # if 'keep parameters' is not checked, change all parameters according to stored data
        if not self.checkBox_keep.isChecked():
            self.colormap, self.n_colors, self.user_lim_col_high, self.user_lim_col_low, self.tmin, self.tmax, self.tmin_shown, self.tmax_shown, self.post_process = self.work_image.get_colormap_data()

            # find correspondances in comboboxes
            a = self._colormap_list.index(self.colormap)
            b = self._out_of_matp.index(self.user_lim_col_high)
            c = self._out_of_matp.index(self.user_lim_col_low)
            d = self._img_post.index(self.post_process)

            # adapt combos
            self.comboBox.setCurrentIndex(a)
            self.comboBox_colors_high.setCurrentIndex(b)
            self.comboBox_colors_low.setCurrentIndex(c)
            self.comboBox_post.setCurrentIndex(d)

            # fill values lineedits
            self.lineEdit_colors.setText(str(self.n_colors))
            self.lineEdit_min_temp.setText(str(round(self.tmin_shown, 2)))
            self.lineEdit_max_temp.setText(str(round(self.tmax_shown, 2)))

            # load radiometeric parameters
            self.thermal_param = self.work_image.thermal_param

            self.lineEdit_emissivity.setText(str(round(self.thermal_param['emissivity'], 2)))
            self.lineEdit_distance.setText(str(round(self.thermal_param['distance'], 2)))
            self.lineEdit_refl_temp.setText(str(round(self.thermal_param['reflection'], 2)))

            # update double slider TODO
            self.range_slider.setLowerValue(self.tmin_shown * 100)
            self.range_slider.setUpperValue(self.tmax_shown * 100)
            self.range_slider.setMinimum(int(self.tmin * 100))
            self.range_slider.setMaximum(int(self.tmax * 100))
            self.range_slider.setHandleColorsFromColormap(self.colormap)
        else:
            # get radiometric data
            self.work_image.thermal_param = self.thermal_param
            self.work_image.update_data(self.thermal_param)

        # clean measurements and annotations
        self.retrace_items()

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
        self.skip_update = True
        self.switch_image_data()

        self.skip_update = False
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

    def update_img_preview(self, refresh_dual=False):
        if self.skip_update:  # allows to skip image update
            return

        """
        Update what is shown in the viewer
        """
        # fetch user choices
        v = self.comboBox_view.currentIndex()

        if v == 1:  # if rgb view
            self.viewer.setPhoto(QPixmap(self.work_image.rgb_path))
            # scale view
            self.viewer.fitInView()

            self.viewer.clean_scene()
            if self.viewer.legendLabel.isVisible():
                self.viewer.toggleLegendVisibility()

        elif v > 1:  # picture-in-picture (or custom)
            self.viewer.setPhoto(QPixmap(self.custom_images[v-2]))
            # scale view
            self.viewer.fitInView()

            self.viewer.clean_scene()
            if self.viewer.legendLabel.isVisible():
                self.viewer.toggleLegendVisibility()

        else:  # IR picture
            self.compile_user_values()  # store combobox choices in img data
            dest_path_post = os.path.join(self.preview_folder, 'preview_post.png')
            img = self.work_image

            # get edge detection parameters
            self.edge_params = [self.edge_method, self.edge_color, self.edge_bil, self.edge_blur, self.edge_blur_size,
                                self.edge_opacity]

            tt.process_raw_data(img, dest_path_post, edges=self.edges, edge_params=self.edge_params)
            self.range_slider.setHandleColorsFromColormap(self.work_image.colormap)

            # add legend
            self.viewer.setPhoto(QPixmap(dest_path_post))
            self.viewer.fitInView()
            self.viewer.setupLegendLabel(self.work_image)

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

    # CONTEXT MENU IN TREEVIEW _________________________________________________
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
                    roi_ir, roi_rgb = interest.compute_data(coords, self.work_image.raw_data_undis, rgb_path, ir_path)
                    roi_ir_path = os.path.join(self.preview_folder, 'roi_ir.JPG')
                    tt.cv_write_all_path(roi_ir, roi_ir_path)

                    if self.has_rgb:
                        roi_rgb_path = os.path.join(self.preview_folder, 'roi_rgb.JPG')
                        tt.cv_write_all_path(roi_rgb, roi_rgb_path)

                        dialog = dia.Meas3dDialog(interest)
                        dialog.dual_view.load_images_from_path(roi_rgb_path, roi_ir_path)

                    else:
                        dialog = dia.Meas3dDialog_simple(interest)

                    dialog.surface_from_image_matplot(self.work_image.colormap, self.n_colors,
                                                      self.user_lim_col_low,
                                                      self.user_lim_col_high)
                    if dialog.exec():
                        pass

        if 'spot' in lookup_text:
            for i, annot in enumerate(self.images[self.active_image].meas_point_list):
                if annot.name == lookup_text:
                    interest = self.images[self.active_image].meas_point_list[i]

        # show dialog:

    # AI METHODS __________________________________________________________________________
    def detect_object(self):
        if self.work_image:
            dialog = dia.DetectionDialog(self.work_image.rgb_path, self)
            dialog.exec()


    # GENERAL GUI METHODS __________________________________________________________________________
    def hand_pan(self):
        # switch back to hand tool
        self.actionHand_selector.setChecked(True)

    def add_item_in_tree(self, parent, line):
        item = QStandardItem(line)
        parent.appendRow(item)

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

    def add_icon(self, img_source, pushButton_object):
        """
        Function to add an icon to a pushButton
        """
        pushButton_object.setIcon(QIcon(img_source))

    def add_all_icons(self):
        # self.add_icon(res.find('img/label.png'), self.pushButton_addCat)
        self.add_icon(res.find('img/folder.png'), self.actionLoad_folder)
        self.add_icon(res.find('img/rectangle.png'), self.actionRectangle_meas)
        self.add_icon(res.find('img/hand.png'), self.actionHand_selector)
        # self.add_icon(res.find('img/forest.png'), self.actionRun)
        self.add_icon(res.find('img/reset.png'), self.actionReset_all)
        # self.add_icon(res.find('img/settings.png'), self.actionParameters)
        self.add_icon(res.find('img/info.png'), self.actionInfo)
        self.add_icon(res.find('img/point.png'), self.actionSpot_meas)
        self.add_icon(res.find('img/line.png'), self.actionLine_meas)
        self.add_icon(res.find('img/compare.png'), self.actionCompose)
        self.add_icon(res.find('img/3d.png'), self.action3D_temperature)
        self.add_icon(res.find('img/maxima.png'), self.actionFind_maxima)
        self.add_icon(res.find('img/robot.png'), self.actionDetect_object)

    def full_reset(self):
        """
        Reset all model parameters (image and categories)
        """
        self.initialize_variables()
        self.initialize_tree_view()

        # clean graphicscene
        self.viewer.clean_complete()


def main(argv=None):
    """
    Creates the main window for the application and begins the \
    QApplication if necessary.

    :param      argv | [, ..] || None

    :return      error code
    """

    # Define installation path
    install_folder = os.path.dirname(__file__)

    app = None
    extra = {

        # Density Scale
        'density_scale': '-3',
    }

    # create the application if necessary
    if (not QApplication.instance()):
        app = QApplication(argv)
        app.setStyle('Fusion')

    # create the main window

    window = DroneIrWindow()
    window.setWindowIcon(QIcon(res.find('img/icone.png')))
    window.showMaximized()

    # run the application if necessary
    if app:
        return app.exec()

    # no errors since we're not running our own event loop
    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))
