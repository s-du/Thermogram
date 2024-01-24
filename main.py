# imports
from PySide6.QtGui import *
from PySide6.QtWidgets import *
from PySide6.QtCore import *

import os
import json

# custom libraries
import widgets as wid
import resources as res
import dialogs as dia
from tools import thermal_tools as tt
from tools import thermal_3d as t3d

"""
TODO:
- Allow to import only thermal pictures
- Implement improved drone data

"""
# FILES
# default lens calibration files
ir_xml_path = res.find('other/cam_calib_m2t_opencv.xml')
rgb_xml_path = res.find('other/rgb_cam_calib_m2t_opencv.xml')

# PARAMETERS
APP_FOLDER = 'DroneIrToolkit'
ORIGIN_THERMAL_IMAGES_NAME = 'Original Thermal Images'
RGB_ORIGINAL_NAME = 'Original RGB Images'
RGB_CROPPED_NAME = 'Cropped RGB Images'
ORIGIN_TH_FOLDER = 'img_th_original'
RGB_CROPPED_FOLDER = 'img_rgb'
PROC_TH_FOLDER = 'img_th_processed'

RECT_MEAS_NAME = 'Rectangle measurements'
POINT_MEAS_NAME = 'Spot measurements'
LINE_MEAS_NAME = 'Line measurements'

OUT_LIM = ['black', 'white', 'red']
OUT_LIM_MATPLOT = ['k', 'w', 'r']
POST_PROCESS = ['none', 'smooth', 'sharpen', 'sharpen strong', 'edge (simple)', 'edge (from rgb)']
COLORMAPS = ['coolwarm', 'Artic', 'Iron', 'Rainbow', 'Greys_r', 'Greys', 'plasma', 'inferno', 'jet',
             'Spectral_r', 'cividis', 'viridis', 'gnuplot2']
VIEWS = ['th. undistorted', 'RGB crop', 'PicInPic']


# USEFUL CLASSES
class ObjectDetectionCategory:
    """
    Class to describe a segmentation category
    """

    def __init__(self):
        self.color = None
        self.name = ''


class DroneIrWindow(QMainWindow):
    """
    Main Window class for the Drone IR Toolkit
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
        wid.loadUi(uifile, self)

        # initialize status
        self.update_progress(nb=100, text="Status: Choose image folder")

        # threadin
        self.__pool = QThreadPool()
        self.__pool.setMaxThreadCount(3)

        # set variables
        self.rgb_shown = False

        # set options
        self.save_colormap_info = True  # if True, the colormap and temperature options will be stored for each picture

        self.list_rgb_paths = ''
        self.list_ir_paths = ''

        self.ir_folder = ''
        self.rgb_folder = ''

        # list thermal images
        self.ir_imgs = ''
        self.rgb_imgs = ''
        self.n_imgs = len(self.ir_imgs)

        # list images classes (where to store all measurements and annotations)
        self.images = []

        # comboboxes content
        self.out_of_lim = OUT_LIM
        self.out_of_matp = OUT_LIM_MATPLOT
        self.img_post = POST_PROCESS
        self.colormap_list = COLORMAPS
        self.view_list = VIEWS

        # add content to comboboxes
        self.comboBox.addItems(self.colormap_list)
        self.comboBox.setCurrentIndex(0)
        self.comboBox_img.addItems(self.ir_imgs)

        self.comboBox_colors_low.addItems(self.out_of_lim)
        self.comboBox_colors_low.setCurrentIndex(0)
        self.comboBox_colors_high.addItems(self.out_of_lim)
        self.comboBox_colors_high.setCurrentIndex(1)
        self.comboBox_post.addItems(self.img_post)
        self.comboBox_view.addItems(self.view_list)

        self.advanced_options = False

        # create validator for qlineedit
        onlyInt = QIntValidator()
        onlyInt.setRange(0, 999)
        self.lineEdit_colors.setValidator(onlyInt)
        self.n_colors = 256  # default number of colors
        self.lineEdit_colors.setText(str(256))

        # default thermal options:
        self.thermal_param = {'emissivity': 0.95, 'distance': 5, 'humidity': 50, 'reflection': 25}

        # image iterator to know which image is active
        self.active_image = 0

        # Create model (for the tree structure)
        self.model = QStandardItemModel()
        self.treeView.setModel(self.model)
        self.treeView.setContextMenuPolicy(Qt.CustomContextMenu)
        self.treeView.customContextMenuRequested.connect(self.onContextMenu)

        # add measurement and annotations categories to tree view
        self.add_item_in_tree(self.model, RECT_MEAS_NAME)
        self.add_item_in_tree(self.model, POINT_MEAS_NAME)
        self.add_item_in_tree(self.model, LINE_MEAS_NAME)
        self.model.setHeaderData(0, Qt.Horizontal, 'Added Data')

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

    def create_connections(self):
        """
        Link signals to slots
        """
        # self.pushButton_addCat.clicked.connect(self.add_cat)
        self.actionLoad_folder.triggered.connect(self.load_folder_phase1)
        self.actionRectangle_meas.triggered.connect(self.rectangle_meas)
        self.actionSpot_meas.triggered.connect(self.point_meas)
        self.actionLine_meas.triggered.connect(self.line_meas)
        self.actionReset_all.triggered.connect(self.reset_roi)
        self.actionInfo.triggered.connect(self.show_info)
        self.actionSave_Image.triggered.connect(self.save_image)
        self.actionProcess_all.triggered.connect(self.process_all_images)
        self.action3D_temperature.triggered.connect(self.show_viz_threed)
        self.actionCompose.triggered.connect(self.compose_pic)

        self.viewer.endDrawing_rect_meas.connect(self.add_rect_meas)
        self.viewer.endDrawing_point_meas.connect(self.add_point_meas)
        self.viewer.endDrawing_line_meas.connect(self.add_line_meas)

        self.pushButton_left.clicked.connect(lambda: self.update_img_to_preview('minus'))
        self.pushButton_right.clicked.connect(lambda: self.update_img_to_preview('plus'))
        self.pushButton_estimate.clicked.connect(self.estimate_temp)
        self.pushButton_advanced.clicked.connect(self.define_options)
        self.pushButton_meas_color.clicked.connect(self.viewer.change_meas_color)
        self.pushButton_match.clicked.connect(self.image_matching)

        # Dropdowns
        self.comboBox.currentIndexChanged.connect(self.update_img_preview)
        self.comboBox_colors_low.currentIndexChanged.connect(self.update_img_preview)
        self.comboBox_colors_high.currentIndexChanged.connect(self.update_img_preview)
        self.comboBox_post.currentIndexChanged.connect(self.update_img_preview)
        self.comboBox_img.currentIndexChanged.connect(lambda: self.update_img_to_preview('other'))
        self.comboBox_view.currentIndexChanged.connect(self.update_img_preview)

        # Line edits
        self.lineEdit_min_temp.editingFinished.connect(self.update_img_preview)
        self.lineEdit_max_temp.editingFinished.connect(self.update_img_preview)
        self.lineEdit_colors.editingFinished.connect(self.update_img_preview)

    def update_img_list(self):
        """
        Save information about the number of images and add processed images classes
        """
        self.ir_folder = self.original_th_img_folder
        self.rgb_folder = self.rgb_crop_img_folder

        # list thermal images
        self.ir_imgs = os.listdir(self.ir_folder)
        self.rgb_imgs = os.listdir(self.rgb_folder)
        self.n_imgs = len(self.ir_imgs)

        if self.n_imgs > 1:
            self.pushButton_right.setEnabled(True)

        self.active_image = 0

        # add classes
        for im in self.ir_imgs:
            print(os.path.join(self.ir_folder, im))
            image = tt.ProcessedImage_bis(os.path.join(self.ir_folder, im))
            self.images.append(image)

        # choose first image for preview
        test_img = self.ir_imgs[self.active_image]
        self.test_img_path = os.path.join(self.ir_folder, test_img)

        # get drone model
        drone_name = tt.get_drone_model(self.test_img_path)
        self.drone_model = tt.DroneModel(drone_name)

        # create temporary folder
        self.preview_folder = os.path.join(self.ir_folder, 'preview')
        if not os.path.exists(self.preview_folder):
            os.mkdir(self.preview_folder)

        # quickly compute temperature delta on first image
        self.tmin = self.images[self.active_image].tmin
        self.tmax = self.images[self.active_image].tmax

        self.lineEdit_min_temp.setText(str(round(self.tmin, 2)))
        self.lineEdit_max_temp.setText(str(round(self.tmax, 2)))

        self.update_img_preview()
        self.comboBox_img.addItems(self.ir_imgs)

    def show_viz_threed(self):
        test_img = self.ir_imgs[self.active_image]
        img_path = os.path.join(self.ir_folder, test_img)
        t3d.run_viz_app(img_path, self.colormap, self.user_lim_col_high, self.user_lim_col_low, self.n_colors)

    def show_info(self):
        dialog = dia.AboutDialog()
        if dialog.exec_():
            pass

    def image_matching(self):
        pass

    # ANNOTATIONS _________________________________________________________________
    def add_rect_meas(self, rect_item):
        """
        Add a region of interest coming from the rectangle tool
        :param rect_item: a rectangle item from the viewer
        """
        # create annotation (object)
        new_rect_annot = tt.RectMeas(rect_item)

        # get image data
        rgb_path = os.path.join(self.rgb_folder, self.rgb_imgs[self.active_image])
        ir_path = self.dest_path_no_post
        coords = new_rect_annot.get_coord_from_item(rect_item)
        roi_ir, roi_rgb = new_rect_annot.compute_data(coords, self.raw_data, rgb_path, ir_path)

        roi_rgb_path = os.path.join(self.preview_folder, 'roi_rgb.JPG')
        roi_ir_path = os.path.join(self.preview_folder, 'roi_ir.JPG')
        tt.cv_write_all_path(roi_rgb, roi_rgb_path)
        tt.cv_write_all_path(roi_ir, roi_ir_path)

        # add interesting data to viewer
        new_rect_annot.compute_highlights()
        new_rect_annot.create_items()
        for item in new_rect_annot.ellipse_items:
            self.viewer.add_item_from_annot(item)
        for item in new_rect_annot.text_items:
            self.viewer.add_item_from_annot(item)

        # create description name
        self.images[self.active_image].nb_meas_rect += 1
        desc = 'rect_measure_' + str(self.images[self.active_image].nb_meas_rect)
        new_rect_annot.name = desc

        # add annotation to the image annotation list
        self.images[self.active_image].meas_rect_list.append(new_rect_annot)

        rect_cat = self.model.findItems(RECT_MEAS_NAME)
        self.add_item_in_tree(rect_cat[0], desc)
        self.treeView.expandAll()

        # bring data 3d figure
        dialog = dia.Meas3dDialog(new_rect_annot)
        dialog.dual_view.load_images_from_path(roi_rgb_path, roi_ir_path)
        dialog.surface_from_image_matplot(self.colormap, self.n_colors, self.user_lim_col_low, self.user_lim_col_high)
        if dialog.exec_():
            pass

        # switch back to hand tool
        self.hand_pan()

    def add_line_meas(self, line_item):
        # create annotation (object)
        new_line_annot = tt.LineMeas(line_item)

        # compute stuff
        new_line_annot.compute_data(self.raw_data)
        self.images[self.active_image].nb_meas_line += 1
        desc = 'line_measure_' + str(self.images[self.active_image].nb_meas_line)
        new_line_annot.name = desc

        # add annotation to the image annotation list
        self.images[self.active_image].meas_line_list.append(new_line_annot)

        line_cat = self.model.findItems(LINE_MEAS_NAME)
        self.add_item_in_tree(line_cat[0], desc)

        # bring data 3d figure
        dialog = dia.MeasLineDialog(new_line_annot.data_roi)
        if dialog.exec_():
            pass

        self.hand_pan()

    def add_point_meas(self, qpointf):
        # create annotation (object)
        new_pt_annot = tt.PointMeas()
        new_pt_annot.temp = self.raw_data[int(qpointf.y()), int(qpointf.x())]
        new_pt_annot.create_items()
        self.viewer.add_item_from_annot(new_pt_annot.ellipse_item)
        self.viewer.add_item_from_annot(new_pt_annot.text_item)

        # create description name
        self.images[self.active_image].nb_meas_point += 1
        desc = 'spot_measure_' + str(self.images[self.active_image].nb_meas_point)
        new_pt_annot.name = desc

        # add annotation to the image annotation list
        self.images[self.active_image].meas_point_list.append(new_pt_annot)

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

    def define_options(self):
        dialog = dia.DialogThParams(self.thermal_param)
        dialog.setWindowTitle("Choose advanced thermal options")

        if dialog.exec_():
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

                self.update_img_preview()

            except ValueError:
                QMessageBox.warning(self, "Warning",
                                    "Oops! Some of the values are not valid!")
                self.define_options()

    def estimate_temp(self):
        ref_pic_name = QFileDialog.getOpenFileName(self, 'Open file',
                                                   self.ir_folder, "Image files (*.jpg *.JPG *.gif)")
        img_path = ref_pic_name[0]
        if img_path != '':
            tmin, tmax = tt.compute_delta(img_path, self.thermal_param)
            self.lineEdit_min_temp.setText(str(round(tmin, 2)))
            self.lineEdit_max_temp.setText(str(round(tmax, 2)))

        self.update_img_preview()

    # LOAD AND SAVE ACTIONS ______________________________________________________________________________
    def load_folder_phase1(self):
        folder = str(QFileDialog.getExistingDirectory(self, "Select Directory"))

        # warning message (new project)
        if self.list_rgb_paths != '':
            qm = QMessageBox
            reply = qm.question(self, '', "Are you sure ? It will create a new project", qm.Yes | qm.No)

            if reply == qm.Yes:
                # reset all data
                self.reset_project()

        # sort images
        if not folder == '':  # if user cancel selection, stop function
            self.main_folder = folder
            self.app_folder = os.path.join(folder, APP_FOLDER)

            # update json path
            self.json_file = os.path.join(self.app_folder, 'data.json')

            # create some sub folders for storing images
            self.original_th_img_folder = os.path.join(self.app_folder, ORIGIN_TH_FOLDER)
            self.rgb_crop_img_folder = os.path.join(self.app_folder, RGB_CROPPED_FOLDER)

            # if the sub folders do not exist, create them
            if not os.path.exists(self.app_folder):
                os.mkdir(self.app_folder)
            if not os.path.exists(self.original_th_img_folder):
                os.mkdir(self.original_th_img_folder)
            if not os.path.exists(self.rgb_crop_img_folder):
                os.mkdir(self.rgb_crop_img_folder)

            # update status
            text_status = 'loading images...'
            self.update_progress(nb=0, text=text_status)
            self.list_rgb_paths, self.list_ir_paths = tt.list_th_rgb_images_from_res(self.main_folder)

            # get drone model
            drone_name = tt.get_drone_model(self.list_ir_paths[0])
            self.drone_model = tt.DroneModel(drone_name)

            print(f'Drone model : {drone_name}')

            dictionary = {
                "Drone model": drone_name,
                "Number of image pairs": str(len(self.list_ir_paths)),
                "rgb_paths": self.list_rgb_paths,
                "ir_paths": self.list_ir_paths
            }
            self.write_json(dictionary)  # store original images paths in a JSON

            text_status = 'copying thermal images...'
            self.update_progress(nb=10, text=text_status)

            # duplicate thermal images
            tt.copy_list_dest(self.list_ir_paths, self.original_th_img_folder)
            # create folder for cropped/resized rgb

            text_status = 'creating rgb miniatures...'
            self.update_progress(nb=20, text=text_status)

            worker_1 = tt.RunnerMiniature(self.list_rgb_paths, self.drone_model, 20, self.rgb_crop_img_folder, 20,
                                          100)
            worker_1.signals.progressed.connect(lambda value: self.update_progress(value))
            worker_1.signals.messaged.connect(lambda string: self.update_progress(text=string))

            self.__pool.start(worker_1)
            worker_1.signals.finished.connect(self.load_folder_phase2)

    def load_folder_phase2(self):
        self.update_progress(nb=100, text="Status: You can now process thermal images!")
        # get list to main window
        self.update_img_list()
        # activate buttons and options

        #   dock widgets
        self.pushButton_advanced.setEnabled(True)
        self.lineEdit_max_temp.setEnabled(True)
        self.lineEdit_min_temp.setEnabled(True)
        self.pushButton_estimate.setEnabled(True)
        self.pushButton_match.setEnabled(True)
        self.comboBox.setEnabled(True)
        self.lineEdit_colors.setEnabled(True)
        self.comboBox_colors_low.setEnabled(True)
        self.comboBox_colors_high.setEnabled(True)
        self.comboBox_post.setEnabled(True)

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
        self.actionCompose.setEnabled(True)
        self.action3D_temperature.setEnabled(True)

    def go_save(self):
        """
        Save measurements
        """
        # load image
        pass

    def save_image(self):
        # Create a QImage with the size of the viewport
        image = QImage(self.viewer.viewport().size(), QImage.Format_ARGB32_Premultiplied)
        image.fill(Qt.transparent)

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

    def process_all_images(self):
        pass

    def switch_image_data(self):
        """
        When the shown picture is change (adapt measurements and user colormap for this picture)
        """
        # load stored data
        self.colormap, self.n_colors, self.user_lim_col_high, self.user_lim_col_low, self.tmin, self.tmax = self.images[
            self.active_image].get_colormap_data()
        print(f'HERE___________{self.tmin}')
        print(f'HERE___________{self.tmax}')

        # find correspondances in comboboxes
        a = self.colormap_list.index(self.colormap)
        b = self.out_of_matp.index(self.user_lim_col_high)
        c = self.out_of_matp.index(self.user_lim_col_low)
        d = self.img_post.index(self.post_process)

        # adapt combos
        self.comboBox.setCurrentIndex(a)
        self.comboBox_colors_high.setCurrentIndex(b)
        self.comboBox_colors_low.setCurrentIndex(c)
        self.comboBox_post.setCurrentIndex(d)

        # fill values lineedits
        self.lineEdit_colors.setText(str(self.n_colors))
        self.lineEdit_min_temp.setText(str(round(self.tmin, 2)))
        self.lineEdit_max_temp.setText(str(round(self.tmax, 2)))

        self.thermal_param = self.images[self.active_image].thermal_param

        # clean measurements and annotations
        self.retrace_items()

    def retrace_items(self):
        self.viewer.clean_scene()

        # Tree operations
        self.model = QStandardItemModel()
        self.treeView.setModel(self.model)

        self.add_item_in_tree(self.model, RECT_MEAS_NAME)
        self.add_item_in_tree(self.model, POINT_MEAS_NAME)
        self.add_item_in_tree(self.model, LINE_MEAS_NAME)

        self.model.setHeaderData(0, Qt.Horizontal, 'Added Data')

        point_cat = self.model.findItems(POINT_MEAS_NAME)
        rect_cat = self.model.findItems(RECT_MEAS_NAME)
        line_cat = self.model.findItems(LINE_MEAS_NAME)

        for i, point in enumerate(self.images[self.active_image].meas_point_list):
            desc = point.name
            self.add_item_in_tree(point_cat[0], desc)

            self.viewer.add_item_from_annot(point.ellipse_item)
            self.viewer.add_item_from_annot(point.text_item)

        for i, rect in enumerate(self.images[self.active_image].meas_rect_list):
            desc = rect.name
            self.add_item_in_tree(rect_cat[0], desc)

            self.viewer.add_item_from_annot(rect.main_item)
            for item in rect.ellipse_items:
                self.viewer.add_item_from_annot(item)
            for item in rect.text_items:
                self.viewer.add_item_from_annot(item)

        for i, line in enumerate(self.images[self.active_image].meas_line_list):
            desc = line.name
            self.add_item_in_tree(line_cat[0], desc)

            self.viewer.add_item_from_annot(line.main_item)

    # VISUALIZE __________________________________________________________________
    def change_meas_color(self):
        self.viewer.change_meas_color()
        self.switch_image_data()

    def compose_pic(self):
        rgb_path = os.path.join(self.rgb_folder, self.rgb_imgs[self.active_image])
        ir_path = self.dest_path_no_post

        dest_path_temp = self.dest_path_no_post[:-4] + '_temp.png'
        dialog = dia.ImageFusionDialog(rgb_path, ir_path, self.raw_data, self.colormap, self.n_colors, dest_path_temp)
        if dialog.exec_():
            pass

    def compile_user_values(self):
        # colormap
        i = self.comboBox.currentIndex()
        self.colormap = self.colormap_list[i]

        try:
            self.n_colors = int(self.lineEdit_colors.text())
        except:
            self.n_colors = 256

        #   temp limits
        try:
            tmin = float(self.lineEdit_min_temp.text())
            tmax = float(self.lineEdit_max_temp.text())

            if tmax > tmin:
                self.tmin = tmin
                self.tmax = tmax
            else:
                raise ValueError

        except ValueError:
            QMessageBox.warning(self, "Warning",
                                "Oops! A least one of the temperatures is not valid.  Try again...")
            self.lineEdit_min_temp.setText(str(round(self.tmin, 2)))
            self.lineEdit_max_temp.setText(str(round(self.tmax, 2)))

        #   out of limits color
        i = self.comboBox_colors_low.currentIndex()
        self.user_lim_col_low = self.out_of_matp[i]

        i = self.comboBox_colors_high.currentIndex()
        self.user_lim_col_high = self.out_of_matp[i]

        #   post process operation
        k = self.comboBox_post.currentIndex()
        self.post_process = self.img_post[k]

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

        test_img = self.ir_imgs[self.active_image]
        self.test_img_path = os.path.join(self.ir_folder, test_img)

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

    def update_img_preview(self):
        """
        Update what is shown in the viewer
        """
        # fetch user choices
        v = self.comboBox_view.currentIndex()
        rgb_path = os.path.join(self.rgb_folder, self.rgb_imgs[self.active_image])

        if v == 1:  # if rgb view
            self.viewer.setPhoto(QPixmap(rgb_path))
            self.viewer.clean_scene()

        elif v == 2:  # picture-in-picture
            pass

        else:
            self.compile_user_values()

            dest_path_no_post = os.path.join(self.preview_folder, 'preview.JPG')
            dest_path_post = os.path.join(self.preview_folder, 'preview_post.JPG')

            read_path = os.path.join(self.rgb_folder, self.rgb_imgs[self.active_image])
            self.raw_data = tt.process_one_th_picture(self.thermal_param, self.drone_model, self.test_img_path,
                                                      dest_path_no_post,
                                                      self.tmin, self.tmax, self.colormap, self.user_lim_col_high,
                                                      self.user_lim_col_low, n_colors=self.n_colors,
                                                      post_process='none',
                                                      rgb_path=read_path)

            cv_img = tt.cv_read_all_path(dest_path_no_post)
            undis, _ = tt.undis(cv_img, ir_xml_path)
            tt.cv_write_all_path(undis, dest_path_no_post)
            self.viewer.setPhoto(QPixmap(dest_path_no_post))

            # set left and right views (in dual viewer)
            self.dual_viewer.load_images_from_path(rgb_path, dest_path_no_post)

            if self.post_process != 'none':  # if a post-process is applied
                _ = tt.process_one_th_picture(self.thermal_param, self.drone_model, self.test_img_path,
                                              dest_path_post,
                                              self.tmin, self.tmax, self.colormap, self.user_lim_col_high,
                                              self.user_lim_col_low, n_colors=self.n_colors,
                                              post_process=self.post_process,
                                              rgb_path=read_path)

                if self.post_process != 'edge (from rgb)':
                    cv_img = tt.cv_read_all_path(dest_path_post)
                    undis, _ = tt.undis(cv_img, ir_xml_path)
                    tt.cv_write_all_path(undis, dest_path_post)

                self.viewer.setPhoto(QPixmap(dest_path_post))
                # set left and right views
                self.dual_viewer.load_images_from_path(rgb_path, dest_path_post)

            # store all colormap data in current image before switching image
            self.images[self.active_image].update_colormap_data(self.colormap, self.n_colors,
                                                                self.user_lim_col_high,
                                                                self.user_lim_col_low, self.post_process, self.tmin,
                                                                self.tmax)
            # store emissivity data
            self.images[self.active_image].thermal_param = self.thermal_param

            self.dest_path_no_post = dest_path_no_post

    # CONTEXT MENU _________________________________________________
    def onContextMenu(self, point):
        # Get the index of the item that was clicked
        index = self.treeView.indexAt(point)
        if not index.isValid():
            return

        item = self.model.itemFromIndex(index)

        # Check if the clicked item is a second level item (annotation object)
        if item and item.parent() and item.parent().text() in [RECT_MEAS_NAME, POINT_MEAS_NAME, LINE_MEAS_NAME]:
            # Create Context Menu
            contextMenu = QMenu(self.treeView)

            deleteAction = contextMenu.addAction("Delete Annotation")
            showAction = contextMenu.addAction("Show Annotation")

            # Connect actions to methods
            deleteAction.triggered.connect(lambda: self.deleteAnnotation(item))
            showAction.triggered.connect(lambda: self.showAnnotation(item))

            # Display the menu
            contextMenu.exec_(self.treeView.viewport().mapToGlobal(point))

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
                    if dialog.exec_():
                        pass
        if 'rect' in lookup_text:
            for i, annot in enumerate(self.images[self.active_image].meas_rect_list):
                if annot.name == lookup_text:
                    interest = self.images[self.active_image].meas_rect_list[i]

                    # bring data 3d figure
                    rgb_path = os.path.join(self.rgb_folder, self.rgb_imgs[self.active_image])
                    ir_path = self.dest_path_no_post
                    coords = interest.get_coord_from_item(interest.main_item)
                    roi_ir, roi_rgb = interest.compute_data(coords, self.raw_data, rgb_path, ir_path)

                    roi_rgb_path = os.path.join(self.preview_folder, 'roi_rgb.JPG')
                    roi_ir_path = os.path.join(self.preview_folder, 'roi_ir.JPG')
                    tt.cv_write_all_path(roi_rgb, roi_rgb_path)
                    tt.cv_write_all_path(roi_ir, roi_ir_path)
                    dialog = dia.Meas3dDialog(interest)
                    dialog.dual_view.load_images_from_path(roi_rgb_path, roi_ir_path)
                    dialog.surface_from_image_matplot(self.colormap, self.n_colors, self.user_lim_col_low,
                                                      self.user_lim_col_high)
                    if dialog.exec_():
                        pass

        if 'spot' in lookup_text:
            for i, annot in enumerate(self.images[self.active_image].meas_point_list):
                if annot.name == lookup_text:
                    interest = self.images[self.active_image].meas_point_list[i]

        # show dialog:

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

    def full_reset_parameters(self):
        """
        Reset all model parameters (image and categories)
        """
        self.images = []
        self.n_imgs = 0
        self.img_paths = []  # paths to images
        self.active_image = 0
        self.image_loaded = False

        # define categories
        self.categories = []
        self.active_category = None

        # Create model (for the tree structure)
        self.model = QStandardItemModel()
        self.treeView.setModel(self.model)

        # clean graphicscene
        self.viewer.clean_scene()

        # clean combobox
        self.comboBox_cat.clear()

    def reset_roi(self):
        # clean tree view
        self.model = QStandardItemModel()
        self.treeView.setModel(self.model)

        # clean roi in each cat
        self.active_image.nb_roi_rect = 0
        self.active_image.item_list_rect = []
        self.active_image.roi_list_rect = []

        for cat in self.categories:
            self.add_item_in_tree(self.model, cat.name)
        self.model.setHeaderData(0, Qt.Horizontal, 'Categories')

        # clean graphicscene
        self.viewer.clean_scene()

    def add_icon(self, img_source, pushButton_object):
        """
        Function to add an icon to a pushButton
        """
        pushButton_object.setIcon(QIcon(img_source))


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
        app.setStyle('Breeze')
        # apply_stylesheet(app, theme='light_blue.xml',extra=extra)

    # create the main window

    window = DroneIrWindow()
    window.setWindowIcon(QIcon(res.find('img/icone.png')))
    window.showMaximized()

    # run the application if necessary
    if app:
        return app.exec_()

    # no errors since we're not running our own event loop
    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))
