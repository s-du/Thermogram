# imports
from PySide6 import QtWidgets, QtGui, QtCore

import os
import json
import platformdirs
import copy
import numpy as np

# custom libraries
import widgets as wid
import resources as res
import dialogs as dia
from tools import thermal_tools as tt

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

# USEFUL CLASSES
class ObjectDetectionCategory:
    """
    Class to describe a segmentation category
    """
    def __init__(self):
        self.color = None
        self.name = ''


class ProcessedImage:
    def __init__(self):
        self.path = ''
        # for annotations
        self.annot_rect_items = []
        self.corresp_cat = []

        # for measurements
        self.nb_meas_rect = 0 # number of rect measurements
        self.meas_rect_items = []
        self.meas_rect_coords = []
        self.nb_meas_point = 0 # number of spot measurements
        self.meas_point_items = []
        self.meas_text_spot_items = []

        self.nb_meas_line = 0  # number of line measurements
        self.meas_line_items = []





class DroneIrWindow(QtWidgets.QMainWindow):
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
        self.__pool = QtCore.QThreadPool()
        self.__pool.setMaxThreadCount(3)

        # set variables
        self.preview_rgb = False

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
        self.out_of_lim = ['black', 'white', 'red']
        self.out_of_matp = ['k', 'w', 'r']
        self.img_post = ['none', 'smooth', 'sharpen', 'sharpen strong', 'edge (simple)', 'edge (from rgb)']
        self.colormap_list = ['coolwarm','Artic', 'Iron', 'Rainbow', 'Greys_r', 'Greys', 'plasma', 'inferno', 'jet',
                              'Spectral_r', 'cividis', 'viridis', 'gnuplot2']
        self.view_list = ['th. undistorted', 'RGB crop']

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
        onlyInt = QtGui.QIntValidator()
        onlyInt.setRange(0, 999)
        self.lineEdit_colors.setValidator(onlyInt)
        self.n_colors = 256  # default number of colors
        self.lineEdit_colors.setText(str(256))

        # default thermal options:
        self.thermal_param = {'emissivity': 0.95, 'distance': 5, 'humidity': 50, 'reflection': 25}

        # image iterator to know which image is active
        self.active_image = 0

        # define segmentation categories
        self.categories = []
        self.active_category = None

        # Create model (for the tree structure)
        self.model = QtGui.QStandardItemModel()
        self.treeView.setModel(self.model)

        # add measurement and annotations categories to tree view
        self.add_item_in_tree(self.model, RECT_MEAS_NAME)
        self.add_item_in_tree(self.model, POINT_MEAS_NAME)
        self.add_item_in_tree(self.model, LINE_MEAS_NAME)
        self.model.setHeaderData(0, QtCore.Qt.Horizontal, 'Added Data')

        # add actions to action group (mutually exclusive functions)
        ag = QtGui.QActionGroup(self)
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

        self.viewer.endDrawing_rect_meas.connect(self.add_rect_meas)
        self.viewer.endDrawing_point_meas.connect(self.add_point_meas)
        self.viewer.endDrawing_line_meas.connect(self.add_line_meas)
        # self.comboBox_cat.currentIndexChanged.connect(self.on_cat_change)

        self.pushButton_left.clicked.connect(lambda: self.update_img_to_preview('minus'))
        self.pushButton_right.clicked.connect(lambda: self.update_img_to_preview('plus'))
        self.pushButton_temp_viz.clicked.connect(self.show_viz_threed)
        self.pushButton_estimate.clicked.connect(self.estimate_temp)
        self.pushButton_advanced.clicked.connect(self.define_options)
        self.pushButton_meas_color.clicked.connect(self.viewer.change_meas_color)

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
            image = ProcessedImage()
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
        self.tmin, self.tmax = self.compute_delta(self.test_img_path)
        self.lineEdit_min_temp.setText(str(round(self.tmin, 2)))
        self.lineEdit_max_temp.setText(str(round(self.tmax, 2)))


        self.update_img_preview()
        self.comboBox_img.addItems(self.ir_imgs)

    def switch_cat_data(self):
        self.viewer.clean_scene()

        # update tree view
        self.model = QtGui.QStandardItemModel()
        self.treeView.setModel(self.model)


        # add object categories to tree view
        for cat in self.categories:
            self.add_item_in_tree(self.model, cat.name)
            self.model.setHeaderData(0, QtCore.Qt.Horizontal, 'Categories')

        counter = []
        for c in range(len(self.categories)):
            counter.append(0)

        # load bboxes
        for i, roi in enumerate(self.images[self.active_image].annot_rect_items):
            print(i)
            # define color and name of object category
            concerned_cat = self.images[self.active_image].corresp_cat[i]
            color = self.categories[concerned_cat].color
            name = self.categories[concerned_cat].name

            self.viewer.draw_box_from_r(roi, color)

            category = self.images[self.active_image].corresp_cat[i]
            counter[category] += 1

            desc = 'rect_zone' + str(counter[category])
            rect_item = self.model.findItems(name)
            self.add_item_in_tree(rect_item[0], desc)

    def show_viz_threed(self):
        data = self.raw_data
        tt.surface_from_image(data, self.colormap, self.n_colors, self.user_lim_col_low, self.user_lim_col_high)

    def show_info(self):
        dialog = dia.AboutDialog()
        if dialog.exec_():
            pass

    def add_cat(self):
        """
        Add a segmentation category (eg. 'wood', 'bricks', ...)
        """
        text, ok = QtWidgets.QInputDialog.getText(self, 'Text Input Dialog',
                                                  'Enter name of category:')
        if ok:
            # add color
            color = QtWidgets.QColorDialog.getColor()
            print(color.rgb())
            if color.isValid():
                # add category to combobox
                self.comboBox_cat.addItem(text)
                self.comboBox_cat.setEnabled(True)

                # add header to ROI list
                self.add_item_in_tree(self.model, text)
                self.model.setHeaderData(0, QtCore.Qt.Horizontal, 'Categories')

                # create category class
                cat = ObjectDetectionCategory()
                cat.name = text
                cat.color = color

                self.categories.append(cat)

                # activate tools
                self.actionRectangle_selection.setEnabled(True)

                # select new cat in combobox
                nb_cat = len(self.categories)
                self.comboBox_cat.setCurrentIndex(nb_cat-1)
                self.on_cat_change()

    def on_cat_change(self):
        """
        When the combobox to choose a segmentation category is activated
        """
        self.active_i = self.comboBox_cat.currentIndex()
        if self.categories:
            self.active_category = self.categories[self.active_i]
            print(f"the active segmentation category is {self.active_category}")

    def add_rect_meas(self, nb):
        """
        Add a region of interest coming from the rectangle tool
        :param nb: number of existing roi's
        """


        img_from_gui = self.viewer.get_current_image()
        self.images[self.active_image].nb_meas_rect = img_from_gui.nb_meas_rect
        self.images[self.active_image].annot_rect_items = img_from_gui.annot_rect_items
        self.images[self.active_image].meas_rect_items = img_from_gui.meas_rect_items

        # run matplotlib on the roi
        last_coords = self.images[self.active_image].meas_rect_coords[-1]
        p1 = last_coords[0]
        p2 = last_coords[1]
        roi_meas = self.raw_data[int(p1.y()):int(p2.y()) , int(p1.x()):int(p2.x())]

        # clear old figure
        dialog = dia.Meas3dDialog()
        dialog.surface_from_image_matplot(roi_meas, self.colormap, self.n_colors, self.user_lim_col_low, self.user_lim_col_high)
        if dialog.exec_():
            pass



        # create description name
        desc = 'rect_measure_' + str(self.images[self.active_image].nb_meas_rect)

        rect_cat = self.model.findItems(RECT_MEAS_NAME)
        self.add_item_in_tree(rect_cat[0], desc)
        self.treeView.expandAll()

        # switch back to hand tool
        self.hand_pan()

    def add_line_meas(self):
        img_from_gui = self.viewer.get_current_image()
        self.images[self.active_image].nb_meas_line = img_from_gui.nb_meas_line
        self.images[self.active_image].meas_line_items = img_from_gui.meas_line_items

        desc = 'line_measure_' + str(self.images[self.active_image].nb_meas_line)
        line_cat = self.model.findItems(LINE_MEAS_NAME)
        self.add_item_in_tree(line_cat[0], desc)
        self.hand_pan()

    def add_point_meas(self):
        img_from_gui = self.viewer.get_current_image()
        self.images[self.active_image].nb_meas_point = img_from_gui.nb_meas_point
        self.images[self.active_image].meas_point_items = img_from_gui.meas_point_items
        self.images[self.active_image].meas_text_spot_items = img_from_gui.meas_text_spot_items

        # create description name
        desc = 'spot_measure_' + str(self.images[self.active_image].nb_meas_point)

        point_cat = self.model.findItems(POINT_MEAS_NAME)
        self.add_item_in_tree(point_cat[0], desc)
        self.hand_pan()

    def hand_pan(self):
        # switch back to hand tool
        self.actionHand_selector.setChecked(True)

    # measurements methods
    def rectangle_meas(self):
        if self.actionRectangle_meas.isChecked():
            self.viewer.set_image(self.images[self.active_image])

            # activate drawing tool
            self.viewer.rect_meas = True
            self.viewer.toggleDragMode()

    def point_meas(self):
        if self.actionSpot_meas.isChecked():
            self.viewer.set_image(self.images[self.active_image])

            # activate drawing tool
            self.viewer.point_meas = True
            self.viewer.toggleDragMode()

    def line_meas(self):
        if self.actionLine_meas.isChecked():
            self.viewer.set_image(self.images[self.active_image])

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
                QtWidgets.QMessageBox.warning(self, "Warning",
                                              "Oops! Some of the values are not valid!")
                self.define_options()

    def compute_delta(self, img_path):
        raw_out = img_path[:-4] + '.raw'
        tt.read_dji_image(img_path, raw_out, self.thermal_param)

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

    def estimate_temp(self):
        ref_pic_name = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file',
                                                             self.ir_folder, "Image files (*.jpg *.JPG *.gif)")
        img_path = ref_pic_name[0]
        if img_path != '':
            tmin, tmax = self.compute_delta(img_path)
            self.lineEdit_min_temp.setText(str(round(tmin, 2)))
            self.lineEdit_max_temp.setText(str(round(tmax, 2)))

        self.update_img_preview()

    def load_folder_phase1(self):
        folder = str(QtWidgets.QFileDialog.getExistingDirectory(self, "Select Directory"))

        # warning message (new project)
        if self.list_rgb_paths != '':
            qm = QtWidgets.QMessageBox
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
        self.pushButton_temp_viz.setEnabled(True)
        self.comboBox_view.setEnabled(True)
        self.comboBox_img.setEnabled(True)

        # enable action
        self.actionHand_selector.setEnabled(True)
        self.actionHand_selector.setChecked(True)

        self.actionRectangle_meas.setEnabled(True)
        self.actionSpot_meas.setEnabled(True)
        self.actionLine_meas.setEnabled(True)

    def change_meas_color(self):
        self.viewer.change_meas_color()
        self.switch_image_data()

    def switch_image_data(self):
        self.viewer.clean_scene()
        # clean tree
        self.model = QtGui.QStandardItemModel()
        self.treeView.setModel(self.model)

        self.add_item_in_tree(self.model, RECT_MEAS_NAME)
        self.add_item_in_tree(self.model, POINT_MEAS_NAME)
        self.add_item_in_tree(self.model, LINE_MEAS_NAME)

        self.model.setHeaderData(0, QtCore.Qt.Horizontal, 'Added Data')

        point_cat = self.model.findItems(POINT_MEAS_NAME)
        rect_cat = self.model.findItems(RECT_MEAS_NAME)

        list_of_items = []
        for i, item in enumerate(self.images[self.active_image].meas_point_items):
            list_of_items.append(item)
            desc = 'spot_measure_' + str(i)
            self.add_item_in_tree(point_cat[0],desc)
        for i, item in enumerate(self.images[self.active_image].meas_rect_items):
            list_of_items.append(item)
            desc = 'rect_measure_' + str(i)
            self.add_item_in_tree(rect_cat[0], desc)
        for item in self.images[self.active_image].meas_text_spot_items:
            list_of_items.append(item)

        self.viewer.draw_all_meas(list_of_items)



    def update_img_to_preview(self, direction):
        self.preview_rgb = False
        if direction == 'minus':
            self.active_image -= 1
            self.comboBox_img.setCurrentIndex(self.active_image)
        elif direction == 'plus':
            self.active_image += 1
            self.comboBox_img.setCurrentIndex(self.active_image)
        else:
            self.active_image = self.comboBox_img.currentIndex()

        # remove all measurements and annotations
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
        # fetch user choices
        v = self.comboBox_view.currentIndex()
        rgb_path = os.path.join(self.rgb_folder, self.rgb_imgs[self.active_image])

        if v == 1: # if rgb view
            self.viewer.setPhoto(QtGui.QPixmap(rgb_path))

        else:
            # colormap
            i = self.comboBox.currentIndex()
            colormap = self.colormap_list[i]
            self.colormap = colormap
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
                QtWidgets.QMessageBox.warning(self, "Warning",
                                              "Oops! A least one of the temperatures is not valid.  Try again...")
                self.lineEdit_min_temp.setText(str(round(self.tmin, 2)))
                self.lineEdit_max_temp.setText(str(round(self.tmax, 2)))

            #   out of limits color
            i = self.comboBox_colors_low.currentIndex()
            user_lim_col_low = self.out_of_matp[i]
            self.user_lim_col_low = user_lim_col_low
            i = self.comboBox_colors_high.currentIndex()
            user_lim_col_high = self.out_of_matp[i]
            self.user_lim_col_high = user_lim_col_high

            #   post process operation
            k = self.comboBox_post.currentIndex()
            post_process = self.img_post[k]

            print(self.preview_folder)
            dest_path_no_post = os.path.join(self.preview_folder, 'preview.JPG')
            dest_path_post = os.path.join(self.preview_folder, 'preview_post.JPG')

            read_path = os.path.join(self.rgb_folder, self.rgb_imgs[self.active_image])
            self.raw_data = tt.process_one_th_picture(self.thermal_param, self.drone_model, self.test_img_path, dest_path_no_post,
                                      self.tmin, self.tmax, colormap, user_lim_col_high,
                                      user_lim_col_low, n_colors=self.n_colors, post_process='none',
                                      rgb_path=read_path)

            cv_img = tt.cv_read_all_path(dest_path_no_post)
            undis, _ = tt.undis(cv_img, ir_xml_path)
            tt.cv_write_all_path(undis, dest_path_no_post)
            self.viewer.setPhoto(QtGui.QPixmap(dest_path_no_post))

            # set left and right views (in dual viewer)
            self.dual_viewer.load_images(rgb_path, dest_path_no_post)

            if k !=0: #if a post-process is applied
                _ = tt.process_one_th_picture(self.thermal_param, self.drone_model, self.test_img_path,
                                                          dest_path_post,
                                                          self.tmin, self.tmax, colormap, user_lim_col_high,
                                                          user_lim_col_low, n_colors=self.n_colors,
                                                          post_process=post_process,
                                                          rgb_path=read_path)

                if post_process != 'edge (from rgb)':
                    cv_img = tt.cv_read_all_path(dest_path_post)
                    undis = tt.undis(cv_img, ir_xml_path)
                    tt.cv_write_all_path(undis, dest_path_post)

                self.viewer.setPhoto(QtGui.QPixmap(dest_path_post))
                # set left and right views
                self.dual_viewer.load_images(rgb_path, dest_path_post)

            self.viewer.set_temperature_data(self.raw_data)

    def go_save(self):
        """
        Save measurements
        """
        # load image
        pass

    # GENERAL GUI METHODS
    def add_item_in_tree(self, parent, line):
        item = QtGui.QStandardItem(line)
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
        self.model = QtGui.QStandardItemModel()
        self.treeView.setModel(self.model)

        # clean graphicscene
        self.viewer.clean_scene()

        # clean combobox
        self.comboBox_cat.clear()

    def reset_roi(self):
        # clean tree view
        self.model = QtGui.QStandardItemModel()
        self.treeView.setModel(self.model)

        # clean roi in each cat
        self.active_image.nb_roi_rect = 0
        self.active_image.item_list_rect = []
        self.active_image.roi_list_rect = []

        for cat in self.categories:
            self.add_item_in_tree(self.model, cat.name)
        self.model.setHeaderData(0, QtCore.Qt.Horizontal, 'Categories')

        # clean graphicscene
        self.viewer.clean_scene()

    def add_icon(self, img_source, pushButton_object):
        """
        Function to add an icon to a pushButton
        """
        pushButton_object.setIcon(QtGui.QIcon(img_source))


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
    if (not QtWidgets.QApplication.instance()):
        app = QtWidgets.QApplication(argv)
        app.setStyle('Breeze')
        # apply_stylesheet(app, theme='light_blue.xml',extra=extra)

    # create the main window

    window = DroneIrWindow()
    window.setWindowIcon(QtGui.QIcon(res.find('img/icone.png')))
    window.showMaximized()

    # run the application if necessary
    if (app):
        return app.exec_()

    # no errors since we're not running our own event loop
    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))