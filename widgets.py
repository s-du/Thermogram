from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *
from PySide6.QtUiTools import QUiLoader

import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

import os
import numpy as np
import resources as res

SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))


class UiLoader(QUiLoader):
    """
    Subclass :class:`~PySide.QtUiTools.QUiLoader` to create the user interface
    in a base instance.

    Unlike :class:`~PySide.QtUiTools.QUiLoader` itself this class does not
    create a new instance of the top-level widget, but creates the user
    interface in an existing instance of the top-level class.

    This mimics the behaviour of :func:`PyQt4.uic.loadUi`.
    """

    def __init__(self, baseinstance, customWidgets=None):
        """
        Create a loader for the given ``baseinstance``.

        The user interface is created in ``baseinstance``, which must be an
        instance of the top-level class in the user interface to load, or a
        subclass thereof.

        ``customWidgets`` is a dictionary mapping from class name to class object
        for widgets that you've promoted in the Qt Designer interface. Usually,
        this should be done by calling registerCustomWidget on the QUiLoader, but
        with PySide 1.1.2 on Ubuntu 12.04 x86_64 this causes a segfault.

        ``parent`` is the parent object of this loader.
        """

        QUiLoader.__init__(self, baseinstance)
        self.baseinstance = baseinstance
        self.customWidgets = customWidgets

    def createWidget(self, class_name, parent=None, name=''):
        """
        Function that is called for each widget defined in ui file,
        overridden here to populate baseinstance instead.
        """

        if parent is None and self.baseinstance:
            # supposed to create the top-level widget, return the base instance
            # instead
            return self.baseinstance

        else:
            if class_name in self.availableWidgets():
                # create a new widget for child widgets
                widget = QUiLoader.createWidget(self, class_name, parent, name)

            else:
                # if not in the list of availableWidgets, must be a custom widget
                # this will raise KeyError if the user has not supplied the
                # relevant class_name in the dictionary, or TypeError, if
                # customWidgets is None
                try:
                    widget = self.customWidgets[class_name](parent)

                except (TypeError, KeyError) as e:
                    raise Exception(
                        'No custom widget ' + class_name + ' found in customWidgets param of UiLoader __init__.')

            if self.baseinstance:
                # set an attribute for the new child widget on the base
                # instance, just like PyQt4.uic.loadUi does.
                setattr(self.baseinstance, name, widget)

                # this outputs the various widget names, e.g.
                # sampleGraphicsView, dockWidget, samplesTableView etc.
                # print(name)

            return widget


def loadUi(uifile, baseinstance=None, customWidgets=None,
           workingDirectory=None):
    """
    Dynamically load a user interface from the given ``uifile``.

    ``uifile`` is a string containing a file name of the UI file to load.

    If ``baseinstance`` is ``None``, the a new instance of the top-level widget
    will be created.  Otherwise, the user interface is created within the given
    ``baseinstance``.  In this case ``baseinstance`` must be an instance of the
    top-level widget class in the UI file to load, or a subclass thereof.  In
    other words, if you've created a ``QMainWindow`` interface in the designer,
    ``baseinstance`` must be a ``QMainWindow`` or a subclass thereof, too.  You
    cannot load a ``QMainWindow`` UI file with a plain
    :class:`~PySide.QtGui.QWidget` as ``baseinstance``.

    ``customWidgets`` is a dictionary mapping from class name to class object
    for widgets that you've promoted in the Qt Designer interface. Usually,
    this should be done by calling registerCustomWidget on the QUiLoader, but
    with PySide 1.1.2 on Ubuntu 12.04 x86_64 this causes a segfault.

    :method:`~PySide.QtCore.QMetaObject.connectSlotsByName()` is called on the
    created user interface, so you can implemented your slots according to its
    conventions in your widget class.

    Return ``baseinstance``, if ``baseinstance`` is not ``None``.  Otherwise
    return the newly created instance of the user interface.
    """

    loader = UiLoader(baseinstance, customWidgets)

    if workingDirectory is not None:
        loader.setWorkingDirectory(workingDirectory)

    widget = loader.load(uifile)
    QMetaObject.connectSlotsByName(widget)
    return widget


class MplCanvas_project3d(FigureCanvasQTAgg):
    """
    Class for embedding matplotlib plots into PySide6
    """
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        super(MplCanvas_project3d, self).__init__(fig)


class TableModel(QAbstractTableModel):
    def __init__(self, data):
        super(TableModel, self).__init__()
        self._data = data

    def data(self, index, role):
        if role == Qt.DisplayRole:
            # See below for the nested-list data structure.
            # .row() indexes into the outer list,
            # .column() indexes into the sub-list
            return self._data[index.row()][index.column()]

    def rowCount(self, index):
        # The length of the outer list.
        return len(self._data)

    def columnCount(self, index):
        # The following takes the first sub-list, and returns
        # the length (only works if all rows are an equal length)
        return len(self._data[0])

def QPixmapFromItem(item):
    """
    Transform a QGraphicsitem into a Pixmap
    :param item: QGraphicsItem
    :return: QPixmap
    """
    pixmap = QPixmap(item.boundingRect().size().toSize())
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    # this line seems to be needed for all items except of a LineItem...
    painter.translate(-item.boundingRect().x(), -item.boundingRect().y())
    painter.setRenderHint(QPainter.Antialiasing, True)
    opt = QStyleOptionGraphicsItem()
    item.paint(painter, opt)  # here in some cases the self is needed
    return pixmap


def QPixmapToArray(pixmap):
    ## Get the size of the current pixmap
    size = pixmap.size()
    h = size.width()
    w = size.height()

    ## Get the QImage Item and convert it to a byte string
    qimg = pixmap.toImage()
    byte_str = qimg.bits().tobytes()

    ## Using the np.frombuffer function to convert the byte string into an np array
    img = np.frombuffer(byte_str, dtype=np.uint8).reshape((w, h, 4))

    return img


# CUSTOM 3D VIEWER
class Custom3dView:
    def __init__(self, cloud_ir, tmin, tmax, loc_tmin, loc_tmax):

        app = gui.Application.instance
        self.window = app.create_window("Open3D - Infrared analyzer", 1024, 768)

        self.window.set_on_layout(self._on_layout)
        self.widget3d = gui.SceneWidget()
        self.window.add_child(self.widget3d)

        self.info = gui.Label("")
        self.info.visible = False
        self.window.add_child(self.info)

        # create all voxel grids
        self.pc_ir = cloud_ir
        self.voxel_grids = []
        self.voxel_size = [2, 5, 10, 20]

        for size in self.voxel_size:
            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(self.pc_ir,
                                                                        voxel_size=size)
            self.voxel_grids.append(voxel_grid)

        self.widget3d.scene = rendering.Open3DScene(self.window.renderer)
        self.widget3d.enable_scene_caching(True)
        self.widget3d.scene.scene.set_sun_light(
            [0.577, -0.577, -0.577],  # direction
            [1, 1, 1],  # color
            100000)  # intensity
        self.widget3d.scene.scene.enable_sun_light(True)
        self.widget3d.set_on_sun_direction_changed(self._on_sun_dir)

        self.mat = rendering.MaterialRecord()
        self.mat.shader = "defaultLit"
        # Point size is in native pixels, but "pixel" means different things to
        # different platforms (macOS, in particular), so multiply by Window scale
        # factor.
        self.mat.point_size = 3 * self.window.scaling

        self.mat_maxi = rendering.MaterialRecord()
        self.mat_maxi.shader = "defaultUnlit"
        self.mat_maxi.point_size = 15 * self.window.scaling

        # show one geometry
        self.widget3d.scene.add_geometry('PC 2', self.voxel_grids[2], self.mat)
        self.current_index = 2

        self.widget3d.scene.show_geometry("Point Cloud IR 0", True)
        self.widget3d.force_redraw()

        em = self.window.theme.font_size
        self.layout = gui.Vert(0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))
        view_ctrls = gui.CollapsableVert("View controls", 0.25 * em,
                                         gui.Margins(em, 0, 0, 0))

        # add combo for lit/unlit/depth
        self._shader = gui.Combobox()
        self.materials = ["defaultLit", "defaultUnlit", "normals", "depth"]
        self.materials_name = ['Sun Light', 'No light', 'Normals', 'Depth']
        self._shader.add_item(self.materials_name[0])
        self._shader.add_item(self.materials_name[1])
        self._shader.add_item(self.materials_name[2])
        self._shader.add_item(self.materials_name[3])
        self._shader.set_on_selection_changed(self._on_shader)
        combo_light = gui.Horiz(0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))
        combo_light.add_child(gui.Label("Rendering"))
        combo_light.add_child(self._shader)

        # add combo for voxel size
        self._voxel = gui.Combobox()

        self.voxel_name = ["2", "5", "10", "20"]
        self._voxel.add_item(self.voxel_name[0])
        self._voxel.add_item(self.voxel_name[1])
        self._voxel.add_item(self.voxel_name[2])
        self._voxel.add_item(self.voxel_name[3])
        self._voxel.set_on_selection_changed(self._on_voxel)

        combo_light = gui.Horiz(0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))
        combo_light.add_child(gui.Label("Rendering"))
        combo_light.add_child(self._shader)
        combo_light.add_child(self._voxel)

        # layout
        view_ctrls.add_child(combo_light)
        self.layout.add_child(view_ctrls)
        self.window.add_child(self.layout)

        bounds = self.widget3d.scene.bounding_box
        center = bounds.get_center()
        self.widget3d.setup_camera(60, bounds, center)
        self.widget3d.look_at(center, center + [0, 0, 400], [0, 0, 1])
        self.widget3d.set_on_mouse(self._on_mouse_widget3d)

        # We are sizing the info label to be exactly the right size,
        # so since the text likely changed width, we need to
        # re-layout to set the new frame.
        self.window.set_needs_layout()
        self.widget3d.set_on_mouse(self._on_mouse_widget3d)

        # add labels for min and max values
        loc_tmin = np.append(loc_tmin, tmin * 10)
        loc_tmax = np.append(loc_tmax, tmax * 10)
        text_max = 'Temp. max.:' + str(round(tmax, 2)) + '°C'
        text_min = 'Temp. min.:' + str(round(tmin, 2)) + '°C'
        lab_tmin = self.widget3d.add_3d_label(loc_tmin, text_min)
        lab_tmax = self.widget3d.add_3d_label(loc_tmax, text_max)
        lab_color_red = gui.Color(1, 0, 0)
        lab_color_blue = gui.Color(0, 0, 1)

        lab_tmin.color = lab_color_blue
        lab_tmax.color = lab_color_red

        # add points
        pcd_maxi = o3d.geometry.PointCloud()
        array = np.array([loc_tmin, loc_tmax])
        pcd_maxi.points = o3d.utility.Vector3dVector(array)
        color_array = np.array([[0, 0, 1], [1, 0, 0]])
        pcd_maxi.colors = o3d.utility.Vector3dVector(color_array)
        self.widget3d.scene.add_geometry('Max/Min', pcd_maxi, self.mat_maxi)

    def choose_material(self, is_enabled):
        pass

    def _on_layout(self, layout_context):
        r = self.window.content_rect
        self.widget3d.frame = r
        pref = self.info.calc_preferred_size(layout_context,
                                             gui.Widget.Constraints())

        width = 17 * layout_context.theme.font_size
        height = min(
            r.height,
            self.layout.calc_preferred_size(
                layout_context, gui.Widget.Constraints()).height)

        self.layout.frame = gui.Rect(r.get_right() - width, r.y, width,
                                     height)

        self.info.frame = gui.Rect(r.x,
                                   r.get_bottom() - pref.height, pref.width,
                                   pref.height)

    def _on_voxel(self, name, index):
        print('ok!')
        old_name = f"PC {self.current_index}"
        print(old_name)
        self.widget3d.scene.remove_geometry(old_name)
        self.widget3d.scene.add_geometry(f"PC {index}", self.voxel_grids[index], self.mat)
        self.current_index = index
        print('everything good')
        self.widget3d.force_redraw()

    def _on_shader(self, name, index):
        material = self.materials[index]
        print(material)
        self.mat.shader = material
        self.widget3d.scene.update_material(self.mat)
        self.widget3d.force_redraw()

    def _on_sun_dir(self, sun_dir):
        self.widget3d.scene.scene.set_sun_light(sun_dir, [1, 1, 1], 100000)
        self.widget3d.force_redraw()

    def _on_mouse_widget3d(self, event):
        # We could override BUTTON_DOWN without a modifier, but that would
        # interfere with manipulating the scene.
        if event.type == gui.MouseEvent.Type.BUTTON_DOWN and event.is_modifier_down(
                gui.KeyModifier.CTRL):

            def depth_callback(depth_image):
                # Coordinates are expressed in absolute coordinates of the
                # window, but to dereference the image correctly we need them
                # relative to the origin of the widget. Note that even if the
                # scene widget is the only thing in the window, if a menubar
                # exists it also takes up space in the window (except on macOS).
                x = event.x - self.widget3d.frame.x
                y = event.y - self.widget3d.frame.y
                # Note that np.asarray() reverses the axes.
                depth = np.asarray(depth_image)[y, x]

                if depth == 1.0:  # clicked on nothing (i.e. the far plane)
                    text = ""
                    coords = []
                else:
                    world = self.widget3d.scene.camera.unproject(
                        event.x, event.y, depth, self.widget3d.frame.width,
                        self.widget3d.frame.height)
                    text = "({:.3f}, {:.3f}, {:.3f})".format(
                        world[0], world[1], world[2])

                    # add 3D label
                    self.widget3d.add_3d_label(world, '._yeah')

                # This is not called on the main thread, so we need to
                # post to the main thread to safely access UI items.
                def update_label():
                    self.info.text = text
                    self.info.visible = (text != "")
                    # We are sizing the info label to be exactly the right size,
                    # so since the text likely changed width, we need to
                    # re-layout to set the new frame.
                    self.window.set_needs_layout()

                gui.Application.instance.post_to_main_thread(
                    self.window, update_label)

            self.widget3d.scene.scene.render_to_depth_image(depth_callback)

            return gui.Widget.EventCallbackResult.HANDLED
        return gui.Widget.EventCallbackResult.IGNORED


# DUAL IMAGE VIEWER
def createLineIterator(P1, P2, img):
    """
    Source: https://stackoverflow.com/questions/32328179/opencv-3-0-lineiterator
    Produces and array that consists of the coordinates and intensities of each pixel in a line between two points

    Parameters:
        -P1: a numpy array that consists of the coordinate of the first point (x,y)
        -P2: a numpy array that consists of the coordinate of the second point (x,y)
        -img: the image being processed

    Returns:
        -it: a numpy array that consists of the coordinates and intensities of each pixel in the radii (shape: [numPixels, 3], row = [x,y,intensity])
    """
    # define local variables for readability
    imageH = img.shape[0]
    imageW = img.shape[1]
    P1X = P1[0]
    P1Y = P1[1]
    P2X = P2[0]
    P2Y = P2[1]

    # difference and absolute difference between points
    # used to calculate slope and relative location between points
    dX = P2X - P1X
    dY = P2Y - P1Y
    dXa = np.abs(dX)
    dYa = np.abs(dY)

    # predefine numpy array for output based on distance between points
    itbuffer = np.empty(shape=(np.maximum(dYa, dXa), 3), dtype=np.float32)
    itbuffer.fill(np.nan)

    # Obtain coordinates along the line using a form of Bresenham's algorithm
    negY = P1Y > P2Y
    negX = P1X > P2X
    if P1X == P2X:  # vertical line segment
        itbuffer[:, 0] = P1X
        if negY:
            itbuffer[:, 1] = np.arange(P1Y - 1, P1Y - dYa - 1, -1)
        else:
            itbuffer[:, 1] = np.arange(P1Y + 1, P1Y + dYa + 1)
    elif P1Y == P2Y:  # horizontal line segment
        itbuffer[:, 1] = P1Y
        if negX:
            itbuffer[:, 0] = np.arange(P1X - 1, P1X - dXa - 1, -1)
        else:
            itbuffer[:, 0] = np.arange(P1X + 1, P1X + dXa + 1)
    else:  # diagonal line segment
        steepSlope = dYa > dXa
        if steepSlope:
            slope = dX.astype(np.float32) / dY.astype(np.float32)
            if negY:
                itbuffer[:, 1] = np.arange(P1Y - 1, P1Y - dYa - 1, -1)
            else:
                itbuffer[:, 1] = np.arange(P1Y + 1, P1Y + dYa + 1)
            itbuffer[:, 0] = (slope * (itbuffer[:, 1] - P1Y)).astype(int) + P1X
        else:
            slope = dY.astype(np.float32) / dX.astype(np.float32)
            if negX:
                itbuffer[:, 0] = np.arange(P1X - 1, P1X - dXa - 1, -1)
            else:
                itbuffer[:, 0] = np.arange(P1X + 1, P1X + dXa + 1)
            itbuffer[:, 1] = (slope * (itbuffer[:, 0] - P1X)).astype(int) + P1Y

    # Remove points outside of image
    colX = itbuffer[:, 0]
    colY = itbuffer[:, 1]
    itbuffer = itbuffer[(colX >= 0) & (colY >= 0) & (colX < imageW) & (colY < imageH)]

    # Get intensities from img ndarray
    itbuffer[:, 2] = img[itbuffer[:, 1].astype(int), itbuffer[:, 0].astype(int)]

    return itbuffer


class DualViewer(QWidget):
    def __init__(self):
        super().__init__()

        # Create the main widget and layout
        self.layout = QHBoxLayout()

        # Create the QGraphicsView widgets
        self.view1 = QGraphicsView()
        self.view2 = QGraphicsView()
        self.view1.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view1.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view2.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view2.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # Create the QGraphicsScenes for each view
        self.scene1 = QGraphicsScene()
        self.scene2 = QGraphicsScene()

        # Set the scenes for the views
        self.view1.setScene(self.scene1)
        self.view2.setScene(self.scene2)

        # Connect the view1's zoom event to the zoom_views function
        self.view1.wheelEvent = lambda event: self.zoom_views(event)
        self.view2.wheelEvent = lambda event: self.zoom_views(event)

        # Connect the view1's mouse events to the pan_views function
        self.view1.mousePressEvent = lambda event: self.pan_views(event)
        self.view1.mouseMoveEvent = lambda event: self.pan_views(event)
        self.view1.mouseReleaseEvent = lambda event: self.pan_views(event)

        # Add the views to the layout
        self.layout.addWidget(self.view1)
        self.layout.addWidget(self.view2)

        # Set the central widget
        self.setLayout(self.layout)

        # Store the zoom scale for synchronization
        self.zoom_scale = 1.0
        self.pan_origin = QPointF()

    def load_images_from_path(self, image_path1, image_path2):
        im1 = Image.open(image_path1)
        w1, _ = im1.size

        im2 = Image.open(image_path2)
        w2, _ = im2.size
        self.ratio = w2 / w1

        # Load the images from the file paths
        pixmap1 = QPixmap(image_path1)
        pixmap2 = QPixmap(image_path2)

        # Clear the scenes
        self.scene1.clear()
        self.scene2.clear()

        # Add the images to the respective scenes
        self.scene1.addPixmap(pixmap1)
        self.scene2.addPixmap(pixmap2)

        # Fit the views to the images
        self.view1.fitInView(self.scene1.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self.view2.fitInView(self.scene2.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

        # Reset the zoom scale and pan origin
        self.zoom_scale = 1.0
        self.pan_origin = QPointF()

    def zoom_views(self, event):
        # Get the zoom factor from the wheel event
        zoom_factor = 1.2 ** (event.angleDelta().y() / 120.0)
        print(zoom_factor, self.ratio)

        # Calculate the new zoom scale
        self.zoom_scale *= zoom_factor

        # Zoom view1
        view1_transform = QTransform()
        view1_transform.scale(self.zoom_scale, self.zoom_scale)
        self.view1.setTransform(view1_transform)

        # Zoom view2
        view2_transform = QTransform()
        view2_transform.scale(self.zoom_scale / self.ratio, self.zoom_scale / self.ratio)
        self.view2.setTransform(view2_transform)

    def pan_views(self, event):
        if event.buttons() == Qt.MouseButtons.LeftButton:
            if event.type() == QEvent.Type.MouseButtonPress:
                # Store the initial mouse position for panning
                self.pan_origin = event.pos()
            elif event.type() == QEvent.Type.MouseMove:
                # Calculate the delta between the current and initial mouse position
                delta = event.pos() - self.pan_origin

                # Pan view1
                view1_horizontal_scroll = self.view1.horizontalScrollBar()
                view1_vertical_scroll = self.view1.verticalScrollBar()
                view1_horizontal_scroll.setValue(view1_horizontal_scroll.value() - delta.x())
                view1_vertical_scroll.setValue(view1_vertical_scroll.value() - delta.y())

                # Pan view2
                view2_horizontal_scroll = self.view2.horizontalScrollBar()
                view2_vertical_scroll = self.view2.verticalScrollBar()
                view2_horizontal_scroll.setValue(view2_horizontal_scroll.value() - delta.x())
                view2_vertical_scroll.setValue(view2_vertical_scroll.value() - delta.y())

                # Update the pan origin for the next mouse move
                self.pan_origin = event.pos()
            elif event.type() == event.MouseButtonRelease:
                # Reset the pan origin
                self.pan_origin = QPointF()


class PhotoViewer(QGraphicsView):
    photoClicked = Signal(QPoint)
    endDrawing_brush_meas = Signal(int)
    endDrawing_rect_meas = Signal(int)
    endDrawing_point_meas = Signal()
    endDrawing_line_meas = Signal()

    def __init__(self, parent):
        super(PhotoViewer, self).__init__(parent)
        self._zoom = 0
        self._empty = True
        self._scene = QGraphicsScene(self)
        self._photo = QGraphicsPixmapItem()
        self._scene.addItem(self._photo)
        self.setScene(self._scene)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.setBackgroundBrush(QBrush(QColor(255, 255, 255)))
        self.setFrameShape(QFrame.NoFrame)

        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.rect_meas = False
        self.point_meas = False
        self.line_meas = False
        self.painting = False
        self.setMouseTracking(True)
        self.origin = QPoint()

        self._current_ellipse_item = None
        self._current_line_item = None
        self._current_rect_item = None
        self._current_path_item = None
        self._current_text_item = None
        self._current_path = None

        # define custom cursor
        cur_img = res.find('img/circle.png')
        self.cur_pixmap = QPixmap(cur_img)
        pixmap_scaled = self.cur_pixmap.scaledToWidth(12)
        self.brush_cur = QCursor(pixmap_scaled)

        # measurement main color
        self.meas_color = QColor(255, 0, 0, a=255)
        self.pen_meas = QPen()
        # self.pen.setStyle(Qt.DashDotLine)
        self.pen_meas.setWidth(2)
        self.pen_meas.setColor(self.meas_color)
        self.pen_meas.setCapStyle(Qt.RoundCap)
        self.pen_meas.setJoinStyle(Qt.RoundJoin)

        self.categories = None
        self.images = None
        self.active_image = None

    def has_photo(self):
        return not self._empty

    def change_meas_color(self):
        self.meas_color = QColorDialog.getColor()
        self.pen_meas.setColor(self.meas_color)

    def showEvent(self, event):
        self.fitInView()
        super(PhotoViewer, self).showEvent(event)

    def set_temperature_data(self, temp):
        self.temperatures = temp
        print(np.shape(np.array(temp)))

    def fitInView(self, scale=True):
        rect = QRectF(self._photo.pixmap().rect())
        print(rect)
        if not rect.isNull():
            self.setSceneRect(rect)
            if self.has_photo():
                unity = self.transform().mapRect(QRectF(0, 0, 1, 1))
                print('unity: ', unity)
                self.scale(1 / unity.width(), 1 / unity.height())
                viewrect = self.viewport().rect()
                print('view: ', viewrect)
                scenerect = self.transform().mapRect(rect)
                print('scene: ', viewrect)
                factor = min(viewrect.width() / scenerect.width(),
                             viewrect.height() / scenerect.height())
                self.scale(factor, factor)
            self._zoom = 0

    def clean_scene(self):
        for item in self._scene.items():
            print(type(item))
            if isinstance(item, QGraphicsPathItem):
                self._scene.removeItem(item)
            elif isinstance(item, QGraphicsRectItem):
                self._scene.removeItem(item)
            elif isinstance(item, QGraphicsEllipseItem):
                self._scene.removeItem(item)
            elif isinstance(item, QGraphicsTextItem):
                self._scene.removeItem(item)
            elif isinstance(item, QGraphicsLineItem):
                self._scene.removeItem(item)

    def draw_all_meas(self, meas_items):
        for item in meas_items:
            if isinstance(item, QGraphicsLineItem) or isinstance(item, QGraphicsRectItem) or isinstance(item,
                                                                                                        QGraphicsEllipseItem):
                item.setPen(self.pen_meas)
            self._scene.addItem(item)

    def setPhoto(self, pixmap=None):
        self._zoom = 0
        if pixmap and not pixmap.isNull():
            self._empty = False
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            self._photo.setPixmap(pixmap)
        else:
            self._empty = True
            self.setDragMode(QGraphicsView.NoDrag)
            self._photo.setPixmap(QPixmap())
        self.fitInView()

    def change_to_brush_cursor(self):
        self.setCursor(self.brush_cur)

    def toggleDragMode(self):
        if not self.rect_meas or self.point_meas or self.painting:
            if self.dragMode() == QGraphicsView.ScrollHandDrag:
                self.setDragMode(QGraphicsView.NoDrag)
            elif not self._photo.pixmap().isNull():
                self.setDragMode(QGraphicsView.ScrollHandDrag)
        else:
            self.setDragMode(QGraphicsView.NoDrag)

    def get_current_image(self):
        return self.active_image

    def get_coord(self, QGraphicsRect):
        rect = QGraphicsRect.rect()
        coord = [rect.topLeft(), rect.bottomRight()]
        print(coord)

        return coord

    # mouse events
    def wheelEvent(self, event):
        print(self._zoom)
        if self.has_photo():
            if event.angleDelta().y() > 0:
                factor = 1.25
                self._zoom += 1
            else:
                factor = 0.8
                self._zoom -= 1
            if self._zoom > 0:
                self.scale(factor, factor)
            elif self._zoom == 0:
                self.fitInView()
            else:
                self._zoom = 0

    def set_image(self, active_im):
        self.active_image = active_im

    def draw_box(self, color, coord1, coord2):
        self.pen_meas.setColor(color)
        box = QGraphicsRectItem()
        box.setPen(self.pen_meas)

        r = QRectF(coord1, coord2)
        box.setRect(r)

        self._scene.addItem(box)

    def draw_box_from_r(self, rect_item, color):
        self.pen_meas.setColor(color)

        rect_item.setPen(self.pen_meas)
        self._scene.addItem(rect_item)

    def mousePressEvent(self, event):
        if self.rect_meas:
            self._current_rect_item = QGraphicsRectItem()
            # self._current_rect_item.setFlag(QGraphicsItem.ItemIsSelectable)
            self._current_rect_item.setPen(self.pen_meas)

            self._scene.addItem(self._current_rect_item)
            self.origin = self.mapToScene(event.pos())
            r = QRectF(self.origin, self.origin)
            self._current_rect_item.setRect(r)

        elif self.line_meas:
            self._current_line_item = QGraphicsLineItem()
            self._current_line_item.setPen(self.pen_meas)

            self._scene.addItem(self._current_line_item)
            self.origin = self.mapToScene(event.pos())

            self._current_line_item.setLine(QLineF(self.origin, self.origin))

        elif self.point_meas:
            self._current_ellipse_item = QGraphicsEllipseItem()
            # self._current_ellipse_item.setFlag(QGraphicsItem.ItemIsSelectable)
            self._current_ellipse_item.setPen(self.pen_meas)

            self._scene.addItem(self._current_ellipse_item)

            self.origin = self.mapToScene(event.pos())
            p1 = QPointF(self.origin.x() - 2, self.origin.y() - 2)
            p2 = QPointF(self.origin.x() + 2, self.origin.y() + 2)

            r = QRectF(p1, p2)
            self._current_ellipse_item.setRect(r)

            # add temperature label
            print(int(self.origin.x()), int(self.origin.y()))
            clicked_temp = self.temperatures[int(self.origin.y()), int(self.origin.x())]
            clicked_temp = round(clicked_temp, 2)
            self._current_text_item = QGraphicsTextItem()
            self._current_text_item.setPos(self.origin)
            self._current_text_item.setHtml(
                "<div style='background-color:rgba(255, 255, 255, 0.3);'>" + str(clicked_temp) + "</div>")
            self._scene.addItem(self._current_text_item)

        else:
            if self._photo.isUnderMouse():
                self.photoClicked.emit(self.mapToScene(event.pos()).toPoint())
        super(PhotoViewer, self).mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.rect_meas:
            if self._current_rect_item is not None:
                new_coord = self.mapToScene(event.pos())
                r = QRectF(self.origin, new_coord)
                self._current_rect_item.setRect(r)
        elif self.line_meas:
            if self._current_line_item is not None:
                self.new_coord = self.mapToScene(event.pos())
                self._current_line_item.setLine(QLineF(self.origin, self.new_coord))


        elif self.painting:
            if self._current_path_item is not None:
                new_coord = self.mapToScene(event.pos())
                self._current_path.lineTo(new_coord)
                self._current_path_item.setPath(self._current_path)

        super(PhotoViewer, self).mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.rect_meas:
            self.rect_meas = False
            self.origin = QPoint()
            if self._current_rect_item is not None:
                coord = self.get_coord(self._current_rect_item)

                # save measurement to image metadata
                self.active_image.meas_rect_items.append(self._current_rect_item)
                self.active_image.meas_rect_coords.append(coord)
                self.active_image.nb_meas_rect += 1

                # emit signal (end of measure)
                self.endDrawing_rect_meas.emit(self.active_image.nb_meas_rect)
                print('rectangle ROI added: ' + str(coord))
            self._current_rect_item = None
            self.toggleDragMode()
        elif self.line_meas:
            self.line_meas = False

            if self._current_line_item is not None:
                # save measurement to image metadata
                self.active_image.meas_line_items.append(self._current_line_item)
                self.active_image.nb_meas_line += 1

                # emit signal (end of measure)
                self.endDrawing_line_meas.emit()
                print('Line meas. added')

            # compute line values
            p1 = np.array([int(self.origin.x()), int(self.origin.y())])
            p2 = np.array([int(self.new_coord.x()), int(self.new_coord.y())])
            print(p1,p2)
            line_values = createLineIterator(p1, p2, self.temperatures)
            plt.plot(line_values[:,2])
            plt.ylabel('Temperature [°C]')
            plt.show()

            self.origin = QPoint()
            self._current_line_item = None
            self.toggleDragMode()

        elif self.point_meas:
            # emit signal (end of measure)
            self.endDrawing_point_meas.emit()

            # save measurement to image metadata
            self.active_image.meas_point_items.append(self._current_ellipse_item)
            self.active_image.meas_text_spot_items.append(self._current_text_item)
            self.active_image.nb_meas_point += 1
            self._current_ellipse_item = None
            self.toggleDragMode()
            self.origin = QPoint()

            self.point_meas = False

        super(PhotoViewer, self).mouseReleaseEvent(event)
