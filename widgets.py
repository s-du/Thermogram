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
from matplotlib.colors import to_hex
from matplotlib import cm



import os
import numpy as np
import resources as res
from tools import thermal_tools as tt

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


class QRangeSlider(QSlider):
    lowerValueChanged = Signal(int)
    upperValueChanged = Signal(int)

    def __init__(self, colormap_name='', parent=None):
        super(QRangeSlider, self).__init__(Qt.Horizontal, parent)

        self._lowerValue = self.minimum()
        self._upperValue = self.maximum()

        self._lowerPressed = False
        self._upperPressed = False
        self.handleSize = 15
        self.back_color = QColor(200, 200, 200)

        # Initialize handle colors
        self.lowerHandleColor = QColor(100,100,100)
        self.upperHandleColor = QColor(100,100,100)

        if colormap_name:
            self.setHandleColorsFromColormap(colormap_name)

    def setHandleColors(self, lowerColor, upperColor):
        self.lowerHandleColor = QColor(lowerColor)
        self.upperHandleColor = QColor(upperColor)
        self.update()  # Refresh the widget

    def setHandleColorsFromColormap(self, colormap_name):
        if colormap_name == 'Artic' or colormap_name == 'Iron' or colormap_name == 'Rainbow':
            custom_cmap = tt.get_custom_cmaps(colormap_name, 256)
        else:
            custom_cmap = cm.get_cmap(colormap_name, 256)
        lower_color = to_hex(custom_cmap(0.0))  # Color at the start of the colormap
        upper_color = to_hex(custom_cmap(1.0))  # Color at the end of the colormap

        self.setHandleColors(lower_color, upper_color)

    def lowerValue(self):
        return self._lowerValue

    def upperValue(self):
        return self._lowerValue

    def setLowerValue(self, value):
        self._lowerValue = value
        self.lowerValueChanged.emit(value)
        self.update()

    def upperValue(self):
        return self._upperValue

    def setUpperValue(self, value):
        self._upperValue = value
        self.upperValueChanged.emit(value)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Draw the slider track
        rect = QRect(0, (self.height() - self.handleSize) // 2, self.width(), self.handleSize)
        painter.setBrush(QBrush(self.back_color))
        painter.drawRect(rect)

        # Draw the lower handle with the specified color
        lower_handle_rect = self.handleRect(self._lowerValue)
        painter.setBrush(QBrush(self.lowerHandleColor))
        painter.drawRect(lower_handle_rect)

        # Draw the upper handle with the specified color
        upper_handle_rect = self.handleRect(self._upperValue)
        painter.setBrush(QBrush(self.upperHandleColor))
        painter.drawRect(upper_handle_rect)

    def handleRect(self, value):
        xPos = self.scalePosition(value)
        return QRect(xPos - self.handleSize // 2, (self.height() - self.handleSize) // 2, self.handleSize,
                     self.handleSize)

    def scalePosition(self, value):
        return ((value - self.minimum()) / (self.maximum() - self.minimum())) * self.width()

    def mousePressEvent(self, event):
        if self.handleRect(self._lowerValue).contains(event.pos()):
            self._lowerPressed = True
        elif self.handleRect(self._upperValue).contains(event.pos()):
            self._upperPressed = True

    def mouseMoveEvent(self, event):
        posValue = self.pixelPosToRangeValue(event.pos())

        if self._lowerPressed:
            # Constrain the lower handle
            if posValue < self.minimum():
                posValue = self.minimum()
            elif posValue > self._upperValue:
                posValue = self._upperValue
            self.setLowerValue(posValue)

        elif self._upperPressed:
            # Constrain the upper handle
            if posValue > self.maximum():
                posValue = self.maximum()
            elif posValue < self._lowerValue:
                posValue = self._lowerValue
            self.setUpperValue(posValue)

    def mouseReleaseEvent(self, event):
        self._lowerPressed = False
        self._upperPressed = False

    def pixelPosToRangeValue(self, pos):
        return int(((pos.x() / self.width()) * (self.maximum() - self.minimum())) + self.minimum())


class MplCanvas_project3d(FigureCanvasQTAgg):
    """
    Class for embedding matplotlib plots into PySide6
    """

    def __init__(self, parent=None):
        fig = Figure()
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


# DUAL IMAGE VIEWER
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
    endDrawing_rect_meas = Signal(QGraphicsRectItem)
    endDrawing_point_meas = Signal(QPointF)
    endDrawing_line_meas = Signal(QGraphicsLineItem)

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

    def add_item_from_annot(self, item):
        if not isinstance(item, QGraphicsTextItem):
            item.setPen(self.pen_meas)

        self._scene.addItem(item)

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
            self.origin = self.mapToScene(event.pos())

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
                # emit signal (end of measure)
                self.endDrawing_rect_meas.emit(self._current_rect_item)
                print('rectangle ROI added')
            self._current_rect_item = None
            self.toggleDragMode()

        elif self.line_meas:
            self.line_meas = False

            if self._current_line_item is not None:
                # save measurement to image metadata

                # emit signal (end of measure)
                self.endDrawing_line_meas.emit(self._current_line_item)
                print('Line meas. added')

            self.origin = QPoint()
            self._current_line_item = None
            self.toggleDragMode()

        elif self.point_meas:
            # emit signal (end of measure)
            self.endDrawing_point_meas.emit(self.origin)

            self._current_ellipse_item = None
            self.toggleDragMode()
            self.origin = QPoint()

            self.point_meas = False

        super(PhotoViewer, self).mouseReleaseEvent(event)
