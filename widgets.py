from PyQt6.QtCore import *  # Changed from PySide6.QtCore
from PyQt6.QtGui import *  # Changed from PySide6.QtGui
from PyQt6.QtWidgets import *  # Changed from PySide6.QtWidgets

from PIL import Image
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.colors import to_hex
from matplotlib import cm

import os
import numpy as np
import resources as res
from tools import thermal_tools as tt

SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))


class QRangeSlider(QSlider):
    lowerValueChanged = pyqtSignal(int)
    upperValueChanged = pyqtSignal(int)

    def __init__(self, colormap_name='', parent=None):
        super(QRangeSlider, self).__init__(Qt.Orientation.Horizontal, parent)

        self._lowerValue = self.minimum()
        self._upperValue = self.maximum()

        self._lowerPressed = False
        self._upperPressed = False
        self.handleWidth = 20
        self.handleHeight = 30
        self.trackHeight = 30
        self.back_color = QColor(200, 200, 200)

        # Initialize handle colors
        self.lowerHandleColor = QColor(100, 100, 100)
        self.upperHandleColor = QColor(100, 100, 100)

        if colormap_name:
            self.setHandleColorsFromColormap(colormap_name)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Draw the slider track
        track_y = (self.height() - self.trackHeight) // 2  # Center the track vertically
        track_rect = QRect(0, track_y, self.width(), self.trackHeight)
        painter.setBrush(QBrush(self.back_color))
        painter.drawRect(track_rect)

        # Draw the lower handle
        lower_handle_rect = self.handleRect(self._lowerValue)
        painter.setBrush(QBrush(self.lowerHandleColor))
        painter.drawRect(lower_handle_rect)

        # Draw the upper handle
        upper_handle_rect = self.handleRect(self._upperValue)
        painter.setBrush(QBrush(self.upperHandleColor))
        painter.drawRect(upper_handle_rect)

    def handleRect(self, value):
        """Calculate handle position using separate width and height."""
        xPos = self.scalePosition(value)
        yPos = (self.height() - self.handleHeight) // 2  # Center handle vertically
        return QRect(
            int(xPos - self.handleWidth // 2),  # Horizontal center
            yPos,                               # Vertical center
            int(self.handleWidth),              # Handle width
            int(self.handleHeight)              # Handle height
        )

    def scalePosition(self, value):
        return ((value - self.minimum()) / (self.maximum() - self.minimum())) * self.width()

    def setHandleDimensions(self, width, height):
        """Allow external control over handle size."""
        self.handleWidth = width
        self.handleHeight = height
        self.update()  # Refresh the widget

    # Example: Set handle colors
    def setHandleColors(self, lowerColor, upperColor):
        self.lowerHandleColor = QColor(lowerColor)
        self.upperHandleColor = QColor(upperColor)
        self.update()  # Refresh the widget

    # Example: Set track height
    def setTrackHeight(self, height):
        self.trackHeight = height
        self.update()  # Refresh the widget

    def setHandleColorsFromColormap(self, colormap_name):
        if colormap_name in tt.LIST_CUSTOM_NAMES:
            all_cmaps = tt.get_all_custom_cmaps(256)
            custom_cmap = all_cmaps[colormap_name]
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
        if role == Qt.ItemDataRole.DisplayRole:
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
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    # this line seems to be needed for all items except of a LineItem...
    painter.translate(-item.boundingRect().x(), -item.boundingRect().y())
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
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
        self.view1.setBackgroundBrush(QBrush(Qt.GlobalColor.transparent))
        self.view1.setFrameShape(QFrame.Shape.NoFrame)
        self.view2.setBackgroundBrush(QBrush(Qt.GlobalColor.transparent))
        self.view2.setFrameShape(QFrame.Shape.NoFrame)
        # Set widget attributes for transparency
        self.view1.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.view1.setStyleSheet("background: transparent")
        self.view2.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.view2.setStyleSheet("background: transparent")

        self.view1.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.view1.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.view2.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.view2.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

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

    def refresh(self):
        # Create the QGraphicsScenes for each view
        self.scene1 = QGraphicsScene()
        self.scene2 = QGraphicsScene()

        # Set the scenes for the views
        self.view1.setScene(self.scene1)
        self.view2.setScene(self.scene2)

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
        if event.buttons() == Qt.MouseButton.LeftButton:
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


class LegendContainer(QGraphicsRectItem):
    def __init__(self, width, height, radius, parent=None):
        super().__init__(parent)
        self.setRect(0, 0, width, height)
        self.radius = radius

    def paint(self, painter, option, widget=None):
        painter.setBrush(QColor(255, 255, 255, 128))  # Semi-transparent white
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(self.rect(), self.radius, self.radius)


class PhotoViewer(QGraphicsView):
    photoClicked = pyqtSignal(QPoint)
    endDrawing_brush_meas = pyqtSignal(int)
    endDrawing_rect_meas = pyqtSignal(QGraphicsRectItem)
    endDrawing_point_meas = pyqtSignal(QPointF)
    endDrawing_line_meas = pyqtSignal(QGraphicsLineItem)

    def __init__(self, parent):
        super(PhotoViewer, self).__init__(parent)
        self._zoom = 0
        self._empty = True
        self._scene = QGraphicsScene(self)
        self._photo = QGraphicsPixmapItem()
        self._scene.addItem(self._photo)
        self.setScene(self._scene)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.setBackgroundBrush(QBrush(Qt.GlobalColor.transparent))
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)

        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # Set widget attributes for transparency
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setStyleSheet("background: transparent")

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
        self.meas_color = QColor(255, 0, 0, 255)
        self.pen_meas = QPen()
        # self.pen.setStyle(Qt.DashDotLine)
        self.pen_meas.setWidth(2)
        self.pen_meas.setColor(self.meas_color)
        self.pen_meas.setCapStyle(Qt.PenCapStyle.RoundCap)
        self.pen_meas.setJoinStyle(Qt.PenJoinStyle.RoundJoin)

        self.legendContainer = QLabel(self)
        self.legendLabel = QLabel(self)
        self.tickLabels = []  # List to store tick labels
        self.has_legend = False

        self.categories = None
        self.images = None
        self.active_image = None

    def setupLegendLabel(self, img_object, legend_type="bar"):
        # Clear existing legend and tick labels
        self.clearLegend()

        if legend_type == "bar":
            self._setupBarLegend(img_object)
        elif legend_type == "colorbar":
            self._setupMatplotlibLegend(img_object)
        else:
            print(f"Unknown legend_type: {legend_type}")

    def _setupBarLegend(self, img_object):
        if img_object.colormap in tt.LIST_CUSTOM_NAMES:
            all_cmaps = tt.get_all_custom_cmaps(256)
            custom_cmap = all_cmaps[img_object.colormap]
        else:
            custom_cmap = cm.get_cmap(img_object.colormap, 256)

        legendPixmap = self.createLegendPixmap(custom_cmap, img_object.n_colors)
        self.legendLabel = QLabel(self)
        self.legendLabel.setPixmap(legendPixmap)
        self.legendLabel.move(50, 50)
        self.legendLabel.show()

        tick_interval = legendPixmap.height() / 4
        for i, temp in enumerate(self.generateTicks(img_object.tmin_shown, img_object.tmax_shown, 5)):
            tick_label = QLabel(f"{temp:.2f}°C", self)
            tick_label_pos_y = 45 + (4 - i) * tick_interval
            tick_label_pos_x = 50 + legendPixmap.width() + 10
            tick_label.move(int(tick_label_pos_x), int(tick_label_pos_y))
            tick_label.setFont(QFont("Calibri", 10, QFont.Weight.Bold))
            tick_label.setStyleSheet("background-color: rgba(255, 255, 255, 128);")
            tick_label.show()
            self.tickLabels.append(tick_label)

    def _setupMatplotlibLegend(self, img_object):
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib import cm
        from io import BytesIO

        tmin = img_object.tmin_shown
        tmax = img_object.tmax_shown
        n_colors = img_object.n_colors
        colormap = img_object.colormap

        # Get the colormap
        if colormap in tt.LIST_CUSTOM_NAMES:
            all_cmaps = tt.get_all_custom_cmaps(n_colors)
            custom_cmap = all_cmaps[colormap]
        else:
            custom_cmap = cm.get_cmap(colormap, n_colors)

        if img_object.user_lim_col_high != 'c':
            custom_cmap.set_over(img_object.user_lim_col_high)
        if img_object.user_lim_col_low != 'c':
            custom_cmap.set_under(img_object.user_lim_col_low)

        # Dummy gradient data
        data = np.linspace(tmin, tmax, 100).reshape(10, 10)

        fig, ax = plt.subplots()
        im = ax.imshow(data, cmap=custom_cmap, vmin=tmin, vmax=tmax)

        ax.axis("off")  # Hide axes without removing

        ticks = np.linspace(tmin, tmax, 5)
        cbar = fig.colorbar(im, ticks=ticks, extend='both')
        cbar.ax.tick_params(labelsize=8)

        # Add a semi-transparent rounded rectangle background
        fig.patch.set_facecolor((1, 1, 1, 0.5))

        ax.remove()

        # Save to memory buffer
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)

        image = QImage.fromData(buf.read())
        pixmap = QPixmap.fromImage(image)

        self.legendLabel = QLabel(self)
        self.legendLabel.setPixmap(pixmap)
        self.legendLabel.show()

    def createLegendPixmap(self, colormap, num_colors):
        height = 300
        width = 15
        pixmap = QPixmap(width, height)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)

        for i in range(num_colors):
            temp_fraction = i / (num_colors - 1)
            rgba = colormap(temp_fraction)
            color = QColor.fromRgbF(*rgba)
            painter.setPen(color)
            painter.setBrush(color)
            y = height - (i * (height / num_colors)) - (height / num_colors)
            painter.drawRect(0, int(y), int(width), int(height / num_colors))

        painter.end()
        return pixmap

    def generateTicks(self, min_temp, max_temp, num_ticks):
        temp_range = max_temp - min_temp
        return [min_temp + i * (temp_range / (num_ticks - 1)) for i in range(num_ticks)]

    def setupLegendContainer(self):
        self.has_legend = True
        containerWidth = 110
        containerHeight = 350
        radius = 7

        # Create a QPixmap with rounded corners
        pixmap = QPixmap(containerWidth, containerHeight)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(QColor(120, 120, 120, 75))
        painter.setPen(Qt.PenStyle.NoPen)
        rectPath = QPainterPath()
        rectPath.addRoundedRect(QRectF(0, 0, containerWidth, containerHeight), radius, radius)
        painter.drawPath(rectPath)
        painter.end()

        self.legendContainer.setPixmap(pixmap)
        self.legendContainer.setFixedSize(pixmap.size())

        self.legendContainer.move(10, 10)  # Adjust position

        print(f"Container Size: {self.legendContainer.width()}x{self.legendContainer.height()}")
        print(f"Container Position: {self.legendContainer.pos()}")

    def toggleLegendVisibility(self):
        # Toggle the visibility of the legend and tick labels
        isVisible = not self.legendLabel.isVisible()
        self.legendLabel.setVisible(isVisible)
        # self.legendContainer.setVisible(isVisible)
        for label in self.tickLabels:
            label.setVisible(isVisible)

    def clearLegend(self):
        if self.legendLabel:
            self.legendLabel.hide()
            self.legendLabel.deleteLater()
            self.legendLabel = None

        if self.legendContainer:
            self.legendContainer.hide()
            self.legendContainer.deleteLater()
            self.legendContainer = None

            # Hide and remove all tick labels
        for tick_label in self.tickLabels:
            if tick_label:
                tick_label.hide()
                tick_label.deleteLater()

        self.tickLabels.clear()

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

    def clean_complete(self):
        self.clean_scene()

        self._scene = QGraphicsScene(self)
        self._photo = QGraphicsPixmapItem()
        self._scene.addItem(self._photo)
        self.setScene(self._scene)

    def draw_all_meas(self, meas_items):
        for item in meas_items:
            if isinstance(item, QGraphicsLineItem) or isinstance(item, QGraphicsRectItem) or isinstance(item,
                                                                                                        QGraphicsEllipseItem):
                item.setPen(self.pen_meas)
            self._scene.addItem(item)

    def setPhoto(self, pixmap=None):
        # self._zoom = 0
        if pixmap and not pixmap.isNull():
            self._empty = False
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self._photo.setPixmap(pixmap)
        else:
            self._empty = True
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self._photo.setPixmap(QPixmap())
        # self.fitInView()

    def change_to_brush_cursor(self):
        self.setCursor(self.brush_cur)

    def toggleDragMode(self):
        if not self.rect_meas or self.point_meas or self.painting:
            if self.dragMode() == QGraphicsView.DragMode.ScrollHandDrag:
                self.setDragMode(QGraphicsView.DragMode.NoDrag)
            elif not self._photo.pixmap().isNull():
                self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        else:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)

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
