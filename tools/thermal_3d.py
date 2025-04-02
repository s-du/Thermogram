import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from matplotlib import cm

import numpy as np
from pathlib import Path

#custom libraries
import resources as res
from tools import thermal_tools as tt

OUT_LIM_MATPLOT = ['k', 'w', 'r']
POST_PROCESS = ['none', 'smooth', 'sharpen', 'sharpen strong', 'edge (simple)', 'edge (from rgb)']
COLORMAPS = ['coolwarm','Artic', 'Iron', 'Rainbow', 'Greys_r', 'Greys', 'plasma', 'inferno', 'jet',
                              'Spectral_r', 'cividis', 'viridis', 'gnuplot2']
VIEWS = ['th. undistorted', 'RGB crop']

SDK_PATH = Path(res.find('dji/dji_irp.exe'))

img_path = Path(res.find('img/M2EA_IR.JPG'))


class Custom3dView:
    def __init__(self, data, colormap, col_high, col_low, n_colors, tmin_shown, tmax_shown):
        self.colormap = colormap
        self.col_high = col_high
        self.col_low = col_low
        self.n_colors = n_colors
        self.tmin_shown = tmin_shown
        self.tmax_shown = tmax_shown

        app = gui.Application.instance
        self.window = app.create_window("Open3D - Infrared voxels", 1800, 900)
        self.window.set_on_layout(self._on_layout)
        self.widget3d = gui.SceneWidget()
        self.window.add_child(self.widget3d)

        self.info = gui.Label("")
        self.info.visible = False
        self.window.add_child(self.info)

        self.widget3d.scene = rendering.Open3DScene(self.window.renderer)
        self.widget3d.scene.set_background([1, 1, 1, 1])
        self.viewopt = self.widget3d.scene.view
        # self.viewopt.set_ambient_occlusion(True, ssct_enabled=True)

        self.widget3d.enable_scene_caching(True)
        self.widget3d.scene.show_axes(True)
        self.widget3d.scene.scene.set_sun_light(
            [0.45, 0.45, -1],  # direction
            [1, 1, 1],  # color
            100000)  # intensity
        self.widget3d.scene.scene.enable_sun_light(True)
        self.widget3d.scene.scene.enable_indirect_light(True)
        self.widget3d.set_on_sun_direction_changed(self._on_sun_dir)

        self.mat = rendering.MaterialRecord()
        self.mat.shader = "defaultLit"
        self.mat.point_size = 3 * self.window.scaling

        self.mat_maxi = rendering.MaterialRecord()
        self.mat_maxi.shader = "defaultUnlit"
        self.mat_maxi.point_size = 15 * self.window.scaling

        # layout
        self.create_layout()

        # load image
        self.load(data)

        # default autorescale on
        self.auto_rescale = True


    def create_layout(self):
        # LAYOUT GUI ELEMENTS
        em = self.window.theme.font_size
        self.layout = gui.Vert(0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))

        view_ctrls = gui.CollapsableVert("View controls", 0.25 * em,
                                         gui.Margins(em, 0, 0, 0))

        # add button to reset camera
        camera_but = gui.Button('Reset view')
        camera_but.set_on_clicked(self._on_reset_camera)

        filter_but = gui.Button('Reset temp. filter')
        filter_but.set_on_clicked(self._on_reset_filter)

        # add checkbox to rescale the colormap
        rescale_check = gui.Checkbox('Auto rescale colormap')
        # set checked
        rescale_check.checked = True
        rescale_check.set_on_checked(self._on_autoscale)

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

        combo_voxel = gui.Horiz(0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))
        combo_voxel.add_child(gui.Label("Size of voxels"))
        combo_voxel.add_child(self._voxel)

        # add editor for max temp
        self.edit_max = gui.Slider(gui.Slider.DOUBLE)
        self.edit_min = gui.Slider(gui.Slider.DOUBLE)

        numlayout_max = gui.Horiz()
        numlayout_max.add_child(gui.Label("Max. temp.:"))
        numlayout_max.add_child(self.edit_max)

        numlayout_min = gui.Horiz()
        numlayout_min.add_child(gui.Label("Min. temp.:"))
        numlayout_min.add_child(self.edit_min)

        # layout
        view_ctrls.add_child(combo_light)
        view_ctrls.add_child(combo_voxel)
        view_ctrls.add_child(numlayout_min)
        view_ctrls.add_child(numlayout_max)
        view_ctrls.add_child(rescale_check)
        view_ctrls.add_child(filter_but)

        view_ctrls.add_child(camera_but)

        self.layout.add_child(view_ctrls)
        self.window.add_child(self.layout)

        self.widget3d.set_on_mouse(self._on_mouse_widget3d)
        self.window.set_needs_layout()


    def choose_material(self, is_enabled):
        pass

    def _on_autoscale(self, status):
        if status:
            self.auto_rescale = True
        else:
            self.auto_rescale = False


    def _on_change_colormap(self):
        pass

    def _on_reset_filter(self):
        self.voxel_grids = []
        for size in self.voxel_size:
            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(self.pc_ir,voxel_size=size)
            self.voxel_grids.append(voxel_grid)

        # show one geometry
        self.widget3d.scene.clear_geometry()
        self.widget3d.scene.add_geometry(f"PC {self.current_index}", self.voxel_grids[self.current_index], self.mat)
        self.widget3d.force_redraw()

    def load(self, data):
        self.data = data
        self.pc_ir, self.points, self.tmax, self.tmin, loc_tmax, loc_tmin, self.factor = surface_from_image(self.data, self.colormap, self.n_colors, self.col_low, self.col_high, self.tmin_shown, self.tmax_shown)

        # store basic properties
        bound = self.pc_ir.get_axis_aligned_bounding_box()
        center = bound.get_center()
        dim = bound.get_extent()

        dim_x = dim[0]
        dim_y = dim[1]
        dim_z = dim[2]

        self.pt1 = [center[0] - dim_x / 2, center[1] - dim_y / 2, center[2] - dim_z / 2]
        self.pt2 = [center[0] + dim_x / 2, center[1] + dim_y / 2, center[2] + dim_z / 2]

        self.min_value = center[2] - dim_z / 2
        self.max_value = center[2] + dim_z / 2

        # create all voxel grids
        self.voxel_grids = []
        self.voxel_size = [2, 5, 10, 20]

        for size in self.voxel_size:
            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(self.pc_ir,
                                                                        voxel_size=size)
            self.voxel_grids.append(voxel_grid)

        # show one geometry
        self.widget3d.scene.add_geometry('PC 0', self.voxel_grids[0], self.mat)
        self.current_index = 0


        self.widget3d.scene.show_geometry("Point Cloud IR 0", True)
        self.widget3d.force_redraw()

        self.voxel_name = ["2", "5", "10", "20"]
        self._voxel.add_item(self.voxel_name[0])
        self._voxel.add_item(self.voxel_name[1])
        self._voxel.add_item(self.voxel_name[2])
        self._voxel.add_item(self.voxel_name[3])
        self._voxel.set_on_selection_changed(self._on_voxel)

        # add labels for min and max values
        loc_tmin = np.append(loc_tmin, self.tmin * self.factor)
        loc_tmax = np.append(loc_tmax, self.tmax * self.factor)
        text_max = 'Temp. max.:' + str(round(self.tmax, 2)) + '°C'
        text_min = 'Temp. min.:' + str(round(self.tmin, 2)) + '°C'
        lab_tmin = self.widget3d.add_3d_label(loc_tmin, text_min)
        lab_tmax = self.widget3d.add_3d_label(loc_tmax, text_max)
        lab_color_red = gui.Color(1, 0, 0)
        lab_color_blue = gui.Color(0, 0, 1)

        lab_tmin.color = lab_color_blue
        lab_tmax.color = lab_color_red

        # adapt temp edit limits
        self.edit_max.set_limits(self.tmin, self.tmax)
        self.edit_max.set_on_value_changed(self._on_edit_max)
        self.edit_max.double_value = self.tmax

        self.edit_min.set_limits(self.tmin, self.tmax)
        self.edit_min.set_on_value_changed(self._on_edit_min)
        self.edit_min.double_value = self.tmin

        # add points
        pcd_maxi = o3d.geometry.PointCloud()
        array = np.array([loc_tmin, loc_tmax])
        pcd_maxi.points = o3d.utility.Vector3dVector(array)
        color_array = np.array([[0, 0, 1], [1, 0, 0]])
        pcd_maxi.colors = o3d.utility.Vector3dVector(color_array)
        self.widget3d.scene.add_geometry('Max/Min', pcd_maxi, self.mat_maxi)

        self._on_reset_camera()

        # add image label

    def _on_edit_min_new(self, value):
        self.min_value = value
        new_points = filter_point_cloud_by_intensity(self.points, value*self.factor, self.max_value*self.factor)
        self.voxel_grids = []

        # create new point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(new_points)

        # create new colors
        color = colorize_pc_height(self.points, self.colormap, self.col_high, self.col_low, self.n_colors)
        pcd.colors = o3d.utility.Vector3dVector(color)

        for size in self.voxel_size:
            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=size)
            self.voxel_grids.append(voxel_grid)

        # show one geometry
        self.widget3d.scene.clear_geometry()
        self.widget3d.scene.add_geometry(f"PC {self.current_index}", self.voxel_grids[self.current_index], self.mat)
        self.widget3d.force_redraw()

        # set max values
        self.edit_max.set_limits(self.min_value, self.tmax)


    def _on_edit_min(self, value):
        self.min_value = value
        self.voxel_grids = []
        # crop point cloud
        pt1 = self.pt1
        pt1[2] = value*self.factor
        pt2 = self.pt2
        pt2[2] = self.max_value*self.factor
        np_points = [pt1, pt2]

        points = o3d.utility.Vector3dVector(np_points)
        crop_box = o3d.geometry.AxisAlignedBoundingBox
        crop_box = crop_box.create_from_points(points)

        point_cloud_crop = self.pc_ir.crop(crop_box)
        for size in self.voxel_size:
            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(point_cloud_crop,voxel_size=size)
            self.voxel_grids.append(voxel_grid)

        # show one geometry
        self.widget3d.scene.clear_geometry()
        self.widget3d.scene.add_geometry(f"PC {self.current_index}", self.voxel_grids[self.current_index], self.mat)
        self.widget3d.force_redraw()

        # set max values
        self.edit_max.set_limits(self.min_value, self.tmax)

    def _on_edit_max(self, value):
        self.max_value = value
        self.voxel_grids = []

        # crop point cloud
        pt1 = self.pt1
        pt1[2] = self.min_value*self.factor
        pt2 = self.pt2
        pt2[2] = value*self.factor
        np_points = [pt1, pt2]
        points = o3d.utility.Vector3dVector(np_points)
        print(pt1,pt2)

        crop_box = o3d.geometry.AxisAlignedBoundingBox
        crop_box = crop_box.create_from_points(points)

        point_cloud_crop = self.pc_ir.crop(crop_box)

        for size in self.voxel_size:
            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(point_cloud_crop,voxel_size=size)
            self.voxel_grids.append(voxel_grid)

        # show one geometry
        self.widget3d.scene.clear_geometry()
        self.widget3d.scene.add_geometry(f"PC {self.current_index}", self.voxel_grids[self.current_index], self.mat)
        self.widget3d.force_redraw()

        # set max values
        self.edit_min.set_limits(self.tmin, self.max_value)

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

    def _on_reset_camera(self):
        # adapt camera
        bounds = self.widget3d.scene.bounding_box
        center = bounds.get_center()
        self.widget3d.setup_camera(30, bounds, center)
        camera = self.widget3d.scene.camera
        self.widget3d.look_at(center, center + [0, 0, 1200], [0, -1, 0])


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

def run_viz_app(data, colormap, high, low, n, tmin_shown, tmax_shown):
    app_vis = gui.Application.instance
    app_vis.initialize()

    viz = Custom3dView(data, colormap, high, low, n, tmin_shown, tmax_shown)
    app_vis.run()


def replace_pixels_between_thresholds(image, lower_threshold, upper_threshold, new_value):
    # Create a copy of the original image to avoid modifying it directly
    modified_image = np.copy(image)

    # Find the indices of pixels that satisfy the condition (lower_threshold < pixel < upper_threshold)
    between_threshold_indices = np.logical_and(image > lower_threshold, image < upper_threshold)

    # Replace the pixels between the thresholds with the new value
    modified_image[between_threshold_indices] = new_value

    return modified_image


def filter_point_cloud_by_intensity(point_cloud, lower_threshold, upper_threshold):
    # Extract the intensity values from the point cloud
    print('ok')
    intensity_values = point_cloud[:, 2]  # Assuming the intensity is in the fourth column (index 3)
    print(intensity_values)

    # Find the indices of points with intensity within the desired range
    valid_indices = np.where(np.logical_and(intensity_values >= lower_threshold, intensity_values <= upper_threshold))[0]

    print('ok')
    # Create the filtered point cloud
    filtered_point_cloud = point_cloud[valid_indices]

    return filtered_point_cloud


def colorize_pc_height(pc, colormap, col_high, col_low, n_colors):
    if colormap in tt.LIST_CUSTOM_NAMES:
        all_cmaps = tt.get_all_custom_cmaps(n_colors)
        custom_cmap = all_cmaps[colormap]
    else:
        custom_cmap = cm.get_cmap(colormap, n_colors)

    custom_cmap.set_over(col_high)
    custom_cmap.set_under(col_low)

    pc_height = pc[:,2]

    # get extreme values from data
    tmax = np.amax(pc_height)
    tmin = np.amin(pc_height)
    """
    indices_max = np.where(pc_height == tmax)
    indices_min = np.where(pc_height == tmin)

    # Check if there are any occurrences of 'a'
    if len(indices_max[0]) > 0:
        # Get the coordinates of the first occurrence
        loc_tmax = np.array([-indices_max[1][0], indices_max[0][0]])

    if len(indices_min[0]) > 0:
        # Get the coordinates of the first occurrence
        loc_tmin = np.array([-indices_min[1][0], indices_min[0][0]])
    """

    # normalized data
    thermal_normalized = (pc_height - tmin) / (tmax - tmin)
    thermal_cmap = custom_cmap(thermal_normalized)

    # thermal_cmap = np.uint8(thermal_cmap)
    color_array = thermal_cmap[:, [0, 1, 2]]
    print(color_array.shape)
    print(color_array)

    return color_array


def surface_from_image(data, colormap, n_colors, col_low, col_high, tmin_shown, tmax_shown):
    if colormap in tt.LIST_CUSTOM_NAMES:
        all_cmaps = tt.get_all_custom_cmaps(n_colors)
        custom_cmap = all_cmaps[colormap]
    else:
        custom_cmap = cm.get_cmap(colormap, n_colors)

    if col_high != 'c':
        custom_cmap.set_over(col_high)
    if col_low != 'c':
        custom_cmap.set_under(col_low)

    # get extreme values from data
    tmax = np.amax(data)
    tmin = np.amin(data)
    indices_max = np.where(data == tmax)
    indices_min = np.where(data == tmin)

    # Check if there are any occurrences of 'a'
    if len(indices_max[0]) > 0:
        # Get the coordinates of the first occurrence
        loc_tmax = np.array([-indices_max[1][0], indices_max[0][0]])

    if len(indices_min[0]) > 0:
        # Get the coordinates of the first occurrence
        loc_tmin = np.array([-indices_min[1][0], indices_min[0][0]])

    # normalized data
    thermal_normalized = (data - tmin_shown) / (tmax_shown - tmin_shown)

    thermal_cmap = custom_cmap(thermal_normalized)
    # thermal_cmap = np.uint8(thermal_cmap)
    color_array = thermal_cmap[:, :, [0, 1, 2]]

    # color_array = np.transpose(color_array, (1, 0, 2))
    print(color_array.shape)

    height, width = data.shape

    # Generate the x and y coordinates
    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))

    # Flatten the arrays
    x = -x_coords.flatten()
    y = y_coords.flatten()
    z = data.flatten()

    # Create the point cloud using the flattened arrays
    # compute range of temp
    range_temp = tmax-tmin
    # compute how the range scales compared to x/y
    factor = width/range_temp/3
    points = np.column_stack((x, y, z*factor))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # get colors
    color_array = color_array.reshape(width*height, 3)
    pcd.colors = o3d.utility.Vector3dVector(color_array)

    return pcd, points, tmax, tmin, loc_tmax, loc_tmin, factor


