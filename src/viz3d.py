import copy

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from src.registration_utils import compute_inliers


def save_image_of_3d(pc_target, save, pc_source=None, transform_source=None):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1000,height=1000)
    tmp_pc_target = copy.deepcopy(pc_target)
    tmp_pc_target.paint_uniform_color([0, 0.651, 0.929])
    vis.add_geometry(tmp_pc_target)
    vis.update_geometry(tmp_pc_target)
    if pc_source is not None:
        if transform_source is None:
            tmp_source_pc = pc_source
        else:
            tmp_source_pc = copy.deepcopy(pc_source)
            tmp_source_pc.transform(transform_source)
            tmp_source_pc.paint_uniform_color([1, 0.706, 0])

        vis.add_geometry(tmp_source_pc)
        vis.update_geometry(tmp_source_pc)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(save)
    vis.destroy_window()
    return


def colorize_segmented_pc(pc, labels):
    colors = plt.get_cmap("tab20")(labels / labels.max())
    pc.colors = o3d.utility.Vector3dVector(colors[:, :3])


def draw_registration_result(source, target, transformation=np.identity(4), winname='', add_color=True, threshold=0.1):
    """
    Interactive 3D visualization

    1,2 toggles visibility of source, target
    3 toggles inlier points / default colorscheme
    https://www.open3d.org/html/tutorial/Advanced/customized_visualization.html
    https://www.open3d.org/html/python_api/open3d.visualization.Visualizer.html#open3d.visualization.Visualizer

    """
    global show_source, show_target, show_inliers
    show_source = True
    show_target = True
    show_inliers = False
    def quit_visualize(vis):
        vis.destroy_window()
        return False

    def toggle_source(vis):
        global show_source
        if show_source:
            vis.remove_geometry(source_temp,False)
        else:
            vis.add_geometry(source_temp,False)
        show_source = not show_source
        return True

    def toggle_target(vis):
        global show_target
        if show_target:
            vis.remove_geometry(target_temp,False)
        else:
            vis.add_geometry(target_temp,False)
        show_target = not show_target
        return True

    def toggle_inliers(vis):
        global show_inliers
        if show_inliers:
            source_temp.colors = o3d.utility.Vector3dVector(source_temp_org_colors)
            vis.update_geometry(source_temp)
        else:
            source_temp.colors = o3d.utility.Vector3dVector(inlier_colors)
            vis.update_geometry(source_temp)
        show_inliers = not show_inliers
        return True


    key_to_callback = {}
    key_to_callback[ord("q")] = quit_visualize
    key_to_callback[ord("1")] = toggle_source
    key_to_callback[ord("2")] = toggle_target
    key_to_callback[ord("3")] = toggle_inliers



    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)

    if add_color:
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp_org_colors = copy.deepcopy(np.asarray(source_temp.colors))
    target_temp_org_colors = np.asarray(target_temp.colors)

    source_temp.transform(transformation)
    source_inliers = compute_inliers(source_temp, target_temp, threshold)
    inlier_colors = np.asarray([[1, 0.706, 0]]*len(source_inliers))
    inlier_colors[source_inliers] = [1, 0, 0]
    o3d.visualization.draw_geometries_with_key_callbacks([source_temp, target_temp],
                                                         key_to_callback,
                                                         window_name=winname,
                                                         width=1000, height=600)

    # o3d.visualization.draw_geometries_with_key_callbacks([source_temp, target_temp],
    #                                                      key_to_callback,
    #                                                      window_name=winname,
    #                                                      width=1000, height=600)
    return


def draw_single_pc(source, transformation=np.identity(4), winname=''):

    def quit_visualize(vis):
        vis.destroy_window()
        return False

    key_to_callback = {}
    key_to_callback[ord("q")] = quit_visualize

    source_temp = copy.deepcopy(source)
    source_temp.paint_uniform_color([1, 0.706, 0])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries_with_key_callbacks([source_temp],
                                                         key_to_callback,
                                                         window_name=winname,
                                                         width=1000, height=600)


# labels - numpy array, same length as source, and values 0 - K-1 (K clusters)
def draw_segmented_pc(source, labels, transformation=np.identity(4), winname=''):

    colors = plt.get_cmap("tab20")(labels / labels.max())

    def quit_visualize(vis):
        vis.destroy_window()
        return False

    key_to_callback = {}
    key_to_callback[ord("q")] = quit_visualize

    source_temp = copy.deepcopy(source)
    source_temp.colors = o3d.Vector3dVector(colors[:, :3])

    source_temp.transform(transformation)
    o3d.visualization.draw_geometries_with_key_callbacks([source_temp],
                                                         key_to_callback,
                                                         window_name=winname,
                                                         width=1000, height=600)

