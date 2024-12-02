# %% imports
import copy

import numpy as np
import open3d as o3d
import point_cloud_utils as pcu


def copy_and_transform_pc(pcd, tr):
    """
    Creates a copy of a point cloud and performs a transformation on the copy.
    """
    pcd2 = copy.deepcopy(pcd)
    pcd2.transform(tr)
    return pcd2


def voxel_inclusion(source, target, voxel_size):
    """
    measures points in target that are inside voxelized source with voxelsize voxel_size
    """

    # measure points in source voxels
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(source, voxel_size=voxel_size)
    queries = np.asarray(target.points)
    output = voxel_grid.check_if_included(o3d.utility.Vector3dVector(queries))
    inclusion_ratio = sum(output) / len(output)
    return inclusion_ratio


def chamfer_distance_pcu(p1, p2):
    """
    Computes the chamfer distance between p1 and p2
    """
    cd = pcu.chamfer_distance(np.asarray(p1.points), np.asarray(p2.points))
    return cd


def get_features(pcd, radius_feature, feat_type='fpfh'):
    """
    Get features for a point cloud.
    Currently, the method only supports the fpfh method.
    """

    if feat_type == 'fpfh':
        feats = o3d.pipelines.registration.compute_fpfh_feature(
            pcd,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    else:
        raise NotImplementedError("Feature type not implemented")

    return feats




