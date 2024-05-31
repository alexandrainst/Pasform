#%% imports
import copy

import jakteristics as jakt
import numpy as np
import open3d as o3d
import point_cloud_utils as pcu
import sklearn.metrics


def copy_and_transform_pc(pcd, tr):
    pcd2 = copy.deepcopy(pcd)
    pcd2.transform(tr)
    return pcd2

def voxel_inclusion(source, target, voxel_size):
    """
    measures points in target that are inside voxelized source with voxelsize voxel_size
    #TODO this measure is not very sensible https://www.open3d.org/docs/latest/tutorial/Advanced/voxelization.html and should be fixed to something else
    """
    
    # measure points in source voxels
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(source,
                                                                voxel_size=voxel_size)
    queries = np.asarray(target.points)
    output = voxel_grid.check_if_included(o3d.utility.Vector3dVector(queries))
    inclusion_ratio = sum(output)/len(output)
    
    return inclusion_ratio

# measure chamfer distance (https://www.fwilliams.info/point-cloud-utils/sections/shape_metrics/) with Point Cloud Utils lib
def chamfer_distance_pcu(p1, p2):

    # Compute the chamfer distance between p1 and p2
    cd = pcu.chamfer_distance(np.asarray(p1.points), np.asarray(p2.points))

    return cd


def pdist(X1, X2):
    dist_mat=sklearn.metrics.pairwise_distances(X1, X2)
    min12 = np.min(dist_mat, axis=0) # minimum distances from feature set 1 to 2
    min21 = np.min(dist_mat, axis=1) # and vice versa..
    #dist = np.std(min12) + np.std(min21)
    dist = np.mean(min12) + np.mean(min21)
    return dist


def get_features(pcd, radius_feature, feat_type='fpfh'):
    
    if feat_type == 'fpfh':
        # print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
        feats = o3d.pipelines.registration.compute_fpfh_feature(
            pcd,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

    elif feat_type == 'jakteristics':
        
        # featnames = ['planarity', 'linearity', 'sphericity', 'number_of_neighbors']
        featnames = jakt.FEATURE_NAMES
        feats = jakt.compute_features(np.asarray(pcd.points), search_radius=radius_feature, feature_names=featnames)

    return feats




