import numpy as np
import open3d as o3d
import open3d.pipelines.registration as treg


def ICP_open3d(source, target, init_trans, max_correspondence_distance):
    """
    Local registration method requires an initial transform that roughly aligns the two pointclouds
    """
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    estimation = treg.TransformationEstimationPointToPoint()
    criteria = treg.ICPConvergenceCriteria(relative_fitness=0.000001,
                                           relative_rmse=0.000001,
                                           max_iteration=500)
    
    # Search distance for Nearest Neighbour Search [Hybrid-Search is used].
    result = treg.registration_icp(source, target, max_correspondence_distance,
                                init_trans, estimation, criteria)
    return result
 

def RANSAC_open3d(source_normal, target_normal, source_features, target_features, distance_threshold):
    """
    RANSAC is a global registration method (unlike ICP which is a local registration method).
    Typically, RANSAC is used to give a starting estimate for ICP if none is known.
    See https://www.open3d.org/docs/latest/tutorial/pipelines/global_registration.html for more info
    """
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint(False)
    var1 = [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9), o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)]
    criteria = o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(source_normal, target_normal, source_features, target_features, True, distance_threshold, p2p, 3, var1, criteria)
    return result


def fast_RANSAC_open3d(source_down, target_down, source_fpfh,target_fpfh, voxel_size):
    distance_threshold = voxel_size * 0.5
    var1 = target_fpfh,o3d.pipelines.registration.FastGlobalRegistrationOption(maximum_correspondence_distance=distance_threshold)
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(source_down, target_down, source_fpfh, var1)
    return result    



def RANSAC_open3d_for_video(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    max_iterations = 1
    distance_threshold = voxel_size * 3
    p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint(False)
    var1 = [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)]
    criteria = o3d.pipelines.registration.RANSACConvergenceCriteria(max_iterations, 0.999)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(source_down, target_down, source_fpfh, target_fpfh, True,distance_threshold,p2p, 3, var1, criteria)
    return result


def boundingbox_intersect(p1t, p2, epsilon=1e-9):
    """
    Finds the intersection of the bounding boxes for two pointclouds
    Returns the intersecting bounding box.

    Epsilon was added since it would randomly cut off the actual corner points otherwise (intersecting a volume with itself would give 2 different pointclouds sometimes otherwise)
    """
    ## cropping - two way
    bb1 = p1t.get_axis_aligned_bounding_box()  # Poor - not do axis-aligned.. NOTE: try out better cropping..
    bb2 = p2.get_axis_aligned_bounding_box()  # Poor - not do axis-aligned.. NOTE: try out better cropping..

    tmpmin = np.vstack([bb1.min_bound, bb2.min_bound])  # tmp bb min var
    tmpmax = np.vstack([bb1.max_bound, bb2.max_bound])
    bbmin = np.max(tmpmin, axis=0)
    bbmax = np.min(tmpmax, axis=0)

    bb1.min_bound = bbmin - epsilon
    bb1.max_bound = bbmax + epsilon

    return bb1


def extract_registration_patches_results(registration_results_patches):
    """
    Extracts a list of registration results into lists of their results.
    """
    n = len(registration_results_patches)
    fitnesses = np.empty(n)
    inlier_rmses = np.empty(n)
    n_elements = np.empty(n)
    for i, result in enumerate(registration_results_patches):
        fitnesses[i] = result.fitness
        inlier_rmses[i] = result.inlier_rmse
        n_elements[i] = len(result.correspondence_set)
    return fitnesses, inlier_rmses, n_elements

def compute_inliers(source,target,threshold):
    """
    Returns a mask of which points are inliers.
    """
    dist = np.asarray(source.compute_point_cloud_distance(target))
    inliers = dist < threshold
    return inliers

