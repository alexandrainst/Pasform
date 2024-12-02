import copy
import glob
import os
from os.path import exists
from pathlib import Path
from time import perf_counter

import numpy as np
import open3d as o3d
from seaborn import color_palette
from sklearn.cluster import KMeans
from tqdm import tqdm

from src.feature_methods import get_features, copy_and_transform_pc
from src.registration_utils import RANSAC_open3d, ICP_open3d, boundingbox_intersect
from src.viz3d import draw_registration_result, save_image_of_3d


def prepare_base_set(path,voxel_size,path_low_res,voxel_size_low_res,max_files=99,output_folder=None, draw_samples=False, indices=None, order_by_boundingbox=True):
    """
    """
    files = glob.glob(os.path.join(path,'') + "*.pcd")
    files.sort()
    files = files[:max_files]
    if indices is not None:
        files = [files[i] for i in indices]

    assert voxel_size_low_res > voxel_size, "Low res voxel size must be larger than normal voxel size if used."
    files_low_res = glob.glob(os.path.join(path_low_res,'') + "*.pcd")
    files_low_res.sort()
    files_low_res = files_low_res[:max_files]
    if indices is not None:
        files_low_res = [files_low_res[i] for i in indices]

    for file, file_low_res in zip(files,files_low_res):
        filename, filename_low_res = os.path.basename(file), os.path.basename(file_low_res)
        assert filename == filename_low_res, f"{filename}, does not match the low resolution version: {file_low_res}. The folder with low resolution files should contain identical files names as the normal resolution one"
    names = [Path(file).stem for file in files]
    print(names)
    output_path_low_res = os.path.join(output_folder,'low_res')
    output_path_high_res = os.path.join(output_folder,'high_res')
    transformations_low_res, fits_low_res, inlier_rmse_low_res, cloud_sizes_low_res = compare_all_to_all(files_low_res,voxel_size_low_res,output_path_low_res,save_image=True, ids=names)
    transformations, fits, inlier_rmse, cloud_sizes = compare_all_to_all(files,voxel_size,output_path_high_res,init_transforms=transformations_low_res,save_image=True, ids=names)

    return names, fits, inlier_rmse, transformations

def compare_all_to_all(files, voxel_size, output_path, init_transforms=None, save_image=False, debug=False,overwrite=False, ids=None):
    os.makedirs(output_path, exist_ok=True)
    checkpoint_file = os.path.join(output_path,f"transformations.npz")
    nfiles = len(files)
    if ids is None:
        ids = [str(i) for i in range(nfiles)]

    if exists(checkpoint_file) and not overwrite:
        data = np.load(checkpoint_file, allow_pickle=True)
        transformations = data['transformations']
        fits = data['fits']
        inlier_rmses = data['inlier_rmses']
        computed = data['computed']
        id_matrix = data['id_matrix']
        cloud_sizes = data['cloud_sizes']
    else:
        transformations = np.empty((nfiles, nfiles), dtype=object)
        fits = np.ones((nfiles, nfiles))
        inlier_rmses = np.zeros((nfiles, nfiles))
        computed = np.zeros((nfiles,nfiles),dtype=bool)
        computed[np.arange(nfiles),np.arange(nfiles)] = True
        id_matrix = np.empty((nfiles,nfiles),dtype=object)
        cloud_sizes = np.ones((nfiles,nfiles))


    m = computed == False
    with tqdm(total=m.sum(), position=0, leave=True, desc=f'Compare all to all. Voxelsize={voxel_size}') as pbar:
        for i,file_target in enumerate(files):
            pc_target = o3d.io.read_point_cloud(file_target)
            p_target = np.asarray(pc_target.points).shape[0]
            for j,file_source in enumerate(files):
                if j == i or computed[i,j]:
                    if save_image:
                        output_file = os.path.join(output_path, f"{i}_{j}.png")
                        save_image_of_3d(pc_target, output_file)

                    continue
                if init_transforms is None:
                    init_transform = None
                else:
                    init_transform = init_transforms[i,j]

                pc_source = o3d.io.read_point_cloud(file_source)
                p_source = np.asarray(pc_source.points).shape[0]

                registration_result = pc_registration2(pc_source, pc_target, voxel_size, init_transform=init_transform,print_performance=debug)

                fits[i,j] = registration_result.fitness
                inlier_rmses[i,j] = registration_result.inlier_rmse
                transformations[i,j] = registration_result.transformation
                id_matrix[i,j] = ids[i] + "+" + ids[j]
                cloud_sizes[i,j] = p_source/p_target
                pbar.update()

                computed[i,j] = True
                np.savez(checkpoint_file, ids=ids, transformations=transformations, fits=fits, inlier_rmses=inlier_rmses, computed=computed, id_matrix=id_matrix, voxel_size=voxel_size, cloud_sizes=cloud_sizes)
                if save_image:
                    output_file = os.path.join(output_path,f"{i}_{j}.png")
                    save_image_of_3d(pc_target, output_file, pc_source=pc_source,transform_source=transformations[i,j])

    return transformations, fits, inlier_rmses, cloud_sizes

                # output_file = os.path.join(output_folder, f"global_reg_{i}_{j}.png")
                # save_image_of_3d(pc_target, output_file, pc_source=pc_source,transform_source=registration_result_global.transformation)
                # output_file = os.path.join(output_folder, f"local_reg_{i}_{j}.png")
                # save_image_of_3d(pc_target, output_file, pc_source=pc_source,transform_source=registration_result_local.transformation)


def pc_registration2(pc_source,pc_target,voxel_size, init_transform=None, local_refinement=True, print_performance=False):
    """
    Takes 2 point clouds and performs registration.
    If no init_transform is given, then a global transformation is performed in order to generate an init_transform
    If only_intersetion=True, then the point clouds are cropped to their intersecting bounding boxes
    Returns the registration_result object
    """
    t0 = perf_counter()
    radius_normal = voxel_size * 2
    feature_radius = voxel_size * 5
    distance_threshold = voxel_size * 3
    search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    t1 = perf_counter()

    pc_source.estimate_normals(search_param) #This estimates the normals and adds them directly to the object
    source_features = get_features(pc_source,feature_radius)

    pc_target.estimate_normals(search_param)
    target_features = get_features(pc_target,feature_radius)
    t2 = perf_counter()

    if init_transform is None:
        global_registration_result = RANSAC_open3d(pc_source, pc_target, source_features, target_features, distance_threshold)
        init_transform = global_registration_result.transformation
    else:
        assert local_refinement == True, "init_transform cannot be given without local_refinement activated (then there would be nothing to do)."
    t3 = perf_counter()

    if local_refinement:
        registration_result = ICP_open3d(pc_source, pc_target, init_transform, distance_threshold)
    else:
        registration_result = global_registration_result
    t4 = perf_counter()
    if print_performance:
        print(f"{t1-t0:2.2f}s, {t2-t1:2.2f}s, {t3-t2:2.2f}s, {t4-t3:2.2f}s")
    return registration_result




def pc_registration(pc_source,pc_target,voxel_size, init_transform=None, only_intersection=False):
    """
    Takes 2 point clouds and performs registration.
    If no init_transform is given, then a global transformation is performed in order to generate an init_transform
    If only_intersetion=True, then the point clouds are cropped to their intersecting bounding boxes
    Returns the registration_result object
    """
    radius_normal = voxel_size * 2
    feature_radius = voxel_size * 5
    distance_threshold = voxel_size * 3
    search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    bb_intersect = None

    pc_source.estimate_normals(search_param) #This estimates the normals and adds them directly to the object
    source_features = get_features(pc_source,feature_radius)

    pc_target.estimate_normals(search_param)
    target_features = get_features(pc_target,feature_radius)

    if init_transform is None:
        global_registration_result = RANSAC_open3d(pc_source, pc_target, source_features, target_features, distance_threshold)
        init_transform = global_registration_result.transformation

    if only_intersection:
        pc_source = copy_and_transform_pc(pc_source, init_transform)
        bb_intersect = boundingbox_intersect(pc_source, pc_target)
        pc_source = pc_source.crop(bb_intersect)
        pc_target = pc_target.crop(bb_intersect)
        pc_source.transform(np.linalg.pinv(init_transform)) # We need to transform the source back to origin again such that it works with the init_transform next

    registration_result = ICP_open3d(pc_source, pc_target, init_transform, distance_threshold)
    return registration_result, bb_intersect


def pc_registration_patches(pc_source,pc_target,voxel_size, init_transform=None, k=50, draw_samples=True):
    """
    Takes 2 point clouds and performs patchwise registration.
    If no init_transform is given, then a global transformation is performed in order to generate an init_transform
    k is the number of patches/clusters that the point cloud is split into
    Returns the registration_result object
    """
    radius_normal = voxel_size * 2
    feature_radius = voxel_size * 5
    distance_threshold = voxel_size * 3
    search_param = o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)

    pc_source.estimate_normals(search_param) #This estimates the normals and adds them directly to the object
    source_features = get_features(pc_source,feature_radius)

    pc_target.estimate_normals(search_param)
    target_features = get_features(pc_target,feature_radius)

    if init_transform is None:
        global_registration_result = RANSAC_open3d(pc_source, pc_target, source_features, target_features, distance_threshold)
        init_transform = global_registration_result.transformation

    p1p = np.asarray(pc_source.points)
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=3, max_iter=50)
    kmeans.fit(p1p)

    registration_best_fit = 0
    registration_inlier_rmse = 99
    registration_results = []
    samples = np.empty(k,dtype=np.int64)
    if draw_samples:
        winname = f"All {k} patches"
        colors = np.empty((len(pc_source.points),3))
        clrs = color_palette('husl', n_colors=k)
        for i in range(k):
            indices = np.where(kmeans.labels_ == i)[0]
            colors[indices] = clrs[i]
        tmp_pc_source = copy.deepcopy(pc_source)
        tmp_pc_source.colors = o3d.utility.Vector3dVector(colors)
        draw_registration_result(tmp_pc_source, pc_target, transformation=init_transform, winname=winname,
                                 threshold=distance_threshold,add_color=False)

    for i in range(k):
        indices = np.where(kmeans.labels_ == i)[0]
        samples[i] = len(indices)
        pc_patch = pc_source.select_by_index(indices)
        registration_result_i = ICP_open3d(pc_patch, pc_target, init_transform, distance_threshold)
        registration_results.append(registration_result_i)
        if draw_samples:
            winname = f"Patch={i}/{k}, samples={samples[i]}, fit={registration_result_i.fitness:2.2f}, rmse={registration_result_i.inlier_rmse:2.2f}"
            draw_registration_result(pc_patch,pc_target, transformation=registration_result_i.transformation,winname=winname, threshold=distance_threshold)
        if registration_result_i.fitness > registration_best_fit:
            registration_best_fit = registration_result_i.fitness
            registration_inlier_rmse = registration_result_i.inlier_rmse

    return registration_results, samples, kmeans.labels_, (registration_best_fit, registration_inlier_rmse)

