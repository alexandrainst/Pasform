import argparse
import os

import numpy as np
import open3d as o3d

from pasform.viz3d import draw_registration_result

"""
A simple script that allows for manual inspection of a pairwise registration.

You can use the keys 1,2,3 during the visualization to toggle visibility of the source, target and inliers.
    1,2 toggles visibility of source, target
    3 toggles inlier points / default colorscheme

"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='./data/results/0.2_1.5/low_res' ,help='Path to all the point cloud registration files.')
    parser.add_argument('--input_file_transformation', type=str, default='./data/results/0.2_1.5/high_res/transformations.npz' ,help='Path to the transformation file.')
    parser.add_argument('--voxel_size', type=float, help='The voxel size used.')
    parser.add_argument('--name_target', type=str, default='C41222', help='The name of the target point cloud as given in the transformation.npz file')
    parser.add_argument('--name_source', type=str, default='Dime_198107_starshaped', help='The name of the source point cloud as given in the transformation.npz file')

    args = parser.parse_args()
    input_path = args.input_path
    transformations_file = args.input_file_transformation
    voxel_size = args.voxel_size
    name_target = args.name_target
    name_source = args.name_source

    name = name_source + '_' + name_target
    data = np.load(transformations_file, allow_pickle=True)
    id_matrix = data['id_matrix']
    idx = np.where(id_matrix == name)
    transformation_matrix = data['transformations']
    transformation = transformation_matrix[idx[1],idx[0]][0]

    file_source = os.path.join(input_path, name_source+'.pcd')
    file_target = os.path.join(input_path, name_target+'.pcd')
    pc_source = o3d.io.read_point_cloud(file_source)
    pc_target = o3d.io.read_point_cloud(file_target)
    threshold = 3 * voxel_size
    draw_registration_result(pc_source, pc_target,transformation=transformation, threshold=threshold)





