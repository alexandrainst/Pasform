import os

import numpy as np
import open3d as o3d

from src.viz3d import draw_registration_result

"""
A simple script that allows for manual inspection of a pairwise registration.

The keys: 1,2,3 can be used during the visualization to highlight various things.

"""


if __name__ == "__main__":
    voxel_size = 0.2
    # input_path = f"./data/pointcloud/voxel_downsampled/{voxel_size}"
    input_path = f"/home/tue/Data/Archaeology/pointcloud/voxel_downsampled/{voxel_size}"
    transformations_file = f"/home/tue/Data/Archaeology/results/0.2_1.5/high_res/transformations.npz"
    voxel_size = float(input_path.split("/")[-1])
    name_source = 'Dime_198107_starshaped'
    name_target = 'C41222'

    name_source = 'C30567_urnes' # u_1
    name_target = 'D11058_urnes' # u_2

    # name_source = 'Dime141357'  # u_4f
    # name_target = 'DIME215055'  # b_5f

    name_source = 'DIME215055'
    name_target = 'B601_a'

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





