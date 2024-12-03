import argparse
import os
from pathlib import Path

import open3d as o3d
from tqdm import tqdm

from pasform.utils import set_seed

"""
This script takes a folder of stl files as input and outputs a bunch of downsampled point clouds based on each stl file.
"""


def convert_stl_to_pointcloud(fn):
    """
    Converts stl files to point clouds
    """
    pcd = o3d.geometry.PointCloud()
    mesh = o3d.io.read_triangle_mesh(fn)
    pcd.points = mesh.vertices
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    return pcd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='./data/stl/' ,help='Path to input stl files')
    parser.add_argument('--output_path', type=str, default='./data/pointcloud/', help='Base output path, where all the results will be saved.')
    parser.add_argument('--seed', type=int, default=1234, help='A seed for the randomizers, to ensure reproduceable results.')
    args = parser.parse_args()

    set_seed(args.seed)

    input_dir = Path(args.input_path)
    base_output_dir = Path(args.output_path)
    output_path = os.path.join(base_output_dir, 'unstructured_point_cloud')
    os.makedirs(output_path, exist_ok=True)

    voxel_sizes = [0.1, 0.2, 0.5, 1.0, 1.5, 2.0]

    stl_files = list(input_dir.glob('*.stl'))
    if len(stl_files) == 0:
        print(f'No stl files found in the input directory {input_dir}')
        exit()

    print("Converting stl files to point clouds at various voxel sizes...")
    for stl_file in tqdm(stl_files):
        pcd = convert_stl_to_pointcloud(str(stl_file))

        fileout = os.path.join(output_path, stl_file.name[:-3] + 'pcd')
        o3d.io.write_point_cloud(fileout, pcd)

        for vx_size in voxel_sizes:
            pcd_d = pcd.voxel_down_sample(voxel_size=vx_size)
            output_path_i = os.path.join(base_output_dir, 'voxel_downsampled/' + str(vx_size))
            os.makedirs(output_path_i, exist_ok=True)
            fileout_i = os.path.join(output_path_i, stl_file.name[:-3] + 'pcd')
            o3d.io.write_point_cloud(fileout_i, pcd_d)

        # print('Processed : ', str(stl_file))
        # print(f'{len(pcd.points)} datapoints')
