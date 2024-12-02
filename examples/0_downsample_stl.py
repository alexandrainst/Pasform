import argparse
import os
from pathlib import Path

import open3d as o3d

"""
This script takes a folder of stl files as input and outputs a bunch of downsampled point clouds based on each stl file.
"""


def load_file_and_to_pc(fn):
    pcd = o3d.geometry.PointCloud()
    mesh = o3d.io.read_triangle_mesh(fn)
    pcd.points = mesh.vertices
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    return pcd


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='./data/stl/' ,help='Path to input stl files')
    parser.add_argument('--output_path', type=str, default='./data/pointcloud/', help='Base output path, where all the results will be saved.')
    args = parser.parse_args()


    input_dir = Path(args.input_path)
    base_output_dir = Path(args.output_path)

    voxel_sizes = [0.1, 0.2, 0.5, 1.0, 1.5, 2.0]

    for fp in input_dir.glob('*.stl'):
        pcd = load_file_and_to_pc(str(fp))

        # store original size
        output_path = base_output_dir + '/unstructured_point_cloud'
        os.makedirs(output_path,exist_ok=True)
        fileout = os.path.join(output_path, fp.name[:-3] + 'pcd')
        o3d.io.write_point_cloud(fileout, pcd)

        for vx_size in voxel_sizes:
            pcd_d = pcd.voxel_down_sample(voxel_size=vx_size)
            output_path = base_output_dir + '/voxel_downsampled/' + str(vx_size)
            os.makedirs(output_path, exist_ok=True)
            fileout = os.path.join(output_path, fp.name[:-3] + 'pcd')
            o3d.io.write_point_cloud(fileout, pcd_d)

        print('Processed : ', str(fp))
        print(f'{len(pcd.points)} datapoints')
