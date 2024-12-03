import argparse
import os

from matplotlib import pyplot as plt

from pasform.registration import prepare_base_set
from pasform.utils import set_seed
from pasform.viz import mshow

"""
This script takes a path to a bunch of point cloud files, in both high and low resolution.
Each point cloud is pairwise registrered against the others, first globally (using the low resolution pc) and then locally (using the high resolution pc)
"""


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='./data/pointcloud/voxel_downsampled/0.2' ,help='Path to input stl files in high resolution')
    parser.add_argument('--input_path_low_res', type=str, default='./data/pointcloud/voxel_downsampled/1.5' ,help='Path to input stl files in low resolution')
    parser.add_argument('--output_path', type=str, default='./data/results/', help='Base output path, where all the results will be saved.')
    parser.add_argument('--ordering_of_artefacts', type=list, default=[4, 6, 7, 9, 10, 0, 1, 2, 3, 8, 5, 11], help='The ordering of the artefacts. This can be used to reorder the artefacts in the confusion matrix.')
    parser.add_argument('--seed', type=int, default=1234, help='A seed for the randomizers, to ensure reproduceable results.')
    args = parser.parse_args()

    set_seed(args.seed)

    input_path = args.input_path
    input_path_low_res = args.input_path_low_res
    output_path_root = args.output_path
    indices = args.ordering_of_artefacts

    voxel_size = float(input_path.split("/")[-1])
    if input_path_low_res is None:
        voxel_size_low_res = None
        output_path = os.path.join(output_path_root,f"{voxel_size}")
    else:
        voxel_size_low_res = float(input_path_low_res.split("/")[-1])
        output_path = os.path.join(output_path_root,f"{voxel_size}_{voxel_size_low_res}")

    os.makedirs(output_path, exist_ok=True)

    names, fits, inlier_rmse, transformations = prepare_base_set(input_path,voxel_size,input_path_low_res,voxel_size_low_res,output_folder=output_path, indices=indices)
    n = len(fits)
    fig, axs = plt.subplots(1, 2, figsize=(15,15))
    mshow(axs[0], fits, n, f'fitness, voxel={voxel_size:1.1f}',labels=names)
    mshow(axs[1], inlier_rmse,n, 'inlier rmse ',labels=names)
    file_out = os.path.join(output_path,"confusion_matrix.png")
    fig.savefig(file_out)
