import os

from matplotlib import pyplot as plt

from src.registration import prepare_base_set
from src.viz import mshow

"""
This script takes a path to a bunch of point cloud files, in both high and low resolution.
Each point cloud is pairwise registrered against the others, first globally (using the low resolution pc) and then locally (using the high resolution pc)
"""


if __name__ == "__main__":
    input_path = "./data/pointcloud/voxel_downsampled/0.2"
    input_path_low_res = "./data/pointcloud/voxel_downsampled/1.5"
    output_path_root = './data/results/'
    # indices = [48, 58, 59, 64, 65, 1, 6, 22, 25, 63, 51, 66] # We only select a certain subset of artefacts among the ones available.
    indices = None
    names = ['DIME215055','C41222','C30567_urnes','D11058_urnes','D160-1997_urnes','Dime141357','Dime141360','Dime_198107_starshaped','559295_hst','B10420','B5884f_a','B601_a']
    names.sort()

    voxel_size = float(input_path.split("/")[-1])
    if input_path_low_res is None:
        voxel_size_low_res = None
        output_path = os.path.join(output_path_root,f"{voxel_size}")
    else:
        voxel_size_low_res = float(input_path_low_res.split("/")[-1])
        output_path = os.path.join(output_path_root,f"{voxel_size}_{voxel_size_low_res}")

    os.makedirs(output_path, exist_ok=True)

    names, fits, inlier_rmse, transformations = prepare_base_set(input_path,voxel_size,input_path_low_res,voxel_size_low_res,output_folder=output_path, draw_samples=False,indices=indices)
    n = len(fits)
    fig, axs = plt.subplots(1, 2, figsize=(15,15))
    mshow(axs[0], fits, n, f'fitness, voxel={voxel_size:1.1f}',labels=names)
    mshow(axs[1], inlier_rmse,n, 'inlier rmse ',labels=names)
    file_out = os.path.join(output_path,"confusion_matrix.png")
    fig.savefig(file_out)
