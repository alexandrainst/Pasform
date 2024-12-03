import argparse
import os

import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

from pasform.utils import set_seed
from pasform.viz import mshow

"""
This script plots the fitness and distance matrix and does a clustering based on the distances.
It also plots an image all the point cloud registrations. 
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_image_path', type=str, default='./data/results/0.2_1.5/low_res' ,help='Path to all the point cloud registration images.')
    parser.add_argument('--input_file_transformation', type=str, default='./data/results/0.2_1.5/low_res/transformations.npz' ,help='Path to the transformation file.')
    parser.add_argument('--output_path', type=str, default='./data/results/clustering', help='Base output path, where all the results will be saved.')
    parser.add_argument('--names', type=list, default=[r'$u_1$', r'$u_2$', r'$u_3$', r'$u_{4,f}$', r'$u_{5,f}$', r'$b_1$', r'$b_2$', r'$b_{3}$', r'$b_{4}$', r'$b_{5f}$', r'$s_{1}$', r'$s_{2f}$'], help='A list of names for all the point clouds being compared')
    parser.add_argument('--seed', type=int, default=1234, help='A seed for the randomizers, to ensure reproduceable results.')
    args = parser.parse_args()

    set_seed(args.seed)
    print("Clustering the results of the point cloud registration and producing plots...")

    np.set_printoptions(precision=2,linewidth=500)
    path_images_in = args.input_image_path
    file_transformation_in = args.input_file_transformation
    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)

    with np.load(file_transformation_in, allow_pickle=True) as data:
        fits = data['fits']
        inlier_rmses = data['inlier_rmses']
        computed = data['computed']
        id_matrix = data['id_matrix']
        cloud_sizes = data['cloud_sizes']
    cloud_sizes[np.arange(cloud_sizes.shape[0]),np.arange(cloud_sizes.shape[0])] = 1
    c1 = cloud_sizes.reshape(-1)
    c2 = 1 / c1
    max_relative_sizes = np.max([c1[:, None], c2[:, None]], axis=0).reshape(cloud_sizes.shape)

    x = fits
    names = np.asarray(args.names)
    n = x.shape[0]
    x_sym = np.maximum(x,x.T)
    dist = 1 - x_sym

    fig, axs = plt.subplots(1, 1, figsize=(8,8))
    mshow(axs, max_relative_sizes, n, f'Relative point cloud sizes',labels=names,vmin='paper')
    file_out = os.path.join(output_path,"relative_size.png")
    fig.savefig(file_out, bbox_inches='tight')


    fig, axs = plt.subplots(1, 1, figsize=(8,8))
    mshow(axs, dist, n, f'Distance',labels=names,vmin='paper')
    file_out = os.path.join(output_path,"clustering_dist_only.png")
    fig.savefig(file_out, bbox_inches='tight')

    fig, axs = plt.subplots(1, 1, figsize=(8, 8))
    mshow(axs, dist/max_relative_sizes, n, f'Size adjusted distance', labels=names, vmin='paper')
    file_out = os.path.join(output_path, "clustering_size_dist_only.png")
    fig.savefig(file_out, bbox_inches='tight')

    fig, axs = plt.subplots(1, 1, figsize=(8,8))
    mshow(axs, x, n, f'Fitness',labels=names,vmin='paper')
    file_out = os.path.join(output_path,"clustering_fitness_only.png")
    fig.savefig(file_out, bbox_inches='tight')


    # Clustering with Kmeans
    means = KMeans(n_clusters=3, random_state=0, n_init=10).fit(dist)
    labels = means.labels_


    indices = np.argsort(labels)
    print(f"The names ordered by clustering groups: \n{names[indices]}")
    print(f"{labels[indices]}")

    ii = np.arange(n)[indices]
    fig1,axs = plt.subplots(n,n, figsize=(n,n), dpi=1000)
    for i in range(n):
        for j in range(n):
            filename = os.path.join(path_images_in,f"{ii[i]}_{ii[j]}.png")
            ax = axs[i,j]
            aa= plt.imread(filename)
            ax.imshow(aa)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
        axs[i,0].set_ylabel(names[i], fontsize=8)
        axs[-1, i].set_xlabel(names[i], fontsize=8)
    file_out = os.path.join(output_path,"all.png")
    fig1.savefig(file_out, bbox_inches='tight')




