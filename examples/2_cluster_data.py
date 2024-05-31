import os

import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

from src.viz import mshow

"""
This script plots the fitness and distance matrix and does a clustering based on the distances.
"""

if __name__ == "__main__":
    np.set_printoptions(precision=2,linewidth=500)
    path_in = './data/results/0.2_1.5/high_res'
    file_in = './data/results/0.2_1.5/high_res/transformations.npz'
    output_path = './data/results/0.2_1.5/high_res/'
    with np.load(file_in, allow_pickle=True) as data:
        fits = data['fits']
        inlier_rmses = data['inlier_rmses']
        computed = data['computed']
        id_matrix = data['id_matrix']

    x = fits
    names = np.asarray([r'$u_1$', r'$u_2$', r'$u_3$', r'$u_{4,f}$', r'$u_{5,f}$', r'$b_1$', r'$b_2$', r'$b_{3}$', r'$b_{4}$', r'$b_{5f}$', r'$s_{1}$', r'$s_{2f}$'])
    n = x.shape[0]
    x_sym = np.maximum(x,x.T)
    dist = 1 - x_sym

    fig, axs = plt.subplots(1, 1, figsize=(8,8))
    mshow(axs, dist, n, f'Distance',labels=names,vmin='paper')
    file_out = os.path.join(output_path,"clustering_dist_only.png")
    fig.savefig(file_out, bbox_inches='tight')

    fig, axs = plt.subplots(1, 1, figsize=(8,8))
    mshow(axs, x, n, f'Fitness',labels=names,vmin='paper')
    file_out = os.path.join(output_path,"clustering_fitness_only.png")
    fig.savefig(file_out, bbox_inches='tight')


    # Clustering with Kmeans
    means = KMeans(n_clusters=3, random_state=0, n_init=10).fit(dist)
    labels = means.labels_


    indices = np.argsort(labels)
    print(labels[indices])
    print(names[indices])

    ii = np.arange(n)[indices]
    fig1,axs = plt.subplots(n,n, figsize=(n,n), dpi=1000)
    for i in range(n):
        for j in range(n):
            filename = os.path.join(path_in,f"{ii[i]}_{ii[j]}.png")
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
    print("done")




