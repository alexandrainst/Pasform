import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from pasform.registration_utils import extract_registration_patches_results

matplotlib.use('agg')



def plot_registration_patches(registration_result_patches, samples, save=None, title=''):
    fitnesses, inlier_rmses, n_elements_corresponding = extract_registration_patches_results(registration_result_patches)
    samples_total = samples.sum()
    idx = np.argsort(fitnesses)
    n = len(fitnesses)
    n_arange = np.arange(n)
    s = 50*n*(samples/samples_total)
    if samples is None:
        fig, axs = plt.subplots(1,2,figsize=(10,10))
    else:
        fig, axs = plt.subplots(1,3,figsize=(10,10))
    fig.suptitle(title)
    axs[0].scatter(n_arange,fitnesses[idx], s=s[idx])
    axs[0].set_title("fitness")
    axs[1].scatter(n_arange,inlier_rmses[idx], s=s[idx])
    axs[1].set_title("inlier_rmse")
    if samples is not None:
        axs[2].scatter(n_arange,samples[idx], s=20)
        axs[2].set_title("number of samples")
    if save is not None:
        fig.savefig(save)
        plt.close(fig)
        return
    plt.pause(1)
    return





def mshow(ax, matrix, max_val, title,labels, vmin=None):
    if vmin is None:
        ax.imshow(matrix, vmin=0, vmax=np.max(matrix), cmap='Wistia')
    elif vmin == 'paper':
        m = matrix > 0
        vmin = np.min(matrix[m])
        ax.imshow(matrix, vmin=vmin, vmax=np.max(matrix), cmap='Wistia')
    else:
        ax.imshow(matrix, vmin=vmin, vmax=np.max(matrix), cmap='Wistia')
    ax.set_title(title, fontsize=20)

    for i in range(max_val):
        for j in range(max_val):
            c = matrix[j, i]
            ax.text(i, j, '{:.2f}'.format(c), va='center', ha='center')
    ax.set(xticks=(np.arange(len(labels))), xticklabels=labels, yticks=(np.arange(len(labels))), yticklabels=labels),# xlabel='source', ylabel='target')
    ax.set_xticklabels(labels,rotation=45, fontsize=18,label='source')
    ax.set_yticklabels(labels, fontsize=18)
    return