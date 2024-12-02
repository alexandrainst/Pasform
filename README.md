# Automated similarity search for 3D scanned archaeological artefacts

This project compares various artefacts through point cloud registration and clustering, for more information see: (paper link coming here) 


## Installation
To get a working environment for the project install the packages in the pyproject.toml file. This can for instance be done with poetry using
```shell
poetry install
```

When you run the code, if you get any errors about src not being part of the module, then you need to add the root folder to you python path environment variable.
In a terminal this can be done as such:
```shell
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

## Getting started

The first thing you will need is a dataset to run the pipeline on. The dataset used in the paper can be found at: https://zenodo.org/records/11401136/files/stl.zip?download=1.
There is also a script provided which will download and unzip the dataset for you. 
```shell
python data/download_data.py
```

The examples folder contains various scripts. The paper pipeline essentially consist on running script 0-2 in sequential order. 
The pipeline does not require much in terms of memory or computational power. The required computational power depends upon the size of the 3D point clouds. For the benchmark dataset 16 GB of memory should be sufficient.  


### 0_downsample_stl.py
This script takes a folder of stl files as input and outputs downsampled point clouds based on each stl file. On a modern laptop, it takes about 5 minutes to run on the benchmark dataset.


### 1_run_pointcloud_registration.py
This script takes a path containing point cloud files, in both high and low resolution.
Each point cloud is pairwise registered against the others, first globally (using the low resolution pc) and then locally (using the high resolution pc)
On a modern laptop, the low resolution comparison should take a few minutes, while the high resolution comparison takes about 2.5 hours. 

### 2_cluster_data.py
This script plots the fitness and distance matrix and does a clustering based on the distances. It also plots an image all the point cloud registrations. 
The clustering should be done almost instantaneously, and the script should only take seconds to run.

### interactive_visualization_of_registration.py
This script is not part of the main pipeline, but is provided as a helper script to interactively visualize the registration of two point clouds.


The keys: 1,2,3 can be used during the visualization to highlight various things.

### Hardware specs
The project was developed and run on a laptop with the following specs:
13th Gen Intel(R) Core(TM) i7-13800H
32 GB ram
NVIDIA GeForce RTX 4060
