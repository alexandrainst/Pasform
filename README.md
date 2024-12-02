# Automated similarity search for 3D scanned archaeological artefacts

This project compares various artefacts through point cloud registration and clustering, for more information see: (paper link coming here) 


## Installation
To get a working environment for the project install the packages in the pyproject.toml file. This can for instance be done with poetry using
```shell
poetry install --no-dev
```

If you get any errors about src not being part of the module, you need to add the root folder to you python path environment variable.
In the terminal this can be done as such:
```shell
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

## Getting started

The examples folder contains various scripts. The paper pipeline essentially consist on running script 0-2 in sequential order. 
The pipeline does not require much in terms of memory or computational power. The required computational power depends upon the size of the 3D point clouds. For the example dataset 


### 0_downsample_stl.py
This script takes a folder of stl files as input and outputs downsampled point clouds based on each stl file.


Note that in order to utilize these scripts you will need a set of 3D files, either mesh files (.stl) or point cloud files (.pc). In the data folder there is a python script for downloading the 3D files for the artefacts used in the article.

