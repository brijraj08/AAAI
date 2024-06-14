#We have leveraged our proposed Content-driven Sessions in the [Transformers4Rec](https://github.com/NVIDIA-Merlin/Transformers4Rec/)

Transformers4Rec is a flexible and efficient library for sequential and session-based recommendation and can work with PyTorch.

## Installation

You can install Transformers4Rec with Pip, Conda, or run a Docker container.

### Installing Transformers4Rec Using Pip

You can install Transformers4Rec with the functionality to use the GPU-accelerated Merlin dataloader.
Installation with the dataloader is highly recommended for better performance.
Those components can be installed as optional arguments for the `pip install` command.

To install Transformers4Rec using Pip, run the following command:

```shell
pip install transformers4rec[pytorch,nvtabular,dataloader]
```

> Be aware that installing Transformers4Rec with `pip` only supports the CPU version of Merlin Dataloader because `pip` does not install cuDF.
> The GPU capabilities of the dataloader are available by using the Docker container or by installing
> the dataloader with Conda first and then performing the `pip` installation within the Conda environment.

### Installing Transformers4Rec Using Conda

To install Transformers4Rec using Conda, run the following command:

```shell
conda install -c nvidia transformers4rec
```

### Installing Transformers4Rec Using Docker

Transformers4Rec is pre-installed in the `merlin-pytorch` container that is available from the NVIDIA GPU Cloud (NGC) catalog.

Refer to the [Merlin Containers](https://nvidia-merlin.github.io/Merlin/main/containers.html) documentation page for information about the Merlin container names, URLs to container images in the catalog, and key Merlin components.

## Notebooks
 
- tr4rec_nvt for each_dataset:This notebook is created using the latest stable merlin-pytorch container. this is used to perform the feature engineering that is needed to model the each dataset which contains a collection of sessions. then saved our processed data frames as parquet files.
- model_evaluation for each dataset:this notebook use processed parquet files to train a session-based recommendation model.

