# Training Graph Neural Networks for Mesh-based Physics Simulations

## Overview

This repository let's you train Graph Neural Networks on meshes (e.g. fluid dynamics, material simulations, etc).
We offer a simple training script to:
- Setup your model's architecture
- Define your dataset with different augmentation functions
- Follow the training live, including live vizualisations

The code is based on Pytorch.

## Datasets

We give access to all datasets (full trajectories) and the mesh used to compute said simulation.

| Dataset                | Description                                                             | Link                                                                         |
|------------------------|-------------------------------------------------------------------------|------------------------------------------------------------------------------|
| CylinderFlow | As .H5 and reduced from MeshGraphNet | [Train](https://storage.googleapis.com/large-physics-model/datasets/cylinder/train.h5) [Test](https://storage.googleapis.com/large-physics-model/datasets/cylinder/test.h5) [Validation](https://storage.googleapis.com/large-physics-model/datasets/cylinder/valid.h5) |
| DeformingPlate | As .H5 and reduced from MeshGraphNet | [Train](https://storage.googleapis.com/large-physics-model/datasets/plate/train.h5) [Test](https://storage.googleapis.com/large-physics-model/datasets/plate/test.h5) [Validation](https://storage.googleapis.com/large-physics-model/datasets/plate/valid.h5) |
| Bezier | As .xdmf | [Train](https://storage.googleapis.com/large-physics-model/datasets/bezier/train1.zip) [Test](https://storage.googleapis.com/large-physics-model/datasets/bezier/test.zip) |
| 2D-Aneurysm | As .XDMF and Sliced from AnxPlore | [Dataset](https://storage.googleapis.com/large-physics-model/datasets/aneurysm/2D_dataset.zip) |
| 3D-CoarseAneurysm | As .XDMF and Interpolated from AnxPlore | [Dataset](https://storage.googleapis.com/large-physics-model/datasets/aneurysm/coarse_03_dataset.zip) |

## Meshs

| Dataset                | Description                                                             | Link                                                                         |
|------------------------|-------------------------------------------------------------------------|------------------------------------------------------------------------------|
| MultipleBezierShapes | 1200 meshs of 1 to 4 bezier shapes at random places                                   | https://storage.googleapis.com/large-physics-model/datasets/meshs/dataset_1_to_4_bezier_shapes.zip |
| 3DAneurysm | 100 Meshes                                   | https://github.com/aurelegoetz/AnXplore/tree/main |

### Tutorials

We offer a Google colab to showcase training on:
- a blood flow inside a 3D Aneurysm with Message Passing
  - [Colab]()
  - dataset is from [AnXplore: a comprehensive fluid-structure interaction study of 101 intracranial aneurysms](https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2024.1433811/full?field&journalName=Frontiers_in_Bioengineering_and_Biotechnology&id=1433811)

## Vizualisations 

We use [Weights and Biases](https://wandb.ai/site) to log most information during training. This includes:
- training and validation loss
  - per step
  - per epoch 
- All Rollout RMSE on validation dataset

## Setup

For local setup, just run the following command to build your environment:
```
make install
```

### Default requirements

```python
import torch

def format_pytorch_version(version):
  return version.split('+')[0]

TORCH = 2.8.0

def format_cuda_version(version):
  return 'cu' + version.replace('.', '')

CUDA_version = torch.version.cuda
CUDA = format_cuda_version(CUDA_version)
```

```
pip install torch-scatter     -f https://pytorch-geometric.com/whl/torch-{TORCH}+{CUDA}.html
pip install torch-sparse      -f https://pytorch-geometric.com/whl/torch-{TORCH}+{CUDA}.html
pip install torch-cluster     -f https://pytorch-geometric.com/whl/torch-{TORCH}+{CUDA}.html
pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-{TORCH}+{CUDA}.html
pip install torch-geometric

pip install numpy=1.26.4
pip install loguru==0.7.2
pip install autoflake==2.3.0
pip install pytest==8.0.1
pip install meshio==5.3.5
pip install h5py==3.10.0

!pip install pyvista lightning==2.5.0 wandb "wandb[media]"
!pip install pytorch-lightning==2.5.0 torchmetrics==1.6.3
```


### WandB

We use Weights and Bias to log most of our metrics and vizualizations during the training. Make sure you create and account, and log in before you start training.

```python
import wandb
wandb.login()
```

## Documentation

Most of setting up a new use case depends on two `.json` files: one to define the dataset details, and one for the training settings.

Let's start with the training settings. An example is available [here](https://github.com/DonsetPG/graph-physics/blob/main/training_config/cylinder.json).

### Dataset

```json 
"dataset": {
    "extension": "h5",
    "h5_path": "dataset/h5_dataset/cylinder_flow/train.h5",
    "meta_path": "dataset/h5_dataset/cylinder_flow/meta.json",
    "targets": ["velocity"],
    "khop": 1
}
```

- `extension`: If the dataset used is h5 or xdmf.
- `h5_path` (`xdmf_folder` for an xdmf dataset): Path to the dataset.

> [!NOTE]  
> You will need a dataset at the same location with `test` instead of `train` in its name for the validation step to work. Otherwise, you can specify its name directly in `training.py`

- `meta_path`: Location to the .json file with the dataset details (see below)
- `targets`: List of the fields to predict.

> [!NOTE]   
> If this key does not exist or if the list is empty, an error will be raised.   
> The field(s) designated as target(s) must be dynamic. If it is not the case, an error will be raised.   
> The order in which the fields are defined in `targets` must be the same as that in the `dataset_config` file.     
> The dynamic fields not designated as targets will be added to `graph.next_data` and will be available for building features, for instance if the user wants to use the velocity boundary conditions of the next time step.

- `khop`: K-hop neighbors size to use. You should start with 1.

You also need to define a few other parameters: 

```json
"index": {
    "feature_index_start": 0,
    "feature_index_end": 2,
    "output_index_start": 0,
    "output_index_end": 2,
    "node_type_index": 2
}
```
- `feature_index_`: This is to define where we should look for nodes features. The end is excluded. For example, if you have 2D velocities at index 0 and 1, and pressure at index 2. If you want to use the pressure you should set `feature_index_start=0` and `feature_index_end=3`, otherwise, `feature_index_end=2`.

- `output_index_`: We define our architectures to predict one of your feature for the next time steps. So you need to tell us where to look. For example, if you want to predict the velocity at the next step, since the velocity is at index 0 and 1, you will set `output_index_start=0` and `output_index_end=2`.

- `node_type_index`: Finally, we use a node type classification for each node:

```python
NORMAL = 0
OBSTACLE = 1
AIRFOIL = 2
HANDLE = 3
INFLOW = 4
OUTFLOW = 5
WALL_BOUNDARY = 6
SIZE = 9
```

> [!WARNING]  
> You should modify this if this is not at all representative of your use case. Those are taken from [Meshgraphnet](https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets) and we found them to be general enough for all of our use cases. 

This means that you either need to have such feature in your dataset, or to define a python function to build them (see below). After that, you need to tell us where to look. For example, if we only have velocity and node type, we will have `node_type_index=2`. If we also had the pressure, we would set `node_type_index=3`

> [!WARNING]  
> H5-based dataloader does not support multiple workers. XDMF can.

### Custom Processing Functions

First, we allow you to add noise to your inputs to make the prediction of a trajectory more robust.

```json
"preprocessing": {
    "noise": 0.02,
    "noise_index_start": [0],
    "noise_index_end": [2],
    "masking": 0
},
```

> [!WARNING]  
> Masking is not implemented yet.

```python
def add_noise(
    graph: Data,
    noise_index_start: Union[int, List[int]],
    noise_index_end: Union[int, List[int]],
    noise_scale: Union[float, List[float]],
    node_type_index: int,
) -> Data:
    """
    Adds Gaussian noise to the specified features of the graph's nodes.

    Parameters:
        graph (Data): The graph to modify.
        noise_index_start (Union[int, List[int]]): The starting index or indices for noise addition.
        noise_index_end (Union[int, List[int]]): The ending index or indices for noise addition.
        noise_scale (Union[float, List[float]]): The standard deviation(s) of the Gaussian noise.
        node_type_index (int): The index of the node type feature.

    Returns:
        Data: The modified graph with noise added to node features.
    """
```

Second, in the case of dealing with multiple meshes, you can add extra edges based on closeness of those different meshes:

```json 
"world_pos_parameters": {
    "use": false,
    "world_pos_index_start": 0,
    "world_pos_index_end": 3
}
```

See the [description](https://arxiv.org/abs/2010.03409) regarding world edges.

Finally, in the case where: 
- you need to build the node type 
- you need to build extra features that were not in your dataset

In `train.py`:

```python
# Build preprocessing function
preprocessing = get_preprocessing(
    param=parameters,
    device=device,
    use_edge_feature=use_edge_feature,
    extra_node_features=None,
)
```

where: 

```python
extra_node_features: Optional[
        Union[Callable[[Data], Data], List[Callable[[Data], Data]]]
    ] = None
```

You can define one or several functions that takes a graph as an input, and returns another graph with the new features. 

> [!NOTE]  
> In the case where you might need the previous graph as well (to compute acceleration for example, you can pass `get_previous_data` in the `get_dataset` function, and you will be able to access it using the `previous_data` attribute: `graph.previous_data`)
> You can check [build_features](https://github.com/DonsetPG/graph-physics/blob/main/graphphysics/external/aneurysm.py) where we use `previous_velocity = torch.tensor(graph.previous_data["Vitesse"], device=device)`
> It's important to note to if you do so, those previous data also need to be updated autoregressively during the validation steps. To do so, we added 2 parameters in `train.py`: `previous_data_start` and `previous_data_end`. By default, they are set to 4 and 7. This works if for example, you set the acceleration (computed using the previous velocity) at indexes 4, 5 and 6.

For example, let's imagine we want to add the nodes position as a feature, one could define the following function: 

```python
def add_pos(graph: Data) -> Data:
    graph.x = torch.cat(
        (
            graph.pos,
            graph.x,
        ),
        dim=1,
    )
    return graph
```

<details>
  <summary>In that case, the settings would need to be updated.</summary>
  ```json
  "index": {
      "feature_index_start": 0,
      "feature_index_end": 4,
      "output_index_start": 2,
      "output_index_end": 4,
      "node_type_index": 4
  }
  ```
</details>

You can find more examples regarding adding features and building node type [here](https://github.com/DonsetPG/graph-physics/tree/main/graphphysics/external).

We simply then call the function `add_pos` in `get_preprocessing`:

```python
# Build preprocessing function
preprocessing = get_preprocessing(
    param=parameters,
    device=device,
    use_edge_feature=use_edge_feature,
    extra_node_features=add_pos,
)
```

### Architecture

```json
"model": {
    "type": "epd",
    "message_passing_num": 5,
    "hidden_size": 32,
    "node_input_size": 2,
    "output_size": 2,
    "edge_input_size": 3
}
```

- `type`: `epd` (message passing)
- `message_passing_num`: Number of Layers
- `hidden_size`: Number of hidden neurons
- `node_input_size`: Number of node features

> [!WARNING]  
> This should not count the node type feature.

- `edge_input_size`: Size of the edge features. 3 in 2D and 4 in 3D. 0 for transformer based model.
- `output_size`: Size of the output
  
### Dataset Settings

You will also need to design a .json to define the dataset details. Those `meta.json` files are inspired from [Meshgraphnet](https://github.com/google-deepmind/deepmind-research/tree/master/meshgraphnets).

You will need to define: 

- `dt`: the time step of your simulation
- `features`: a set of features used, including at least `cells` and `mesh_pos` for the .h5 dataset.
- `field_names`: the list of all features
- `trajectory_length`: the number of time steps per trajectory
