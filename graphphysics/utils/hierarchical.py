from typing import Any, Dict, Optional, Tuple

import h5py
import numpy as np
from torch_geometric.data import Data

from graphphysics.utils.torch_graph import meshdata_to_graph


def get_traj_as_meshes(
    file_handle: h5py.File, traj_number: str, meta: Dict[str, Any]
) -> Dict[str, np.ndarray]:
    """Retrieves mesh data for an entire trajectory from an H5 file.

    This function iterates over the specified trajectory in the H5 file, converting
    each feature into its appropriate data type and shape as defined in the metadata,
    and collects them into a dictionary.

    Parameters:
        file_handle (h5py.File): An open H5 file handle.
        traj_number (str): The key of the trajectory to retrieve.
        meta (Dict[str, Any]): A dictionary containing metadata about the dataset.

    Returns:
        Dict[str, np.ndarray]: A dictionary where keys are feature names and values are
        NumPy arrays containing the data for each feature across the entire trajectory.
    """
    features = file_handle[traj_number]
    meshes = {}

    for key, field in meta["features"].items():
        data = features[key][()].astype(field["dtype"])
        data = data.reshape(field["shape"])
        meshes[key] = data

    return meshes


def get_frame_as_mesh(
    traj: Dict[str, np.ndarray],
    frame: int,
    targets: list[str] = None,
    frame_target: Optional[int] = None,
) -> Tuple[
    np.ndarray, np.ndarray, Dict[str, np.ndarray], Optional[Dict[str, np.ndarray]]
]:
    """Retrieves mesh data for a given frame from an H5 file.

    This function extracts mesh position, cell data, and additional point data
    (e.g., node type, velocity, pressure) for a specified frame. If a target frame is
    provided, it also retrieves the target frame's data.

    Parameters:
        traj (Dict[str, np.ndarray]): A dictionary where keys are feature names and values
            are NumPy arrays containing the data for each feature across the entire trajectory.
        frame (int): The index of the frame to retrieve data for.
        targets (list[str]): A list of target names to retrieve.
        frame_target (int, optional): The index of the target frame to retrieve data for.

    Returns:
        Tuple: A tuple containing the following elements:
            - np.ndarray: The positions of the mesh points.
            - np.ndarray: The indices of points forming each cell.
            - Dict[str, np.ndarray]: A dictionary containing point data.
            - Optional[Dict[str, np.ndarray]]: A dictionary containing the target frame's point data,
              similar to point_data.
    """
    target_point_data = None
    next_data = None

    if frame_target is not None:
        target_point_data = {key: traj[key][frame_target] for key in targets}
        next_data = {
            key: traj[key][frame_target]
            for key in traj.keys()
            if key not in ["mesh_pos", "cells", "node_type"] and key not in targets
        }

    point_data = {
        key: traj[key][frame]
        for key in traj.keys()
        if key not in ["mesh_pos", "cells", "node_type"]
    }
    point_data["node_type"] = traj["node_type"][0]

    mesh_pos = (
        traj["mesh_pos"][frame] if traj["mesh_pos"].ndim > 1 else traj["mesh_pos"]
    )
    cells = traj["cells"][frame] if traj["cells"].ndim > 1 else traj["cells"]

    return mesh_pos, cells, point_data, target_point_data, next_data


def get_frame_as_graph(
    traj: Dict[str, np.ndarray],
    frame: int,
    meta: Dict[str, Any],
    targets: list[str] = None,
    frame_target: Optional[int] = None,
) -> Data:
    """Converts mesh data for a given frame into a graph representation.

    This function first retrieves mesh data using `get_frame_as_mesh` and then
    converts this data into a graph representation using the `meshdata_to_graph`
    function from the `torch_graph` module.

    Parameters:
        traj (Dict[str, np.ndarray]): A dictionary where keys are feature names and values
            are NumPy arrays containing the data for each feature across the entire trajectory.
        frame (int): The index of the frame to retrieve and convert.
        meta (Dict[str, Any]): A dictionary containing metadata about the dataset.
        targets (list[str]): A list of target names to retrieve.
        frame_target (int, optional): The index of the target frame to retrieve and convert.

    Returns:
        torch_geometric.data.Data: A PyTorch Geometric Data object representing the graph.
    """
    points, cells, point_data, target, next_data = get_frame_as_mesh(
        traj, frame, targets, frame_target
    )
    time = frame * meta.get("dt", 1)
    return meshdata_to_graph(
        points=points,
        cells=cells,
        point_data=point_data,
        time=time,
        target=target,
        next_data=next_data,
    )
