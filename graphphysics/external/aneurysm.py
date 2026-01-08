import torch
from torch_geometric.data import Data

from graphphysics.utils.nodetype import NodeType

device = "cuda" if torch.cuda.is_available() else "cpu"


def aneurysm_node_type(graph: Data) -> torch.Tensor:
    node_type = graph.x[:, -2]
    node_type[node_type < 0] = 0
    node_type[node_type > 6] = 6
    node_type = node_type.to(torch.int)
    return node_type.to(device)


def aneurysm_time(graph: Data) -> torch.Tensor:
    time = graph.x[:, -1]
    time = torch.round(time, decimals=3)
    return time.to(device)


def build_features(graph: Data) -> Data:
    node_type = aneurysm_node_type(graph)
    time = aneurysm_time(graph)

    current_velocity = graph.x[:, 0:3]
    target_velocity = graph.y[:, 0:3]
    previous_velocity = torch.tensor(graph.previous_data["Vitesse"], device=device)

    acceleration = current_velocity - previous_velocity
    next_acceleration = target_velocity - current_velocity

    graph.x[:, 3] = graph.x[:, 3] / 1000
    current_cup = graph.x[:, 3]
    previous_cup = (
        torch.tensor(graph.previous_data["CUP"], device=device).squeeze(1) / 1000
    )
    acceleration_cup = current_cup - previous_cup

    graph.x[:, 4] = graph.x[:, 4] / 1000
    current_cap = graph.x[:, 4]
    previous_cap = (
        torch.tensor(graph.previous_data["CAP"], device=device).squeeze(1) / 1000
    )
    acceleration_cap = current_cap - previous_cap

    current_dbp = graph.x[:, 5]
    previous_dbp = torch.tensor(graph.previous_data["PhiDBP"], device=device).squeeze(1)
    acceleration_dbp = current_dbp - previous_dbp

    not_inflow_mask = node_type != NodeType.INFLOW
    next_acceleration[not_inflow_mask] = 0
    next_acceleration_unique = next_acceleration.unique()

    mean_next_accel = torch.ones(node_type.shape, device=device) * torch.mean(
        next_acceleration_unique
    )
    min_next_accel = torch.ones(node_type.shape, device=device) * torch.min(
        next_acceleration_unique
    )
    max_next_accel = torch.ones(node_type.shape, device=device) * torch.max(
        next_acceleration_unique
    )

    velocity_norm = torch.norm(graph.x[:, 0:3], dim=1)
    velocity_norm = velocity_norm.to(device).unsqueeze(1)

    x_mask = [0, 1, 3, 4, 5]
    graph.x = graph.x[:, x_mask]
    graph.x = torch.cat(
        (
            graph.x,
            acceleration[:, :2],
            acceleration_cup.unsqueeze(1),
            acceleration_cap.unsqueeze(1),
            acceleration_dbp.unsqueeze(1),
            graph.pos[:, :2],
            mean_next_accel.unsqueeze(1),
            min_next_accel.unsqueeze(1),
            max_next_accel.unsqueeze(1),
            time.unsqueeze(1),
            node_type.unsqueeze(1),
        ),
        dim=1,
    )

    y_mask = [0, 1, 3, 4, 5]
    graph.y = graph.y[:, y_mask]
    graph.y[:, 2:4] = graph.y[:, 2:4] / 1000

    return graph
