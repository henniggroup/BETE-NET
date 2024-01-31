import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as tg
import torch_scatter

import e3nn
from e3nn import o3

from torch_geometric.data import Data
from torch_cluster import radius_graph

from e3nn import o3
from e3nn.math import soft_one_hot_linspace
from e3nn.nn import Gate, Dropout
from e3nn.nn.models.gate_points_2101 import Convolution, smooth_cutoff, tp_path_exists

from typing import Dict, Union

bar_format = '{l_bar}{bar:10}{r_bar}{bar:-10b}'
default_dtype = torch.float64
torch.set_default_dtype(default_dtype)

class CustomCompose(torch.nn.Module):
    def __init__(self, first, second):
        super().__init__()
        self.first = first
        self.second = second
        self.irreps_in = self.first.irreps_in
        self.irreps_out = self.second.irreps_out

    def forward(self, *input):
        x = self.first(*input)
        self.first_out = x.clone()
        x = self.second(x)
        self.second_out = x.clone()
        return x


class Network(torch.nn.Module):
    r"""equivariant neural network
    Parameters
    ----------
    irreps_in : `e3nn.o3.Irreps` or None
        representation of the input features
        can be set to ``None`` if nodes don't have input features
    irreps_hidden : `e3nn.o3.Irreps`
        representation of the hidden features
    irreps_out : `e3nn.o3.Irreps`
        representation of the output features
    irreps_node_attr : `e3nn.o3.Irreps` or None
        representation of the nodes attributes
        can be set to ``None`` if nodes don't have attributes
    irreps_edge_attr : `e3nn.o3.Irreps`
        representation of the edge attributes
        the edge attributes are :math:`h(r) Y(\vec r / r)`
        where :math:`h` is a smooth function that goes to zero at ``max_radius``
        and :math:`Y` are the spherical harmonics polynomials
    layers : int
        number of gates (non linearities)
    max_radius : float
        maximum radius for the convolution
    number_of_basis : int
        number of basis on which the edge length are projected
    radial_layers : int
        number of hidden layers in the radial fully connected network
    radial_neurons : int
        number of neurons in the hidden layers of the radial fully connected network
    num_neighbors : float
        typical number of nodes at a distance ``max_radius``
    num_nodes : float
        typical number of nodes in a graph
    """

    def __init__(
        self,
        irreps_in,
        irreps_out,
        irreps_node_attr,
        layers,
        mul,
        lmax,
        max_radius,
        number_of_basis=10,
        radial_layers=1,
        radial_neurons=100,
        num_neighbors=1.0,
        num_nodes=1.0,
        reduce_output=True,
        p=0.2,
        dropout=False
    ) -> None:
        super().__init__()
        self.dropout=dropout
        self.mul = mul
        self.lmax = lmax
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.num_neighbors = num_neighbors
        self.num_nodes = num_nodes
        self.reduce_output = reduce_output

        self.irreps_in = o3.Irreps(irreps_in) if irreps_in is not None else None
        self.irreps_hidden = o3.Irreps(
            [(self.mul, (l, p)) for l in range(lmax + 1) for p in [-1, 1]]
        )
        self.irreps_out = o3.Irreps(irreps_out)
        self.irreps_node_attr = (
            o3.Irreps(irreps_node_attr)
            if irreps_node_attr is not None
            else o3.Irreps("0e")
        )
        self.irreps_edge_attr = o3.Irreps.spherical_harmonics(lmax)

        self.input_has_node_in = irreps_in is not None
        self.input_has_node_attr = irreps_node_attr is not None

        irreps = self.irreps_in if self.irreps_in is not None else o3.Irreps("0e")

        act = {1: torch.nn.functional.silu, -1: torch.tanh}
        act_gates = {1: torch.sigmoid, -1: torch.tanh}

        self.layers = torch.nn.ModuleList()
        self.drop_outs = torch.nn.ModuleList() ## ADDED JG

        for _ in range(layers):
            irreps_scalars = o3.Irreps(
                [
                    (mul, ir)
                    for mul, ir in self.irreps_hidden
                    if ir.l == 0 and tp_path_exists(irreps, self.irreps_edge_attr, ir)
                ]
            )
            irreps_gated = o3.Irreps(
                [
                    (mul, ir)
                    for mul, ir in self.irreps_hidden
                    if ir.l > 0 and tp_path_exists(irreps, self.irreps_edge_attr, ir)
                ]
            )
            ir = "0e" if tp_path_exists(irreps, self.irreps_edge_attr, "0e") else "0o"
            irreps_gates = o3.Irreps([(mul, ir) for mul, _ in irreps_gated])

            gate = Gate(
                irreps_scalars,
                [act[ir.p] for _, ir in irreps_scalars],  # scalar
                irreps_gates,
                [act_gates[ir.p] for _, ir in irreps_gates],  # gates (scalars)
                irreps_gated,  # gated tensors
            )
            conv = Convolution(
                irreps,
                self.irreps_node_attr,
                self.irreps_edge_attr,
                gate.irreps_in,
                number_of_basis,
                radial_layers,
                radial_neurons,
                num_neighbors,
            )
            irreps = gate.irreps_out
            self.layers.append(CustomCompose(conv, gate))

        self.layers.append(
            Convolution(
                irreps,
                self.irreps_node_attr,
                self.irreps_edge_attr,
                self.irreps_out,
                number_of_basis,
                radial_layers,
                radial_neurons,
                num_neighbors,
            )
        )
        self.drop_outs.append(Dropout(self.irreps_out,p))

    def preprocess(self, data: Union[Data, Dict[str, torch.Tensor]]) -> torch.Tensor:
        if "batch" in data:
            batch = data["batch"]
        else:
            batch = data["pos"].new_zeros(data["pos"].shape[0], dtype=torch.long)

        if "edge_index" in data:
            edge_src = data["edge_index"][0]  # edge source
            edge_dst = data["edge_index"][1]  # edge destination
            edge_vec = data["edge_vec"]

        else:
            edge_index = radius_graph(data["pos"], self.max_radius, batch)
            edge_src = edge_index[0]
            edge_dst = edge_index[1]
            edge_vec = data["pos"][edge_src] - data["pos"][edge_dst]

        return batch, edge_src, edge_dst, edge_vec

    def forward(self, data: Union[Data, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """evaluate the network
        Parameters
        ----------
        data : `torch_geometric.data.Data` or dict
            data object containing
            - ``pos`` the position of the nodes (atoms)
            - ``x`` the input features of the nodes, optional
            - ``z`` the attributes of the nodes, for instance the atom type, optional
            - ``batch`` the graph to which the node belong, optional
        """
        batch, edge_src, edge_dst, edge_vec = self.preprocess(data)
        edge_sh = o3.spherical_harmonics(
            self.irreps_edge_attr, edge_vec, True, normalization="component"
        )
        edge_length = edge_vec.norm(dim=1)
        edge_length_embedded = soft_one_hot_linspace(
            x=edge_length,
            start=0.0,
            end=self.max_radius,
            number=self.number_of_basis,
            basis="gaussian",
            cutoff=False,
        ).mul(self.number_of_basis ** 0.5)
        edge_attr = smooth_cutoff(edge_length / self.max_radius)[:, None] * edge_sh

        if self.input_has_node_in and "x" in data:
            assert self.irreps_in is not None
            x = data["x"]
        else:
            assert self.irreps_in is None
            x = data["pos"].new_ones((data["pos"].shape[0], 1))

        if self.input_has_node_attr and "z" in data:
            z = data["z"]
        else:
            assert self.irreps_node_attr == o3.Irreps("0e")
            z = data["pos"].new_ones((data["pos"].shape[0], 1))

        for i, lay in enumerate(self.layers):
            x = lay(x, z, edge_src, edge_dst, edge_attr, edge_length_embedded)
            ############## ADDED JG ##############
            if self.dropout:
                do = self.drop_outs[i]
                x = do(x)

        if self.reduce_output:
            return scatter(x, batch, dim=0).div(self.num_nodes ** 0.5)
        else:
            return x


class PeriodicNetwork(Network):
    def __init__(self, in_dim, em_dim, **kwargs):
        # override the `reduce_output` keyword to instead perform an averge over atom contributions
        self.pool = False
        if kwargs["reduce_output"] == True:
            kwargs["reduce_output"] = False
            self.pool = True

        super().__init__(**kwargs)

        # embed the mass-weighted one-hot encoding
        self.em = nn.Linear(in_dim, em_dim)

    def forward(
        self, data: Union[tg.data.Data, Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        data.x = F.relu(self.em(data.x))
        data.z = F.relu(self.em(data.z))
        output = super().forward(data)
        output = torch.relu(output)

        # if pool_nodes was set to True, use scatter_mean to aggregate
        if self.pool == True:
            output = torch_scatter.scatter_mean(
                output, data.batch, dim=0
            )  # take mean over atoms per example

        return output
    
class PeriodicNetworkPhdos(Network):
    def __init__(self, in_dim, em_dim,out_dim, **kwargs):
        # override the `reduce_output` keyword to instead perform an averge over atom contributions
        self.pool = False
        if kwargs["reduce_output"] == True:
            kwargs["reduce_output"] = False
            self.pool = True

        super().__init__(**kwargs)

        # embed the mass-weighted one-hot encoding
        self.em = nn.Linear(in_dim, em_dim)
        self.output = nn.Linear(out_dim*2,out_dim)

    def forward(
        self, data: Union[tg.data.Data, Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        data.x = F.relu(self.em(data.x))
        data.z = F.relu(self.em(data.z))
        output = super().forward(data)
        output = torch.relu(output)
        output = torch_scatter.scatter_mean(
            output, data.batch, dim=0
        )  # take mean over atoms per example
        output = torch.cat((output,data.phdos),dim=1)
        output = self.output(output)
        output = torch.relu(output)        
        return output    

class EMDLoss(nn.Module):
    def __init__(self):
        super(EMDLoss, self).__init__()

    def forward(self, p, q):
        print(f' p = {p.shape}')
        cdf_p = torch.cumsum(p, dim=1)
        cdf_q = torch.cumsum(q, dim=1)
        #print(cdf_q.shape)
        #print(f'sum{torch.sum(torch.abs(cdf_p - cdf_q), dim=1).shape}')

        emd = torch.sum(torch.abs(cdf_p - cdf_q), dim=1).mean()

        return emd
    
class weighted_MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,inputs,targets,weights):
        return torch.mean(((inputs - targets)**2 ) * weights)   

