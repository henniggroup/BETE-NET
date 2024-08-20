import torch
import torch_geometric as tg

from ase.neighborlist import neighbor_list  # load data
from ase import Atom, Atoms

import numpy as np
from scipy.signal import savgol_filter
from tqdm import tqdm

bar_format = '{l_bar}{bar:10}{r_bar}{bar:-10b}'
default_dtype = torch.float64
torch.set_default_dtype(default_dtype)

Freq_final =np.arange(0.25,101,2)
Freq_final_E =np.arange(-50,50,1)

# type_onehot, am_onehot, type_encoding = get_onehot()
type_encoding = {}
specie_am = []
for Z in range(1, 119):
    specie = Atom(Z)
    type_encoding[specie.symbol] = Z - 1
    specie_am.append(specie.mass)

type_onehot = torch.eye(len(type_encoding))
am_onehot = torch.diag(torch.tensor(specie_am))

def build_data(
    entry, r_max=4.0, embed_ph_dos=True, embed_e_dos=True, fine=False
):
    """
    Construct a data object suitable for graph-based models using PyTorch Geometric.

    Parameters:
    - entry (object): Input object containing structure information about a molecule or crystal.
    - r_max (float, optional): Cutoff radius for neighbor list calculation. Default is 4.0.
    - embed_ph_dos (bool, optional): Flag to indicate if phonon density of states (phDOS) data should be embedded. Default is True.
    - embed_e_dos (bool, optional): Flag to indicate if electronic density of states (eDOS) data should be embedded. Default is True.
    - fine (bool, optional): Flag to determine the granularity of phonon density of states. Default is True.
    - avg (bool, optional): Flag to determine whether to average some properties. Default is False.

    Returns:
    - tg.data.Data: A PyTorch Geometric data object containing various attributes like node features, edge indices, etc., which represent the molecular/crystalline structure.

    Notes:
    - This function assumes certain global variables and helper functions like `neighbor_list`, `type_encoding`, `am_onehot`, `process_edos`, etc. are available in the environment.
    - It is recommended to ensure all dependencies are imported and necessary global variables are initialized before invoking this function.
    """
#     type_onehot, am_onehot, type_encoding = get_onehot()
    symbols = list(entry.structure.symbols).copy()
    positions = torch.from_numpy(entry.structure.positions.copy())
    lattice = torch.from_numpy(entry.structure.cell.array.copy()).unsqueeze(0)

    # edge_src and edge_dst are the indices of the central and neighboring atom, respectively
    # edge_shift indicates whether the neighbors are in different images or copies of the unit cell
    edge_src, edge_dst, edge_shift = neighbor_list(
        "ijS", a=entry.structure, cutoff=r_max, self_interaction=True
    )

    # compute the relative distances and unit cell shifts from periodic boundaries
    edge_batch = positions.new_zeros(positions.shape[0], dtype=torch.long)[
        torch.from_numpy(edge_src)
    ]
    edge_vec = (
        positions[torch.from_numpy(edge_dst)]
        - positions[torch.from_numpy(edge_src)]
        + torch.einsum(
            "ni,nij->nj",
            torch.tensor(edge_shift, dtype=default_dtype),
            lattice[edge_batch],
        )
    )

    # compute edge lengths (rounded only for plotting purposes)
    edge_len = np.around(edge_vec.norm(dim=1).numpy(), decimals=2)

    x = am_onehot[[type_encoding[specie] for i, specie in enumerate(symbols)]]
    z = type_onehot[
        [type_encoding[specie] for specie in symbols]
    ]  # Not updated at each convolution
    if embed_ph_dos and embed_e_dos:
        p_ph_dos = process_phdos(entry, fine=fine)
        p_e_dos = process_edos(entry, fine=fine)

        x = torch.cat((x, torch.ones_like(p_ph_dos), torch.ones_like(p_e_dos)), 1)
        z = torch.cat((z, p_ph_dos, p_e_dos), 1)

    elif embed_ph_dos:
        p_ph_dos = process_phdos(entry, fine=fine)

        x = torch.cat((x, torch.ones_like(p_ph_dos)), 1)
        z = torch.cat((z, p_ph_dos), 1)

    elif embed_e_dos:
        p_e_dos = process_edos(entry, fine=fine)

        x = torch.cat((x, torch.ones_like(p_e_dos)), 1)
        z = torch.cat((z, p_e_dos), 1)

    data = tg.data.Data(
        pos=positions,
        lattice=lattice,
        symbol=symbols,
        x=x,
        z=z,  # atom type (node attribute)
        edge_index=torch.stack(
            [torch.LongTensor(edge_src), torch.LongTensor(edge_dst)], dim=0
        ),
        edge_shift=torch.tensor(edge_shift, dtype=default_dtype),
        edge_vec=edge_vec,
        edge_len=edge_len,
        target=torch.from_numpy(np.asarray(entry.target)).unsqueeze(0),
    )
    return data


def get_target(df):
    x = df.Freq_meV
    y = df.a2F
    xl = np.arange(0.25, 101, 0.1)
    y = np.interp(xl, x, y)
    Y = savgol_filter(y, 101, 3, mode="interp")
    Y = np.interp(Freq_final, xl, Y)
    Y = np.asarray([y if y > 0.0 else 0.0 for y in Y])
    return Y  # /max(Y)


def get_phdos(df):
    x = df.PhFreq_meV
    y = df.Tot_PhDOS
    xl = np.arange(0.25, 101, 0.1)
    y = np.interp(xl, x, y)
    Y = savgol_filter(y, 101, 3, mode="interp")
    Y = np.interp(Freq_final, xl, Y)
    return np.asarray([y if y > 0.0 else 0.0 for y in Y])


def process_edos(entry):
    ys = entry.Site_proj_eDOS
    x = entry.Site_proj_eDOS_eng_meV
    Y_proc = []

    for y in ys:
        Y = savgol_filter(y, 101, 3, mode="interp")
        Y = np.interp(Freq_final_E, x, Y)
        Y = [y if y > 0.0 else 0.0 for y in Y]
        Y_proc.append(Y.copy())
    return torch.Tensor(Y_proc)


def process_phdos(entry, fine=False):
    Y_proc = []
    if fine:
        x = entry.PhFreq_meV_dense
        ys = entry.Site_Proj_PhDOS_dense
    else:
        x = entry.Ph_2x2x2_interpolated_Freq_meV
        ys = entry.Ph_2x2x2_interpolated_Site_Proj_DOS

    for y in ys:

        xl = np.arange(0.25, 101, 0.1)
        y = np.interp(xl, x, y)
        Y = savgol_filter(y, 101, 3, mode="interp")
        Y = np.interp(Freq_final, xl, Y)
        Y = [y if y > 0.0 else 0.0 for y in Y]
        Y_proc.append(Y.copy())
    return torch.Tensor(Y_proc)

def get_neighbors(df, idx):
    n = []
    for entry in df.itertuples():
        N = entry.data.pos.shape[0]
        for i in range(N):
            n.append(len((entry.data.edge_index[0] == i).nonzero()))
    return np.array(n)
