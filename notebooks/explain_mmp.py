# %%
import os
import sys
import random
print(os.getcwd())
# extend `../` to the working directory
os.chdir('/home/zhuyeqi/gdrsnc/')
sys.path.append(os.getcwd())

from src.utils.explainer import explain_graph
from src.data.gdrcm_datamodule import GDRCMDataModule
import lightning as L
import dgl
from src.models.grsn_counting2 import GRSNCounting
from dgl.nn.pytorch.explain import GNNExplainer
import torch
import networkx as nx
import matplotlib.pyplot as plt
from src.utils.explain_utils import component_map, edge_label_map

L.seed_everything(12345, workers=True)
dgl.seed(12345)

# Load model
ckpt_path = '''
/home/zhuyeqi/gdrsnc/logs/train/runs/2024-11-08_07-21-02/checkpoints/epoch_068.ckpt
'''.strip()

# no W1
# ckpt_path = '''
# /home/zhuyeqi/gdrsnc/logs/train/runs/2024-11-14_07-23-26/checkpoints/epoch_153.ckpt
# '''.strip()
device = 'cpu'
model = GRSNCounting.load_from_checkpoint(ckpt_path,
                                          map_location=torch.device(device))
model.eval()

# explainer related args
num_hops = 2
lr = 0.001
beta1 = 0.8
beta2 = 0.2
num_epochs = 200
log = False

# initialize explainer
explainer = GNNExplainer(model=model,
                         num_hops=num_hops,
                         lr=lr,
                         beta1=beta1,
                         beta2=beta2,
                         num_epochs=num_epochs,
                         log=log)

# data related args
data_name = 'Tox21_MMP_training'
walk_length = [20, 30, 15, 7, 5, 9]
num_sample = 7
num_structure_feat = 64  # Note: this is from the hidden_size in the config file
batch_size = 32
p = 2
q = 0.6
dsl = 1
down_sample_rate = 1.1
sampling_method = 'node2vec'
# load data
dm = GDRCMDataModule(name=data_name,
                     dsl=dsl,
                     down_sample_rate=down_sample_rate,
                     re_gen_ds_labels=False,
                     walk_length=walk_length,
                     sampling_method=sampling_method,
                     num_sample=num_sample,
                     batch_size=batch_size,
                     train_val_test_split=(0.7, 0.2, 0.1),
                     p=p,
                     q=q,
                     seed=12345,
                     shuffle=True, 
                     )
dm.setup()
dataset = dm.data_test
print('dataset loaded')

# %%
# other args
start_graph_id = 1
end_graph_id = 5
show = True
save_fig = True

#%%
# explain
explain_graph(model=model,
              explainer=explainer,
              data_name=data_name,
              start_graph_id=start_graph_id,
              end_graph_id=end_graph_id,
              num_structure_feat=num_structure_feat,
              show=show,
              save_fig=save_fig,
              num_hops=num_hops,
              dataset=dataset)

# %%
# iter the dataset to find a graph whose label=1
for i, graph in enumerate(dataset):
    if graph[1].item() == dsl:
        print(i)
        # break

# %%
# filter label=dsl and label!=dsl
anomalous_samples = [graph for graph in dataset if graph[1].item() == dsl]
normal_samples = [graph for graph in dataset if graph[1].item() != dsl]

# %%
# analyze the two types of samples separately
start_graph_id = 2
end_graph_id = 3
show = True
save_fig = False

explain_graph(model=model,
              explainer=explainer,
              data_name=data_name,
              start_graph_id=start_graph_id,
              end_graph_id=end_graph_id,
              num_structure_feat=num_structure_feat,
              show=show,
              save_fig=save_fig,
              num_hops=num_hops,
              dataset=anomalous_samples,
              postfix='anomalous')

# %%
start_graph_id = 0
end_graph_id = None

explain_graph(model=model,
              explainer=explainer,
              data_name=data_name,
              start_graph_id=start_graph_id,
              end_graph_id=end_graph_id,
              num_structure_feat=num_structure_feat,
              show=show,
              save_fig=save_fig,
              num_hops=num_hops,
              dataset=normal_samples,
              postfix='normal')

#%%
# --------------------------------------------------------------------------
# In this cell, we only analyze the anomalous samples
graph_id = random.randint(0, len(anomalous_samples)-1)
print(f'random graph_id: {graph_id}')

#%%
graph_id = 102
#%%
graph = anomalous_samples[graph_id][0]
graph = graph.remove_self_loop()
# Convert DGL graph to NetworkX graph
nx_graph = graph.to_networkx()

# Add edge labels based on edge attributes
edge_labels = {}
for i, (src, dst) in enumerate(nx_graph.edges()):
    bond_type = graph.edata['edge_labels'][i].item()  
    edge_labels[(src, dst)] = edge_label_map[bond_type]

atom_labels = {i: component_map[graph.ndata['node_labels_raw'][i].item()] for i in range(graph.number_of_nodes())}

pos = nx.spring_layout(nx_graph, iterations=1000, k=0.15)
plt.figure(figsize=(8, 6))
nx.draw(
    nx_graph, pos,
    with_labels=True, labels=atom_labels,
    node_size=500, node_color="skyblue", font_size=8
)
nx.draw_networkx_edge_labels(
    nx_graph, pos,
    edge_labels=edge_labels, font_size=8, label_pos=0.5
)
plt.title(f"Molecular Graph with Atom and Bond Types, label={dataset[graph_id][1].item()}")
# show the graph with high resolution
plt.savefig(f'{data_name}_anomalous_graph_{graph_id}.png', dpi=300)
plt.show()

#%%
# Print mapping of node indices to atom types
print("\nNode index to atom type mapping:")
for node_idx in range(graph.number_of_nodes()):
    atom_type = component_map[graph.ndata['node_labels_raw'][node_idx].item()]
    print(f"Node {node_idx}: {atom_type}")

#%%
# Print mapping of edge pairs to bond types
print("\nEdge pairs and their bond types:")
for (src, dst), bond_type in edge_labels.items():
    src_atom = component_map[graph.ndata['node_labels_raw'][src].item()]
    dst_atom = component_map[graph.ndata['node_labels_raw'][dst].item()]
    print(f"Edge ({src}:{src_atom}) -- {bond_type} -- ({dst}:{dst_atom})")


# %% 
# --------------------------------------------------------------------------
# In this cell, we only analyze the normal samples
graph_id = 24
graph = normal_samples[graph_id][0]
# remove self-loops
graph = graph.remove_self_loop()
# Convert DGL graph to NetworkX graph
nx_graph = graph.to_networkx()

edge_labels = {}
for i, (src, dst) in enumerate(nx_graph.edges()):
    bond_type = graph.edata['edge_labels'][i].item()  
    edge_labels[(src, dst)] = edge_label_map[bond_type]


# Add atom or bond type labels for better visualization
atom_labels = {i: component_map[graph.ndata['node_labels_raw'][i].item()] for i in range(graph.number_of_nodes())}

pos = nx.spring_layout(nx_graph)
plt.figure(figsize=(8, 6))
nx.draw(
    nx_graph, pos,
    with_labels=True, labels=atom_labels,
    node_size=500, node_color="skyblue", font_size=8
)
nx.draw_networkx_edge_labels(
    nx_graph, pos,
    edge_labels=edge_labels, font_size=8, label_pos=0.5
)
plt.title(f"Molecular Graph with Atom and Bond Types, label={dataset[graph_id][1].item()}")
# show the graph with high resolution
plt.savefig(f'{data_name}_normal_graph_{graph_id}.png', dpi=300)
plt.show()


# %%
# use rdkit to get the smiles of the graph
from rdkit import Chem
from rdkit.Chem import Draw

# edge_label_map = {0: '-', 1: '=', 2: ':', 3: '#'}

# Create an empty RDKit molecule
mol = Chem.RWMol()

# Add atoms
for i in range(graph.number_of_nodes()):
    atom_type = component_map[graph.ndata['node_labels_raw'][i].item()]
    atom = Chem.Atom(atom_type)
    mol.AddAtom(atom)

# Add bonds - only add each edge once
added_bonds = set()  # 用来跟踪已添加的边
for i, (src, dst) in enumerate(nx_graph.edges()):
    # 将边的两个端点排序，确保 (1,2) 和 (2,1) 被视为相同的边
    bond_pair = tuple(sorted([int(src), int(dst)]))
    if bond_pair not in added_bonds:
        bond_type = graph.edata['edge_labels'][i].item()
        if bond_type == 0:  # Single bond '-'
            bond_type = Chem.BondType.SINGLE
        elif bond_type == 1:  # Double bond '='
            bond_type = Chem.BondType.DOUBLE
        elif bond_type == 2:  # Aromatic bond ':'
            bond_type = Chem.BondType.AROMATIC
        elif bond_type == 3:  # Triple bond '#'
            bond_type = Chem.BondType.TRIPLE
        
        mol.AddBond(bond_pair[0], bond_pair[1], bond_type)
        added_bonds.add(bond_pair)

# Convert to mol and get SMILES
mol = mol.GetMol()
smiles = Chem.MolToSmiles(mol)
print(f"SMILES representation: {smiles}")

#%%
# Draw the molecule
img = Draw.MolToImage(mol)
# show
img

# img.save(f'{data_name}_rdkit_mol_{graph_id}.png')


# %%
# 统计
