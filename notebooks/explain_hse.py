# %%
import os
import sys
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

L.seed_everything(12345, workers=True)
dgl.seed(12345)

# Load model
ckpt_path = '''
/home/zhuyeqi/gdrsnc/logs/train/runs/2024-11-18_01-07-12/checkpoints/epoch_075.ckpt
'''.strip()
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
data_name = 'Tox21_HSE_training'
walk_length = [20,30,15,25,11]
num_sample = 15
num_structure_feat = num_sample * len(walk_length)
batch_size = 32
p = 1
q = 0.2
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
                     shuffle=True
                     )
dm.setup()
dataset = dm.data_test
print('dataset loaded')

# %%
# other args
start_graph_id = 0
end_graph_id = None
show = False
save_fig = True

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

n # %%
# iter the dataset to find a graph whose label=1
# for i, graph in enumerate(dataset):
#     if graph[1].item() == dsl:
#         print(i)
#         # break
