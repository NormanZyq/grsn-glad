# %%
import os
import random

import dgl
import torch
import torch.nn.functional as F
from dgl.data import DGLDataset
from dgl.data.tu import TUDataset

from src.data.components.brains_dataset import BrainsDataset
from src.data.components.p53_dataset import P53Dataset


def random_sampling(graph, nodes, walk_length, restart_prob=0.5):
    return dgl.sampling.random_walk(
        graph,
        nodes,
        length=walk_length,
        restart_prob=restart_prob
    )


def node2vec_sampling(graph, nodes, p, q, walk_length):
    return dgl.sampling.node2vec_random_walk(
        graph,
        nodes,
        p=p,
        q=q,
        walk_length=walk_length
    )


def neighbor_sampling():
    pass


class GDRCDataset(DGLDataset):
    def __init__(self, name="AIDS",
                 down_sample_label=0,
                 down_sample_rate=0.1,
                 num_sample=5,
                 walk_length=None,
                 re_gen_ds_labels=False,
                 sampling_method='random_walk',
                 url=None,
                 raw_dir="data/raw",
                 save_dir="data/processed",
                 stage: str = "train",
                 **kwargs):
        super().__init__(name=name, url=url, raw_dir=raw_dir, save_dir=save_dir)
        if walk_length is None:
            # default: set `walk_length` to [5, 6, 8, 10]
            walk_length = [5, 6, 8, 10]
        if isinstance(walk_length, str):
            walk_length = eval(walk_length)
        self.stage = stage
        if 'tox' in name.lower():
            tu_dataset_delegate = P53Dataset(name, raw_dir)
            tu_dataset_delegate.process()
        elif 'KKI' == name or 'OHSU' == name:
            tu_dataset_delegate = BrainsDataset(name, raw_dir=raw_dir)
            tu_dataset_delegate.process()
        else:
            tu_dataset_delegate = TUDataset(name, raw_dir=raw_dir)
        graphs = tu_dataset_delegate.graph_lists
        labels = tu_dataset_delegate.graph_labels

        # load down sampling result if exists
        # if not os.path.exists('{}/downsample_ids'.format(raw_dir)):
        #     os.makedirs('{}/downsample_ids'.format(raw_dir))
        # if os.path.exists('{}/downsample_ids/{}_ds_ids.txt'.format(raw_dir, name)) and not re_gen_ds_labels:
        #     with open('{}/downsample_ids/{}_ds_ids.txt'.format(raw_dir, name), 'r') as f:
        #         use_ids = eval(f.readline())
        # else:
        #     use_ids = []
        #     for i in range(len(labels)):
        #         if labels[i].tolist()[0] == down_sample_label:
        #             if random.random() <= down_sample_rate:
        #                 use_ids.append(i)
        #         else:
        #             use_ids.append(i)
        #     # save to file
        #     with open('{}/downsample_ids/{}_ds_ids.txt'.format(raw_dir, name), 'w') as f:
        #         f.write(str(use_ids))

        self.cfg_seed = kwargs['seed']
        random.seed(12345)      # we fix seed to 12345 for data generation and processing
        print('seed has been fixed to', 12345)
        use_ids = []
        for i in range(len(labels)):
            if labels[i].tolist()[0] == down_sample_label:
                if random.random() <= down_sample_rate:
                    use_ids.append(i)
            else:
                use_ids.append(i)

        # log: building data
        self.graphs = []
        self.labels = []
        num_node_labels = 0
        use_random_feat = kwargs['random_str_feat'] if 'random_str_feat' in kwargs else False
        print('Building data...')
        print('Require random feat: ', use_random_feat)
        print('Perform random walk or node2vec sampling: ', not use_random_feat)
        for idx in use_ids:
            # perform random walk on graph[idx]
            nodes = graphs[idx].nodes()
            if 'node_labels' in graphs[idx].ndata:
                num_node_labels = max(graphs[idx].ndata['node_labels'].max(), num_node_labels)
            if use_random_feat:
                # init with random feat which will replace the `sub_attr` with the same size
                feat_dim = num_sample * len(walk_length)
                graphs[idx].ndata['sub_attr'] = torch.randn((graphs[idx].num_nodes(), feat_dim))
            else:
                # perform random walk
                for wl in walk_length:
                    # iterate `walk_length` and sample `num_sample` times for a single length
                    # so the final extended structural feature size would be `num_sample` * walk_length
                    # Note: 重复实验之前必须要以共通的随机种子构造一次数据集，本项目默认为12345，保存为pkl之后会直接读取，
                    #       此后设置随机种子不会再影响数据特征的挖掘，只会影响模型参数的初始化
                    for it in range(num_sample):
                        # use dgl to impl random walk
                        if sampling_method == 'random' or sampling_method == 'random_walk':
                            traces, _ = random_sampling(
                                graphs[idx],
                                nodes,
                                walk_length=wl,
                                restart_prob=0.5
                            )
                        elif sampling_method == 'node2vec':
                            traces = node2vec_sampling(
                                graphs[idx],
                                nodes,
                                p=kwargs['p'],
                                q=kwargs['q'],
                                walk_length=wl,
                            )
                        else:
                            raise NotImplementedError('sampling method {} not implemented'.format(sampling_method))
                        # count the unique nodes in the traces
                        # dim of traces is expected to be [num_sample, walk_length]
                        # create temp feature
                        substructure_node_counting_feat = []
                        for tr in traces:
                            substructure_node_counting_feat.append(tr.unique().numel())
                        substructure_node_counting_feat = torch.tensor(substructure_node_counting_feat)
                        if 'sub_attr' in graphs[idx].ndata:
                            graphs[idx].ndata['sub_attr'] = torch.cat(
                                (graphs[idx].ndata['sub_attr'], substructure_node_counting_feat.unsqueeze(1).float()),
                                dim=1
                            )
                        else:
                            graphs[idx].ndata['sub_attr'] = substructure_node_counting_feat.unsqueeze(1).float()
            self.graphs.append(graphs[idx].add_self_loop())
            self.labels.append(labels[idx])
        if 'tox' not in name.lower() and 'node_labels' in self.graphs[0].ndata:
            # 对于非p53的数据，重新遍历一次graphs，将node_labels转换为one hot 存进去
            num_node_labels += 1
            for idx in range(len(self.graphs)):
                self.graphs[idx].ndata['node_labels'] = F.one_hot(self.graphs[idx].ndata['node_labels'].squeeze(),
                                                                  num_classes=num_node_labels)

        random.seed(self.cfg_seed)
        print('seed has been reset to your config, ', self.cfg_seed)

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx].squeeze(0)

    def __len__(self):
        return len(self.graphs)
