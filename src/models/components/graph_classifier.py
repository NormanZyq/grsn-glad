import dgl
import torch
from dgl.nn.pytorch import AvgPooling
import torch.nn.functional as F

class GraphClassifier(torch.nn.Module):
    def __init__(self, input_size, output_size, dropout=0.2):
        super().__init__()

        self.avg_pool = AvgPooling()
        # self.graph_readout = torch.nn.AvgPool1d()
        input_size = eval(input_size) if isinstance(input_size, str) else input_size
        self.linear = torch.nn.Linear(input_size, output_size)
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, g: dgl.DGLGraph, feat: torch.Tensor):
        feat = feat.squeeze()
        out = self.avg_pool(g, feat)
        out = self.dropout(out)
        out = F.relu(out)
        out = self.linear(out)
        return out
