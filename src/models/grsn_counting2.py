# Note: grsn_counting = Graph Random Substructure Network + substructure counting
# Added constraint about A in this version
from typing import Any, Dict, Tuple

import dgl
import lightning.pytorch as pl
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import AUROC, F1Score, Accuracy, Precision, Recall


class CustomLoss(torch.nn.Module):
    def __init__(self, pos_weight):
        super(CustomLoss, self).__init__()
        self.pos_weight = pos_weight
        # BCEWithLogitsLoss is with Sigmoid, do not call it
        if self.pos_weight is None:
            loss_fn = torch.nn.BCEWithLogitsLoss()
        else:
            loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.pos_weight]))
        self.loss_fn = loss_fn

    def forward(self, pred, target):
        loss = self.loss_fn(pred, target)
        return loss


class GRSNCounting(pl.LightningModule):
    def __init__(
            self,
            attr_feat_size: int,
            structure_feat_size,
            structure_hidden_size: int,
            str_dropout: float,
            graph_conv: torch.nn.Module,
            graph_classifier: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler,
            compile: bool,
            pos_weight=1.0
    ) -> None:
        super().__init__()
        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.graph_conv = graph_conv
        self.graph_classifier = graph_classifier
        if isinstance(structure_feat_size, str):
            self.structure_feat_size = structure_feat_size = eval(structure_feat_size)
        assert (attr_feat_size > 0) or (structure_feat_size > 0)
        self.attr_feat_size = attr_feat_size
        self.W1 = torch.nn.Sequential(
            torch.nn.Linear(structure_feat_size, structure_hidden_size),
            torch.nn.Dropout(p=str_dropout),
            torch.nn.LeakyReLU()
        )
        self._lambda = torch.nn.Parameter(torch.tensor(0.1).clamp(0, 1))
        self.fusion_layer = torch.nn.Linear(attr_feat_size + structure_hidden_size,
                                            attr_feat_size + structure_hidden_size)

        self.label_crit = CustomLoss(pos_weight=pos_weight)
        self.struct_crit = torch.nn.MSELoss()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # eval metrics - AUC
        self.train_auc = AUROC(task='binary')
        self.val_auc = AUROC(task='binary')
        self.test_auc = AUROC(task='binary')

        # eval metrics - Precision
        self.train_precision = Precision(task='binary')
        self.val_precision = Precision(task='binary')
        self.test_precision = Precision(task='binary')

        # eval metrics - Recall
        self.train_recall = Recall(task='binary')
        self.val_recall = Recall(task='binary')
        self.test_recall = Recall(task='binary')

        # eval metrics - Accuracy
        self.train_acc = Accuracy(task='binary')
        self.val_acc = Accuracy(task='binary')
        self.test_acc = Accuracy(task='binary')

        # eval metrics - F1
        self.train_f1 = F1Score(task='binary')
        self.val_f1 = F1Score(task='binary')
        self.test_f1 = F1Score(task='binary')

        self.val_auc_best = MaxMetric()
        self.test_h_graph_list = []
        self.test_label_list = []

    def concat_attrs(self, graph):
        feat_str = feat_node_attr = feat_node_labels = None
        if 'sub_attr' in graph.ndata and self.structure_feat_size > 0:
            feat_str = graph.ndata['sub_attr']
            feat_str = F.normalize(feat_str, p=2, dim=1)
            feat_str = self.W1(feat_str)
        if 'node_attr' in graph.ndata and self.attr_feat_size > 0:
            # node attr
            feat_node_attr = graph.ndata['node_attr'].float()  # [N_b * num_sample, hidden]
        if 'node_labels' in graph.ndata and self.attr_feat_size > 0:  # 让attr_feat_size对node label同时控制
            # if the data contains node labels
            feat_node_labels = graph.ndata['node_labels'].float()  # [N_b * num_sample, node_classes] (in one hot)
        feats_tmp = [feat_node_attr, feat_node_labels, feat_str]  # 按顺序
        feat = []
        for f in feats_tmp:
            if f is not None:
                feat.append(f)
        # concat
        feat = torch.cat(feat, dim=1)
        return feat

    def get_h_graph(self, graph: dgl.DGLGraph):
        feat = self.concat_attrs(graph)
        # fusion layer
        feat = self.fusion_layer(feat)
        # activation
        feat = F.leaky_relu(feat)

        # A_hat = torch.matmul(feat, feat.T)  # [N, N]
        feat = torch.mm(graph.adjacency_matrix().to_dense(), feat)

        # 卷积
        h_graph = self.graph_conv(graph, feat)

        return h_graph

    def forward(self, graph: dgl.DGLGraph, **kwargs):
        h_graph = self.get_h_graph(graph)
        return h_graph
        # # 计算A_hat
        # A_hat = torch.mm(h_graph, h_graph.T)  # [N, N]
        # A_hat = F.sigmoid(A_hat)
        # # decode attrs
        # labels_pred = self.graph_classifier(graph, h_graph)
        # return labels_pred, A_hat

    def model_step(self, batch: Tuple[dgl.DGLGraph, torch.Tensor], ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        graph, labels = batch
        labels = labels.float()
        # get the forward output
        h_graph = self.forward(graph)  # [N_b, 1]

        # 计算A_hat
        A_hat = torch.mm(h_graph, h_graph.T)  # [N, N]
        A_hat = F.sigmoid(A_hat)
        # decode attrs
        labels_pred = self.graph_classifier(graph, h_graph).squeeze(1)  # [N_b]

        # label loss
        loss_label = self.label_crit(labels_pred, labels)
        # str loss
        # loss_str = self.struct_crit(graph.adjacency_matrix().to_dense(), A_hat)
        loss_str = 0
        return loss_label, loss_str, labels_pred

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        _, labels = batch
        loss_label, loss_str, labels_pred = self.model_step(batch)
        loss = loss_label + self._lambda * loss_str
        self.train_loss(loss.item())
        self.train_auc(labels_pred, labels)
        labels_pred = labels_pred.sigmoid()
        labels_pred = torch.where(labels_pred > 0.5, torch.ones_like(labels_pred), torch.zeros_like(labels_pred))
        self.train_precision(labels_pred, labels)
        self.train_recall(labels_pred, labels)
        self.train_acc(labels_pred, labels)
        self.train_f1(labels_pred, labels)
        self.log('train/loss', self.train_loss, on_step=False, on_epoch=True, batch_size=1, prog_bar=True)
        self.log('train/auc', self.train_auc, on_step=False, on_epoch=True, batch_size=1, prog_bar=True)
        self.log('train/f1', self.train_f1, on_step=False, on_epoch=True, batch_size=1, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx: int) -> None:
        _, labels = batch
        loss_label, loss_str, labels_pred = self.model_step(batch)
        loss = loss_label + self._lambda * loss_str
        self.val_loss(loss.item())
        self.val_auc(labels_pred, labels)
        labels_pred = labels_pred.sigmoid()
        labels_pred = torch.where(labels_pred > 0.5, torch.ones_like(labels_pred), torch.zeros_like(labels_pred))
        self.val_precision(labels_pred, labels)
        self.val_recall(labels_pred, labels)
        self.val_acc(labels_pred, labels)
        self.val_f1(labels_pred, labels)
        self.log('val/loss', self.val_loss, on_step=False, on_epoch=True, batch_size=1, prog_bar=True)
        self.log('val/auc', self.val_auc, on_step=False, on_epoch=True, batch_size=1, prog_bar=True)
        self.log('val/f1', self.val_f1, on_step=False, on_epoch=True, batch_size=1, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        auc = self.val_auc.compute()  # get current val acc
        self.val_auc_best(auc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/auc_best", self.val_auc_best.compute(), sync_dist=True, prog_bar=True)

    def test_step(self, batch, batch_idx: int) -> None:
        graph, labels = batch
        loss_label, loss_str, labels_pred = self.model_step(batch)
        loss = loss_label + self._lambda * loss_str
        self.test_loss(loss.item())
        self.test_auc(labels_pred, labels)
        labels_pred = labels_pred.sigmoid()
        labels_pred = torch.where(labels_pred > 0.5, torch.ones_like(labels_pred), torch.zeros_like(labels_pred))
        self.test_precision(labels_pred, labels)
        self.test_recall(labels_pred, labels)
        self.test_acc(labels_pred, labels)
        self.test_f1(labels_pred, labels)
        # logging
        self.log('test/auc', self.test_auc, on_step=False, on_epoch=True, batch_size=1, prog_bar=True)
        self.log('test/f1', self.test_f1, on_step=False, on_epoch=True, batch_size=1, prog_bar=True)
        self.log('test/acc', self.test_acc, on_step=False, on_epoch=True, batch_size=1, prog_bar=True)
        self.log('test/precision', self.test_precision, on_step=False, on_epoch=True, batch_size=1, prog_bar=True)
        self.log('test/recall', self.test_recall, on_step=False, on_epoch=True, batch_size=1, prog_bar=True)
        self.log('test/loss', self.test_loss, on_step=False, on_epoch=True, batch_size=1, prog_bar=True)

        # get the h_graph, and save it for later visualization
        with torch.no_grad():
            h_graph = self.get_h_graph(graph)
            h_graph = h_graph.squeeze()
            h_graph = self.graph_classifier.avg_pool(graph, h_graph)
            h_graph = self.graph_classifier.dropout(h_graph)
            h_graph = F.relu(h_graph)
            h_graph = h_graph.detach().cpu().numpy()
            self.test_h_graph_list.extend(h_graph.tolist())
            self.test_label_list.extend(labels.detach().int().cpu().numpy().tolist())


    # def on_test_end(self) -> None:
    #     super().on_test_end()
    #     print('final lambda', self._lambda)
    #     h_graph_list = np.array(self.test_h_graph_list)  # [N, d]
    #     labels = np.array(self.test_label_list)
    #     # 降维
    #     h_graph_list = TSNE(n_components=2).fit_transform(h_graph_list)  # [N, 2]
    #     # visualize the scatter with different colors
    #     colors = np.array(['r', 'b'])
    #     # 创建图形
    #     # fig = plt.figure()
    #     # ax = fig.add_subplot(111, projection='3d')
    #
    #     # 绘制3D散点图
    #     # ax.scatter(h_graph_list[:, 0], h_graph_list[:, 1], h_graph_list[:, 2], c=colors[labels])
    #
    #     # 设置坐标轴标签
    #     # ax.set_xlabel('X Label')
    #     # ax.set_ylabel('Y Label')
    #     # ax.set_zlabel('Z Label')
    #     plt.scatter(h_graph_list[:, 0], h_graph_list[:, 1], c=colors[labels])
    #     # save fig
    #     plt.savefig('tsne_visualization.png')

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.parameters())
        return {"optimizer": optimizer}
