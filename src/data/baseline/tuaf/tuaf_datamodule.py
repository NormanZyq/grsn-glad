# %%
from typing import Any, Dict, Optional, Tuple

from dgl.data.utils import split_dataset
from lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader

from src.data.baseline.tuaf.tuaf_dataset import TUAFDataset, GraphSampler


class TUAFDataModule(LightningDataModule):
    def __init__(
            self,
            name: str = 'AIDS',
            dsl: int = 0,
            down_sample_rate: float = 0.1,
            default_feat_dim: int = -1,
            re_gen_ds_labels=False,
            data_dir: str = "data/",
            train_val_test_split: Tuple[int, int, int] = (0.7, 0.2, 0.1),
            shuffle: bool = False,
            seed: int = 12345,
            batch_size: int = 4,
            num_workers: int = 0,
            pin_memory: bool = False,
    ) -> None:
        """Initialize a `DFDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_val_test_split: The train, validation and test split. Defaults to `(55_000, 5_000, 10_000)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()
        # self.data_dir = os.path.join(data_dir, '', 'raw')

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.name = name  # data name
        self.down_sample_label = dsl
        self.down_sample_rate = down_sample_rate
        self.re_gen_ds_labels = re_gen_ds_labels
        self.default_feat_dim = default_feat_dim

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            dataset = TUAFDataset(name=self.name,
                                  down_sample_label=self.down_sample_label,
                                  down_sample_rate=self.down_sample_rate,
                                  re_gen_ds_labels=self.re_gen_ds_labels,
                                  default_feat_dim=self.default_feat_dim,
                                  seed=self.hparams.seed)
            data_train, data_val, data_test = split_dataset(
                dataset=dataset,
                frac_list=self.hparams.train_val_test_split,
                shuffle=self.hparams.shuffle,
                random_state=12345
            )
            # 进一步的sample
            train_graphs = []
            val_graphs = []
            test_graphs = []
            for idx in data_train.indices:
                # Using the normal graphs in the training process --- authors
                if dataset[idx][1] != self.down_sample_label:
                    train_graphs.append(dataset[idx][0])
            for idx in data_val.indices:
                val_graphs.append(dataset[idx][0])
            for idx in data_test.indices:
                test_graphs.append(dataset[idx][0])

            self.data_train = GraphSampler(train_graphs, normalize=True, max_num_triples=dataset.max_triples_num)
            self.data_val = GraphSampler(val_graphs, normalize=True, max_num_triples=dataset.max_triples_num)
            self.data_test = GraphSampler(test_graphs, normalize=True, max_num_triples=dataset.max_triples_num)

        # ----------------------- do some statistical things -----------------------
        num_train_anomaly = 0
        num_val_anomaly = 0
        num_test_anomaly = 0
        for s in self.data_train:
            if s['graph_label'][0] == self.down_sample_label:
                num_train_anomaly += 1
        for s in self.data_val:
            if s['graph_label'][0] == self.down_sample_label:
                num_val_anomaly += 1
        for s in self.data_test:
            if s['graph_label'][0] == self.down_sample_label:
                num_test_anomaly += 1
        print('''
        -------------
        Anomalies in train: {}, All: {}, rate={:.2%}
        Anomalies in val: {}, All: {}, rate={:.2%}
        Anomalies in test: {}, All: {}, rate={:.2%}
        -------------
        '''.format(num_train_anomaly, len(self.data_train), num_train_anomaly / len(self.data_train),
                   num_val_anomaly, len(self.data_val), num_val_anomaly / len(self.data_val),
                   num_test_anomaly, len(self.data_test), num_test_anomaly / len(self.data_test)
                   ))
        # ----------------------- end of that -----------------------

    def train_dataloader(self) -> DataLoader:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass
