import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm  # tqdm 추가
import scipy.sparse as sp
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import svds

from hyperopt import hp
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.search.hyperopt import HyperOptSearch

from datasets import Dataset
import layers
import metrics

# 디바이스 설정 (MPS 지원)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class Model(nn.Module):
    def __init__(self, dataset, **config):
        super(Model, self).__init__()
        self.dataset = dataset
        self.config = config
        self._init_params()

        self.alpha = float(self.config.get('alpha', 0))
        self.beta = float(self.config.get('beta', 0))
        self.normed = self.config.get('normed', False)
        self.sparse = self.config.get('sparse', True)

        self.user_graph = self.dataset.side_info.get('user_graph', None)
        self.item_graph = self.dataset.side_info.get('item_graph', None)

        self.user_lap = self._prepare_laplacian(self.user_graph) if self.user_graph is not None else None
        self.item_lap = self._prepare_laplacian(self.item_graph) if self.item_graph is not None else None

        self.range = self.dataset.range
        self.min_val = self.dataset.min

    def _init_params(self):
        pass

    def _prepare_laplacian(self, g):
        # SciPy sparse laplacian 계산 후 torch tensor 변환
        s = laplacian(sp.coo_matrix(g), normed=self.normed).astype(np.float32)
        s_coo = sp.coo_matrix(s)
        return torch.tensor(s_coo.toarray(), dtype=torch.float32, device=device)

    def forward(self, user_id, item_id):
        r_pred = self.min_val + self.range * torch.sigmoid(self._r_pred(user_id, item_id))
        return r_pred

    def _r_pred(self, user_id, item_id):
        raise NotImplementedError()

    def loss_fn(self, user_id, item_id, r_true):
        r_pred = self.forward(user_id, item_id)
        mse = ((r_pred - r_true)**2).mean()
        reg = self._reg()
        loss = (1 - self.alpha) * mse / (self.range**2) + self.alpha * reg
        return loss, mse

    def _reg(self):
        reg_l2 = 0.0
        reg_graph = 0.0
        for name, param in self.named_parameters():
            if param.requires_grad:
                reg_l2 += torch.sum(param**2)
                if 'user_factor' in name and self.user_lap is not None:
                    reg_graph += self._graph_reg(self.user_lap, param)
                elif 'item_factor' in name and self.item_lap is not None:
                    reg_graph += self._graph_reg(self.item_lap, param)
                else:
                    reg_graph += torch.sum(param**2)
        return (1 - self.beta) * reg_l2 + self.beta * reg_graph

    def _graph_reg(self, lap, w):
        if w.dim() == 1:
            w = w.unsqueeze(1)
        return torch.trace(w.t() @ lap @ w)

    def train_model(self, max_updates=100000, n_check=100, patience=float('inf'), batch_size=None):
        optimizer = optim.Adam(self.parameters())
        best = {'updates': 0, 'loss': float('inf'), 'rmse_tune': float('inf')}

        for i in tqdm(range(max_updates), desc="Training Progress"):
            batch = self.dataset.get_batch(mode='train', size=batch_size)
            user_id = torch.tensor(batch.user_id.values, dtype=torch.long, device=device)
            item_id = torch.tensor(batch.item_id.values, dtype=torch.long, device=device)
            r_true = torch.tensor(batch.rating.values, dtype=torch.float32, device=device)

            optimizer.zero_grad()
            loss, _ = self.loss_fn(user_id, item_id, r_true)
            loss.backward()
            optimizer.step()

            if i % n_check == 0 or i == max_updates - 1:
                rmse_tune = self.evaluate_rmse(mode='tune')
                if len(self.dataset.tune) == 0 or rmse_tune < best['rmse_tune']:
                    rmse_test = self.evaluate_rmse(mode='test')
                    best = {'updates': i, 'loss': loss.item(), 'rmse_tune': rmse_tune, 'rmse_test': rmse_test}
                tqdm.write(str(best))
                if (i - best['updates']) // n_check > patience:
                    break
        return best

    def evaluate_rmse(self, mode='test'):
        if len(getattr(self.dataset, mode)) == 0:
            return float('inf')
        batch = self.dataset.get_batch(mode=mode, size=None)
        if len(batch) == 0:
            return float('inf')
        user_id = torch.tensor(batch.user_id.values, dtype=torch.long, device=device)
        item_id = torch.tensor(batch.item_id.values, dtype=torch.long, device=device)
        r_true = batch.rating.values
        user_ids = batch.user_id.values
        item_ids = batch.item_id.values
        r_pred = self.forward(user_id, item_id).detach().cpu().numpy()
        return metrics.rmse(r_pred, r_true, user_ids, item_ids)

    def test_model(self):
        batch = self.dataset.get_batch(mode='test', size=None)
        user_id = torch.tensor(batch.user_id.values, dtype=torch.long, device=device)
        item_id = torch.tensor(batch.item_id.values, dtype=torch.long, device=device)
        r_true = batch.rating.values
        user_ids = batch.user_id.values
        item_ids = batch.item_id.values
        r_pred = self.forward(user_id, item_id).detach().cpu().numpy()
        return {
            'rmse': metrics.bootstrap(metrics.rmse, r_pred, r_true, user_ids, item_ids)
        }



class SVD(Model):
    def _init_params(self):
        self.rank = int(self.config.get('rank', 1))
        self.user_factors = nn.Embedding(self.dataset.n_user, self.rank)
        self.item_factors = nn.Embedding(self.dataset.n_item, self.rank)
        self.user_bias = nn.Embedding(self.dataset.n_user, 1)
        self.item_bias = nn.Embedding(self.dataset.n_item, 1)
        self.bias = nn.Parameter(torch.zeros(1))

        nn.init.normal_(self.user_factors.weight, std=0.01)
        nn.init.normal_(self.item_factors.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def _r_pred(self, user_id, item_id):
        uf = self.user_factors(user_id)
        if_ = self.item_factors(item_id)
        ub = self.user_bias(user_id).squeeze()
        ib = self.item_bias(item_id).squeeze()
        return (uf * if_).sum(dim=1) + ub + ib + self.bias


if __name__ == '__main__':
    MODEL = SVD
    CWD = os.getcwd()
    DATASET_PATH = CWD + '/data/datasets/Movielens100K'
    METRICS_PATH = CWD + '/data/results/metrics'
    BATCH_SIZE = None
    DATASET = Dataset.load(DATASET_PATH)
    NAME = '{}_{}'.format(MODEL.__name__, DATASET.name)

    config = {
        'alpha': 0.01,
        'beta': 0.01,
        'rank': 10,
        'max_updates': 10000,
    }

    data = DATASET.data[['user_id', 'item_id', 'rating', 'is_test']]
    dataset = Dataset(data, **DATASET.side_info)

    model = MODEL(dataset, **config).to(device)
    model.train_model(max_updates=config['max_updates'], batch_size=BATCH_SIZE)
    results = model.test_model()
    print(pd.DataFrame(results).describe())
    pd.DataFrame(results).to_csv('{}/{}.csv'.format(METRICS_PATH, NAME), index=False)
