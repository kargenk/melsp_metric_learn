from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm

from dataloader import MelspMetricDataset
from model import MelspMap

class MelspMetric(nn.Module):
    """
    メルスペクトログラムを128次元の特徴ベクトルに落とし込み，
    ターゲットのリアルデータ集合のうち最近傍のデータ点との距離を損失として返すクラス.
    """
    def __init__(self, model=None, dataloader=None, metrics=None):
        super().__init__()
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(device).eval()
        self.dataloader = dataloader
        self.metrics = NearestNeighbors  # sklearnのニアレストネイバー
        self.speakers = ['jvs001', 'jvs010', 'jvs015', 'jvs018', 'jvs037', 'jvs076']
        self.real_data = {}
        self.nn_models = {}

        # リアルデータ集合とNearest Neighborのモデルを作成
        self._store_real_data()
        self._make_nnmodels()

    def _store_real_data(self):
        print('storing real data ...')
        for speaker in self.speakers:
            self.real_data[speaker] = []

        for melsps, labels, speakers in tqdm(self.dataloader):
            melsps = melsps.to(self.device)
            feature = self.model(melsps).squeeze().detach().cpu().numpy()
            for i, s in enumerate(speakers):
                self.real_data[s].append(feature[i])

    def _make_nnmodels(self):
        """最近傍法に用いるターゲットごとのデータ集合の辞書を返す"""
        print('make nearest neighbor models ...')
        for speaker in self.speakers:
            X = self.real_data[speaker]
            nn_model = self.metrics(n_neighbors=1, algorithm='ball_tree').fit(X)
            self.nn_models[speaker] = nn_model

    def nearest_neighbor(self, melsps: np.ndarray, target_speakers: torch.Tensor) -> torch.Tensor:
        """
        ニアレストネイバー法を用いてリアルのデータ集合内の最近傍点との距離を返す.

        Args:
            melsps (np.ndarray): メルスペクトログラム
            target_speakers (torch.Tensor): 変換先の話者

        Returns:
            torch.Tensor: 最近傍点との距離
        """
        distances = []
        for melsp, target_speaker in zip(melsps, target_speakers):
            feature = self.model(melsp.unsqueeze(dim=0)).squeeze().detach().cpu().numpy()
            nn_model = self.nn_models[target_speaker]
            _distance, _indices = nn_model.kneighbors([feature])
            distances.append(_distance)
        # テスト用
        # X = self.real_data['test']
        # nn_model = self.metrics(n_neighbors=1, algorithm='ball_tree').fit(X)
        # distance, indices = nn_model.kneighbors([[2, 3]])

        # plt.figure()
        # plt.title('Nearest neighbors')
        # plt.scatter(self.real_data['test'][:, 0], self.real_data['test'][:, 1], marker='o', s=75, color='k')
        # plt.scatter(self.real_data['test'][indices][0][:][:, 0], self.real_data['test'][indices][0][:][:, 1],
        #             marker='o', s=250, color='k', facecolors='none')
        # plt.scatter(2, 3, marker='x', s=75, color='k')
        # plt.savefig('nn_test.png')

        return distances

if __name__ == '__main__':
    torch.manual_seed(42)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    data_root = Path.cwd().joinpath('data/log_melsp/train/origin/')

    dataset = MelspMetricDataset(data_root)
    metric_dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    melsp_dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    metric = MelspMetric(model=MelspMap(), dataloader=metric_dataloader)

    num_epoch = 1
    for i in range(1, num_epoch + 1):
        for melsp, label, speaker in tqdm(melsp_dataloader):
            melsp = melsp.to(device)
            loss_nn = metric.nearest_neighbor(melsps=melsp, target_speakers=speaker)
            print(loss_nn)
            break
