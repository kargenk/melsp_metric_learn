from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader import MelspMetricDataset
from model import MelspMap

def draw_heatmap(data: np.ndarray, row_labels: list, col_labels: list):
    """
    ヒートマップを描画する関数.

    Args:
        data (np.ndarray): 正方行列
        row_labels (list): 行メモリ,sklearnのコサイン類似度行列ならx_feature
        column_labels (list): 列メモリ,sklearnのコサイン類似度行列ならy_feature

    Returns:
        [type]: [description]
    """
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(data, cmap=plt.cm.Blues)

    # 画素値の中央にメモリがくるように調整
    ax.set_xticks(np.arange(data.shape[0]) + 0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[1]) + 0.5, minor=False)

    # y軸を上から下に
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    # 軸メモリの指定
    ax.set_xticklabels(row_labels, minor=False)
    ax.set_yticklabels(col_labels, minor=False)

    # タイトル
    ax.set_xlabel('y_feature')
    ax.set_ylabel('x_feature')
    plt.title('cosine simularity')

    plt.colorbar(heatmap)
    plt.savefig('cossim_matrix.png')

    return heatmap

def cossim_matrix(x, y, row_labels, col_labels):
    cossim = cosine_similarity(x, y)
    draw_heatmap(cossim, row_labels=row_labels, col_labels=col_labels)
    print(row_labels)
    print(col_labels)
    print(cossim)

def test(model, dataloader, device):
    model.eval()
    _predicted_metrics = []
    _true_labels = []
    with torch.inference_mode():
        for i, (inputs, labels, _) in enumerate(tqdm(dataloader)):
            inputs, labels = inputs.to(device), labels.to(device)
            metric = model(inputs).detach().cpu().numpy()
            metric = metric.reshape(metric.shape[0], metric.shape[1])
            _predicted_metrics.append(metric)
            _true_labels.append(labels.detach().cpu().numpy())
    return np.concatenate(_predicted_metrics), np.concatenate(_true_labels)

if __name__ == '__main__':
    torch.manual_seed(42)
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    batch_size = 128

    test_dir = Path.cwd().joinpath('data/log_melsp/for_cossim/')
    test_dataset = MelspMetricDataset(test_dir)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # モデルの読み込み
    model = MelspMap()
    weights = torch.load('models/ep100.pt')
    model.load_state_dict(weights)
    model.to(device)

    # テストデータの写像と真のラベル
    test_predicted_metrics, test_true_labels = test(model, test_loader, device)

    # コサイン類似度をみてみる
    cossim_matrix(test_predicted_metrics[:12], test_predicted_metrics[12:],
                  row_labels=test_true_labels[12:], col_labels=test_true_labels[:12])

    # # tSNEで2次元に
    # tSNE_metrics = TSNE(n_components=2, random_state=42).fit_transform(test_predicted_metrics)

    # # プロット
    # plt.scatter(tSNE_metrics[:, 0], tSNE_metrics[:, 1], c=test_true_labels)
    # plt.colorbar()
    # plt.savefig('train_test_output.png')
