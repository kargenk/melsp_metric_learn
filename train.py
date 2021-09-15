from pathlib import Path

import numpy as np
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.losses import TripletMarginLoss
from pytorch_metric_learning.miners import TripletMarginMiner
from pytorch_metric_learning.reducers import ThresholdReducer
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader import MelspMetricDataset
from logger import Logger
from model import MelspMap

def train(model, criterion, miner, dataloader, optimizer, logger, epoch, device):
    model.train()
    for i, (inputs, labels, _) in enumerate(tqdm(dataloader)):
        optimizer.zero_grad()
        inputs, labels = inputs.to(device), labels.to(device)
        embeddings = model(inputs)
        indices = miner(embeddings, labels)
        loss = criterion(embeddings, labels, indices)
        loss.backward()
        optimizer.step()
        # TensorBoardのログに表示
        logger.scalar_summary(f'train/loss', loss, i)
        if i % 10 == 0 and i != 0:
            print(f'Epoch {epoch} Iteration {i}: Loss = {loss:.4f}, Number of mined triplets = {miner.num_triplets}')
    print()

def test(model, dataloader, epoch, device):
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
    epochs = 100
    learning_rate = 1e-4
    batch_size = 8
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

    # データセットの準備
    train_dir = Path.cwd().joinpath('data/log_melsp/train/')
    test_dir = Path.cwd().joinpath('data/log_melsp//test/')
    train_dataset = MelspMetricDataset(train_dir)
    test_dataset = MelspMetricDataset(test_dir)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # モデルと損失関数，最適化手法
    model = MelspMap().to(device)
    logger = Logger('logs')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    distance = CosineSimilarity()
    reducer = ThresholdReducer(low=0)
    criterion = TripletMarginLoss(margin=0.2, distance=distance, reducer=reducer)
    miner = TripletMarginMiner(margin=0.2, distance=distance)

    test_predicted_metrics = []
    test_true_labels = []

    for epoch in range(1, epochs + 1):
        print(f'Epoch {epoch}/{epochs}')
        print('-' * 20)

        train(model, criterion, miner, train_loader, optimizer, logger, epoch, device)
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f'models/ep{epoch}.pt')
        # _temp_metrics, _temp_labels = test(model, test_loader, epoch, device)
        # test_predicted_metrics.append(_temp_metrics)
        # test_true_labels.append(_temp_labels)
