import pathlib
from pathlib import Path

import numpy as np
from sklearn.preprocessing import LabelBinarizer
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class MelspMetricDataset(Dataset):

    def __init__(self, melsp_dir: pathlib.PosixPath,
                 speakers: list = ['jvs001', 'jvs010', 'jvs015', 'jvs018', 'jvs037', 'jvs076'],
                 train: bool = True):
        self.melsp_paths = self._make_melsp_paths(melsp_dir)
        self.n_melsp = len(self.melsp_paths)
        self.train = train

        self._speakers = speakers
        self.encoder = LabelBinarizer().fit(self._speakers)

    def __len__(self):
        return self.n_melsp

    def __getitem__(self, index: int):
        melsp_path = self.melsp_paths[index]
        out_melsp = self._load_melsp(melsp_path)
        out_label = self._make_label(melsp_path)
        # print(melsp_path)

        return out_melsp, out_label, melsp_path.stem[:6]

    def _make_label(self, melsp_path: pathlib.PosixPath) -> torch.Tensor:
        speaker = melsp_path.stem[:6]
        if speaker in self._speakers:
            return torch.tensor(1)  # real
        else:
            return torch.tensor(0)  # fake

    # ================================================================================
    #
    # Instance Method
    #
    # ================================================================================

    @staticmethod
    def _make_melsp_paths(melsp_dir: pathlib.PosixPath) -> list:
        # npyのみを対象とする
        return [path for path in tqdm(list(melsp_dir.glob('**/*.npy')))]

    @staticmethod
    def _load_melsp(melsp_path: pathlib.PosixPath) -> torch.Tensor:
        melsp = np.load(melsp_path)
        return torch.from_numpy(melsp).unsqueeze(0)


if __name__ == '__main__':
    root = Path.cwd().joinpath('data/log_melsp')
    print(root)

    dataset = MelspMetricDataset(root)
    print(len(dataset))
    melsp, label, speaker = dataset[0]
    print(melsp.shape, label, speaker)

    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    num_epoch = 1
    for i in range(1, num_epoch + 1):
        for melsp, label, speaker in tqdm(dataloader):
            print(melsp.shape, label, speaker)
            break
