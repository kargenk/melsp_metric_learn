import torch
import torch.nn as nn
from torchinfo import summary


class ConvBN(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, strides=1, padding=0, bias=True, padding_mode='replicate'):
        super().__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=dim_in,
                      out_channels=dim_out,
                      kernel_size=kernel_size,
                      stride=strides,
                      padding=padding,
                      bias=bias),
            nn.BatchNorm2d(dim_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_layer(x)


class MelspMap(nn.Module):
    """メルスペクトログラムを入力として誰の声かの分類結果を出力するネットワーク."""
    def __init__(self):
        super().__init__()
        self.conv_branch1 = nn.Sequential(
            ConvBN(1, 32, kernel_size=(3, 1), strides=(1, 1), padding=(1, 0)),
            ConvBN(32, 32, kernel_size=(1, 3), strides=(1, 1), padding=(0, 1)),
            ConvBN(32, 64, kernel_size=(3, 1), strides=(1, 1), padding=(1, 0)),
            ConvBN(64, 64, kernel_size=(1, 3), strides=(1, 1), padding=(0, 1)),
        )
        self.conv_branch2 = nn.Sequential(
            ConvBN(1, 32, kernel_size=(9, 1), strides=(1, 1), padding=(4, 0)),
            ConvBN(32, 32, kernel_size=(1, 9), strides=(1, 1), padding=(0, 4)),
            ConvBN(32, 64, kernel_size=(9, 1), strides=(1, 1), padding=(4, 0)),
            ConvBN(64, 64, kernel_size=(1, 9), strides=(1, 1), padding=(0, 4)),
        )
        self.conv_branch3 = nn.Sequential(
            ConvBN(1, 32, kernel_size=(21, 1), strides=(1, 1), padding=(10, 0)),
            ConvBN(32, 32, kernel_size=(1, 21), strides=(1, 1), padding=(0, 10)),
            ConvBN(32, 64, kernel_size=(21, 1), strides=(1, 1), padding=(10, 0)),
            ConvBN(64, 64, kernel_size=(1, 21), strides=(1, 1), padding=(0, 10)),
        )
        self.conv_branch4 = nn.Sequential(
            ConvBN(1, 32, kernel_size=(39, 1), strides=(1, 1), padding=(19, 0)),
            ConvBN(32, 32, kernel_size=(1, 39), strides=(1, 1), padding=(0, 19)),
            ConvBN(32, 64, kernel_size=(39, 1), strides=(1, 1), padding=(19, 0)),
            ConvBN(64, 64, kernel_size=(1, 39), strides=(1, 1), padding=(0, 19)),
        )
        self.conv_after = nn.Sequential(
            ConvBN(256, 128, kernel_size=(5, 1), strides=(1, 1), padding=(2, 0)),
            ConvBN(128, 128, kernel_size=(5, 1), strides=(1, 1), padding=(2, 0)),
        )
        self.classification = nn.Sequential(
            nn.Linear(128, 7),
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        b1 = self.conv_branch1(x)
        b2 = self.conv_branch2(x)
        b3 = self.conv_branch3(x)
        b4 = self.conv_branch4(x)

        # それぞれのブランチをチャネル方向で結合, [N, 256, 80, 128]
        concat = torch.cat([b1, b2, b3, b4], dim=1)

        # 畳み込みを行なったのち, Global Average Poolingで各チャネルごとの値を一つに集約
        after = self.conv_after(concat)
        after_gap = torch.mean(after, dim=(2, 3))  # [N, 128, 80, 128] -> [N, 128]

        # out = self.classification(after_gap)  # for classification
        return after_gap


if __name__ == '__main__':
    input_tensor = torch.rand(1, 1, 80, 128)
    model = MelspMap()
    summary(model, input_size=input_tensor.shape)
    out = model(input_tensor)
    print(out.shape)
