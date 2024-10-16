# models/cnn_model.py
import torch.nn as nn
import torch.nn.functional as F


class DNN(nn.Module):
    def __init__(self, args):
        super(DNN, self).__init__()
        input_size = args.image_size * args.image_size * 3
        layers = []
        # 输入层
        layers.extend([
            nn.Linear(input_size, 512),
            nn.ReLU(inplace=True),
        ])
        # 动态设置隐藏层（减去输入层）
        in_channels = 512
        for i in range(args.num_layers - 1):
            layers.extend([
                nn.Linear(in_channels, in_channels // 2),
                nn.ReLU(inplace=True),
            ])
            in_channels //= 2
        # 输出层
        layers.append(nn.Linear(in_channels, args.num_classes))
        self.dnn = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平
        x = self.dnn(x)
        return x


class CNN(nn.Module):
    def __init__(self, args):
        super(CNN, self).__init__()

        in_channels, out_channels = 3, 32
        layers = []
        # 动态创建卷积层
        for i in range(args.num_layers):
            layers.extend([
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            ])
            in_channels = out_channels
            out_channels *= 2
        self.conv = nn.Sequential(*layers)

        # 计算全连接层的输入特征数
        image_size = args.image_size // pow(2, args.num_layers)
        self.feature_size = image_size * image_size * in_channels

        # 输出层
        self.fc = nn.Linear(in_features=self.feature_size, out_features=args.num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc(x)
        return x


class RNN(nn.Module):
    def __init__(self, args):
        super(RNN, self).__init__()
        # 图像展平成一维: height * width * channels
        input_size = args.image_size * args.image_size * 3  # 将图像展平成一维
        hidden_size = 128

        # 根据 args.num_layers 动态设置 LSTM 的层数
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=args.num_layers, batch_first=True)

        # 输出层
        self.fc = nn.Linear(hidden_size, args.num_classes)

    def forward(self, x):
        # (batch_size, channels, height, width) -> (batch_size, 1, channels * height * width)
        x = x.view(x.size(0), 1, -1)
        lstm_out, (hn, cn) = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])

        return out
