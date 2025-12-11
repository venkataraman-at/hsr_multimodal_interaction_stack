# model_crnn.py
import torch
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, num_classes=9):
        super(CRNN, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
        )

        # LSTM input size = (channels * mel_after_pooling)
        # mel_after_pooling = 64 mel becomes 8 mel after 3 poolings
        lstm_input_size = 128 * 8   # 128 channels * 8 mel bins

        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )

        self.fc = nn.Sequential(
            nn.Linear(128*2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x = (B, 1, 64, T)
        x = self.cnn(x)  # → (B, 128, 8, T')

        # rearrange: (B, 128, 8, T') → (B, T', 128*8)
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, T', 128, 8)
        x = x.view(x.size(0), x.size(1), -1)    # (B, T', 1024)

        out, _ = self.lstm(x)  # (B, T', 256)

        # last time-step
        out = out[:, -1, :]

        return self.fc(out)

