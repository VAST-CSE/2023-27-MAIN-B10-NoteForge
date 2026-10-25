
import torch
import torch.nn as nn


class PolyphonicAcousticModel(nn.Module):

    def __init__(self, input_bins=229, num_keys=88):
        super().__init__()

        # 1. CNN: Extract patterns from the Mel-spectrogram
        self.cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=48,
                kernel_size=(3, 3),
                padding=1
            ),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),

            nn.Conv2d(
                in_channels=48,
                out_channels=48,
                kernel_size=(3, 3),
                padding=1
            ),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2))
        )

        # Two pooling layers reduce 229 Mel bins to 57
        cnn_out_size = 48 * (input_bins // 4)

        # 2. Two-layer Bidirectional LSTM
        self.rnn = nn.LSTM(
            input_size=cnn_out_size,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

        # 3. Onset head: Predict when notes begin
        self.onset_head = nn.Sequential(
            nn.Linear(512, num_keys),
            nn.Sigmoid()
        )

        # 4. Frame head: Predict which notes are active
        self.frame_head = nn.Sequential(
            nn.Linear(512, num_keys),
            nn.Sigmoid()
        )

    def forward(self, mel_spectrogram):

        # Expected input: [Batch, 1, Time, 229]
        x = self.cnn(mel_spectrogram)

        batch_size, channels, time_frames, freq_bins = x.shape

        # Rearrange dimensions for the LSTM
        x = x.permute(0, 2, 1, 3).contiguous()

        # Combine CNN channels and frequency features
        x = x.reshape(
            batch_size,
            time_frames,
            channels * freq_bins
        )

        # Learn temporal patterns
        rnn_out, _ = self.rnn(x)

        # Generate note predictions
        onsets = self.onset_head(rnn_out)
        frames = self.frame_head(rnn_out)

        return onsets, frames
