from torch import nn


class CNNGRU(nn.Module):
    """Classify three-channel pulse waveforms.

    Input:  [batch_size, sequence_length, 3]
    Output: [batch_size, 11]
    """

    def __init__(
        self,
        input_size=3,
        hidden_size=256,
        num_classes=11,
    ):
        super().__init__()

        # Extract local waveform features and shorten the sequence.
        self.cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=input_size,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.MaxPool1d(2),
            nn.MaxPool1d(2),
        )

        # Learn temporal patterns in the extracted features.
        self.gru = nn.GRU(
            input_size=64,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # [batch, time, channels] -> [batch, channels, time]
        x = x.transpose(1, 2)

        x = self.cnn(x)

        # Convert back to the sequence format expected by GRU.
        x = x.transpose(1, 2)

        sequence_output, _ = self.gru(x)
        last_output = sequence_output[:, -1, :]

        return self.classifier(last_output)