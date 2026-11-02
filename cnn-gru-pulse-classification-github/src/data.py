import torch
from torch.utils.data import Dataset, DataLoader


class PulseDataset(Dataset):
    """Store processed pulse waveforms and class labels."""

    def __init__(self, features, labels):
        self.features = torch.as_tensor(
            features, dtype=torch.float32
        )
        self.labels = torch.as_tensor(
            labels, dtype=torch.long
        ).reshape(-1)

        if self.features.ndim != 3 or self.features.shape[-1] != 3:
            raise ValueError(
                "Features must have shape [samples, time, 3]."
            )

        if len(self.features) != len(self.labels):
            raise ValueError(
                "Features and labels must have the same length."
            )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]


def build_dataloader(
    features,
    labels,
    batch_size=8,
    shuffle=True,
):
    dataset = PulseDataset(features, labels)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )