# ================================================================================
#                              DataLoader
# ================================================================================
# Scans dataset creating x, y pairs using the window size
# Allows for x to be shorter than WINDOW_SIZE by padding with pad_token
# This ensures the model can predict even with fewer chords preventing cold start
# ================================================================================


import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from src.config import WINDOW_SIZE, BATCH_SIZE

TEST_SPLIT = 0.2


class ChordDataset(Dataset):
    def __init__(self, songs, vocabulary, window_size=WINDOW_SIZE):
        self.vocabulary = vocabulary
        self.window_size = window_size

        self.samples = []  # list of (song_indices, start, target)
        pad_idx = self.vocabulary.chord_to_idx[self.vocabulary.pad_token]

        for song in tqdm(songs, desc="Preparing index"):
            if len(song) < 2:
                continue

            song_indices = list(self.vocabulary.to_indices(song))
            padded_song = [pad_idx] * (self.window_size - 1) + song_indices

            num_targets = len(song_indices) - 1
            max_start = len(padded_song) - self.window_size  # last valid start

            for i in range(num_targets):
                if i > max_start:
                    break  # can't have a complete window
                target = song_indices[i + 1]
                self.samples.append((padded_song, i, target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        padded_song, start, target = self.samples[idx]
        window = padded_song[start : start + self.window_size]

        # security: assert that the window has always seq_len
        assert len(window) == self.window_size, f"Got window len {len(window)}"

        x = torch.tensor(window, dtype=torch.long)
        y = torch.tensor(target, dtype=torch.long)
        return x, y



def create_loaders(songs, vocabulary, batch_size=BATCH_SIZE, window_size=WINDOW_SIZE, test_split=TEST_SPLIT, num_workers=0, pin_memory=False,):
    dataset = ChordDataset(songs, vocabulary, window_size)

    # Split into train and test
    test_size = int(len(dataset) * test_split)
    train_size = len(dataset) - test_size

    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else None,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=4 if num_workers > 0 else None,
    )

    return train_loader, test_loader