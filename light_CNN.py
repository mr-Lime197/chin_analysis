"""
Audio classification with a Mel-Spectrogram + CNN in PyTorch.

Includes:
  - load_audio_ffmpeg: decodes any audio file via the ffmpeg binary
    directly (subprocess), sidestepping torchaudio's I/O backends
    (torchcodec / soundfile / sox) entirely
  - MelSpecAudioDataset: loads waveforms, converts to log-mel spectrograms
  - AudioCNN: a small CNN classifier (<10M parameters)
  - train_one_epoch / validate: single-epoch loops
  - EarlyStopping: stops training when validation loss stops improving
  - fit: full training loop tying everything together
  - InferenceMelSpecDataset / predict: run the trained model on a plain
    list of audio file paths (no labels required)

Requires the ffmpeg binary to be installed and available on PATH
(e.g. `apt-get install ffmpeg` / `brew install ffmpeg`). torchaudio is
still used, but only for its pure-tensor MelSpectrogram/AmplitudeToDB
transforms, which don't touch any audio I/O backend.

Usage: build a list of (filepath, label) pairs for train/val, then call fit().
See the __main__ block at the bottom for a worked example.
"""

import copy
import shutil
import subprocess
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# --------------------------------------------------------------------------- #
# Audio loading via the ffmpeg binary (bypasses torchaudio I/O backends)
# --------------------------------------------------------------------------- #
def load_audio_ffmpeg(path, sample_rate=16000):
    """
    Decode any audio file ffmpeg understands into a mono float32 waveform,
    resampled to `sample_rate`, by shelling out to the ffmpeg binary and
    reading raw PCM off stdout.

    This avoids torchaudio.load() and its backend dependencies
    (torchcodec/soundfile/sox) altogether -- useful if those are giving
    you install/runtime trouble, since it only needs the `ffmpeg`
    executable on PATH.

    Returns
    -------
    torch.Tensor
        Shape (1, num_samples), mono, float32.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg binary not found on PATH. Install it first, "
            "e.g. `apt-get install -y ffmpeg` or `brew install ffmpeg`."
        )

    cmd = [
        "ffmpeg",
        "-v", "error",       # suppress ffmpeg's banner/progress noise
        "-i", str(path),
        "-f", "f32le",       # raw little-endian float32 PCM
        "-acodec", "pcm_f32le",
        "-ac", "1",           # downmix to mono
        "-ar", str(sample_rate),
        "-",
    ]

    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed to decode '{path}': {stderr}") from e

    audio = np.frombuffer(proc.stdout, dtype=np.float32)
    if audio.size == 0:
        raise RuntimeError(f"ffmpeg produced no audio data for '{path}' (empty/corrupt file?)")

    waveform = torch.from_numpy(audio.copy()).unsqueeze(0)  # (1, num_samples)
    return waveform


def _pad_or_trim(waveform, num_samples):
    if waveform.shape[1] < num_samples:
        pad = num_samples - waveform.shape[1]
        waveform = F.pad(waveform, (0, pad))
    else:
        waveform = waveform[:, :num_samples]
    return waveform


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class MelSpecAudioDataset(Dataset):
    """
    Generic audio classification dataset.

    Parameters
    ----------
    file_label_pairs : list[(str, int)]
        List of (audio_filepath, integer_label) pairs.
    sample_rate : int
        Target sample rate; files are resampled to this if needed.
    n_mels, n_fft, hop_length : int
        Mel-spectrogram parameters.
    duration : float
        Fixed clip length in seconds. Shorter clips are zero-padded,
        longer clips are truncated.
    augment : bool
        If True, applies SpecAugment-style frequency/time masking
        (use only for the training split).
    """

    def __init__(self, file_label_pairs, sample_rate=16000, n_mels=64,
                 n_fft=1024, hop_length=256, duration=4.0, augment=False):
        self.pairs = file_label_pairs
        self.sample_rate = sample_rate
        self.duration = duration
        self.num_samples = int(sample_rate * duration)
        self.augment = augment

        self.mel_spec = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        self.db_transform = T.AmplitudeToDB(top_db=80)

        if augment:
            self.freq_mask = T.FrequencyMasking(freq_mask_param=10)
            self.time_mask = T.TimeMasking(time_mask_param=20)

    def __len__(self):
        return len(self.pairs)

    def _load_audio(self, path):
        # ffmpeg handles decoding, mono downmix, and resampling in one go.
        waveform = load_audio_ffmpeg(path, sample_rate=self.sample_rate)
        return _pad_or_trim(waveform, self.num_samples)

    def __getitem__(self, idx):
        path, label = self.pairs[idx]
        waveform = self._load_audio(path)

        mel = self.mel_spec(waveform)          # (1, n_mels, T)
        mel_db = self.db_transform(mel)

        if self.augment:
            mel_db = self.freq_mask(mel_db)
            mel_db = self.time_mask(mel_db)

        # Per-sample normalization
        mean, std = mel_db.mean(), mel_db.std() + 1e-6
        mel_db = (mel_db - mean) / std

        return mel_db, label


# --------------------------------------------------------------------------- #
# Inference-only dataset: just a list of file paths, no labels
# --------------------------------------------------------------------------- #
class InferenceMelSpecDataset(Dataset):
    """
    Same preprocessing as MelSpecAudioDataset, but built from a plain list
    of file paths (no labels needed, since this is for prediction only).

    Files that fail to load (corrupt/missing/unsupported) are not skipped
    silently: __getitem__ returns (None, idx, error_message) for them, and
    the collate_fn below routes those into an "errors" bucket so `predict`
    can still report a result for every input path.
    """

    def __init__(self, file_paths, sample_rate=16000, n_mels=64,
                 n_fft=1024, hop_length=256, duration=4.0):
        self.file_paths = file_paths
        self.sample_rate = sample_rate
        self.duration = duration
        self.num_samples = int(sample_rate * duration)

        self.mel_spec = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        self.db_transform = T.AmplitudeToDB(top_db=80)

    def __len__(self):
        return len(self.file_paths)

    def _load_audio(self, path):
        # ffmpeg handles decoding, mono downmix, and resampling in one go.
        waveform = load_audio_ffmpeg(path, sample_rate=self.sample_rate)
        return _pad_or_trim(waveform, self.num_samples)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        try:
            waveform = self._load_audio(path)
        except Exception as e:  # noqa: BLE001 - report any load failure per-file
            return None, idx, str(e)

        mel = self.mel_spec(waveform)
        mel_db = self.db_transform(mel)

        mean, std = mel_db.mean(), mel_db.std() + 1e-6
        mel_db = (mel_db - mean) / std

        return mel_db, idx, None


def _inference_collate(batch):
    """Splits a batch into (successfully loaded tensors) and (per-file errors)."""
    valid = [(mel, idx) for mel, idx, err in batch if mel is not None]
    errors = [(idx, err) for _, idx, err in batch if err is not None]

    if valid:
        mels = torch.stack([m for m, _ in valid])
        idxs = torch.tensor([i for _, i in valid], dtype=torch.long)
    else:
        mels, idxs = None, None

    return mels, idxs, errors


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class AudioCNN(nn.Module):
    """
    Small CNN over log-mel spectrograms. Well under 10M parameters
    (typically ~350K-400K depending on n_classes).
    """

    def __init__(self, n_classes, n_mels=64, class_names=None):
        super().__init__()

        def conv_block(in_c, out_c, pool=(2, 2)):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(pool),
            )
        self.class_names=class_names
        self.features = nn.Sequential(
            conv_block(1, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 128),
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        x = x.flatten(1)
        return self.classifier(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --------------------------------------------------------------------------- #
# Early stopping
# --------------------------------------------------------------------------- #
class EarlyStopping:
    """
    Tracks a metric (default: validation loss, mode='min') and signals
    when training should stop after `patience` epochs without improvement.
    Also keeps a copy of the best model's state_dict.
    """

    def __init__(self, patience=7, min_delta=0.0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.best_state = None

    def step(self, metric, model):
        score = -metric if self.mode == 'min' else metric

        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            self.best_state = copy.deepcopy(model.state_dict())
            improved = True
        else:
            self.counter += 1
            improved = False
            if self.counter >= self.patience:
                self.early_stop = True

        return improved


# --------------------------------------------------------------------------- #
# Train / validate loops
# --------------------------------------------------------------------------- #
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for x, y in tqdm(loader):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        correct += (out.argmax(dim=1) == y).sum().item()
        total += y.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)

        running_loss += loss.item() * x.size(0)
        correct += (out.argmax(dim=1) == y).sum().item()
        total += y.size(0)

    return running_loss / total, correct / total


def fit(model, train_loader, val_loader, device,
        epochs=50, lr=1e-3, weight_decay=1e-4, patience=7,
        save_path='best_model.pt'):
    """
    Full training loop with per-epoch validation and early stopping.
    Restores and saves the best-performing model weights (lowest val loss).
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    early_stopping = EarlyStopping(patience=patience, mode='min')
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        improved = early_stopping.step(val_loss, model)
        dt = time.time() - t0

        print(f"Epoch {epoch:03d} | "
              f"train_loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"val_loss {val_loss:.4f} acc {val_acc:.4f} | "
              f"{'*' if improved else ' '} {dt:.1f}s")

        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch}.")
            break

    if early_stopping.best_state is not None:
        model.load_state_dict(early_stopping.best_state)
        torch.save(model.state_dict(), save_path)
        print(f"Best model (val_loss={-early_stopping.best_score:.4f}) saved to {save_path}")

    return model, history


# --------------------------------------------------------------------------- #
# Inference: run the trained model on a plain list of file names
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict(model, file_paths, device, batch_size=32, num_workers=0,
            sample_rate=16000, n_mels=64, n_fft=1024, hop_length=256,
            duration=4.0, class_names=None):
    """
    Run the trained model over a list of audio file paths and return a
    prediction for each one. This is the inference entry point: the only
    required input describing the data is `file_paths` (a plain list of
    strings) -- no labels are needed.

    Parameters
    ----------
    model : nn.Module
        A trained AudioCNN (already loaded with the desired weights).
    file_paths : list[str]
        Paths to the audio files to classify. If the list contains
        duplicate paths, later entries overwrite earlier ones in the
        result dict (same limitation as any dict keyed by filename).
    device : torch.device
    batch_size, num_workers : int
        DataLoader settings.
    sample_rate, n_mels, n_fft, hop_length, duration :
        Must match the values the model was trained with.
    class_names : list[str], optional
        Human-readable class names, in the same order as the model's
        output logits. If omitted, class indices ("0", "1", ...) are
        used as keys instead.

    Returns
    -------
    dict[str, dict[str, float]]
        Maps each file path to a dict of {class_name: probability},
        e.g. {"clip_0.wav": {"dog": 0.83, "cat": 0.05, ...}, ...}.
        A file that failed to load maps to {"error": <message>} instead.
    """
    model.eval()
    model.to(device)

    if class_names is None:
        n_classes = model.classifier[-1].out_features
        class_names = [str(i) for i in range(n_classes)]

    dataset = InferenceMelSpecDataset(
        file_paths, sample_rate=sample_rate, n_mels=n_mels,
        n_fft=n_fft, hop_length=hop_length, duration=duration,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=_inference_collate,
    )

    results = {}

    for mels, idxs, errors in tqdm(loader, desc="Predicting"):
        # Files that failed to load: record the error, keep going.
        for idx, err in errors:
            results[file_paths[idx]] = {"error": err}

        if mels is None:
            continue  # entire batch failed to load

        mels = mels.to(device)
        logits = model(mels)
        probs = F.softmax(logits, dim=1)

        for i, idx in enumerate(idxs.tolist()):
            prob_vector = probs[i].cpu().tolist()
            results[file_paths[idx]] = dict(zip(class_names, prob_vector))

    return results


# --------------------------------------------------------------------------- #
# Example usage
# --------------------------------------------------------------------------- #
# if __name__ == "__main__":
#     N_CLASSES = 21
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     model = AudioCNN(n_classes=N_CLASSES)
#     print(f"Trainable parameters: {count_parameters(model):,} (limit: 10,000,000)")

#     # --- Training (see previous examples for train_pairs / val_pairs) -------
#     # model, history = fit(model, train_loader, val_loader, device, ...)

#     # --- Inference on a plain list of file names -----------------------------
#     file_list = ["test.wav"]
#     model.load_state_dict(torch.load("models/best_model.pt", map_location=device))
#     predictions = predict(model, file_list, device)
#     # predictions = {
#     #     "clip_001.wav": {"dog": 0.83, "cat": 0.05, ...},
#     #     "clip_002.wav": {"dog": 0.12, "cat": 0.61, ...},
#     #     ...
#     # }
#     for filename, class_probs in predictions.items():
#         print(filename, class_probs)