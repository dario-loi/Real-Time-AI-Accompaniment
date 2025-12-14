import sys
import os
import pickle

# Add project root to Python path (critical for imports when running from nested directories)
project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

torch.set_float32_matmul_precision("high")

from src.utils.logger import setup_logger
from src.model.lstm_model import ChordLSTM
from src.model.vocabulary import Vocabulary
from src.model.train.dataloader import create_loaders
from src.config import (
    CLEAN_DATA_PKL,
    MODEL_PATH,
    VOCAB_PATH,
    WINDOW_SIZE,
    HIDDEN_SIZE,
    EMBEDDING_DIM,
    NUM_LAYERS,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    DROPOUT,
    NUM_WORKERS,
)


logger = setup_logger()


class ChordLitModule(pl.LightningModule):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        learning_rate: float,
        total_steps: int,
        pad_idx: int,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = ChordLSTM(
            vocab_size,
            embedding_dim,
            hidden_size,
            num_layers,
            dropout,
            padding_idx=pad_idx,
        )
        self.pad_idx = pad_idx
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=pad_idx)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch  # x: (B, T), y: (B, T)
        logits = self(x)  # (B, T, V)
        loss = self._compute_loss(logits, y)
        acc = self._compute_accuracy(logits, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=x.size(0))
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=x.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self._compute_loss(logits, y)
        acc = self._compute_accuracy(logits, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=x.size(0))
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True, batch_size=x.size(0))

    def configure_optimizers(self):
        decay, no_decay = [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.endswith("bias") or "norm" in name or "embedding" in name:
                no_decay.append(param)
            else:
                decay.append(param)

        optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": 1e-4},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.hparams.learning_rate,
            amsgrad=True,
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.hparams.learning_rate,
            total_steps=max(int(self.hparams.total_steps), 1),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "name": "lr",
            },
        }

    def _compute_loss(self, logits, targets):
        # logits: (B, T, V), targets: (B, T)
        vocab_size = logits.size(-1)
        loss = self.criterion(logits.view(-1, vocab_size), targets.view(-1))
        return loss

    def _compute_accuracy(self, logits, targets):
        preds = torch.argmax(logits, dim=-1)
        mask = targets.ne(self.pad_idx)
        if mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device)
        correct = (preds == targets) & mask
        return correct.float().sum() / mask.sum().float()


def _select_accelerator_and_precision():
    if torch.cuda.is_available():
        return "gpu", "bf16-mixed"
    if torch.backends.mps.is_available():
        return "mps", "32-true"
    return "cpu", "32-true"


def train():
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    
    logger.info("Loading data...")
    if not os.path.exists(CLEAN_DATA_PKL):
        logger.error(f"Data file not found at {CLEAN_DATA_PKL}")
        return

    with open(CLEAN_DATA_PKL, "rb") as f:
        songs = pickle.load(f)

    logger.info("Loading vocabulary...")
    if not os.path.exists(VOCAB_PATH):
        logger.error(f"Vocabulary not found at {VOCAB_PATH}. Run src/model/vocabulary.py first to build it.")
        return

    vocab = Vocabulary()
    vocab.load_vocab(VOCAB_PATH)
    logger.info(f"Vocabulary size: {len(vocab)}")

    logger.info("Creating dataloaders...")
    train_loader, val_loader = create_loaders(
        songs,
        vocab,
        batch_size=BATCH_SIZE,
        window_size=WINDOW_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    steps_per_epoch = len(train_loader)
    total_steps = EPOCHS * steps_per_epoch

    pad_idx = vocab.chord_to_idx.get(vocab.pad_token, 0)

    model = ChordLitModule(
        vocab_size=len(vocab),
        embedding_dim=EMBEDDING_DIM,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        learning_rate=LEARNING_RATE,
        total_steps=total_steps,
        pad_idx=pad_idx,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        filename="best",
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    log_dir = os.path.join(project_root, "logs")
    tb_logger = TensorBoardLogger(save_dir=log_dir, name="chord_lstm")

    accelerator, precision = _select_accelerator_and_precision()

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator=accelerator,
        devices=1,
        logger=tb_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        gradient_clip_val=1.0,
        precision=precision,
        log_every_n_steps=50,
    )

    logger.info(f"Training on accelerator={accelerator} with precision={precision}")
    trainer.fit(model, train_loader, val_loader)

    best_ckpt_path = checkpoint_callback.best_model_path
    if best_ckpt_path:
        logger.info(f"Best checkpoint: {best_ckpt_path}")
        best_model = ChordLitModule.load_from_checkpoint(best_ckpt_path)
        torch.save(best_model.model.state_dict(), MODEL_PATH)
        logger.info(f"Best model weights saved to {MODEL_PATH}")
        if checkpoint_callback.best_model_score is not None:
            logger.info(f"Best Val Acc: {checkpoint_callback.best_model_score.item() * 100:.2f}%")
    else:
        logger.warning("No checkpoint was saved during training.")


if __name__ == "__main__":
    train()
