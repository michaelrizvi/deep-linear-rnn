import random
from abc import ABC, abstractmethod
from typing import Any, Iterable, Literal

import torch
from beartype import beartype
from lightning import Callback, LightningModule
from torch import Tensor
from torch.nn import functional as F
import math

def sequential_ce_loss(
    input: Tensor,
    target: Tensor,
    ignore_index: int = -1,
) -> Tensor:
    """
    Compute the negative log likelihood loss for a sequence of predictions.
    Args:
        input (Tensor): The predicted probabilities (shape: batch_size, seq_len, vocab_size).
        target (Tensor): The target indices (shape: batch_size, seq_len, vocab_size).
    Returns:
        Tensor: The computed loss.
    """
    # Reshape input and target to match the expected dimensions
    input = input.float()
    input = input.reshape(-1, input.size(-1))
    target = target.view(-1)

    # Compute the negative log likelihood loss
    loss = F.cross_entropy(input, target, ignore_index=ignore_index)
    return loss


class CopyTaskRegression(LightningModule):
    @beartype
    def __init__(
        self,
        model: Any,
        lr: float = 1e-4,
    ):
        super().__init__()
        self.model = model
        self.save_hyperparameters()  # ignore the instance of nn.Module that are already stored ignore=['my_module'])

    @beartype
    def forward(self, x: Any) -> Any:
        outputs =self.model(x)
        return outputs 


    @beartype
    def training_step(self, data, batch_idx) -> Tensor:
        x, y = data
        # Forward pass
        preds = self.forward(x)
        # Create mask for nonzero vectors (shape: seq_len, batch)
        mask = (y.abs().sum(dim=-1) != 0)
        # Compute loss only on nonzero vectors
        loss = self.loss_function(y[mask], preds[mask])

        # Log loss
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=False)

        return loss

    @beartype
    def validation_step(self, data, batch_idx) -> Tensor:
        x, y = data
        # Forward pass
        preds = self.forward(x)
        # Create mask for nonzero vectors (shape: seq_len, batch)
        mask = (y.abs().sum(dim=-1) != 0)
        # Compute loss only on nonzero vectors
        loss = self.loss_function(y[mask], preds[mask])
        # Log loss
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    @beartype
    def test_step(self, data, batch_idx) -> Tensor:
        x, y = data
        # Forward pass
        preds = self.forward(x)
        # Create mask for nonzero vectors (shape: seq_len, batch)
        mask = (y.abs().sum(dim=-1) != 0)
        # Compute loss only on nonzero vectors
        loss = self.loss_function(y[mask], preds[mask])
        # Log loss
        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    @abstractmethod
    def loss_function(self, target: Any, preds: Any) -> Tensor:
        """Loss function to be used in the training loop."""
        loss = F.mse_loss(preds, target)
        return loss

    def configure_optimizers(self) -> None:
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


class CopyTaskTokenized(LightningModule):
    @beartype
    def __init__(
        self,
        model: Any,
        lr: float = 1e-4,
        ignore_index: int = -1,
    ):
        super().__init__()
        self.model = model
        self.save_hyperparameters()  # ignore the instance of nn.Module that are already stored ignore=['my_module'])
        self.loss_function = sequential_ce_loss 
        self.ignore_index = ignore_index

    @beartype
    def forward(self, x: Any) -> Any:
        outputs = self.model(x)

        return outputs 
    
    @beartype
    def training_step(self, data, batch_idx) -> Tensor:
        x, y = data
        # Forward pass
        preds = self.forward(x)

        # Compute loss only on nonzero vectors
        loss = self.loss_function(preds, y)

        # Log loss
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=False)

        return loss

    @beartype
    def validation_step(self, data, batch_idx) -> Tensor:
        x, y = data
        # Forward pass
        preds = self.forward(x)
        # Compute loss
        loss = self.loss_function(preds, y)
        # Compute mean accuracy
        mask = y != self.ignore_index 
        accuracy = torch.logical_and(preds.argmax(dim=-1) == y, mask).float().sum() / mask.float().sum() if mask.any() else torch.tensor(0.0)
        # Log loss
        self.log("val_accuracy", accuracy, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def test_step(self, data, batch_idx) -> Tensor:
        x, y = data
        # Forward pass
        preds = self.forward(x)
        # Compute loss
        loss = self.loss_function(preds, y)
        # Compute mean accuracy
        mask = y != self.ignore_index 
        accuracy = torch.logical_and(preds.argmax(dim=-1) == y, mask).float().sum() / mask.float().sum() if mask.any() else torch.tensor(0.0)
        # Log loss
        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test_accuracy", accuracy, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self) -> None:
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


class MyCustomCallback(Callback):
    def __init__(self) -> None:
        super().__init__()

    def on_train_epoch_start(self, trainer, pl_module) -> None:  # pl_module is the LightningModule
        # setattr(pl_module, "my_param", new_param)
        pass


class ShakespeareTask(LightningModule):
    def __init__(self, model, lr=1e-3):
        super().__init__()
        self.model = model
        self.lr = lr
        self.loss_fn = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        # Get model output sequence (batch, seq_len, vocab_size)
        outputs = self.model(x)

        return outputs

    def _shared_step(self, batch):
        input_ids = batch["input_ids"]
        x = input_ids[:, :-1]
        y = input_ids[:, 1:]
        logits = self(x)
        vocab_size = logits.size(-1)
        loss = self.loss_fn(logits.reshape(-1, vocab_size), y.reshape(-1))
        return loss, logits, y

    def training_step(self, batch, batch_idx):
        loss, _, _ = self._shared_step(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, targets = self._shared_step(batch)
        ppl = torch.exp(loss)
        self.log_dict({"val_loss": loss, "val_ppl": ppl}, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, logits, targets = self._shared_step(batch)
        ppl = torch.exp(loss)
        self.log_dict({"test_loss": loss, "test_ppl": ppl}, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class ShakespeareCharTask(ShakespeareTask):
    def validation_step(self, batch, batch_idx):
        loss, logits, targets = self._shared_step(batch)
        bpc = loss.item() / math.log(2)
        self.log_dict({"val_loss": loss, "val_bpc": bpc}, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, logits, targets = self._shared_step(batch)
        bpc = loss.item() / math.log(2)
        self.log_dict({"val_loss": loss, "val_bpc": bpc}, prog_bar=True)


class PautomacTask(LightningModule):
    def __init__(self, model, lr=1e-3, ignore_index=0):
        super().__init__()
        self.model = model
        self.lr = lr
        self.loss_fn = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)
    
    def forward(self, x):
        # Get model output sequence (batch, seq_len, vocab_size)
        outputs = self.model(x)
        return outputs

    def _shared_step(self, batch):
        x = batch[:, :-1]
        y = batch[:, 1:]
        logits = self(x)
        vocab_size = logits.size(-1)
        loss = self.loss_fn(logits.view(-1, vocab_size), y.reshape(-1))
        return loss, logits, y
    
    def training_step(self, batch, batch_idx):
        loss, _, _ = self._shared_step(batch)
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, logits, targets = self._shared_step(batch)
        ppl = torch.exp(loss)
        self.log_dict({"val_loss": loss, "val_ppl": ppl}, prog_bar=True)
    
    def test_step(self, batch, batch_idx):
        loss, logits, targets = self._shared_step(batch)
        ppl = torch.exp(loss)
        self.log_dict({"test_loss": loss, "test_ppl": ppl}, prog_bar=True)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def test_int_task():
    seq_len = 10
    vocab_size = 2
    lookahead = 5
    embedding_size = 2
    hidden_size = 16 
    n_layers = 2

    # Example usage
    train_dataset = SyntheticCopyDataset(n_samples=1000, seq_len=seq_len, vocab_size=vocab_size, lookahead=lookahead, datatype="int")
    val_dataset = SyntheticCopyDataset(n_samples=200, seq_len=seq_len, vocab_size=vocab_size, lookahead=lookahead, datatype="int")
    data_module = SyntheticCopyDataModule(train_dataset, val_dataset, batch_size=32)

    #model = DeepRNNWithEmbedding(vocab_size, embedding_size, hidden_size, vocab_size, n_layers)
    model = S4ModelWithEmbedding(vocab_size, embedding_size, vocab_size, hidden_size, n_layers)
    task = CopyTaskTokenized(model=model, lr=1e-3)
    # Initialize the trainer
    from lightning import Trainer
    trainer = Trainer(max_epochs=10, accelerator="cpu")
    # Train the model
    trainer.fit(task, data_module)
    # Validate the model
    trainer.validate(task, data_module)


def test_regression_task():
    seq_len = 10
    vocab_size = 2
    lookahead = 5
    hidden_size = 16 
    n_layers = 2


    # Example usage
    train_dataset = SyntheticCopyDataset(n_samples=1000, seq_len=seq_len, vocab_size=vocab_size, lookahead=lookahead, datatype="real")
    val_dataset = SyntheticCopyDataset(n_samples=200, seq_len=seq_len, vocab_size=vocab_size, lookahead=lookahead, datatype="real")
    data_module = SyntheticCopyDataModule(train_dataset, val_dataset, batch_size=1)

    #model = SimpleSeq2SeqRNN(vocab_size, embedding_size, hidden_size, vocab_size) 
    model = S4Model(vocab_size, vocab_size, hidden_size, n_layers)
    task = CopyTaskRegression(model=model, lr=1e-3)

    # Initialize the trainer
    from lightning import Trainer
    trainer = Trainer(max_epochs=200, accelerator="cpu")
    # Train the model
    trainer.fit(task, data_module)
    # Validate the model
    trainer.validate(task, data_module)


def test_shakespeare(tokenizer_type: Literal["gpt2", "char"] = "gpt2"):
    # Example usage
    if tokenizer_type == "gpt2":
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
    elif tokenizer_type == "char":
        import string
        chars = string.printable
        model_max_length = 512
        tokenizer = CharacterTokenizer(chars, model_max_length)

    data_module = ShakesepeareDataModule(
        tokenizer=tokenizer,
        input_seq_len=64,
        batch_size=32,
    )

    vocab_size = data_module.vocab_size

    #model = S4ModelWithEmbedding(d_input=vocab_size, embedding_dim=126, d_output=vocab_size, d_model=64, n_layers=2)
    # model = CPRNN(vocab_size, 64, vocab_size, rank=8)
    model = DeepCPRNN(vocab_size, 64, 2, vocab_size, rank=8)

    if tokenizer_type == "char":
        task = ShakespeareCharTask(model=model, lr=1e-3)
    elif tokenizer_type == "gpt2":
        task = ShakespeareTask(model=model, lr=1e-3)

    # Initialize the trainer
    from lightning import Trainer
    trainer = Trainer(max_epochs=1, accelerator="cpu")
    # Train the model
    trainer.fit(task, data_module)
    # Validate the model
    trainer.validate(task, data_module)


if __name__ == "__main__":
    from model import DeepRNN, DeepRNNWithEmbedding, SimpleSeq2SeqRNN, S4Model, S4ModelWithEmbedding
    from dataset import SyntheticCopyDataset, SyntheticCopyDataModule
    from shakespeare import ShakesepeareDataModule
    from pautomac import PautomacDataModule, PautomacDataset
    from transformers import AutoTokenizer
    from charactertokenizer import CharacterTokenizer
    from cprnn import CPRNN, DeepCPRNN

    test_shakespeare(tokenizer_type="char")
    #dataset = PautomacDataset(automata_name="4.spice.train")
    #datamodule = PautomacDataModule(dataset, batch_size=2)
    #print(dataset.vocab_size)
    
    #model = S4ModelWithEmbedding(d_input=dataset.vocab_size, embedding_dim=2, d_output=dataset.vocab_size, d_model=18, n_layers=2, padding_idx=0,dropout=0.1)

    #task = PautomacTask(model=model, lr=1e-3)

    ## Initialize the trainer
    #from lightning import Trainer 
    #trainer = Trainer(max_epochs=400, accelerator="cpu")
    ## Train the model
    #trainer.fit(task, datamodule)
    ## Validate the model
    #trainer.validate(task, datamodule)







