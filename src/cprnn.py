import math
from typing import Union, Optional, Any

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from charactertokenizer import CharacterTokenizer


class CPRNN(nn.Module):
    """CP-Factorized LSTM. Outputs logits (no softmax)

    Args:
        input_size: Dimension of input features.
        hidden_size: Dimension of hidden features.
        vocab_size: Size of vocabulary
        use_embedding: Whether to use embedding layer or one-hot encoding
        rank: Rank of cp factorization
        tokenizer: Character tokenizer
        batch_first: Whether to use batch first or not
        dropout: Dropout rate

    """
    def __init__(self, input_size: int, hidden_size: int, vocab_size: int, use_embedding: bool = False, rank: int = 8,
                 tokenizer: CharacterTokenizer = None, batch_first: bool = True, dropout: float = 0.5,
                 gate: str = 'tanh', **kwargs):
        super().__init__()

        self.dropout = dropout
        self.batch_first = batch_first
        self.tokenizer = tokenizer
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.rank = rank
        self.gate = {"tanh": torch.tanh, "sigmoid": torch.sigmoid, "identity": lambda x: x}[gate]

        # Define embedding and decoder layers
        if use_embedding:
            self.embedding = nn.Embedding(self.vocab_size, self.input_size)

        else:
            # One hot version
            self.embedding = lambda x: torch.nn.functional.one_hot(x, vocab_size).float()
            self.input_size = vocab_size

        self.decoder = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.vocab_size)
        )

        # Encoder using CP factors
        self.A = nn.Parameter(torch.Tensor(self.hidden_size, self.rank))
        self.B = nn.Parameter(torch.Tensor(self.input_size, self.rank))
        self.C = nn.Parameter(torch.Tensor(self.hidden_size, self.rank))
        self.U = nn.Parameter(torch.Tensor(self.input_size, self.hidden_size))
        self.V = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.d = nn.Parameter(torch.Tensor(self.hidden_size))
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def init_hidden(self, batch_size, device=torch.device('cpu')):
        h = torch.zeros(batch_size, self.hidden_size).to(device)
        return h

    def predict(self, inp: Union[torch.LongTensor, str], init_states: tuple = None, top_k: int = 1,
                device=torch.device('cpu')):

        with torch.no_grad():

            if isinstance(inp, str):
                if self.tokenizer is None:
                    raise ValueError("Tokenizer not defined. Please provide a tokenizer to the model.")
                x = torch.tensor(self.tokenizer.char_to_ix(inp)).reshape(1, 1).to(device)
            else:
                x = inp.to(device)

            output, init_states = self.forward(x, init_states)
            output_conf = torch.softmax(output, dim=-1)  # [S, B, Din]
            output_topk = torch.topk(output_conf, top_k, dim=-1)  # [S, B, K]

            prob = output_topk[0].reshape(-1) / output_topk[0].reshape(-1).sum()
            k_star = np.random.choice(np.arange(top_k), p=prob.cpu().numpy())
            output_ids = output_topk[1][:, :, k_star]

            if isinstance(inp, str):
                output_char = self.tokenizer.ix_to_char(output_ids.item())
                return output_char, init_states
            else:
                return output_ids, init_states

    def forward(self, inp: torch.LongTensor, init_states: torch.Tensor = None):

        if self.batch_first:
            inp = inp.transpose(0, 1)

        if len(inp.shape) != 2:
            raise ValueError("Expected input tensor of order 2, but got order {} tensor instead".format(len(inp.shape)))

        x = self.embedding(inp)  # [S, B, D_in] (i.e. [sequence, batch, input_size])
        sequence_length, batch_size, _ = x.size()
        hidden_seq = []

        device = x.device

        if init_states is None:
            h_t = torch.zeros(batch_size, self.hidden_size).to(x.device)

        else:
            h_t = init_states
            h_t = h_t.to(device)

        for t in range(sequence_length):
            x_t = x[t, :, :]

            A_prime = h_t @ self.A
            B_prime = x_t @ self.B

            h_t = self.gate(
                torch.einsum("br,br,hr -> bh", A_prime, B_prime, self.C) + h_t @ self.V + x_t @ self.U + self.d
            )

            hidden_seq.append(h_t.unsqueeze(0))

        hidden_seq = torch.cat(hidden_seq, dim=0)
        output = self.decoder(hidden_seq.contiguous())

        if self.batch_first:
            output = output.transpose(0, 1)

        return output


class CPRNN_cell(nn.Module):
    """CP-Factorized LSTM, single cell.

    Args:
        input_size: Input size
        hidden_size: Dimension of hidden features.
        rank: Rank of cp factorization
        tokenizer: Character tokenizer
        batch_first: Whether to use batch first or not
        dropout: Dropout rate
        gate: Gate function (activation from t to t+1)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        rank: int,
        tokenizer, 
        batch_first: bool,
        dropout: float,
        gate: callable,
        **kwargs
    ):
        super().__init__()

        self.dropout = dropout
        self.batch_first = batch_first
        self.tokenizer = tokenizer
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.rank = rank
        self.gate = gate

        # Encoder using CP factors
        self.A = nn.Parameter(torch.Tensor(self.hidden_size, self.rank))
        self.B = nn.Parameter(torch.Tensor(self.input_size, self.rank))
        self.C = nn.Parameter(torch.Tensor(self.hidden_size, self.rank))
        self.U = nn.Parameter(torch.Tensor(self.input_size, self.hidden_size))
        self.V = nn.Parameter(torch.Tensor(self.hidden_size, self.hidden_size))
        self.d = nn.Parameter(torch.Tensor(self.hidden_size))
        self.tokenizer = tokenizer
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def predict(
        self,
        inp: Union[torch.LongTensor, str],
        init_states: tuple = None,
        top_k: int = 1,
        device=torch.device("cpu"),
    ):

        with torch.no_grad():

            if isinstance(inp, str):
                if self.tokenizer is None:
                    raise ValueError(
                        "Tokenizer not defined. Please provide a tokenizer to the model."
                    )
                x = (
                    torch.tensor(self.tokenizer.char_to_ix(inp))
                    .reshape(1, 1)
                    .to(device)
                )
            else:
                x = inp.to(device)

            output, init_states = self.forward(x, init_states)
            output_conf = torch.softmax(output, dim=-1)  # [S, B, Din]
            output_topk = torch.topk(output_conf, top_k, dim=-1)  # [S, B, K]

            prob = output_topk[0].reshape(-1) / output_topk[0].reshape(-1).sum()
            k_star = np.random.choice(np.arange(top_k), p=prob.cpu().numpy())
            output_ids = output_topk[1][:, :, k_star]

            if isinstance(inp, str):
                output_char = self.tokenizer.ix_to_char(output_ids.item())
                return output_char, init_states
            else:
                return output_ids, init_states

    def forward(self, x, h):
        if h is None:
            h = torch.zeros(x.size(0), self.hidden_size, dtype=x.dtype, device=x.device)
        A_prime = h @ self.A
        B_prime = x @ self.B

        h_next = self.gate(
            torch.einsum("br,br,hr -> bh", A_prime, B_prime, self.C)
            + h @ self.V
            + x @ self.U
            + self.d
        )

        return h_next


class DeepCPRNN(nn.Module):
    """CP-Factorized LSTM. Outputs logits (no softmax)

    Args:
        hidden_size: Dimension of hidden features.
        vocab_size: Size of vocabulary
        num_layers: Number of layers
        embedding_dim: Dimension of the embedding (`None` means no embedding)
        rank: Rank of cp factorization
        tokenizer: Character tokenizer
        batch_first: Whether to use batch first or not
        dropout: Dropout rate
        activation: Activation function (activation from l to l+1)
        readout_activation: Readout activation function
        gate: Gate function (activation from t to t+1)
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        embedding_dim: Optional[int] = None,
        rank: int = 8,
        tokenizer: CharacterTokenizer = None,
        batch_first: bool = True,
        dropout: float = 0.5,
        dropout_between_layers: bool = False,
        activation="identity",
        readout_activation="identity",
        gate: str = "identity",
        **kwargs
    ):
        super().__init__()

        self.dropout = dropout
        self.dropout_between_layers = dropout_between_layers
        self.batch_first = batch_first
        self.tokenizer = tokenizer
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        
        # here specifically
        self.output_size = self.vocab_size

        if activation == "relu":
            self.activation_fn = F.relu
        elif activation == "tanh":
            self.activation_fn = torch.tanh
        elif activation == "identity":
            self.activation_fn = lambda x: x
        else:
            raise ValueError("activation must be 'relu', 'tanh', or 'identity'")

        if readout_activation == "relu":
            self.readout_activation_fn = F.relu
        elif readout_activation == "tanh":
            self.readout_activation_fn = torch.tanh
        elif readout_activation == "identity":
            self.readout_activation_fn = lambda x: x
        else:
            raise ValueError("readout_activation must be 'relu', 'tanh', or 'identity'")

        self.rank = rank
        self.gate = {
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
            "identity": lambda x: x,
        }[gate]

        # Define embedding and decoder layers
        if embedding_dim != None:
            self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
            self.input_size = self.embedding_dim
        else:
            self.input_size = self.vocab_size


        self.init_weights()

        self.cprnn_cell = CPRNN_cell
        self.cprnn_layers = nn.ModuleList(
            [
                self.cprnn_cell(
                    self.input_size if i == 0 else self.hidden_size,
                    self.hidden_size,
                    self.rank,
                    self.tokenizer,
                    self.batch_first,
                    self.dropout,
                    self.gate,
                )
                for i in range(self.num_layers)
            ]
        )

        self.decoder = nn.Linear(self.hidden_size, self.output_size)

    # what is this used for? not clear
    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def predict(
        self,
        inp: Union[torch.LongTensor, str],
        init_states: tuple = None,
        top_k: int = 1,
        device=torch.device("cpu"),
    ):

        with torch.no_grad():

            if isinstance(inp, str):
                if self.tokenizer is None:
                    raise ValueError(
                        "Tokenizer not defined. Please provide a tokenizer to the model."
                    )
                x = (
                    torch.tensor(self.tokenizer.char_to_ix(inp))
                    .reshape(1, 1)
                    .to(device)
                )
            else:
                x = inp.to(device)

            output, init_states = self.forward(x, init_states)
            output_conf = torch.softmax(output, dim=-1)  # [S, B, Din]
            output_topk = torch.topk(output_conf, top_k, dim=-1)  # [S, B, K]

            prob = output_topk[0].reshape(-1) / output_topk[0].reshape(-1).sum()
            k_star = np.random.choice(np.arange(top_k), p=prob.cpu().numpy())
            output_ids = output_topk[1][:, :, k_star]

            if isinstance(inp, str):
                output_char = self.tokenizer.ix_to_char(output_ids.item())
                return output_char, init_states
            else:
                return output_ids, init_states


    def forward(self, x: torch.LongTensor):
        if self.batch_first:
            x = x.transpose(0, 1)

        if self.embedding_dim is not None:
            if len(x.shape) != 2:
                raise ValueError(
                    "Expected input tensor of order 2, but got order {} tensor instead".format(
                        len(x.shape)
                    )
                )
            x = self.embedding(x)  # [S, B, D_in] (i.e. [sequence, batch, input_size])
        seq_length, batch_size, _ = x.size()
        device = x.device
        # not input possible for init states for now, like DeepRNN
        h = self.init_hidden(batch_size, device)
        outputs = []

        for t in range(seq_length):
            out, h = self.forward_one_timestep(x[t], h)
            # Below: out.unsqueeze(0) -> out 
            # unsqueeze(0) was messing the computation 
            outputs.append(out)


        outputs = torch.stack(outputs,dim=0)
        # I don't know what this does but I'm keeping it
        outputs = outputs.contiguous()
        # This reproduces CPRNN behaviour: Dropout after last layer, after stacking
        # just before decoding
        if not self.dropout_between_layers:
            outputs = nn.Dropout(self.dropout)(outputs)
        outputs = self.decoder(outputs)

        if self.batch_first:
            outputs = outputs.transpose(0, 1)
        
        return outputs

    def forward_one_timestep(self, x, h):
        h_depth = []
        h_rec = []
        for i, cprnn in enumerate(self.cprnn_layers):
            h_i = cprnn(x if i == 0 else h_depth[i - 1], h[i])
            h_rec.append(h_i)
            h_i = self.activation_fn(h_i)
            # This reproduces S4 behaviour: Dropout between layers, 
            # after activation
            if self.dropout_between_layers: 
                h_i = nn.Dropout(self.dropout)(h_i)
            h_depth.append(h_i)
        out = h_depth[-1]
        out = self.readout_activation_fn(out)

        return out, torch.stack(h_rec)
    
    def init_hidden(self, batch_size, device=torch.device("cpu")):
        return [
            torch.zeros(batch_size, self.hidden_size).to(device) for _ in range(self.num_layers)
        ]

if __name__ == "__main__":
    # Example usage
    vocab_size = 100
    input_size = 50
    hidden_size = 128
    rank = 8
    batch_size = 32
    seq_len = 10

    model = CPRNN(input_size, hidden_size, vocab_size, rank=rank)
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    output = model(x)
    print(output.shape)  # Should be [batch_size, seq_len, vocab_size]