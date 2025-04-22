from abc import ABC, abstractmethod
from typing import Any

import torch
from beartype import beartype
from torch import Tensor, nn
from torch.nn import functional as F
from ssm import S4Block as S4



# Dropout broke in PyTorch 1.11
if tuple(map(int, torch.__version__.split('.')[:2])) == (1, 11):
    print("WARNING: Dropout is bugged in PyTorch 1.11. Results may be worse.")
    dropout_fn = nn.Dropout
if tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 12):
    dropout_fn = nn.Dropout1d
else:
    dropout_fn = nn.Dropout2d


class LinearRNNCell(nn.Module):
    """
    A linear RNN cell without nonlinearities, similar to PyTorch's RNNCell but without activation functions.
    
    Args:
        input_size: The number of expected features in the input x
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights b_ih and b_hh
    """
    
    def __init__(self, input_size, hidden_size, bias=True):
        super(LinearRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        
        # Weight matrices
        self.weight_ih = nn.Parameter(torch.Tensor(hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        
        # Optional bias
        if bias:
            self.bias_ih = nn.Parameter(torch.Tensor(hidden_size))
            self.bias_hh = nn.Parameter(torch.Tensor(hidden_size))
        else:
            self.register_parameter('bias_ih', None)
            self.register_parameter('bias_hh', None)
            
        self.reset_parameters()
        
    def reset_parameters(self):
        # Use standard initialization method from PyTorch
        nn.init.xavier_uniform_(self.weight_ih)
        nn.init.xavier_uniform_(self.weight_hh)
        if self.bias:
            nn.init.zeros_(self.bias_ih)
            nn.init.zeros_(self.bias_hh)
            
    def forward(self, input, hx=None):
        """
        Forward pass of the linear RNN cell.
        
        Args:
            input: tensor of shape (batch, input_size) containing input features
            hx: tensor of shape (batch, hidden_size) containing the initial hidden state
                or None for zero initial hidden state
                
        Returns:
            h': tensor of shape (batch, hidden_size) containing the next hidden state
        """
        if hx is None:
            hx = torch.zeros(input.size(0), self.hidden_size, 
                             dtype=input.dtype, device=input.device)
        
        # Linear transformations
        h_next = F.linear(input, self.weight_ih, self.bias_ih) + \
                 F.linear(hx, self.weight_hh, self.bias_hh)
        
        # No activation function
        return h_next


class S4Model(nn.Module):

    def __init__(
        self,
        d_input,
        d_output=10,
        d_model=128,
        n_layers=4,
        dropout=0.2,
        prenorm=False,
    ):
        super().__init__()

        self.prenorm = prenorm

        self.encoder = nn.Linear(d_input, d_model)

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                S4(d_model, dropout=dropout, transposed=True)
            )
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(dropout_fn(dropout))

        # Linear decoder
        self.decoder = nn.Linear(d_model, d_output)

    def forward(self, x):
        """
        Input x is shape (B, L, d_input)
        """
        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)

        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)

            # Apply S4 block: we ignore the state input and output
            z, _ = layer(z)

            # Dropout on the output of the S4 block
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        x = x.transpose(-1, -2)

        # Decode the outputs
        x = self.decoder(x)  # (B, L, d_model) -> (B, L, d_output)

        return x


class DeepRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, activation='relu', readout_activation='identity', rnncell='rnn'):
        super(DeepRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        if activation == 'relu':
            self.activation_fn = F.relu
        elif activation == 'tanh':
            self.activation_fn = torch.tanh
        elif activation == 'identity':
            self.activation_fn = lambda x: x
        else:
            raise ValueError("activation must be 'relu', 'tanh', or 'identity'")
        
        if readout_activation == 'relu':
            self.readout_activation_fn = F.relu
        elif readout_activation == 'tanh':
            self.readout_activation_fn = torch.tanh
        elif readout_activation == 'identity':
            self.readout_activation_fn = lambda x: x
        else:
            raise ValueError("readout_activation must be 'relu', 'tanh', or 'identity'")
        
        if rnncell == 'linear':
            self.rnn_cell = LinearRNNCell
        elif rnncell == 'rnn':
            self.rnn_cell = nn.RNNCell
        
        self.rnn_layers = nn.ModuleList([
            self.rnn_cell(input_size if i == 0 else hidden_size, hidden_size)
            for i in range(num_layers)
        ])
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: Any, batch_first=True) -> Any:
        if batch_first:
            # Swap the first two dimensions (batch, seq_len, ...) -> (seq_len, batch, ...)
            x = x.transpose(0, 1)

        # Extract sequence length and batch size for either 2D or 3D input
        seq_length = x.size(0)
        batch_size = x.size(1)
        h = self.init_hidden(batch_size)
        outputs = []
        
        for t in range(seq_length):
            out, h = self.forward_one_timestep(x[t], h)
            outputs.append(out)

        outputs = torch.stack(outputs)

        if batch_first:
            # Swap back two dimensions (seq_len, batch, ...) -> (batch, seq_len, ...)
            outputs = outputs.transpose(0, 1)

        return outputs 

    def forward_one_timestep(self, x, h):
        h_depth = []
        h_rec = []
        for i, rnn in enumerate(self.rnn_layers):
            h_i = rnn(x if i == 0 else h_depth[i-1], h[i])
            h_rec.append(h_i)
            h_i = self.activation_fn(h_i)
            h_depth.append(h_i)
        out = self.fc(h_depth[-1])
        out = self.readout_activation_fn(out)
        
        return out, torch.stack(h_rec)
        
    def init_hidden(self, batch_size):
        return [torch.zeros(batch_size, self.hidden_size) for _ in range(self.num_layers)]


class SimpleSeq2SeqRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size):
        super(SimpleSeq2SeqRNN, self).__init__()
        self.hidden_size = hidden_size
        
        # Embedding layer to convert input tokens to vectors
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Single RNN cell
        self.rnn = nn.RNNCell(embedding_dim, hidden_size)
        
        # Output projection to vocabulary size (no activation - raw logits)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden=None):
        """
        x: Input sequence tensor of shape (seq_len, batch_size) containing token indices
        hidden: Initial hidden state of shape (batch_size, hidden_size) or None
        
        Returns:
        - outputs: Tensor of shape (seq_len, batch_size, output_size) containing logits
        - hidden: Final hidden state
        """
        seq_len, batch_size = x.size()
        
        # Initialize hidden state if not provided
        if hidden is None:
            hidden = torch.zeros(batch_size, self.hidden_size, device=x.device)
        
        # Process the sequence one step at a time
        outputs = []
        for t in range(seq_len):
            # Get current input tokens and convert to embeddings
            emb = self.embedding(x[t])  # (batch_size, embedding_dim)
            
            # Update hidden state
            hidden = self.rnn(emb, hidden)
            
            # Project to output size (logits)
            output = self.fc(hidden)  # (batch_size, output_size)
            outputs.append(output)
        
        # Stack outputs along sequence dimension
        outputs = torch.stack(outputs)  # (seq_len, batch_size, output_size)
        
        return outputs, hidden


class DeepRNNWithEmbedding(DeepRNN):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size, num_layers, activation='relu', readout_activation='identity', rnncell='rnn'):
        super().__init__(embedding_dim, hidden_size, output_size, num_layers, activation, readout_activation, rnncell)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
    
    def forward(self, x, batch_first=True):
        x = self.embedding(x)
        return super().forward(x, batch_first=batch_first)


class S4ModelWithEmbedding(S4Model):
    def __init__(
        self,
        d_input,
        embedding_dim,
        d_output=10,
        d_model=128,
        n_layers=4,
        dropout=0.2,
        prenorm=False,
        padding_idx=None,
    ):
        super().__init__(embedding_dim, d_output, d_model, n_layers, dropout, prenorm)
        self.embedding = nn.Embedding(d_input, embedding_dim, padding_idx=padding_idx)

    def forward(self, x):
        x = self.embedding(x)
        return super().forward(x)


def test_seq2seq_rnn(vocab_size=100, embedding_dim=64, hidden_size=128):
    vocab_size = 100
    embedding_dim = 64
    hidden_size = 128
    output_size = vocab_size
    model = SimpleSeq2SeqRNN(vocab_size, embedding_dim, hidden_size, output_size)

    # Example input: a sequence of token indices
    seq_len = 10
    batch_size = 32
    input_tensor = torch.randint(0, vocab_size, (seq_len, batch_size))  # Random token indices
    print(input_tensor.shape)  # Should be (seq_len, batch_size)
    output, hidden = model(input_tensor)
    print("Output shape:", output.shape)  # Should be (seq_len, batch_size, output_size)

def test_deep_rnn(input_size=100, hidden_size=20, output_size=5, num_layers=2):
    model = DeepRNN(input_size, hidden_size, output_size, num_layers, rnncell='linear')
    
    # Example input: a sequence of vectors
    batch_size = 32 
    seq_len = 10 
    x = torch.randn(batch_size, seq_len, input_size)
    print("Input shape:", x.shape)  # Should be (batch_size, seq_len, input_size)

    outputs = model(x)
    
    print("Output shape:", outputs.shape)  # Should be (batch_size, output_size)

def test_deep_rnn_with_embedding(vocab_size=100, embedding_dim=64, hidden_size=128):
    model = DeepRNNWithEmbedding(vocab_size, embedding_dim, hidden_size, vocab_size, num_layers=2)

    # Example input: a sequence of token indices
    seq_len = 10
    batch_size = 32
    input_tensor = torch.randint(0, vocab_size, (seq_len, batch_size))  # Random token indices
    print(input_tensor.shape)  # Should be (seq_len, batch_size)

    outputs = model(input_tensor)
    print("Output shape:", outputs.shape)  # Should be (seq_len, batch_size, output_size)

def test_s4():
    s4_model = S4Model(d_input=100, d_output=5, d_model=128, n_layers=4, dropout=0.2)
    batch_size = 32 
    seq_len =10 
    x = torch.randn(batch_size, seq_len, 100)
    print("Input shape:", x.shape)  # Should be (batch_size, seq_len, d_input)
    outs = s4_model(x)
    print("S4 Model Output shape:", outs.shape)  # Should be (batch_size, seq_len, d_output)


if __name__ == "__main__":
    print("Testing SimpleSeq2SeqRNN...")
    test_seq2seq_rnn()
    print("Testing DeepRNN...")
    test_deep_rnn()
    print("Testing DeepRNNWithEmbedding...")
    test_deep_rnn_with_embedding()
    print("Testing S4Model...")
    test_s4()