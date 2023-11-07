import torch
import torch.nn as nn
import torch.nn.functional as F


class Sine(nn.Module):
    """Sine activation function"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)


class CNNEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_sizes, num_filters):
        super(CNNEncoder, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_sizes = kernel_sizes
        self.num_filters = num_filters

        self.convs = nn.ModuleList([
            nn.Conv1d(input_dim, num_filters, kernel_size)
            for kernel_size in kernel_sizes
        ])
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, output_dim)

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        x = x.permute(0, 2, 1)  # (batch_size, input_dim, seq_len)
        conv_outputs = []
        for conv in self.convs:
            conv_outputs.append(nn.functional.relu(conv(x)))
        pooled_outputs = [torch.max(output, dim=-1)[0] for output in conv_outputs]
        concat_output = torch.cat(pooled_outputs, dim=1)
        out = self.fc(concat_output)
        return out

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(Encoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        _, (h_n, c_n) = self.lstm(x)
        # h_n: (num_layers * num_directions, batch_size, hidden_dim)
        # c_n: (num_layers * num_directions, batch_size, hidden_dim)
        out = self.fc(h_n[-1])
        # out: (batch_size, output_dim)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        self.query_linear = nn.Linear(input_dim, input_dim)
        self.key_linear = nn.Linear(input_dim, input_dim)
        self.value_linear = nn.Linear(input_dim, input_dim)

        self.output_linear = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, input_dim = x.size()

        # Project inputs into query, key, and value representations
        query = self.query_linear(x)  # shape: (batch_size, seq_len, input_dim)
        key = self.key_linear(x)  # shape: (batch_size, seq_len, input_dim)
        value = self.value_linear(x)  # shape: (batch_size, seq_len, input_dim)

        # Split query, key, and value into multiple heads
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Compute scaled dot-product attention scores
        scores = torch.matmul(query, key.transpose(2, 3)) / (self.head_dim ** 0.5)
        # scores shape: (batch_size, num_heads, seq_len, seq_len)

        # Apply softmax activation to obtain attention weights
        attn_weights = F.softmax(scores, dim=-1)
        # attn_weights shape: (batch_size, num_heads, seq_len, seq_len)

        # Compute weighted sum using attention weights
        attn_output = torch.matmul(attn_weights, value)
        # attn_output shape: (batch_size, num_heads, seq_len, head_dim)

        # Concatenate and reshape multi-head outputs
        attn_output = attn_output.view(batch_size, seq_len, self.num_heads * self.head_dim)

        # Project concatenated output to original input dimension
        attn_output = self.output_linear(attn_output)
        # attn_output shape: (batch_size, seq_len, input_dim)

        return attn_output


class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers):
        super(TransformerEncoder, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        # Embedding layer
        self.embedding_layer = nn.Linear(input_dim, hidden_dim)

        # Multi-head self-attention layers
        self.attention_layers = nn.ModuleList([
            MultiHeadAttention(hidden_dim, num_heads)
            for _ in range(num_layers)
        ])

        # Position-wise feedforward layers
        self.feedforward_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.ReLU(),
                nn.Linear(hidden_dim * 4, hidden_dim)
            )
            for _ in range(num_layers)
        ])

        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        batch_size, seq_len, input_dim = x.size()

        # Embed inputs into hidden dimension
        hidden = self.embedding_layer(x)  # shape: (batch_size, seq_len, hidden_dim)

        # Apply multi-head self-attention and residual connection
        for i in range(self.num_layers):
            attn_output = self.attention_layers[i](hidden)
            hidden = hidden + attn_output

            # Apply layer normalization
            hidden = self.layer_norms[i](hidden)

            # Apply position-wise feedforward network and residual connection
            ff_output = self.feedforward_layers[i](hidden)
            hidden = hidden + ff_output

            # Apply layer normalization
            hidden = self.layer_norms[i](hidden)

        # Compute mean of sequence outputs
        output = torch.mean(hidden, dim=1)

        return output


class Neural_Net(nn.Module):
    """A simple multilayer perceptron (MLP) neural network.

    Parameters:
        layers_list (List): Number of input, hidden and output neurons
        activation_name (str): Type of activation function. Default is `Sine`
        init_method (str): Weight initialization method. Default is `xavier_normal`
    """

    def __init__(self, layers_list,
                 activation_name="sine",
                 init_method="xavier_normal"):
        super().__init__()
        input_dim = 5
        hidden_dim = 20
        num_heads = 2
        num_layers = 2
        self.encoder = TransformerEncoder(input_dim, hidden_dim, num_heads, num_layers)

        activation_dict = {"sine": Sine(), "tanh": nn.Tanh(), "swish": nn.SiLU(), "relu": nn.ReLU(), "leakyrelu":nn.LeakyReLU()}
        self.act = activation_name
        self.init_method = init_method
        self.num_layers = len(layers_list) - 1

        self.base = nn.Sequential()
        for i in range(0, self.num_layers - 1):
            if i == 0:
                self.base.add_module(
                    f"{i}  Linear", nn.Linear(layers_list[i] + hidden_dim, layers_list[i + 1])
                )
            else:
                self.base.add_module(
                    f"{i}  Linear", nn.Linear(layers_list[i], layers_list[i + 1])
                )
            self.base.add_module(f"{i} Act_fn", activation_dict[self.act])
        self.base.add_module(
            f"{self.num_layers - 1}  Linear",
            nn.Linear(layers_list[self.num_layers - 1],
                      layers_list[self.num_layers])
        )

        self.init_weights()

    def init_weights(self):
        # for name, param in self.encoder.named_parameters():
        #     if self.init_method == "xavier_normal":
        #         if name.endswith("weight"):
        #             nn.init.xavier_normal_(param)
        #         elif name.endswith("bias"):
        #             nn.init.zeros_(param)
        #     elif self.init_method == "xavier_uniform":
        #         if name.endswith("weight"):
        #             nn.init.xavier_uniform_(param)
        #         elif name.endswith("bias"):
        #             nn.init.zeros_(param)
        #     else:
        #         raise ValueError(f"{self.init_method} Not implemented yet!")
        for name, param in self.base.named_parameters():
            if self.init_method == "xavier_normal":
                if name.endswith("weight"):
                    nn.init.xavier_normal_(param)
                elif name.endswith("bias"):
                    nn.init.zeros_(param)
            elif self.init_method == "xavier_uniform":
                if name.endswith("weight"):
                    nn.init.xavier_uniform_(param)
                elif name.endswith("bias"):
                    nn.init.zeros_(param)
            else:
                raise ValueError(f"{self.init_method} Not implemented yet!")

    def forward(self, x, y, seq):
        z = self.encoder(seq)
        X = torch.cat([x, y, z], dim=1).requires_grad_(True)
        out = self.base(X)
        n = out.size(1)

        return torch.tensor_split(out, n, dim=1)

    @property
    def model_capacity(self):
        num_learnable_params = sum(p.numel() for p in self.parameters()
                                   if p.requires_grad)
        num_layers = len(list(self.parameters()))
        print(f"\nNumber of layers: {num_layers}\n"
              f"Number of trainable parameters: {num_learnable_params}")


class ResidualBlock(nn.Module):
    """Residual block class for the Residual Network"""

    def __init__(self, in_dim, hidden_dim, out_dim, activation_name="sine"):
        super().__init__()

        activation_dict = {"sine": Sine(), "tanh": nn.Tanh(), "swish": nn.SiLU(), "relu": nn.ReLU(),
                           "leakyrelu": nn.LeakyReLU()}

        self.act = activation_name
        self.block = nn.Sequential()
        self.block.add_module("Act 0", activation_dict[self.act])
        self.block.add_module("Linear 0", nn.Linear(in_dim, hidden_dim))
        self.block.add_module("Act 1", activation_dict[self.act])
        self.block.add_module("Linear 1", nn.Linear(hidden_dim, out_dim))

    def forward(self, x):
        identity = x
        out = self.block(x)

        return out + identity


class ResNet(nn.Module):
    """MLP with residual connections.

    Parameters:
        layers_list (List): Number of input, hidden and output neurons
        activation_name (str): Type of activation function. Default is `Sine`
        init_method (str): Weight initialization method. Default is `xavier_normal`
    """

    def __init__(self, layers_list,
                 activation_name="sine", init_method="xavier_normal"):
        super().__init__()

        activation_dict = {"sine": Sine(), "tanh": nn.Tanh(), "swish": nn.SiLU(), "relu": nn.ReLU(),
                           "leakyrelu": nn.LeakyReLU()}

        self.init_method = init_method
        self.num_res_blocks = len(layers_list) - 2
        self.blocks = nn.Sequential()
        self.blocks.add_module("Linear 0",
                               nn.Linear(layers_list[0], layers_list[1])
                               )
        for i in range(self.num_res_blocks):
            res_blocks = ResidualBlock(layers_list[1],
                                       layers_list[1],
                                       layers_list[1],
                                       activation_name=activation_name)
            self.blocks.add_module(f"ResBlock {i + 1}", res_blocks)
        # self.blocks.add_module("Norm1D", nn.BatchNorm1d(layers_list[-2]))
        self.blocks.add_module("Final_Act", activation_dict[activation_name])
        self.blocks.add_module(
            "Linear_last", nn.Linear(layers_list[-2], layers_list[-1])
        )

        self.init_weights()

    def init_weights(self):
        for name, param in self.blocks.named_parameters():
            if self.init_method == "xavier_normal":
                if name.endswith("weight"):
                    nn.init.xavier_normal_(param)
                elif name.endswith("bias"):
                    nn.init.zeros_(param)
            elif self.init_method == "xavier_uniform":
                if name.endswith("weight"):
                    nn.init.xavier_uniform_(param)
                elif name.endswith("bias"):
                    nn.init.zeros_(param)
            else:
                raise ValueError(f"{self.init_method} Not implemented!")

    def forward(self, x, y, k):
        X = torch.cat([x, y, k], dim=1).requires_grad_(True)
        out = self.blocks(X)
        n = out.size(1)

        return torch.tensor_split(out, n, dim=1)

    @property
    def model_capacity(self):
        num_learnable_params = sum(p.numel() for p in self.parameters()
                                   if p.requires_grad)
        num_layers = len(list(self.parameters()))
        print(f"\nNumber of layers: {num_layers}\n"
              f"Number of trainable parameters: {num_learnable_params}")

        return


class DenseResNet(nn.Module):
    """Dense Residual Neural network class with Fourier features, to enable multilayer perceptron
    (MLP) to learn high-frequency functions in low-dimensional problem domains.

    References:
    1. M. Tancik, P.P. Srinivasan, B. Mildenhall, S. Fridovich-Keil, N. Raghavan, U. Singhal,
        R. Ramamoorthi, J.T. Barron and R. Ng, "Fourier Features Let Networks Learn High
        Frequency Functions in Low Dimensional Domains", NeurIPS, 2020.

    Parameters:
        layers_list (List): Number of input, hidden and output neurons
        num_res_blocks (int): Number of residual network blocks. Default=5
        num_layers_per_block (int): Number of layers per block. Default=2
        fourier_features (bool): If to use fourier features. Default is True
        tune_beta (bool):
        m_freqs (int): fourier frequency. Default = 100
        sigma (int): std value for tuning fourier features. Default = 10
        activation_name (str): Type of activation function. Default is `Sine`
        init_method (str): Weight initialization method. Default is `xavier_normal`
    """

    def __init__(self, layers_list, num_res_blocks=5, num_layers_per_block=2,
                 activation_name="sine", init_method="xavier_normal",
                 fourier_features=True, tune_beta=True, m_freqs=100, sigma=10):
        super().__init__()

        activation_dict = {
            "sine": Sine(), "tanh": nn.Tanh(), "swish": nn.SiLU()
        }
        self.layers_list = layers_list
        self.num_res_blocks = num_res_blocks
        self.num_layers_per_block = num_layers_per_block
        self.activation = activation_dict[activation_name]
        self.fourier_features = fourier_features
        self.init_method = init_method
        self.tune_beta = tune_beta

        if tune_beta:
            self.beta0 = nn.Parameter(torch.ones(1, 1))
            self.beta = nn.Parameter(torch.ones(self.num_res_blocks,
                                                self.num_layers_per_block))

        self.first = nn.Linear(layers_list[0], layers_list[1])
        self.resblocks = nn.ModuleList([
            nn.ModuleList([nn.Linear(layers_list[1], layers_list[1])
                           for _ in range(num_layers_per_block)])
            for _ in range(num_res_blocks)])
        self.last = nn.Linear(layers_list[1], layers_list[-1])

        if fourier_features:
            self.first = nn.Linear(2 * m_freqs, layers_list[1])
            self.B = nn.Parameter(sigma * torch.randn(layers_list[0], m_freqs))

        self.init_weights()

    def init_weights(self):
        for name, param in (self.first.named_parameters() and
                            self.resblocks.named_parameters() and
                            self.last.named_parameters()):
            if self.init_method == "xavier_normal":
                if name.endswith("weight"):
                    nn.init.xavier_normal_(param)
                elif name.endswith("bias"):
                    nn.init.zeros_(param)
            elif self.init_method == "xavier_uniform":
                if name.endswith("weight"):
                    nn.init.xavier_uniform_(param)
                elif name.endswith("bias"):
                    nn.init.zeros_(param)
            else:
                raise ValueError(f"{self.init_method} Not implemented!")

    def forward(self, x, y, t):
        X = torch.cat([x, y, t], dim=1).requires_grad_(True)
        # self.scaler.fit(X)
        # X = self.scaler.transform(X)

        if self.fourier_features:
            cosx = torch.cos(torch.matmul(X, self.B))
            sinx = torch.sin(torch.matmul(X, self.B))
            X = torch.cat((cosx, sinx), dim=1)
            X = self.activation(self.beta0 * self.first(X))
        else:
            X = self.activation(self.beta0 * self.first(X))

        for i in range(self.num_res_blocks):
            z = self.activation(self.beta[i][0] * self.resblocks[i][0](X))
            for j in range(1, self.num_layers_per_block):
                z = self.activation(self.beta[i][j] * self.resblocks[i][j](z))
                X = z + X

        out = self.last(X)
        n = out.size(1)
        return torch.tensor_split(out, n, dim=1)

    @property
    def model_capacity(self):
        num_learnable_params = sum(p.numel() for p in self.parameters()
                                   if p.requires_grad)
        num_layers = len(list(self.parameters()))
        print(f"\nNumber of layers: {num_layers}\n"
              f"Number of trainable parameters: {num_learnable_params}")

        return


if __name__ == '__main__':
    net = Neural_Net([2, 10, 3])
    input_dim = 5
    batch_size = 16  # 样本数
    seq_len = 9  # 序列数

    k = torch.randn(batch_size, seq_len, input_dim)
    x = torch.randn(batch_size, 1)
    y = torch.randn(batch_size, 1)

    print(net(x, y, k))