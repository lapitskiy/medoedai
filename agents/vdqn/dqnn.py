import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Keep torch.compile tolerant to modules that opt out.
torch._dynamo.config.suppress_errors = True


class NoisyLinear(nn.Module):
    """Noisy linear layer used for exploration instead of epsilon-greedy."""

    def __init__(self, in_features: int, out_features: int, std_init: float = 0.1):
        super().__init__()
        self._no_compile = True  # Do not try to torch.compile this helper

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))

        self._reset_parameters()
        self._reset_noise()

    def _reset_parameters(self) -> None:
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size: int) -> torch.Tensor:
        eps = torch.randn(size)
        return eps.sign().mul(eps.abs().sqrt())

    def _reset_noise(self) -> None:
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.register_buffer('weight_epsilon', epsilon_out.outer(epsilon_in))
        self.register_buffer('bias_epsilon', epsilon_out)

    def reset_noise(self) -> None:
        self._reset_noise()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            device = x.device
            weight_epsilon = self.weight_epsilon.to(device)
            bias_epsilon = self.bias_epsilon.to(device)
            weight = self.weight_mu + self.weight_sigma * weight_epsilon
            bias = self.bias_mu + self.bias_sigma * bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


_SUPPORTED_ACTIVATIONS = {'relu', 'gelu', 'silu', 'mish'}


class ResidualLinearBlock(nn.Module):
    """Linear block with optional SwiGLU gating and residual skip connection."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        activation: str = 'relu',
        dropout: float = 0.0,
        layer_norm: bool = True,
        residual: bool = True,
        use_swiglu: bool = False,
    ) -> None:
        super().__init__()
        activation = activation.lower()
        if activation not in _SUPPORTED_ACTIVATIONS:
            raise ValueError(f'Unsupported activation "{activation}"')

        self.use_residual = residual
        self.use_layer_norm = layer_norm
        self.use_swiglu = use_swiglu
        self.activation = activation

        if use_swiglu and out_dim % 2 != 0:
            raise ValueError('use_swiglu requires even out_dim')

        hidden_multiplier = 2 if use_swiglu else 1
        self.linear = nn.Linear(in_dim, out_dim * hidden_multiplier)
        nn.init.kaiming_uniform_(self.linear.weight, nonlinearity='relu')
        nn.init.zeros_(self.linear.bias)

        if residual and in_dim != out_dim:
            self.residual_proj = nn.Linear(in_dim, out_dim)
            nn.init.kaiming_uniform_(self.residual_proj.weight, nonlinearity='linear')
            nn.init.zeros_(self.residual_proj.bias)
        else:
            self.residual_proj = None

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm = nn.LayerNorm(out_dim) if layer_norm else nn.Identity()

    def _activate(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_swiglu:
            a, b = x.chunk(2, dim=-1)
            return F.silu(a) * b
        if self.activation == 'relu':
            return F.relu(x)
        if self.activation == 'gelu':
            return F.gelu(x)
        if self.activation == 'silu':
            return F.silu(x)
        if self.activation == 'mish':
            return F.mish(x)
        raise RuntimeError('Unhandled activation')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.linear(x)
        x = self._activate(x)
        x = self.dropout(x)
        if self.use_residual:
            if self.residual_proj is not None:
                residual = self.residual_proj(residual)
            x = x + residual
        x = self.norm(x)
        return x


def _build_feature_layers(
    input_dim: int,
    hidden_sizes: tuple[int, ...],
    dropout_rate: float,
    layer_norm: bool,
    activation: str,
    use_residual: bool,
    use_swiglu: bool,
) -> nn.ModuleList:
    if not hidden_sizes:
        raise ValueError('hidden_sizes must contain at least one element')
    layers = nn.ModuleList()
    prev_dim = input_dim
    for size in hidden_sizes:
        layers.append(
            ResidualLinearBlock(
                prev_dim,
                size,
                activation=activation,
                dropout=dropout_rate,
                layer_norm=layer_norm,
                residual=use_residual,
                use_swiglu=use_swiglu,
            )
        )
        prev_dim = size
    return layers


class DuelingDQN(nn.Module):
    """Feed-forward Dueling DQN backbone with enhanced MLP blocks."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: tuple[int, ...],
        dropout_rate: float = 0.2,
        layer_norm: bool = True,
        activation: str = 'relu',
        use_residual: bool = True,
        use_swiglu: bool = False,
    ) -> None:
        super().__init__()
        self._no_compile = True

        self.act_dim = act_dim
        self.feature_layers = _build_feature_layers(
            obs_dim,
            hidden_sizes,
            dropout_rate,
            layer_norm,
            activation,
            use_residual,
            use_swiglu,
        )

        feature_dim = hidden_sizes[-1]

        self.value_stream = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim // 2, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim // 2, act_dim),
        )

        for module in self.value_stream.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        for module in self.advantage_stream.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.feature_layers:
            x = layer(x)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        advantage_centered = advantage - advantage.mean(dim=-1, keepdim=True)
        return value + advantage_centered


class NoisyDuelingDQN(nn.Module):
    """Dueling DQN with NoisyLinear layers in the advantage head."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: tuple[int, ...],
        dropout_rate: float = 0.2,
        layer_norm: bool = True,
        activation: str = 'relu',
        use_residual: bool = True,
        use_swiglu: bool = False,
    ) -> None:
        super().__init__()
        self._no_compile = True

        self.act_dim = act_dim
        self.feature_layers = _build_feature_layers(
            obs_dim,
            hidden_sizes,
            dropout_rate,
            layer_norm,
            activation,
            use_residual,
            use_swiglu,
        )

        feature_dim = hidden_sizes[-1]

        self.value_stream = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim // 2, 1),
        )
        self.advantage_stream = nn.Sequential(
            NoisyLinear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            NoisyLinear(feature_dim // 2, act_dim),
        )

        for module in self.value_stream.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.feature_layers:
            x = layer(x)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        advantage_centered = advantage - advantage.mean(dim=-1, keepdim=True)
        return value + advantage_centered

    def reset_noise(self) -> None:
        for module in self.advantage_stream.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


class DistributionalDQN(nn.Module):
    """Distributional DQN head returning categorical distributions over returns."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: tuple[int, ...],
        n_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        dropout_rate: float = 0.2,
        layer_norm: bool = True,
        activation: str = 'relu',
        use_residual: bool = True,
        use_swiglu: bool = False,
    ) -> None:
        super().__init__()
        self._no_compile = True

        self.act_dim = act_dim
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (v_max - v_min) / (n_atoms - 1)
        self.support = torch.linspace(v_min, v_max, n_atoms)

        self.feature_layers = _build_feature_layers(
            obs_dim,
            hidden_sizes,
            dropout_rate,
            layer_norm,
            activation,
            use_residual,
            use_swiglu,
        )

        feature_dim = hidden_sizes[-1]
        self.distribution_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim // 2, act_dim * n_atoms),
        )

        for module in self.distribution_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        for layer in self.feature_layers:
            x = layer(x)
        batch_size = x.size(0)
        logits = self.distribution_head(x)
        logits = logits.view(batch_size, self.act_dim, self.n_atoms)
        distributions = F.softmax(logits, dim=-1)
        q_values = torch.sum(distributions * self.support.to(x.device), dim=-1)
        return q_values, distributions

    def get_q_values(self, x: torch.Tensor) -> torch.Tensor:
        q_values, _ = self.forward(x)
        return q_values


class DQNN(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: tuple[int, ...],
        dropout_rate: float = 0.2,
        layer_norm: bool = True,
        dueling: bool = True,
        activation: str = 'relu',
        use_residual: bool = True,
        use_swiglu: bool = False,
    ) -> None:
        super().__init__()
        self._no_compile = True

        if dueling:
            self.net = DuelingDQN(
                obs_dim,
                act_dim,
                hidden_sizes,
                dropout_rate=dropout_rate,
                layer_norm=layer_norm,
                activation=activation,
                use_residual=use_residual,
                use_swiglu=use_swiglu,
            )
        else:
            layers = []
            prev_dim = obs_dim
            for size in hidden_sizes:
                layers.append(
                    ResidualLinearBlock(
                        prev_dim,
                        size,
                        activation=activation,
                        dropout=dropout_rate,
                        layer_norm=layer_norm,
                        residual=use_residual,
                        use_swiglu=use_swiglu,
                    )
                )
                prev_dim = size
            output_layer = nn.Linear(prev_dim, act_dim)
            nn.init.xavier_uniform_(output_layer.weight)
            nn.init.zeros_(output_layer.bias)
            layers.append(output_layer)
            self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
