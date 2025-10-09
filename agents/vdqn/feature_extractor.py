import torch
import torch.nn as nn
import torch.nn.functional as F


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


class FeatureExtractor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: tuple[int, ...],
        dropout_rate: float,
        layer_norm: bool,
        activation: str,
        use_residual: bool,
        use_swiglu: bool,
    ) -> None:
        super().__init__()
        if not hidden_sizes:
            raise ValueError('hidden_sizes must contain at least one element')
        layers = nn.ModuleList()
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
        self.feature_layers = layers
        self._feature_dim = hidden_sizes[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.feature_layers:
            x = layer(x)
        return x

    @property
    def feature_dim(self) -> int:
        return self._feature_dim
