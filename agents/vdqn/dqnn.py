import torch.nn as nn

class DQNN(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: tuple[int, ...]):
        """
        obs_dim  — размер входного вектора состояния
        act_dim  — число дискретных действий
        hidden_sizes — (128, 64, …) любой кортеж размеров скрытых слоёв
        """
        super().__init__()
        
        
        layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, act_dim))
        self.net = nn.Sequential(*layers)

        # --- инициализация ---
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)