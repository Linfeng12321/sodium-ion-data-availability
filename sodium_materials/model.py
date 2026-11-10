"""Four role attention network described in Appendix C."""

import torch
from torch import nn
from .preprocessing import ROLES


class MaterialSystemModel(nn.Module):
    def __init__(self, dimensions, config):
        super().__init__()
        if set(dimensions) != set(ROLES) or any(
            type(v) is not int or v < 1 for v in dimensions.values()
        ):
            raise ValueError(
                "Provide one positive input dimension for each component role"
            )
        d, h = config["width"], config["heads"]
        if d < 1 or h < 1 or d % h or config["classes"] != 4:
            raise ValueError("Invalid attention width, heads or task class count")
        self.dimensions = dimensions
        self.encoders = nn.ModuleDict(
            {
                r: nn.Sequential(
                    nn.Linear(dimensions[r], d), nn.GELU(approximate="none")
                )
                for r in ROLES
            }
        )
        self.role_embeddings = nn.Parameter(torch.empty(4, d))
        p = config["dropout"]
        self.attention = nn.MultiheadAttention(d, h, dropout=p, batch_first=True)
        self.attention_dropout = nn.Dropout(p)
        self.norm1 = nn.LayerNorm(d, eps=config["layer_norm_epsilon"])
        self.feedforward = nn.Sequential(
            nn.Linear(d, config["feedforward_width"]),
            nn.GELU(approximate="none"),
            nn.Dropout(p),
            nn.Linear(config["feedforward_width"], d),
            nn.Dropout(p),
        )
        self.norm2 = nn.LayerNorm(d, eps=config["layer_norm_epsilon"])

        def head(n):
            return nn.Sequential(
                nn.Linear(d, config["head_width"]),
                nn.GELU(approximate="none"),
                nn.Dropout(p),
                nn.Linear(config["head_width"], n),
            )

        self.capacity_head = head(1)
        self.rmi_head = head(4)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        # Initialize query, key and value matrices individually, not as one tall matrix.
        for projection in self.attention.in_proj_weight.chunk(3, dim=0):
            nn.init.xavier_uniform_(projection)
        nn.init.zeros_(self.attention.in_proj_bias)
        nn.init.normal_(
            self.role_embeddings, mean=0.0, std=config["role_embedding_std"]
        )

    def forward(self, inputs):
        if set(inputs) != set(ROLES):
            raise ValueError("Exactly four role tensors are required")
        shapes = [inputs[r].shape for r in ROLES]
        if (
            any(
                len(s) != 2 or s[1] != self.dimensions[r] for r, s in zip(ROLES, shapes)
            )
            or len({s[0] for s in shapes}) != 1
        ):
            raise ValueError(
                "Role tensors must have matching batch sizes and fitted input dimensions"
            )
        x = torch.stack([self.encoders[r](inputs[r]) for r in ROLES], dim=1)
        x = x + self.role_embeddings.unsqueeze(0)
        attended, _ = self.attention(x, x, x, need_weights=False)
        x = self.norm1(x + self.attention_dropout(attended))
        x = self.norm2(x + self.feedforward(x))
        pooled = x.mean(dim=1)
        return self.capacity_head(pooled).squeeze(-1), self.rmi_head(pooled)


def make_optimizer(model, config):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if name.endswith("weight") and parameter.ndim == 2:
            decay.append(parameter)
        else:
            no_decay.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": config["weight_decay"]},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=config["learning_rate"],
        betas=tuple(config["betas"]),
        eps=config["optimizer_epsilon"],
        amsgrad=False,
        foreach=False,
    )
