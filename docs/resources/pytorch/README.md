# PyTorch Resources

## Key Concepts for BC

```python
import torch
import torch.nn as nn

# Neural network for policy
class Policy(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, act_dim),
        )
    
    def forward(self, obs):
        return self.net(obs)

# Training
policy = Policy(17, 6)
optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# One training step
pred = policy(obs_batch)
loss = loss_fn(pred, expert_actions)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

---

## Documentation

- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [PyTorch nn.Module](https://pytorch.org/docs/stable/nn.html)
- [PyTorch DataLoader](https://pytorch.org/docs/stable/data.html)
