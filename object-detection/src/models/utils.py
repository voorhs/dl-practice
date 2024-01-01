from torch import nn


class Conv(nn.Module):
    def __init__(self, pool=False, **kwargs):
        super().__init__()

        self.pool = pool
        for k, v in kwargs.items():
            setattr(self, k, v)

        self.conv = nn.Conv2d(**kwargs)
        self.relu = nn.ReLU(inplace=True)

        self._init()

    def _init(self):
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        if self.pool:
            x = nn.functional.max_pool2d(x, kernel_size=2)
        x = self.relu(self.conv(x))
        return x
