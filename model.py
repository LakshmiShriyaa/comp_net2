"""CNN-LSTM traffic-matrix forecasters (PyTorch)."""
import torch
import torch.nn as nn


class CNNLSTM(nn.Module):
    """Input : (B, W, N, N)  - W past traffic matrices
    Output: (B, H, N, N)  - H future traffic matrices

    CNN   : extracts spatial structure from each N x N matrix
    LSTM  : models how those features evolve over the W time steps
    Head  : maps the last hidden state to the next H matrices
    """

    def __init__(self, n_nodes: int, horizon: int = 1, cnn_channels: int = 16,
                 feat_dim: int = 128, hidden: int = 128, lstm_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.n, self.h = n_nodes, horizon
        self.cnn = nn.Sequential(
            nn.Conv2d(1, cnn_channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(cnn_channels, cnn_channels * 2, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(cnn_channels * 2 * 16, feat_dim), nn.ReLU(),
        )
        self.lstm = nn.LSTM(feat_dim, hidden, num_layers=lstm_layers, batch_first=True,
                            dropout=dropout if lstm_layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, horizon * n_nodes * n_nodes))

    def forward(self, x):
        B, W, N, _ = x.shape
        f = self.cnn(x.reshape(B * W, 1, N, N)).reshape(B, W, -1)
        out, _ = self.lstm(f)
        y = self.head(out[:, -1])
        return y.reshape(B, self.h, N, N)


class PlainLSTM(nn.Module):
    """Baseline without the CNN: flattens each matrix and feeds it to an LSTM."""

    def __init__(self, n_nodes: int, horizon: int = 1, hidden: int = 128):
        super().__init__()
        self.n, self.h = n_nodes, horizon
        self.lstm = nn.LSTM(n_nodes * n_nodes, hidden, batch_first=True)
        self.head = nn.Linear(hidden, horizon * n_nodes * n_nodes)

    def forward(self, x):
        B, W, N, _ = x.shape
        out, _ = self.lstm(x.reshape(B, W, N * N))
        return self.head(out[:, -1]).reshape(B, self.h, N, N)


class Residual(nn.Module):
    """Predict the CHANGE from the last observed matrix:  out = last_input + base(x).

    The base network's final Linear layer is zero-initialised, so training starts
    exactly at the naive (last value) forecast and can only move away from it if
    that lowers the loss. Works for any base model exposing `.head` and `.h`.
    """

    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base
        last = None
        for m in base.head.modules():
            if isinstance(m, nn.Linear):
                last = m
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, x):
        last = x[:, -1:].expand(-1, self.base.h, -1, -1)
        return last + self.base(x)


class SharedCNNLSTM(nn.Module):
    """Weight-shared CNN-LSTM: ONE small network forecasts every OD pair.

    Why: the dense models above give every one of the N*N matrix entries its own output
    weights, so with ~7,400 training windows they overfit at once. Here the same few
    thousand weights are applied to every OD pair, in units of that pair's typical level
    (`scale`), so the network learns the common dynamics of traffic instead.

      1. z = log1p(x / scale)             per-pair relative level (compresses spikes)
      2. CNN (3x3 convs, size preserved)  gives each pair a feature vector that also sees
                                          its neighbours in the matrix (rows/columns = shared nodes)
      3. LSTM over the W time steps, run separately for every pair with shared weights
      4. Linear head -> predicted change for each of the H steps, in units of `scale`
      5. output = last observed matrix + change * scale     (starts exactly at the naive forecast)

    use_cnn=False drops step 2 (each pair sees only its own history): an ablation that
    shows whether the CNN's spatial context helps.

    x: (B, W, N, N) min-max scaled matrices.  scale: (N, N) typical level per pair in the
    same scaled units.  Returns (B, H, N, N).
    """

    def __init__(self, n_nodes: int, horizon: int, scale, use_cnn: bool = True,
                 channels: int = 16, hidden: int = 32, local_scale: bool = False, use_tod: bool = False):
        super().__init__()
        self.n, self.h, self.use_cnn = n_nodes, horizon, use_cnn
        self.local_scale = local_scale
        self.use_tod = use_tod
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32).reshape(1, 1, n_nodes, n_nodes))
        c_in = 1
        if use_cnn:
            self.cnn = nn.Sequential(
                nn.Conv2d(1, channels, 3, padding=1), nn.ReLU(),
                nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(),
            )
            c_in = channels
        self.lstm = nn.LSTM(c_in + (2 if use_tod else 0), hidden, batch_first=True)
        self.head = nn.Linear(hidden, horizon)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x, tod=None):
        B, W, N, _ = x.shape
        scale = self.scale
        if self.local_scale:
            # per-window level: each pair's mean over the input window, floored at 10% of its global level
            scale = torch.maximum(x.mean(dim=1, keepdim=True), 0.1 * self.scale)       # (B, 1, N, N)
        z = torch.log1p(torch.clamp(x / scale, min=0.0))
        if self.use_cnn:
            f = self.cnn(z.reshape(B * W, 1, N, N))                                  # (B*W, C, N, N)
            C = f.shape[1]
            f = f.reshape(B, W, C, N * N).permute(0, 3, 1, 2).reshape(B * N * N, W, C)
        else:
            f = z.reshape(B, W, N * N).permute(0, 2, 1).reshape(B * N * N, W, 1)
        if self.use_tod:
            # (sin, cos) of the time of day at every input step, the same for all pairs of a window
            f = torch.cat([f, tod.repeat_interleave(N * N, dim=0)], dim=2)                # (B*N*N, W, C+2)
        out, _ = self.lstm(f)                                                        # (B*N*N, W, hidden)
        delta = self.head(out[:, -1]).reshape(B, N, N, self.h).permute(0, 3, 1, 2)   # (B, H, N, N)
        return x[:, -1:] + delta * scale
