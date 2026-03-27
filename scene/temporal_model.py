import torch
import torch.nn as nn
import torch.optim as optim
import os
from utils.system_utils import searchForMaxIteration
from utils.general_utils import get_expon_lr_func


class TemporalEncoder(nn.Module):
    """
    Bidirectional LSTM temporal encoder

    Input: (batch, seq_len, flatten_dim)
        - flatten_dim = N × (d_xyz + rotation + scaling)

    Output: (batch, flatten_dim)
        - Predicted deformation parameters for next frame
    """

    def __init__(self, input_dim, hidden_dim=8, num_layers=8, output_dim=None, dropout=0.2):
        super(TemporalEncoder, self).__init__()
        if output_dim is None:
            output_dim = input_dim

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout, bidirectional=True
        )

        # Layer Normalization
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)

        # Output head
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim*2)
        lstm_out = self.layer_norm(lstm_out)
        last_output = lstm_out[:, -1, :]
        pred = self.fc(last_output)
        for i in range(int(pred.size(-1) / x.size(-1))):
            pred[..., i*x.size(-1):(i+1)*x.size(-1)] = (
                pred[..., i*x.size(-1):(i+1)*x.size(-1)] + x[:, -1, :]
            )

        return pred


class TemporalModel:
    def __init__(self, input_dim, hidden_dim=8, num_layers=8, output_dim=None, lr=1e-3):
        if output_dim is None:
            output_dim = input_dim

        self.net = TemporalEncoder(input_dim, hidden_dim, num_layers, output_dim).cuda()
        self._lr_init = lr
        self.optimizer = None

    def step(self, x):
        return self.net(x)

    def train_mode(self):
        self.net.train()

    def eval_mode(self):
        self.net.eval()

    def train_setting(self, training_args):
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self._lr_init)

    def save_weights(self, model_path, iteration):
        out_weights_path = os.path.join(model_path, "temporal/iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save(self.net.state_dict(), os.path.join(out_weights_path, 'temporal.pth'))

    def load_weights(self, model_path, iteration=-1):
        temporal_dir = os.path.join(model_path, "temporal")

        # If directory doesn't exist, this is first training
        if not os.path.exists(temporal_dir):
            return None

        if iteration == -1:
            loaded_iter = searchForMaxIteration(temporal_dir)
        else:
            loaded_iter = iteration

        weights_path = os.path.join(model_path, "temporal/iteration_{}/temporal.pth".format(loaded_iter))

        if os.path.exists(weights_path):
            self.net.load_state_dict(torch.load(weights_path))
            print(f"Loaded temporal weights from {weights_path}")
            return loaded_iter
        else:
            print(f"Warning: Temporal weights not found at {weights_path}")
            return None
        return None
