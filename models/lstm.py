import torch
import torch.nn as nn

class CNN_LSTM(nn.Module):
    def __init__(
        self,
        input_dim=1,
        num_classes=2,
        base=32,
        num_layers=5,
        pool_layers=2,
        dropout_cnn=0.3,
        use_bn=True,
        use_1x1_proj=True,
        lstm_input_dim=128,
        lstm_hidden=64,
        lstm_layers=1,
        bidirectional=False,
        dropout_lstm_head=0.2
    ):
        super().__init__()

        layers = []
        in_ch = input_dim
        pools_done = 0
        for i in range(num_layers):
            out_ch = base * (2 ** i)  # 32, 64, 128, 256
            layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1, bias=not use_bn))
            if use_bn:
                layers.append(nn.BatchNorm1d(out_ch))
            layers.append(nn.ReLU(inplace=True))

            if pools_done < pool_layers:
                layers.append(nn.MaxPool1d(kernel_size=2, stride=2 if pools_done == 0 else 1))
                pools_done += 1

            if i >= 2 and dropout_cnn > 0:
                layers.append(nn.Dropout(p=dropout_cnn))

            in_ch = out_ch

        self.cnn = nn.Sequential(*layers)

        self.proj = None
        lstm_in = in_ch
        if use_1x1_proj and in_ch != lstm_input_dim:
            self.proj = nn.Conv1d(in_ch, lstm_input_dim, kernel_size=1, bias=False)
            lstm_in = lstm_input_dim

        self.lstm = nn.LSTM(
            input_size=lstm_in,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=0.0 if lstm_layers == 1 else 0.2
        )

        lstm_feat = lstm_hidden * (2 if bidirectional else 1)

        self.head = nn.Sequential(
            nn.Dropout(p=dropout_lstm_head) if dropout_lstm_head > 0 else nn.Identity(),
            nn.Linear(lstm_feat, num_classes)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)

        if self.proj is not None:
            x = self.proj(x)

        x = x.permute(0, 2, 1)

        _, (h_n, _) = self.lstm(x)
        out = h_n[-1]
        return self.head(out)