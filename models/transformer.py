import torch
import torch.nn as nn
import math

class CNN_Transformer(nn.Module):
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
        rnn_input_dim=128,
        d_model=None,
        nhead=4,
        num_encoder_layers=2,
        dim_feedforward=256,
        dropout_trans=0.1,
        activation="gelu",
        use_cls_token=False,
        dropout_head=0.2
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
        rnn_in = in_ch
        if use_1x1_proj and in_ch != rnn_input_dim:
            self.proj = nn.Conv1d(in_ch, rnn_input_dim, kernel_size=1, bias=False)
            rnn_in = rnn_input_dim

        self.d_model = d_model if d_model is not None else rnn_in
        if self.d_model != rnn_in:
            self.to_d_model = nn.Conv1d(rnn_in, self.d_model, kernel_size=1, bias=False)
        else:
            self.to_d_model = None

        assert self.d_model % nhead == 0, f"d_model ({self.d_model}) 必须能被 nhead ({nhead}) 整除"

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout_trans,
            activation=activation,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
            norm=nn.LayerNorm(self.d_model)
        )

        self.pos_encoder = SinusoidalPositionalEncoding(self.d_model)

        self.use_cls_token = use_cls_token
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
            nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.head = nn.Sequential(
            nn.Dropout(p=dropout_head) if dropout_head > 0 else nn.Identity(),
            nn.Linear(self.d_model, num_classes)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)

        if self.proj is not None:
            x = self.proj(x)

        if self.to_d_model is not None:
            x = self.to_d_model(x)

        x = x.permute(0, 2, 1)
        x = self.pos_encoder(x)

        if self.use_cls_token:
            B = x.size(0)
            cls = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls, x], dim=1)

        x = self.transformer(x)

        if self.use_cls_token:
            feat = x[:, 0, :]
        else:
            feat = x.mean(dim=1)

        return self.head(feat)

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x):
        L = x.size(1)
        return x + self.pe[:, :L, :]