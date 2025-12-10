import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self, num_layers=3, input_channels=1, num_classes=2, num_base=32):
        super().__init__()
        layers = []
        in_ch = input_channels
        base = num_base
        for i in range(num_layers):
            out_ch = base * (2 ** i)
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
            ]
            if i < 2:
                layers.append(nn.MaxPool1d(kernel_size=2, stride=2))  # L: 10 -> 5 -> 2
            if i >= 1:
                layers.append(nn.Dropout(p=0.2))
            in_ch = out_ch

        self.cnn = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool1d(output_size=1)
        self.head = nn.Sequential(
            nn.Flatten(),                  
            nn.Dropout(p=0.2),             
            nn.Linear(in_ch, num_classes),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = self.gap(x)
        return self.head(x)