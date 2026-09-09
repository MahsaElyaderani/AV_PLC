import torch
import torch.nn as nn

class AV_bLSTM(nn.Module):
    def __init__(self,
        hidden_size=250, num_layers=3, dropout=0.1,
        asr_loss=False,  bi_dir=True, vocab_size=40,
        spec_len=300, spec_features=80,
        motion_len=75, motion_features=40, motion_coords=2):

        super(AV_bLSTM, self).__init__()

        self.dropout = dropout
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.bi_dir = bi_dir
        self.dir_coef = 2 if self.bi_dir else 1

        self.asr_loss = asr_loss
        self.vocab_size = vocab_size

        self.spec_len = spec_len
        self.spec_features = spec_features

        self.motion_len = motion_len
        self.motion_features = motion_features
        self.motion_coords = motion_coords
        self.motion_flat_size = self.motion_features * self.motion_coords
        self.av_features = self.spec_features + self.motion_flat_size

        self.lstm_layers = nn.ModuleList([
            nn.LSTM(
                input_size=self.av_features if i == 0 else self.hidden_size * self.dir_coef,
                hidden_size=self.hidden_size,
                num_layers=1,
                bidirectional=True,
                batch_first=True,
                dropout=self.dropout if i < self.num_layers - 1 else 0
            ) for i in range(self.num_layers)
        ])

        self.fc = nn.Linear(self.dir_coef * self.hidden_size, self.spec_features)
        if self.asr_loss:
            self.asr_fc = nn.Linear(self.dir_coef * self.hidden_size, self.vocab_size)

        self._init_weights()

    def _init_weights(self):
        for lstm in self.lstm_layers:
            for name, param in lstm.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)
        if self.asr_loss:
            nn.init.xavier_uniform_(self.asr_fc.weight)
            nn.init.zeros_(self.asr_fc.bias)

    def forward(self, spec, motion_features):

        if spec.dim() == 4:
            spec = spec.squeeze(1)
        spec = spec.permute(0, 2, 1) # b, t, f
        batch_size, spec_len, _ = spec.size()

        x = torch.cat([spec, motion_features], dim=2).contiguous()
        for lstm in self.lstm_layers:
            x, _ = lstm(x)

        output = self.fc(x)
        if self.asr_loss:
            asr_logits = self.asr_fc(x)
            return output.permute(0, 2, 1), asr_logits

        return output.permute(0, 2, 1)