import torch
import torch.nn as nn

class ReVoice_v1(nn.Module):
    def __init__(self,
        bi_dir=True, hidden_size=250,
        num_layers=3, dropout=0.1,
        asr_loss=False, vocab_size=28,
        spec_len=300, spec_features=80,
        motion_len=75, motion_features=40, motion_coords=2):

        super(ReVoice_v1, self).__init__()

        self.bi_dir = bi_dir
        self.dir_coef = 2 if self.bi_dir else 1
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = dropout

        self.asr_loss = asr_loss
        self.vocab_size = vocab_size

        self.spec_len = spec_len
        self.spec_features = spec_features

        self.motion_len = motion_len
        self.motion_features = motion_features
        self.motion_coords = motion_coords
        self.motion_flat_size = self.motion_features * self.motion_coords

        self.encoder = nn.ModuleList([
            nn.LSTM(
                input_size=self.motion_flat_size if i == 0 else self.hidden_size * self.dir_coef,
                hidden_size=self.hidden_size,
                num_layers=1,
                bidirectional=self.bi_dir,
                batch_first=True,
                dropout=self.dropout if i < self.num_layers - 1 else 0
            ) for i in range(self.num_layers)
        ])

        self.decoder = nn.ModuleList([
            nn.LSTM(
                input_size=self.spec_features + self.hidden_size * self.dir_coef if i == 0 else self.hidden_size * self.dir_coef,
                hidden_size=self.hidden_size,
                num_layers=1,
                bidirectional=self.bi_dir,
                batch_first=True,
                dropout=self.dropout if i < self.num_layers - 1 else 0
            ) for i in range(self.num_layers)
        ])

        #self.enc_fc = nn.Linear(self.hidden_size * self.dir_coef, self.hidden_size * self.dir_coef)
        self.dec_fc = nn.Linear(self.hidden_size * self.dir_coef, self.spec_features)

        if self.asr_loss:
            self.asr_fc = nn.Linear(self.hidden_size * self.dir_coef, self.vocab_size)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

    def forward(self, spec, motion_vectors):

        if spec.dim() == 4:
            spec = spec.squeeze(1)
        spec = spec.permute(0, 2, 1)

        x = motion_vectors
        x = x.permute(0, 2, 1)  # (batch, feats, seq_len)
        x = torch.nn.functional.interpolate(x, size=self.spec_len,
                                        mode='linear', align_corners=False)
        x = x.permute(0, 2, 1)

        for lstm in self.encoder:
            x, _ = lstm(x)
        #x = self.enc_fc(x)

        # print("enc output max: ", torch.max(x))
        # print("enc output min: ", torch.min(x))
        # #
        # print("spec max: ", torch.max(spec))
        # print("spec min: ", torch.min(spec))
        #spec = spec / torch.max(spec)

        y = torch.cat((spec, x), dim=-1)
        for lstm in self.decoder:
            y, _ = lstm(y)

        output = self.dec_fc(y)
        if self.asr_loss:
            asr_output = self.asr_fc(x)
            return output.permute(0, 2, 1), asr_output

        return output.permute(0, 2, 1)

