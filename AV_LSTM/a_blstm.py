import torch
import torch.nn as nn

class A_bLSTM(nn.Module):
    def __init__(self,
                 bi_dir=True, asr_loss=False, hidden_size=250,
                 num_layers=3, dropout=0.1, vocab_size=41,
                 spec_len=300, spec_features=80):
        super(A_bLSTM, self).__init__()

        self.bi_dir = bi_dir
        self.asr_loss = asr_loss
        self.dir_coef = 2 if self.bi_dir else 1
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = dropout

        self.spec_len = spec_len
        self.spec_features = spec_features
        self.vocab_size = vocab_size


        self.lstm_layers = nn.ModuleList([
            nn.LSTM(
                input_size=self.spec_features if i == 0 else self.hidden_size * self.dir_coef,
                hidden_size=self.hidden_size,
                num_layers=1,
                bidirectional=self.bi_dir,
                batch_first=True,
                dropout=self.dropout if i < self.num_layers - 1 else 0
            ) for i in range(self.num_layers)
        ])
        #self.layer_norm = nn.LayerNorm(self.hidden_size * self.dir_coef)
        self.spec_fc = nn.Linear(self.hidden_size * self.dir_coef, self.spec_features)
        if self.asr_loss:
            self.ctc_fc = nn.Linear(self.hidden_size * self.dir_coef, self.vocab_size)

        self._init_weights()

    def _init_weights(self):
        for lstm in self.lstm_layers:
            for name, param in lstm.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
        nn.init.xavier_uniform_(self.spec_fc.weight)
        nn.init.zeros_(self.spec_fc.bias)

        if self.asr_loss:
            nn.init.xavier_uniform_(self.ctc_fc.weight)
            nn.init.zeros_(self.ctc_fc.bias)

    def forward(self, spec):

        if spec.dim() == 4:  # (batch, 1, T, spec_features) -> (batch, T, spec_features)
           spec = spec.squeeze(1)
        spec = spec.permute(0, 2, 1)

        x = spec
        for lstm in self.lstm_layers:
            x, _ = lstm(x)
            #x = self.layer_norm(x)

        rec_spec = self.spec_fc(x) #(B, T, spec_features)
        if self.asr_loss:
            text_logits = self.ctc_fc(x) #(B, T, vocab_size)
            return rec_spec.permute(0, 2, 1), text_logits

        return rec_spec.permute(0, 2, 1)