import math
import torch
import torch.nn as nn

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class AttentionBlock(nn.Module):
    def __init__(self, hidden_size=512, num_heads=8, masking=True):
        super(AttentionBlock, self).__init__()
        self.masking = masking
        self.multihead_attn = nn.MultiheadAttention(hidden_size,
                                                    num_heads=num_heads,
                                                    batch_first=True,
                                                    dropout=0.1)

    def forward(self, x_in, kv_in, key_mask=None):
        if self.masking:
            bs, l, h = x_in.shape
            mask = torch.triu(torch.ones(l, l, device=x_in.device), 1).bool()
        else:
            mask = None
        return self.multihead_attn(x_in, kv_in, kv_in,
                                   attn_mask=mask, key_padding_mask=key_mask)[0]

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size=512, num_heads=8, is_decoder=False, masking=True):
        super(TransformerBlock, self).__init__()
        self.is_decoder = is_decoder

        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn1 = AttentionBlock(hidden_size=hidden_size, num_heads=num_heads, masking=masking)
        if self.is_decoder:
            self.norm2 = nn.LayerNorm(hidden_size)
            self.attn2 = AttentionBlock(hidden_size=hidden_size, num_heads=num_heads, masking=False)

        self.norm_mlp = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(nn.Linear(hidden_size, hidden_size * 2),
                                 nn.GELU(),
                                 nn.Linear(hidden_size * 2, hidden_size))

    def forward(self, x, input_key_mask=None, cross_key_mask=None, kv_cross=None):

        x = self.attn1(x, x, key_mask=input_key_mask) + x
        x = self.norm1(x)

        if self.is_decoder:
            x = self.attn2(x, kv_cross, key_mask=cross_key_mask) + x
            x = self.norm2(x)

        x = self.mlp(x) + x
        return self.norm_mlp(x)


class Encoder(nn.Module):
    def __init__(self, audio_dim=80, hidden_size=512, num_layers=6, num_heads=8):
        super(Encoder, self).__init__()

        self.audio_emb = nn.Sequential(nn.Linear(audio_dim, hidden_size),
                                       nn.ELU(),
                                       nn.Linear(hidden_size, hidden_size),
                                       nn.ELU())
        self.pos_emb = SinusoidalPosEmb(hidden_size)
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, is_decoder=False, masking=False) for _ in range(num_layers)
        ])

    def forward(self,  audio, padding_mask=None):

        a_emb = self.audio_emb(audio)
        ba, la, fa = a_emb.shape
        seq_indx_a = torch.arange(la, device=a_emb.device)
        pos_emb_a = self.pos_emb(seq_indx_a).reshape(1, la, fa).expand(ba, la, fa)
        a_emb = a_emb + pos_emb_a

        embs = a_emb
        for block in self.blocks:
            embs = block(embs, input_key_mask=padding_mask)

        return embs

class Decoder(nn.Module):
    def __init__(self, audio_dim=80, hidden_size=512, num_layers=7, num_heads=8):
        super(Decoder, self).__init__()

        self.audio_emb = nn.Sequential(nn.Linear(audio_dim, hidden_size),
                                       nn.ELU(),
                                       nn.Linear(hidden_size, hidden_size),
                                       nn.ELU())
        self.pos_emb = SinusoidalPosEmb(hidden_size)

        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, is_decoder=True, masking=True) for _ in range(num_layers)
        ])

        layers = [nn.Linear(hidden_size, hidden_size * 4),
                  nn.ELU(),
                  nn.Linear(hidden_size * 4, hidden_size * 4),
                  nn.ELU(),
                  nn.Linear(hidden_size * 4, audio_dim),
                  nn.ELU()]
        self.fc_out = nn.Sequential(*layers)

    def forward(self, audio, encoder_output, input_padding_mask=None, encoder_padding_mask=None):

        a_emb = self.audio_emb(audio)
        ba, la, fa = a_emb.shape
        seq_indx_a = torch.arange(la, device=a_emb.device)
        pos_emb_a = self.pos_emb(seq_indx_a).reshape(1, la, fa).expand(ba, la, fa)
        embs = a_emb + pos_emb_a

        for block in self.blocks:
            embs = block(embs,
                         input_key_mask=input_padding_mask,
                         cross_key_mask=encoder_padding_mask,
                         kv_cross=encoder_output)

        return self.fc_out(embs)

class EncoderDecoder(nn.Module):
    def __init__(self, audio_dim=80, hidden_size=512, num_layers=(6, 7), num_heads=8):
        super(EncoderDecoder, self).__init__()

        self.encoder = Encoder(audio_dim=audio_dim, hidden_size=hidden_size,
                               num_layers=num_layers[0], num_heads=num_heads)

        self.decoder = Decoder(audio_dim=audio_dim, hidden_size=hidden_size,
                               num_layers=num_layers[1], num_heads=num_heads)

    def forward(self, audio):

        input_key_mask = None #input_seq == 0
        output_key_mask = None #target_seq == 0

        encoded_seq = self.encoder(audio=audio,
                                   padding_mask=input_key_mask)

        decoded_seq = self.decoder(audio=audio,
                                   encoder_output=encoded_seq,
                                   input_padding_mask=output_key_mask,
                                   encoder_padding_mask=input_key_mask)

        return decoded_seq