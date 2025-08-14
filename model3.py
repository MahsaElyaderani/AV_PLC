import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

from conformer import Conformer

from resnet_ import ResNetModel
from av_l_dataloader import AVDataloader
from trainer import Trainer, setup_logging


class FusionModule(nn.Module):
    def __init__(self, audio_dim=80):
        super(FusionModule, self).__init__()

        self.fusion_module = Conformer(
            dim=80,
            depth=2,
            dim_head=64,
            heads=4,
            ff_mult=4,
            conv_expansion_factor=2,
            conv_kernel_size=31,
            attn_dropout=0.1,
            ff_dropout=0.1,
            conv_dropout=0.1
        )

        self.mel_proj = nn.Linear(80, 80)
        self.frame_proj = nn.Linear(80, 80)

    def drop_modality(self, audio, video, drop_prob=0.3):
        b = audio.shape[0]
        audio_drop = torch.rand(b) < drop_prob
        video_drop = torch.rand(b) < drop_prob

        for i in range(b):
            if audio_drop[i]:
                audio[i] = torch.zeros(audio[i].shape)
            if video_drop[i]:
                video[i] = torch.zeros(video[i].shape)

        return audio, video

    def forward(self, audio, video, training=False):

        t_a = audio.size(1)
        if training:
            audio, video = self.drop_modality(audio, video)
        fused = torch.cat([audio, video], dim=1)  # [b, t, dim]
        output = self.fusion_module(fused)

        audio_output = output[:, :t_a, :]
        audio_output = self.mel_proj(audio_output)

        video_output = output[:, t_a:, :]
        video_output = self.frame_proj(video_output)

        return audio_output, video_output



class Video_Encoder(nn.Module):
    def __init__(self,  conformer_block=6, hidden_size=512//2,
                 num_heads=4, spkr_vec=256, mel_dim=80, dropout=0.1):
        super(Video_Encoder, self).__init__()

        self.frontend = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(5, 7, 7),
                      stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(True),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )

        self.resnet = ResNetModel(
            layers=18,
            output_dim=hidden_size,
            pretrained=False,
            large_input=False
        )

        self.encoder = Conformer(
            dim=hidden_size + spkr_vec,
            depth=conformer_block,
            dim_head=64,
            heads=num_heads,
            ff_mult=4,
            conv_expansion_factor=2,
            conv_kernel_size=31,
            attn_dropout=dropout,
            ff_dropout=dropout,
            conv_dropout=dropout
        )
        self.norm_layer = nn.LayerNorm(hidden_size + spkr_vec)
        self.mel_proj = nn.Linear(hidden_size + spkr_vec,
                                  4 * mel_dim)  # mel bins are 80 and the output of mel_proj was 160 in the original code

    def forward(self, frame, spk_emb):
        b, t_v, c, h, w = frame.shape
        x = frame.permute(0, 2, 1, 3, 4)  # [b, c=1, t_v, h, w]

        x = self.frontend(x)  # [b, c, t_v, h', w']
        x = self.resnet(x)  # [b, t_v, 512]

        spk_x = torch.cat([x, spk_emb.unsqueeze(1).repeat(1, x.size(1), 1)], dim=-1)
        spk_x = self.encoder(spk_x)  # x: (b, t_v, f)
        spk_x = self.norm_layer(spk_x)

        mel = self.mel_proj(spk_x)
        b, t_v, d = mel.shape
        mel = mel.reshape(b, t_v, d // 4, 4).transpose(-1, -2).reshape(b, t_v * 4, d // 4)

        return mel.permute(0, 2, 1)


class Audio_Decoder(nn.Module):

    def __init__(self, mel_emb=1*80, mel_dim=80, hidden_size=512//2,
                 conformer_block=4, num_heads=4, dropout=0.1):
        super().__init__()

        self.audio_emb = nn.Linear(mel_emb, hidden_size)

        self.conformer = Conformer(
            dim=hidden_size,
            depth=conformer_block,
            dim_head=64,
            heads=num_heads,
            ff_mult=4,
            conv_expansion_factor=2,
            conv_kernel_size=31, #15
            attn_dropout=dropout,
            ff_dropout=dropout,
            conv_dropout=dropout
        )

        self.norm_layer = nn.LayerNorm(hidden_size)
        self.mel_proj = nn.Linear(hidden_size, mel_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.audio_emb(x)
        x = self.conformer(x)
        #x = self.fc_out(x)
        x = self.norm_layer(x)
        x = self.mel_proj(x)
        return x.permute(0, 2, 1)


class AV_ReVoice(nn.Module):
    def __init__(self, l2s_loss=False, hidden_size=256,
                 mel_dim=80, dropout=0.1, num_heads=1,):
        super(AV_ReVoice, self).__init__()

        self.hidden_size = hidden_size
        self.dropout = dropout
        self.num_heads = num_heads

        self.l2s_loss = l2s_loss

        if self.l2s_loss:
            self.video_enc = Video_Encoder()
            video_checkpoint_path = 'checkpoints/video_plc_sc(grid)/best_model.pt'
            video_best_checkpoint = torch.load(video_checkpoint_path)
            self.video_enc.load_state_dict(video_best_checkpoint['model_state_dict'])

            self.audio_dec = Audio_Decoder(mel_emb=1*80)
            self.fusion_module = FusionModule()

        else:
            self.audio_dec = Audio_Decoder(mel_emb=80)

        audio_checkpoint_path = 'checkpoints/audio_plc_pesq_0.01(grid)/best_model.pt'
        best_audio_checkpoint = torch.load(audio_checkpoint_path)
        #consume_prefix_in_state_dict_if_present(best_audio_checkpoint['model_state_dict'], "audio_dec.")
        #audio_state = self.filter_state_keys(self.audio_dec, best_audio_checkpoint['model_state_dict'])
        self.audio_dec.load_state_dict(best_audio_checkpoint['model_state_dict'])

    def filter_state_keys(self, model, checkpoint):
        #checkpoint = torch.load(checkpoint_path)
        model_state = model.state_dict()
        # Filter keys
        clean_state = {k: v for k, v in checkpoint.items() if k in model_state}
        #model.load_state_dict(clean_state, strict=False)
        return clean_state

    def forward(self, dec_input, enc_input, spk_emb):

        if self.l2s_loss:
            x = self.video_enc(enc_input, spk_emb) # output shape: [b, d_a=80, t_a]
            y = self.audio_dec(dec_input)# output shape: [b, d_a=80, t_a]
            audio_output, video_output = self.fusion_module(y.permute(0, 2, 1),
                                                            x.permute(0, 2, 1))  # [b, t_a=200, 2*d_a=160]
            return audio_output.permute(0, 2, 1), x #video_output.permute(0, 2, 1)
        else:
            y = self.audio_dec(dec_input)

            return y


if __name__ == "__main__":

    batch_size = 16
    num_epochs = 200
    learning_rate = 0.0001

    pesq_flag = True
    l2s_flags = [True]
    dataset_names = ['grid'] #['voxceleb2']
    plc_loss_rates = ['20', '30', '40', '50', '60', 'rand']


    log_dir = 'logs'
    checkpoint_dir = 'checkpoints'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for dataset_name in dataset_names:
        for l2s_flag in l2s_flags:
                model_name = f"av_plc{'_sc' if l2s_flag else ''}{'_pesq' if pesq_flag else ''}({dataset_name})"
                logger = setup_logging(model_name, log_dir)
                logger.info(f"Using device: {device}")

                model = AV_ReVoice(l2s_loss=l2s_flag)

                logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
                #model = torch.compile(model)

                logger.info("Initializing dataloaders...")
                av_dataloader = AVDataloader(mode='av', dataset_name=dataset_name,
                                             batch_size=batch_size, num_workers=2)
                train_loader = av_dataloader.train_dataloader()
                val_loader = av_dataloader.val_dataloader()

                #vocoder_path = '/home/ai/Projects/Mahsa/sources/stable_diffusion/dataset/hifigan/checkpoints/seg_len_4096/model-best.pt'

                trainer = Trainer(
                    model=model,
                    mode='av',
                    l2s_loss=l2s_flag,
                    pesq_loss=pesq_flag,
                    model_name=model_name,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    learning_rate=learning_rate,
                    device=device,
                    vocoder_path=None, #vocoder_path,
                    checkpoint_dir=checkpoint_dir,
                    log_dir=log_dir
                )

                # logger.info(f"Starting training for {num_epochs} epochs...")
                # start_epoch = trainer.load_checkpoint(load_best=False)
                # trainer.train(num_epochs=num_epochs, start_epoch=start_epoch)

                for plc_loss_rate in plc_loss_rates:
                    test_loader = av_dataloader.test_dataloader(plc_loss_rate)
                    logger.info("Evaluating model on test set...")
                    test_loss = trainer.evaluate(test_loader, plc_loss_rate)
                    logger.info(f"Final test loss: {test_loss:.4f}")
