import torch
from AV_PLC.multimodal_decoder import AV_PLC

# 1. Instantiate the model with the SAME config used for training
#    (must match fusion_type="local_cross" + local_radius/local_heads used in that run —
#    check e.g. AV_PLC/logs/av_plc_local_cross_r12_l1_only_enc_loss(grid)/fusion_l1_ablation_config.json)
model = AV_PLC(
    fusion_type="local_cross",
    local_radius=12,
    local_heads=4,
)

# 2. Load the checkpoint
ckpt = torch.load("/home/ai/Projects/Mahsa/models/AV_PLC/checkpoints/av_plc_local_cross_r12_l1_only_enc_loss(grid)/best_model.pt", map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# 3. Read the gates
with torch.no_grad():
    video_gate = torch.sigmoid(model.fusion.video_gate_logit).item()
    audio_gate = torch.sigmoid(model.fusion.audio_gate_logit).item()

print("audio → video gate (how much audio-derived context is blended into video):", video_gate)
print("video → audio gate (how much video-derived context is blended into audio gaps):", audio_gate)



# --- capture internal tensors via hooks (no source edits needed) ---
captured = {}

def save_output(name):
    def hook(module, inputs, output):
        captured[name] = output.detach()
    return hook

h1 = model.fusion.norm_a.register_forward_hook(save_output("a"))
h2 = model.fusion.audio_from_video.register_forward_hook(save_output("a_update"))

# --- run one real batch through the model ---
from AV_PLC.av_dataloader import AVDataloader
av_dataloader = AVDataloader(mode="av",
                             dataset_name="grid",
                             batch_size=1, num_workers=0,
                             dropout_modality=False, video_aug=False,)
val_loader = av_dataloader.val_dataloader()
batch = next(iter(val_loader))
visual_feats, spk_emb, masked_spec, spec, video_aligned_spec, audio_length, text, mask, path, avail = batch
with torch.no_grad():
    fused_mel, amel, vmel = model(
        dec_input=masked_spec,
        enc_input=visual_feats,
        spk_emb=spk_emb,
        audio_length=audio_length,
        audio_mask=mask,
        avail=avail,
    )

h1.remove()
h2.remove()

a = captured["a"]              # [N_both, T, D] — post-norm audio, pre-fusion
a_update = captured["a_update"]  # [N_both, T, D] — raw cross-attn output (pre-gate, pre-dropout)

# --- recompute audio_reliability exactly as AV_PLC.forward does ---
audio_reliability_full = AV_PLC._audio_reliability(
    dec_input=masked_spec, audio_mask=mask, target_steps=a.size(1)
)
both = avail[:, 0] & avail[:, 1]
audio_reliability = audio_reliability_full[both]

gap = (1.0 - audio_reliability).bool()

base_norm = a.norm(dim=-1)
update_norm = (torch.sigmoid(model.fusion.audio_gate_logit) * a_update).norm(dim=-1)

effective_ratio = (update_norm[gap] / base_norm[gap].clamp_min(1e-6)).mean()

print("Effective video/audio ratio inside gaps:", effective_ratio.item())

captured = {}

def save_output(name):
    def hook(module, inputs, output):
        captured[name] = output.detach()
    return hook

h1 = model.fusion.norm_a.register_forward_hook(save_output("a"))
h2 = model.fusion.post_a.register_forward_hook(save_output("a_enhanced"))

model.eval()

with torch.no_grad():
    fused_mel, amel, vmel = model(
        dec_input=masked_spec,
        enc_input=visual_feats,
        spk_emb=spk_emb,
        audio_length=audio_length,
        audio_mask=mask,
        avail=avail,
    )

h1.remove()
h2.remove()

a = captured["a"]
a_enhanced = captured["a_enhanced"]

# What the representation would be without the video update.
with torch.no_grad():
    a_baseline = model.fusion.post_a(a)

audio_reliability_full = AV_PLC._audio_reliability(
    dec_input=masked_spec,
    audio_mask=mask,
    target_steps=a.size(1),
)

both = avail[:, 0] & avail[:, 1]
audio_reliability = audio_reliability_full[both]
gap = audio_reliability < 0.5

post_norm_change = (
    (a_enhanced - a_baseline).norm(dim=-1)
    / a_baseline.norm(dim=-1).clamp_min(1e-6)
)[gap].mean()

cosine = torch.nn.functional.cosine_similarity(
    a_enhanced,
    a_baseline,
    dim=-1,
)[gap].mean()

print("Change after post_a:", post_norm_change.item())
print("Cosine similarity after post_a:", cosine.item())