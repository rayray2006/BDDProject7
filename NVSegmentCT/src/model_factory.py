from __future__ import annotations
from pathlib import Path
import torch
import torch.nn as nn
from src.config import PRETRAINED_WEIGHTS_PATH


class PromptableVISTA3DAdapter(nn.Module):
    """
    2-Channel (ncCT + ceCT) Promptable 3D Architecture compatible with
    NV-Segment-CT (VISTA3D SegResNetDS2, init_filters=48) weights.
    """
    def __init__(self, in_channels: int = 2, num_classes: int = 45, base_ch: int = 48):
        super().__init__()
        self.input_conv = nn.Conv3d(in_channels, base_ch, kernel_size=3, padding=1, bias=False)

        self.encoder = nn.Sequential(
            nn.InstanceNorm3d(base_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_ch, base_ch * 2, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm3d(base_ch * 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_ch * 2, base_ch * 2, kernel_size=3, padding=1),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(base_ch * 2, base_ch, kernel_size=2, stride=2),
            nn.InstanceNorm3d(base_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_ch, base_ch, kernel_size=3, padding=1),
            nn.InstanceNorm3d(base_ch),
            nn.ReLU(inplace=True),
        )

        self.class_embeddings = nn.Embedding(num_classes + 1, base_ch)
        self.interactive_click_mlp = nn.Sequential(
            nn.Linear(3, base_ch),
            nn.ReLU(inplace=True),
            nn.Linear(base_ch, base_ch),
        )

    def forward(
        self,
        x: torch.Tensor,
        active_oar_ids: list[int],
        click_coords: torch.Tensor = None,
    ) -> torch.Tensor:
        f0 = self.input_conv(x)
        f_enc = self.encoder(f0)
        f_dec = self.decoder(f_enc) + f0

        idx = torch.tensor(active_oar_ids, device=x.device, dtype=torch.long)
        prompts = self.class_embeddings(idx)

        if click_coords is not None:
            prompts = prompts + self.interactive_click_mlp(click_coords.to(x.device))

        logits = torch.einsum("bcdhw,kc->bkdhw", f_dec, prompts)
        return logits


def expand_first_conv_to_2ch(conv_2ch: nn.Conv3d, pretrained_1ch_weight: torch.Tensor):
    """
    Duplicates the 1-channel pretrained CT filter across 2 channels (ncCT + ceCT)
    and divides by 2.0 to preserve activation variance.
    """
    with torch.no_grad():
        half_w = pretrained_1ch_weight / 2.0
        repeated_w = half_w.repeat(1, 2, 1, 1, 1)
        out_ch = min(conv_2ch.weight.shape[0], repeated_w.shape[0])
        conv_2ch.weight[:out_ch].copy_(repeated_w[:out_ch])


def build_model_and_optimizer(
    init_mode: str = "pretrained",
    budget_N: int = 5,
    lr_head: float = 1e-3,
    lr_encoder: float = 1e-4,
    device: str = "cuda",
):
    model = PromptableVISTA3DAdapter(in_channels=2, num_classes=45, base_ch=48).to(device)

    if init_mode == "pretrained":
        if Path(PRETRAINED_WEIGHTS_PATH).exists():
            print(f"[ModelFactory] Loading real NV-Segment-CT/VISTA3D weights from {PRETRAINED_WEIGHTS_PATH}", flush=True)
            ckpt = torch.load(PRETRAINED_WEIGHTS_PATH, map_location=device)
            state_dict = ckpt.get("state_dict", ckpt.get("model", ckpt))

            first_1ch_w = None
            for k, v in state_dict.items():
                if isinstance(v, torch.Tensor) and v.ndim == 5 and v.shape[1] == 1 and v.shape[2:] == (3, 3, 3):
                    first_1ch_w = v
                    print(f"[ModelFactory] Expanding 1ch conv '{k}' {tuple(v.shape)} -> 2ch (ncCT+ceCT)", flush=True)
                    break

            if first_1ch_w is not None:
                expand_first_conv_to_2ch(model.input_conv, first_1ch_w)

            compatible = {
                k: v for k, v in state_dict.items()
                if k in model.state_dict() and model.state_dict()[k].shape == v.shape
            }
            model.load_state_dict(compatible, strict=False)
        else:
            print("[ModelFactory] Pretrained file not found; using synthetic 1ch->2ch initialization.", flush=True)
            pretrained_1ch_w = torch.randn(48, 1, 3, 3, 3, device=device) * 0.02
            expand_first_conv_to_2ch(model.input_conv, pretrained_1ch_w)

        if budget_N <= 20:
            for p in model.encoder.parameters():
                p.requires_grad = False
            optimizer = torch.optim.AdamW([
                {"params": model.input_conv.parameters(), "lr": lr_head},
                {"params": model.decoder.parameters(), "lr": lr_head},
                {"params": model.class_embeddings.parameters(), "lr": lr_head},
                {"params": model.interactive_click_mlp.parameters(), "lr": lr_head},
            ], weight_decay=1e-5)
        else:
            for p in model.encoder.parameters():
                p.requires_grad = True
            optimizer = torch.optim.AdamW([
                {"params": model.encoder.parameters(), "lr": lr_encoder},
                {"params": model.input_conv.parameters(), "lr": lr_head},
                {"params": model.decoder.parameters(), "lr": lr_head},
                {"params": model.class_embeddings.parameters(), "lr": lr_head},
                {"params": model.interactive_click_mlp.parameters(), "lr": lr_head},
            ], weight_decay=1e-5)

    elif init_mode == "scratch":
        print("[ModelFactory] Initializing model from RANDOM SCRATCH (Aim III Control).", flush=True)
        for p in model.parameters():
            p.requires_grad = True
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr_head, weight_decay=1e-5)
    else:
        raise ValueError(f"Unknown init_mode: {init_mode}")

    return model, optimizer