from __future__ import annotations

import cv2
import numpy as np


def _bgr_to_tensor(face_bgr: np.ndarray, device: str):
    import torch

    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)
    t = (t - 0.5) / 0.5
    return t.to(device)


def _tensor_to_bgr(tensor) -> np.ndarray:
    import torch

    t = tensor.squeeze(0).detach().cpu().clamp(-1, 1)
    t = (t + 1) / 2
    rgb = (t.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


class GFPGANRestorer:
    def __init__(self, model_path: str, device: str = "cuda"):
        import torch
        from gfpgan.archs.gfpganv1_clean_arch import GFPGANv1Clean

        self.device = device
        self.net = GFPGANv1Clean(
            out_size=512,
            num_style_feat=512,
            channel_multiplier=2,
            decoder_load_path=None,
            fix_decoder=False,
            num_mlp=8,
            input_is_latent=True,
            different_w=True,
            narrow=1,
            sft_half=True,
        )
        loadnet = torch.load(model_path, map_location="cpu")
        keyname = "params_ema" if "params_ema" in loadnet else "params"
        self.net.load_state_dict(loadnet[keyname], strict=False)
        self.net.eval().to(device)

    def restore(self, aligned_bgr: np.ndarray) -> np.ndarray:
        import torch

        x = _bgr_to_tensor(aligned_bgr, self.device)
        with torch.no_grad():
            out = self.net(x, return_rgb=False)[0]
        return _tensor_to_bgr(out)


class CodeFormerRestorer:
    def __init__(self, model_path: str, fidelity_weight: float = 0.5, device: str = "cuda"):
        import torch

        self.device = device
        self.w = float(fidelity_weight)
        self.net = self._build_arch().to(device)
        ckpt = torch.load(model_path, map_location="cpu")
        state = ckpt.get("params_ema", ckpt.get("params", ckpt))
        self.net.load_state_dict(state, strict=False)
        self.net.eval()

    @staticmethod
    def _build_arch():
        candidates = []
        try:
            from basicsr.utils.registry import ARCH_REGISTRY
            try:
                import basicsr.archs.codeformer_arch
            except Exception:
                pass
            try:
                from codeformer.basicsr.archs.codeformer_arch import CodeFormer
                candidates.append(CodeFormer)
            except Exception:
                pass
            try:
                return ARCH_REGISTRY.get("CodeFormer")(
                    dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
                    connect_list=["32", "64", "128", "256"],
                )
            except Exception:
                pass
        except Exception:
            pass
        try:
            from basicsr.archs.codeformer_arch import CodeFormer
            candidates.append(CodeFormer)
        except Exception:
            pass
        try:
            from codeformer.archs.codeformer_arch import CodeFormer
            candidates.append(CodeFormer)
        except Exception:
            pass

        if not candidates:
            raise ImportError(
                "Arsitektur CodeFormer tidak ditemukan. Install salah satu:\n"
                "  pip install codeformer-pip\n"
                "atau clone https://github.com/sczhou/CodeFormer lalu tambahkan "
                "foldernya ke PYTHONPATH."
            )
        Arch = candidates[0]
        return Arch(
            dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
            connect_list=["32", "64", "128", "256"],
        )

    def restore(self, aligned_bgr: np.ndarray) -> np.ndarray:
        import torch

        x = _bgr_to_tensor(aligned_bgr, self.device)
        with torch.no_grad():
            out = self.net(x, w=self.w, adain=True)[0]
        return _tensor_to_bgr(out)


def build_restorer(method: str, model_path: str, device: str = "cuda", **kwargs):
    method = method.lower()
    if method == "gfpgan":
        return GFPGANRestorer(model_path, device=device)
    if method == "codeformer":
        return CodeFormerRestorer(model_path, device=device, **kwargs)
    raise ValueError(f"Metode restorasi wajah tidak dikenal: {method}")
