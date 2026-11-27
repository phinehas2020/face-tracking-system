"""Local OSNet person appearance features; model weights never leave the host.

Architecture: KaiyangZhou/deep-person-reid (MIT, see OSNET_LICENSE).
Weights: https://huggingface.co/kaiyangzhou/osnet (MSMT17, x1.0).
"""

from __future__ import annotations

from .accelerator import mps_guard, serialized_mps


class AppearanceEncoder:
    def __init__(self, weights, *, device="auto"):
        import torch
        from torchvision import transforms

        from .osnet_vendor import osnet_x1_0

        if device not in {"auto", "cpu", "mps"}:
            raise ValueError("Appearance device must be auto, cpu, or mps")
        self.device = device
        if device == "auto":
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        with mps_guard(self.device):
            state = torch.load(weights, map_location="cpu", weights_only=True)
            state = state.get("state_dict", state)
            state = {k.removeprefix("module."): v for k, v in state.items()}
            self.model = osnet_x1_0(
                num_classes=state["classifier.weight"].shape[0], pretrained=False
            )
            self.model.load_state_dict(state)
            self.model.eval().to(self.device)
            # Compile the first batch before anybody reaches the doorway.
            with torch.inference_mode():
                self.model(torch.zeros(1, 3, 256, 128, device=self.device))
            if self.device == "mps":
                torch.mps.synchronize()
        # Thread counts are process-global; never mutate them per encoder while
        # other camera inference may already be running.
        self.transform = transforms.Compose(
            [
                transforms.Resize((256, 128)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    @serialized_mps("device")
    def encode(self, frame, tracks):
        import cv2
        import torch
        from PIL import Image

        if not tracks:
            return {}
        tensors = []
        for t in tracks:
            x, y, r, b = map(int, t.bbox)
            crop = frame[
                max(0, y) : min(frame.shape[0], max(y + 1, b)),
                max(0, x) : min(frame.shape[1], max(x + 1, r)),
            ]
            if not crop.size:
                raise ValueError("Empty person crop")
            tensors.append(self.transform(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))))
        with torch.inference_mode():
            features = (
                torch.nn.functional.normalize(
                    self.model(torch.stack(tensors).to(self.device)), dim=1
                )
                .cpu()
                .numpy()
            )
        return {t.track_id: v for t, v in zip(tracks, features, strict=True)}
