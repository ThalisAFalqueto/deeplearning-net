"""Dataset DSB2018 (Data Science Bowl 2018).

Formato padrão do Kaggle:
    data_dir/
        stage1_train/
            <id>/
                images/
                    <id>.png
                masks/
                    <mask_id>.png  (uma por núcleo)
        stage1_val/
            ...

Cada imagem é uma foto de microscópio. As máscaras são PNGs binários,
uma por núcleo. A classe combina todas as máscaras de uma imagem em um
label map de instâncias (0=fundo, 1..N=núcleos).
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import os


class DSB2018(Dataset):
    """Dataset DSB2018 — imagens de microscópio com máscaras de núcleos.

    Args:
        data_dir: caminho para ``stage1_train`` ou ``stage1_val``.
        size: resolução alvo (redimensiona mantendo proporção).
    """

    def __init__(self, data_dir: str, size: int = 256):
        self.root = Path(os.getcwd())
        self.data_dir = self.root / Path(data_dir)
        self.size = size
        self.samples = sorted(
            d for d in self.data_dir.iterdir() if d.is_dir()
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample_dir = self.samples[idx]

        # --- imagem ---
        img_path = sample_dir / "images" / f"{sample_dir.name}.png"
        img = Image.open(img_path).convert("L")
        img = img.resize((self.size, self.size), Image.BILINEAR)
        img_np = np.array(img, dtype=np.float32) / 255.0

        # --- máscara de instância ---
        mask_dir = sample_dir / "masks"
        mask_files = sorted(mask_dir.glob("*.png"))

        label = np.zeros((self.size, self.size), dtype=np.int64)
        for i, mf in enumerate(mask_files, start=1):
            m = Image.open(mf).convert("L")
            m = m.resize((self.size, self.size), Image.NEAREST)
            m_np = np.array(m) > 127
            label[m_np] = i

        image = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0)
        return image, torch.tensor(label, dtype=torch.int64)
