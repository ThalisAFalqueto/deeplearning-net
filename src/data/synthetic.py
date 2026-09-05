"""Dataset sintético de elipses para a Parte 0.

Gera imagens 128x128 com 5-20 elipses de tamanhos variados, muitas delas se
tocando (p_touch sorteado por imagem), com ruído e contraste variáveis.
As máscaras de instância são geradas gratuitamente pelo próprio desenho.

Usado como "teste unitário" do enunciado: o modelo deve treinar em menos de
5 minutos em CPU e reportar a métrica de instância.
"""

from pathlib import Path

import numpy as np
import torch
from scipy import ndimage
from skimage.draw import ellipse
from torch.utils.data import Dataset


class SyntheticEllipses(Dataset):
    """Dataset sintético de elipses sobrepostas.

    Cada imagem contém 5-20 elipses (raios 5-14 px) desenhadas sobre um fundo
    escuro. O ``p_touch`` é sorteado por imagem em [0, 0.9]: quando ativado,
    uma nova elipse é centralizada perto de uma existente, produzindo fusão
    de objetos (60% dos casos, conforme especificado na PARTE_0.md).

    Args:
        n_samples: número de imagens a gerar.
        size: resolução das imagens quadradas (default 128).
        seed: semente para reprodutibilidade.
        min_obj: número mínimo de elipses por imagem.
        max_obj: número máximo de elipses por imagem.
    """

    def __init__(
        self,
        n_samples: int,
        size: int = 128,
        seed: int = 42,
        min_obj: int = 5,
        max_obj: int = 20,
    ):
        self.n_samples = n_samples
        self.size = size
        self.min_obj = min_obj
        self.max_obj = max_obj
        self.rng = np.random.RandomState(seed)

        # Pré-gera tudo em memória (num_workers=0 no config, dataset é leve)
        self._images = []
        self._labels = []
        for _ in range(n_samples):
            img, lbl = self._generate_one()
            self._images.append(img)
            self._labels.append(lbl)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = torch.tensor(self._images[idx], dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self._labels[idx], dtype=torch.int64)
        return image, label

    def _generate_one(self) -> tuple[np.ndarray, np.ndarray]:
        """Gera uma imagem e sua máscara de instância."""
        n_obj = self.rng.randint(self.min_obj, self.max_obj + 1)
        p_touch = self.rng.uniform(0, 0.9)

        # Fundo escuro com intensidade variável
        bg_intensity = self.rng.uniform(0.0, 0.15)
        image = np.full((self.size, self.size), bg_intensity, dtype=np.float32)
        label = np.zeros((self.size, self.size), dtype=np.int64)

        centers = []  # centros de elipses já desenhadas (para p_touch)

        for obj_id in range(1, n_obj + 1):
            r_radius = self.rng.randint(5, 15)
            c_radius = self.rng.randint(5, 15)

            if centers and self.rng.random() < p_touch:
                # Coloca perto de um elipse existente → fusão
                ref_r, ref_c = centers[self.rng.randint(len(centers))]
                spread = max(r_radius, c_radius)
                r = ref_r + self.rng.uniform(-spread, spread)
                c = ref_c + self.rng.uniform(-spread, spread)
            else:
                r = self.rng.uniform(r_radius, self.size - r_radius)
                c = self.rng.uniform(c_radius, self.size - c_radius)

            # Garantir dentro da imagem
            r = np.clip(r, r_radius, self.size - r_radius - 1)
            c = np.clip(c, c_radius, self.size - c_radius - 1)

            rotation = self.rng.uniform(0, 2 * np.pi)

            rr, cc = ellipse(
                r, c, r_radius, c_radius,
                rotation=rotation, shape=(self.size, self.size),
            )

            # Intensidade da elipse (foreground mais claro que fundo)
            intensity = self.rng.uniform(0.5, 0.95)
            image[rr, cc] = intensity
            label[rr, cc] = obj_id

            centers.append((r, c))

        # Ruído gaussiano (variado por imagem)
        noise_sigma = self.rng.uniform(0.02, 0.1)
        image += self.rng.normal(0, noise_sigma, image.shape).astype(np.float32)

        # Variação de contraste e brilho
        gain = self.rng.uniform(0.8, 1.2)
        offset = self.rng.uniform(-0.05, 0.05)
        image = image * gain + offset

        # Clip final para [0, 1]
        image = np.clip(image, 0.0, 1.0)

        # Renumber labels para garantir 0, 1, 2, ..., N contíguos
        unique = np.unique(label)
        unique = unique[unique != 0]
        for new_id, old_id in enumerate(unique, start=1):
            label[label == old_id] = new_id

        return image, label
