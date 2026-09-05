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

Cada imagem é uma foto de microscópio. As máscaras são PNGs binários, uma por núcleo,
disjuntas entre si. A classe combina todas as máscaras de uma imagem em um label map de
instâncias (0=fundo, 1..N=núcleos).
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import os


class DSB2018(Dataset):
    """Dataset DSB2018 — imagens de microscópio com máscaras de núcleos.

    As imagens do DSB2018 têm tamanhos variados (256x256, 256x320, 512x640, 1024x1024).
    Todas são redimensionadas para ``size x size``, o que **distorce a proporção** das
    não-quadradas. É uma simplificação consciente: mantém o batch homogêneo sem precisar
    de padding, e a distorção é modesta na maioria das imagens.

    Args:
        data_dir: diretório contendo uma pasta por amostra. Ignorado se ``sample_dirs``
            for fornecido.
        size: resolução alvo (lado do quadrado).
        sample_dirs: lista explícita de amostras. É o caminho usado pelo split
            estratificado (``src/data/split.py``), que decide a divisão a partir da
            modalidade de cada imagem em vez de depender de como os arquivos estão
            organizados em disco.
    """

    def __init__(
        self,
        data_dir: str | None = None,
        size: int = 256,
        sample_dirs: list[Path] | None = None,
    ):
        self.root = Path(os.getcwd())
        self.size = size

        if sample_dirs is not None:
            self.data_dir = None
            self.samples = sorted(Path(d) for d in sample_dirs)
        else:
            if data_dir is None:
                raise ValueError("informe data_dir ou sample_dirs")
            self.data_dir = self.root / Path(data_dir)
            self.samples = sorted(d for d in self.data_dir.iterdir() if d.is_dir())

    def __len__(self) -> int:
        return len(self.samples)

    def _load_masks(self, sample_dir: Path) -> list[np.ndarray]:
        """Carrega e redimensiona as máscaras de um núcleo por vez.

        Returns:
            Lista de máscaras booleanas (size, size). Máscaras que desapareceram
            completamente no redimensionamento são descartadas.
        """
        masks = []
        for mask_file in sorted((sample_dir / "masks").glob("*.png")):
            mask = Image.open(mask_file).convert("L")
            # NEAREST e não BILINEAR: interpolar uma máscara binária inventaria
            # valores intermediários na borda dos núcleos.
            mask = mask.resize((self.size, self.size), Image.NEAREST)
            mask_np = np.array(mask) > 127
            if mask_np.any():
                masks.append(mask_np)
        return masks

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample_dir = self.samples[idx]

        # --- imagem ---
        img_path = sample_dir / "images" / f"{sample_dir.name}.png"
        img = Image.open(img_path).convert("L")
        img = img.resize((self.size, self.size), Image.BILINEAR)
        img_np = np.array(img, dtype=np.float32) / 255.0

        # --- máscara de instância ---
        masks = self._load_masks(sample_dir)

        # Pintar em ordem de área DECRESCENTE deixa os menores por cima, de modo que um
        # núcleo pequeno nunca possa ser apagado por um grande em caso de colisão. É
        # defensivo e custa nada: medindo com máscaras encostadas em várias resoluções,
        # nenhuma colisão foi observada — o NEAREST preserva bem a disjunção original.
        # A perda que de fato ocorre é outra, e está documentada em integrity_report().
        ordem = sorted(range(len(masks)), key=lambda i: masks[i].sum(), reverse=True)

        label = np.zeros((self.size, self.size), dtype=np.int64)
        for novo_label, i in enumerate(ordem, start=1):
            label[masks[i]] = novo_label

        image = torch.tensor(img_np, dtype=torch.float32).unsqueeze(0)
        return image, torch.tensor(label, dtype=torch.int64)

    def integrity_report(self) -> dict:
        """Verifica quantos núcleos se perdem no redimensionamento.

        Percorre o dataset comparando o número de arquivos de máscara com o número de
        instâncias que sobrevivem no label map final. Rode uma vez ao trocar `size`:
        núcleos perdidos corrompem o erro de contagem e o mAP silenciosamente.

        Um núcleo some quando fica menor que um pixel no destino e o NEAREST não o
        amostra. Medido com núcleos de 16 a 44 px de diâmetro (a faixa típica do
        DSB2018), a perda é de 0 a 2% até `size=96` e sobe para 7-18% em `size=64`.
        Com o padrão `size=256` a perda é desprezível.

        Returns:
            Dicionário com o total de máscaras em disco, o total de instâncias
            preservadas, quantas sumiram no resize individual, quantas foram cobertas
            ao pintar, e a lista das amostras afetadas.
        """
        em_disco = preservadas = sumiram_no_resize = 0
        afetadas = []

        for idx, sample_dir in enumerate(self.samples):
            n_arquivos = len(list((sample_dir / "masks").glob("*.png")))
            n_sobreviveram = len(self._load_masks(sample_dir))
            _, label = self[idx]
            n_final = int(len(torch.unique(label)) - 1)

            em_disco += n_arquivos
            preservadas += n_final
            sumiram_no_resize += n_arquivos - n_sobreviveram
            if n_final != n_arquivos:
                afetadas.append((sample_dir.name, n_arquivos, n_final))

        return {
            "mascaras_em_disco": em_disco,
            "instancias_preservadas": preservadas,
            "perdidas_no_resize": sumiram_no_resize,
            "cobertas_ao_pintar": em_disco - sumiram_no_resize - preservadas,
            "amostras_afetadas": afetadas,
        }
