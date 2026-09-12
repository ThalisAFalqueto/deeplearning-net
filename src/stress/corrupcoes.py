"""Parte 6 — as três corrupções do enunciado: borrão, ruído e brilho/contraste.

Cada uma é uma função pura sobre o tensor da imagem ``(1, H, W)`` em [0, 1], com os rótulos
intocados. São aplicadas na imagem **já em 256²**, o espaço de entrada do modelo: o que se
testa aqui é o modelo, não o pipeline de dados, e assim o gabarito continua alinhado pixel a
pixel.

Todas têm um valor **neutro** (0) em que a função é a identidade bit a bit, o que dá à curva
de degradação um ponto de partida honesto: a severidade 0 é a mesma imagem que a Parte 2
avaliou, e o mAP nesse ponto tem que bater com o dela.

Duas decisões sobre faixa de valores:

- **o borrão não precisa de corte**. Ele é uma média ponderada de pesos não-negativos que
  somam 1, então a saída nunca sai da faixa da entrada. A borda usa ``reflect``, o padrão do
  scipy: com ``constant`` apareceria uma moldura preta, que a rede leria como borda de objeto
  e contaria como degradação sem ser.
- **o ruído precisa**. Ele empurra valores para fora de [0, 1] — medido, uma imagem clara
  satura no 1,0 —, e cortar ali é o que um sensor real faz. Sem cortar, a rede receberia
  números que nunca viu no treino, e a curva mediria isso em vez da corrupção.
- **o brilho/contraste não chega a precisar**: com força ``f`` ele leva [0, 1] em [``f``, 1]
  (medido nos três níveis), então o corte nunca dispara. Ficou no código como defesa, caso
  alguém mexa nos parâmetros.
"""

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

# fixa o ruído. A semente de cada imagem combina esta base com o índice da imagem e com a
# intensidade, de modo que o resultado não dependa da ordem em que as imagens são percorridas.
SEMENTE_BASE = 20260912


def _virgula(x: float, casas: int = 2) -> str:
    return f"{x:.{casas}f}".replace(".", ",")


def borrao(imagem: torch.Tensor, sigma: float, idx: int = 0) -> torch.Tensor:
    """Borrão gaussiano de desvio ``sigma`` pixels. ``sigma=0`` é a identidade."""
    if sigma == 0:
        return imagem
    arr = imagem.detach().cpu().numpy()
    return torch.from_numpy(gaussian_filter(arr[0], sigma=float(sigma))).unsqueeze(0)


def ruido(imagem: torch.Tensor, sigma: float, idx: int = 0) -> torch.Tensor:
    """Ruído gaussiano aditivo de desvio ``sigma``, cortado em [0, 1].

    A semente vem de ``(SEMENTE_BASE, idx, sigma)``: a mesma imagem na mesma intensidade dá
    sempre o mesmo ruído, e imagens diferentes recebem ruídos diferentes.
    """
    if sigma == 0:
        return imagem
    gerador = np.random.default_rng([SEMENTE_BASE, idx, int(round(sigma * 10_000))])
    barulho = gerador.normal(0.0, float(sigma), size=tuple(imagem.shape)).astype(np.float32)
    return torch.clamp(imagem + torch.from_numpy(barulho), 0.0, 1.0)


def brilho_contraste(imagem: torch.Tensor, forca: float, idx: int = 0) -> torch.Tensor:
    """Reduz contraste e aumenta brilho ao mesmo tempo, cortando em [0, 1].

    ``contraste = 1 - forca`` e ``brilho = forca / 2``, aplicados como
    ``(img - 0,5) * contraste + 0,5 + brilho``. Com ``forca=0`` os dois são neutros (contraste
    1, brilho 0) e a função é a identidade.

    Os dois andam juntos num eixo só porque o enunciado trata "brilho/contraste" como um item.
    """
    if forca == 0:
        return imagem
    contraste, brilho = 1.0 - float(forca), float(forca) / 2.0
    return torch.clamp((imagem - 0.5) * contraste + 0.5 + brilho, 0.0, 1.0)


@dataclass(frozen=True)
class Corrupcao:
    """Uma família de corrupção e as três intensidades que o enunciado pede.

    Args:
        nome: chave usada no JSON e nas figuras.
        funcao: ``(imagem, intensidade, idx) -> imagem``.
        unidade: o que a intensidade significa, para o eixo e a legenda.
        intensidades: exatamente três valores, em ordem crescente de severidade.
        rotulo: formata uma intensidade para leitura humana.
    """

    nome: str
    funcao: Callable[[torch.Tensor, float, int], torch.Tensor]
    unidade: str
    intensidades: tuple
    rotulo: Callable[[float], str]

    def aplicar(self, imagem: torch.Tensor, intensidade: float, idx: int = 0) -> torch.Tensor:
        return self.funcao(imagem, intensidade, idx)


CORRUPCOES = {
    c.nome: c for c in (
        Corrupcao(
            nome="borrao",
            funcao=borrao,
            unidade="σ do borrão (px)",
            # o núcleo mediano tem 13 px de extensão (medido na Parte 5): σ=4 borra cerca de
            # um terço de um núcleo típico
            intensidades=(1.0, 2.0, 4.0),
            rotulo=lambda s: f"σ={s:g} px",
        ),
        Corrupcao(
            nome="ruido",
            funcao=ruido,
            unidade="σ do ruído (fração de [0,1])",
            intensidades=(0.02, 0.05, 0.10),
            rotulo=lambda s: f"σ={_virgula(s)}",
        ),
        Corrupcao(
            nome="brilho_contraste",
            funcao=brilho_contraste,
            unidade="força (contraste 1−f, brilho f/2)",
            intensidades=(0.2, 0.4, 0.6),
            rotulo=lambda f: f"c={_virgula(1 - f, 1)} b=+{_virgula(f / 2)}",
        ),
    )
}

NEUTRO = 0.0
