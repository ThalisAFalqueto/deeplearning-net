"""Inferência numa imagem qualquer, sem retreinar — o que o `inferencia.ipynb` usa.

O enunciado pede um notebook que *"recebe o caminho de uma imagem qualquer, devolve a máscara
de instâncias colorida e a contagem"*. A lógica mora aqui, e não dentro do notebook, por um
motivo prático: código de notebook não entra na suíte de testes, e este é justamente o
arquivo que o avaliador vai rodar.

O caminho de ida e volta importa. A rede foi treinada em imagens 256×256 em tons de cinza,
então qualquer imagem precisa ser levada a esse formato antes de entrar. Mas devolver a
máscara em 256² seria inútil para quem passou uma imagem 1024×768: os rótulos não casariam
com os pixels dela. Por isso o label map volta **no tamanho original**, reamostrado por
vizinho mais próximo — que, diferente de uma interpolação, não inventa rótulos intermediários
entre dois objetos vizinhos.
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image

import yaml

from src.evaluation.config import EvalConfig
from src.models.checkpoint import load_checkpoint
from src.models.factory import ModelFactoryRegistry

LADO_DE_TREINO = 256


def carregar_modelo(config: str = "configs/p2_dsb2018.yaml",
                    checkpoint: str = "outputs/p2/best.pth",
                    device=None):
    """Monta o modelo do config e carrega os pesos. Chamar uma vez, prever várias.

    Aceita checkpoints salvos antes e depois da separação backbone/cabeça — quem resolve isso
    é o ``load_checkpoint`` (``src/models/checkpoint.py``).

    Returns:
        ``(modelo em modo eval, EvalConfig, device)``.
    """
    caminho = Path(checkpoint)
    if not caminho.exists():
        raise FileNotFoundError(
            f"checkpoint não encontrado: {caminho}. O modelo final do projeto é "
            f"outputs/p2/best.pth; veja a seção 'Checkpoint' do README.")

    # lê o YAML direto em vez de passar pelo AppConfig: aquele é singleton, e num processo
    # que já tenha criado um AppConfig ele devolveria o config em cache — silenciosamente, com
    # o arquivo errado.
    cfg = EvalConfig.from_dict(yaml.safe_load(Path(config).read_text()))
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = ModelFactoryRegistry.build(cfg).to(device)
    load_checkpoint(caminho, modelo, device)
    modelo.eval()
    return modelo, cfg, device


def _preparar(caminho) -> tuple:
    """Imagem de disco → tensor (1, 256, 256) em [0, 1], e o tamanho original (L, A)."""
    caminho = Path(caminho)
    if not caminho.exists():
        raise FileNotFoundError(f"imagem não encontrada: {caminho}")

    original = Image.open(caminho).convert("L")      # qualquer modo de cor vira tons de cinza
    tamanho = original.size                          # (largura, altura), convenção do PIL
    reduzida = original.resize((LADO_DE_TREINO, LADO_DE_TREINO), Image.BILINEAR)
    arr = np.asarray(reduzida, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0), tamanho


def _reamostrar_rotulos(rotulos: np.ndarray, tamanho: tuple) -> np.ndarray:
    """Leva o label map de volta ao tamanho original por vizinho mais próximo.

    Feito à mão, com indexação, em vez de uma função de imagem: garante que nenhum valor novo
    apareça — o conjunto de rótulos da saída é subconjunto do da entrada.
    """
    largura, altura = tamanho
    alt_atual, larg_atual = rotulos.shape
    ys = np.minimum((np.arange(altura) * alt_atual) // altura, alt_atual - 1)
    xs = np.minimum((np.arange(largura) * larg_atual) // largura, larg_atual - 1)
    return rotulos[ys[:, None], xs[None, :]]


@torch.no_grad()
def prever(caminho, modelo, cfg, device=None, tamanho_original: bool = True) -> dict:
    """Roda o modelo numa imagem de disco e devolve as instâncias e a contagem.

    Args:
        caminho: arquivo de imagem, em qualquer formato que o Pillow abra e qualquer tamanho.
        modelo, cfg, device: o que ``carregar_modelo`` devolveu.
        tamanho_original: ``True`` devolve os rótulos no tamanho da imagem de entrada;
            ``False`` devolve em 256², como a rede viu.

    Returns:
        Dicionário com ``rotulos`` (H, W), ``contagem`` (int), ``probabilidade`` (256, 256) e
        ``imagem`` (a entrada em tons de cinza, no mesmo tamanho de ``rotulos``).
    """
    device = device or next(modelo.parameters()).device
    entrada, tamanho = _preparar(caminho)

    saida = modelo(entrada.unsqueeze(0).to(device))
    logits = (tuple(s[0].cpu() for s in saida) if isinstance(saida, tuple) else saida[0].cpu())
    probabilidade = modelo.foreground_prob(saida)[0].cpu().numpy()
    rotulos = np.asarray(modelo.decode(logits, cfg.decode))

    imagem = entrada[0].numpy()
    if tamanho_original:
        rotulos = _reamostrar_rotulos(rotulos, tamanho)
        imagem = np.asarray(Image.open(caminho).convert("L"), dtype=np.float32) / 255.0

    return {
        "rotulos": rotulos,
        "contagem": int(len(np.unique(rotulos[rotulos > 0]))),
        "probabilidade": probabilidade,
        "imagem": imagem,
    }
