"""Parte 4 — roda as estratégias de costura nos mosaicos e mede o antes e o depois.

Estratégias, todas com o mesmo modelo e os mesmos mosaicos:

    referencia       imagem inteira numa passada (a rede é totalmente convolucional) —
                     o que os tiles deveriam reproduzir
    ingenua          cada tile decodificado sozinho; cada pixel vem do seu tile dono
                     (a "parte interna" do slide 83) — o antes
    fusao_iou        ingênua + fusão dos pedaços por IoU na faixa de sobreposição — correção A
    mapas_interna    os MAPAS DENSOS costurados pela parte interna, sem média; decodifica uma
                     vez — separa "decodificar uma vez" de "fazer média"
    media_simples    média simples dos mapas densos entre tiles; decodifica uma vez — o
                     "average the results" do slide, literal
    media_interior   média ponderada pelo interior do tile; decodifica uma vez — correção B

Além de mAP, erro de contagem e IoU semântico, mede o **recall dos objetos que cruzam uma
divisa** entre tiles: é nesse grupo (~10% dos núcleos) que o efeito aparece; no mAP total ele
se dilui. O recall do resto serve de controle.

Os mosaicos juntam imagens da **mesma modalidade**. Misturando, uma imagem clara de
brightfield encosta numa escura de fluorescência, e a rede lê a emenda como borda de núcleo:
aparecem objetos falsos ao longo dela, que não têm nada a ver com os tiles (medido: recortado
de volta por imagem, o mAP da referência caía de 0,496 para 0,410 só por estar num mosaico
misto).
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.data.factory import DatasetFactoryRegistry
from src.data.modality import classify_samples
from src.metrics.instance import CountError, IoUMatrix, MeanAveragePrecision
from src.metrics.semantic import IoU
from src.models.checkpoint import load_checkpoint
from src.models.factory import ModelFactoryRegistry
from src.mosaic.build import montar_mosaicos
from src.mosaic.stitch import (
    costura_densa,
    costura_ingenua,
    fusao_por_iou,
    inferir_inteira,
    inferir_tiles,
    media_densa,
    separar,
)
from src.mosaic.tiling import grade_tiles, mapa_de_donos, peso_interior
from src.utils import colorir

ESTRATEGIAS = {
    "referencia": "imagem inteira, sem tiles",
    "ingenua": "tiles + parte interna (antes)",
    "fusao_iou": "correção A: fusão por IoU",
    "mapas_interna": "mapas pela parte interna, decod. 1×",
    "media_simples": "média simples dos mapas",
    "media_interior": "correção B: média pelo interior",
}

# um objeto do gabarito conta como achado se alguma predição o cobre com IoU >= 0,5
IOU_ACHADO = 0.5


def ids_que_cruzam(gt: np.ndarray, donos: np.ndarray) -> np.ndarray:
    """Objetos do gabarito cujos pixels pertencem a mais de um tile dono."""
    fg = gt > 0
    n_tiles = int(donos.max()) + 1
    pares = np.unique(gt[fg].astype(np.int64) * n_tiles + donos[fg])
    ids, contagem = np.unique(pares // n_tiles, return_counts=True)
    return ids[contagem > 1]


def _melhor_iou(pred: np.ndarray, mascara: np.ndarray) -> float:
    """Maior IoU entre um objeto (máscara) e qualquer instância prevista."""
    rotulos, inter = np.unique(pred[mascara], return_counts=True)
    melhor = 0.0
    for r, i in zip(rotulos, inter):
        if r == 0:
            continue
        uniao = mascara.sum() + (pred == r).sum() - i
        melhor = max(melhor, i / uniao)
    return float(melhor)


class MosaicRunner:
    """Monta os mosaicos da validação, roda as estratégias e grava métricas e figuras.

    Args:
        app_config: o ``AppConfig`` do modelo (dados, arquitetura e ``decode``).
        checkpoint: pesos a avaliar.
        tile: lado do tile, em pixels. O padrão é o tamanho de treino.
        passo: distância entre o início de tiles vizinhos; sobreposição = tile − passo.
        limiar_fusao: IoU mínimo na faixa para a fusão juntar dois pedaços.
        saida: pasta de resultados e figuras.
        lado: o mosaico é ``lado × lado`` imagens do dataset.
    """

    def __init__(self, app_config, checkpoint, tile: int = 256, passo: int = 192,
                 limiar_fusao: float = 0.5, saida: str | Path = "outputs/p4", lado: int = 4):
        self.app_config = app_config
        self.checkpoint = Path(checkpoint)
        self.tile = tile
        self.passo = passo
        self.limiar_fusao = limiar_fusao
        self.saida = Path(saida)
        self.lado = lado
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run(self) -> dict:
        cfg = self.app_config.get_eval_config()
        limiar_fg = cfg.decode["threshold"]

        val = DatasetFactoryRegistry.get(self.app_config).build_val(cfg)
        ordem, modalidades = self._ordem_por_modalidade(val)
        mosaicos = montar_mosaicos(val, self.lado, ordem)
        if not mosaicos:
            raise ValueError(f"validação com {len(val)} imagens: poucas para um mosaico "
                             f"{self.lado}×{self.lado}")

        model = ModelFactoryRegistry.build(cfg).to(self.device)
        load_checkpoint(self.checkpoint, model, self.device)
        model.eval()

        altura, largura = mosaicos[0][1].shape
        grade = grade_tiles(altura, largura, self.tile, self.passo)
        donos = mapa_de_donos(altura, largura, self.tile, self.passo)
        peso = peso_interior(self.tile, rampa=self.tile - self.passo)

        def decodificar(denso, tamanhos):
            return np.asarray(model.decode(separar(denso, tamanhos), cfg.decode))

        por_mosaico, exemplo = [], None
        for m, (imagem, gt, fontes) in enumerate(mosaicos):
            cruzam = ids_que_cruzam(gt, donos)

            inteira, tamanhos = inferir_inteira(model, imagem)
            saidas, _ = inferir_tiles(model, imagem, grade, self.tile)
            rotulos_tiles = [decodificar(s, tamanhos) for s in saidas]
            por_dono = costura_densa(saidas, grade, donos, self.tile)
            simples = media_densa(saidas, grade, altura, largura)
            interior = media_densa(saidas, grade, altura, largura, peso)

            # (instâncias, mapa denso de onde sai a máscara semântica)
            preds = {
                "referencia": (decodificar(inteira, tamanhos), inteira),
                "ingenua": (costura_ingenua(rotulos_tiles, grade, donos, self.tile), por_dono),
                "fusao_iou": (fusao_por_iou(rotulos_tiles, grade, donos, self.tile,
                                            self.limiar_fusao), por_dono),
                "mapas_interna": (decodificar(por_dono, tamanhos), por_dono),
                "media_simples": (decodificar(simples, tamanhos), simples),
                "media_interior": (decodificar(interior, tamanhos), interior),
            }

            linha = {"mosaico": m, "fontes": fontes,
                     "modalidade": modalidades[fontes[0]] if modalidades else None,
                     "n_gt": int(len(np.unique(gt)) - 1), "n_cruzam": int(len(cruzam))}
            for nome, (rotulos, denso) in preds.items():
                # canal 0 é o logit de foreground nas duas cabeças (binária e Trilha C)
                fg = torch.sigmoid(denso[0]) > limiar_fg
                linha[nome] = self._metricas(rotulos, gt, fg, cruzam)
            por_mosaico.append(linha)
            print(f"mosaico {m + 1}/{len(mosaicos)}: {linha['n_gt']} núcleos, "
                  f"{linha['n_cruzam']} cruzam divisa | mAP ingênua "
                  f"{linha['ingenua']['map']:.4f} → média interior "
                  f"{linha['media_interior']['map']:.4f}")

            # o exemplo da figura: o melhor candidato entre todos os mosaicos
            alvo = self._objeto_partido(gt, cruzam, preds, bloco=altura // self.lado)
            if alvo is not None and (exemplo is None or alvo[0] > exemplo[0]):
                exemplo = (alvo[0], (imagem, gt, {k: v[0] for k, v in preds.items()}, alvo[1]))

        resumo = self._resumir(por_mosaico, len(mosaicos), altura, largura, len(grade))
        self._gravar(resumo, por_mosaico)
        self._figuras(mosaicos[0], exemplo[1] if exemplo else None, grade, donos, resumo)
        self._imprimir(resumo)
        return resumo

    # ------------------------------------------------------------------------- métricas

    def _metricas(self, pred: np.ndarray, gt: np.ndarray, fg: torch.Tensor,
                  cruzam: np.ndarray) -> dict:
        pred_t, gt_t = torch.from_numpy(pred), torch.from_numpy(gt)
        m_ap, _ = MeanAveragePrecision()(pred_t, gt_t)

        # melhor IoU de cada objeto do gabarito com qualquer predição
        matriz = IoUMatrix()(pred_t, gt_t)
        gt_ids = np.unique(gt[gt > 0])
        melhor = (matriz.max(dim=0).values.numpy() if matriz.shape[0]
                  else np.zeros(len(gt_ids)))
        na_divisa = np.isin(gt_ids, cruzam)
        achado = melhor >= IOU_ACHADO

        return {
            "map": float(m_ap),
            "count_error": int(CountError()(pred_t, gt_t)),
            "iou": float(IoU()(fg, gt_t > 0)),
            "n_pred": int(len(np.unique(pred[pred > 0]))),
            "recall_divisa": float(achado[na_divisa].mean()) if na_divisa.any() else float("nan"),
            "recall_resto": float(achado[~na_divisa].mean()) if (~na_divisa).any() else float("nan"),
        }

    def _ordem_por_modalidade(self, val):
        """Ordem das imagens para os mosaicos: agrupadas por modalidade.

        Cada modalidade contribui só com múltiplos de ``lado²`` imagens, para que nenhum
        mosaico misture modalidades; o que sobra fica de fora. Dentro de cada modalidade,
        mantém a ordem do dataset. Datasets sem ``samples`` (o sintético) usam a ordem dele.

        Returns:
            ``(ordem, modalidade_de_cada_indice)`` ou ``(None, None)``.
        """
        amostras = getattr(val, "samples", None)
        if not amostras:
            return None, None
        mods = classify_samples(amostras)
        modalidades = [mods[Path(s)] for s in amostras]
        grupos: dict[str, list[int]] = {}
        for i, m in enumerate(modalidades):
            grupos.setdefault(m, []).append(i)
        n = self.lado * self.lado
        ordem = []
        for _, indices in sorted(grupos.items(), key=lambda kv: -len(kv[1])):
            ordem += indices[:len(indices) // n * n]
        return ordem, modalidades

    def _objeto_partido(self, gt, cruzam, preds, bloco: int):
        """Um núcleo que a costura ingênua partiu, para a figura.

        Partido = pelo menos dois rótulos da ingênua cobrindo, cada um, ≥ 20% dele. Um bom
        exemplo, além disso, fica longe das emendas do mosaico (a borda de cada imagem-fonte,
        onde o núcleo pode estar cortado pelo próprio gabarito) e é recuperado pela
        referência e pelas duas correções. Entre esses, prefere o maior.

        Returns:
            ``(prioridade, id)`` ou ``None``.
        """
        candidatos = []
        for g in cruzam:
            mascara = gt == g
            rotulos, n = np.unique(preds["ingenua"][0][mascara], return_counts=True)
            if ((rotulos > 0) & (n >= 0.2 * mascara.sum())).sum() < 2:
                continue
            ys, xs = np.nonzero(mascara)
            longe_da_emenda = all(
                (c.min() % bloco) > 4 and (c.max() % bloco) < bloco - 5
                and c.min() // bloco == c.max() // bloco for c in (ys, xs))
            recuperado = all(_melhor_iou(preds[k][0], mascara) >= IOU_ACHADO
                             for k in ("referencia", "fusao_iou", "media_interior"))
            candidatos.append(((recuperado and longe_da_emenda, recuperado,
                                int(mascara.sum())), int(g)))
        return max(candidatos) if candidatos else None

    def _resumir(self, por_mosaico, n_mosaicos, altura, largura, n_tiles) -> dict:
        estrategias = {}
        for nome in ESTRATEGIAS:
            chaves = por_mosaico[0][nome].keys()
            estrategias[nome] = {k: float(np.nanmean([l[nome][k] for l in por_mosaico]))
                                 for k in chaves}
        return {
            "checkpoint": str(self.checkpoint),
            "mosaicos": n_mosaicos,
            "tamanho": [altura, largura],
            "tile": self.tile,
            "passo": self.passo,
            "sobreposicao": self.tile - self.passo,
            "tiles_por_mosaico": n_tiles,
            "limiar_fusao": self.limiar_fusao,
            "nucleos": int(sum(l["n_gt"] for l in por_mosaico)),
            "nucleos_na_divisa": int(sum(l["n_cruzam"] for l in por_mosaico)),
            "estrategias": estrategias,
        }

    def _gravar(self, resumo, por_mosaico) -> None:
        self.saida.mkdir(parents=True, exist_ok=True)
        (self.saida / "summary.json").write_text(json.dumps(resumo, indent=2), encoding="utf-8")
        (self.saida / "per_mosaic.json").write_text(json.dumps(por_mosaico, indent=2),
                                                    encoding="utf-8")

    def _imprimir(self, resumo) -> None:
        print(f"\n{resumo['mosaicos']} mosaicos {resumo['tamanho'][0]}×{resumo['tamanho'][1]}, "
              f"{resumo['nucleos']} núcleos ({resumo['nucleos_na_divisa']} cruzam uma divisa) | "
              f"tile {resumo['tile']}, passo {resumo['passo']}, "
              f"sobreposição {resumo['sobreposicao']}\n")
        print(f"  {'estratégia':34s} {'mAP':>7s} {'erro cont.':>10s} {'IoU':>7s} "
              f"{'recall divisa':>13s} {'recall resto':>12s}")
        for nome, descricao in ESTRATEGIAS.items():
            e = resumo["estrategias"][nome]
            print(f"  {descricao:34s} {e['map']:7.4f} {e['count_error']:10.2f} {e['iou']:7.4f} "
                  f"{e['recall_divisa']:13.3f} {e['recall_resto']:12.3f}")
        print(f"\n  resultados: {self.saida / 'summary.json'}")
        print(f"  figuras:    {self.saida}/p4_*.png")

    # -------------------------------------------------------------------------- figuras

    def _figuras(self, primeiro, exemplo, grade, donos, resumo) -> None:
        self.saida.mkdir(parents=True, exist_ok=True)
        self._figura_mosaico(primeiro, grade, donos)
        if exemplo is not None:
            self._figura_fronteira(*exemplo, donos)
        self._figura_barras(resumo)

    def _figura_mosaico(self, primeiro, grade, donos) -> None:
        """O mosaico com a grade de tiles (contorno) e as divisas entre donos (tracejado)."""
        imagem, gt, _ = primeiro
        fig, eixos = plt.subplots(1, 2, figsize=(12, 6))
        eixos[0].imshow(imagem[0], cmap="gray")
        for y, x in grade:
            eixos[0].add_patch(plt.Rectangle((x - 0.5, y - 0.5), self.tile, self.tile,
                                             fill=False, lw=0.8, ec="tab:orange", alpha=0.8))
        eixos[0].contour(donos, levels=np.arange(donos.max()) + 0.5, colors="cyan",
                         linewidths=0.8, linestyles="--")
        eixos[0].set_title(f"mosaico, {len(grade)} tiles {self.tile}² (laranja) "
                           f"e divisas (tracejado)", fontsize=9)
        eixos[1].imshow(colorir(gt))
        eixos[1].contour(donos, levels=np.arange(donos.max()) + 0.5, colors="white",
                         linewidths=0.6, linestyles="--")
        eixos[1].set_title(f"gabarito: {len(np.unique(gt)) - 1} núcleos", fontsize=9)
        for e in eixos:
            e.axis("off")
        plt.tight_layout()
        plt.savefig(self.saida / "p4_mosaico.png", dpi=110)
        plt.close(fig)

    def _figura_fronteira(self, imagem, gt, rotulos, alvo, donos) -> None:
        """Zoom num núcleo que cruza uma divisa: o que cada estratégia fez com ele."""
        ys, xs = np.nonzero(gt == alvo)
        margem = 24
        y0, y1 = max(ys.min() - margem, 0), min(ys.max() + margem + 1, gt.shape[0])
        x0, x1 = max(xs.min() - margem, 0), min(xs.max() + margem + 1, gt.shape[1])
        recorte = (slice(y0, y1), slice(x0, x1))
        donos_r = donos[recorte]

        paineis = [("imagem", None), ("gabarito", gt)] + [
            (ESTRATEGIAS[k], rotulos[k]) for k in
            ("referencia", "ingenua", "fusao_iou", "media_simples", "media_interior")]
        fig, eixos = plt.subplots(1, len(paineis), figsize=(2.4 * len(paineis), 2.8))
        no_nucleo = gt[recorte] == alvo
        for eixo, (titulo, rot) in zip(eixos, paineis):
            if rot is None:
                eixo.imshow(imagem[0][recorte], cmap="gray")
            else:
                r = rot[recorte]
                # conta só rótulos que cobrem ≥ 10% do núcleo: fiapos de 2 px não são "partes"
                ids, n = np.unique(r[no_nucleo], return_counts=True)
                partes = int(((ids > 0) & (n >= 0.1 * no_nucleo.sum())).sum())
                eixo.imshow(colorir(r))
                titulo = f"{titulo}\n{partes} parte(s) no núcleo" if rot is not gt else titulo
            if len(np.unique(donos_r)) > 1:
                eixo.contour(donos_r, levels=np.unique(donos_r)[:-1] + 0.5, colors="white",
                             linewidths=1.0, linestyles="--")
            eixo.contour(gt[recorte] == alvo, levels=[0.5], colors="yellow", linewidths=0.8)
            eixo.set_title(titulo, fontsize=7)
            eixo.axis("off")
        fig.suptitle("núcleo na divisa entre tiles (contorno amarelo = gabarito; "
                     "tracejado = divisa)", fontsize=8)
        plt.tight_layout()
        plt.savefig(self.saida / "p4_fronteira.png", dpi=130)
        plt.close(fig)

    def _figura_barras(self, resumo) -> None:
        nomes = list(ESTRATEGIAS)
        rotulos = [ESTRATEGIAS[n] for n in nomes]
        cores = ["gray", "tab:red", "tab:green", "tab:olive", "tab:blue", "tab:purple"]
        e = resumo["estrategias"]
        fig, eixos = plt.subplots(1, 3, figsize=(15, 5))
        for eixo, (chave, titulo) in zip(eixos, [("map", "mAP (todos os núcleos)"),
                                                 ("recall_divisa", "recall — núcleos na divisa"),
                                                 ("count_error", "erro de contagem por mosaico")]):
            valores = [e[n][chave] for n in nomes]
            barras = eixo.bar(range(len(nomes)), valores, color=cores)
            eixo.bar_label(barras, fmt="%.3f" if chave != "count_error" else "%.1f", fontsize=7)
            eixo.set_xticks(range(len(nomes)), rotulos, fontsize=7, rotation=35, ha="right")
            eixo.set_title(titulo, fontsize=9)
        plt.tight_layout()
        plt.savefig(self.saida / "p4_barras.png", dpi=110)
        plt.close(fig)
