"""Parte 5 — galeria de falhas: onde o modelo final erra feio, e por quê.

Compara o campo receptivo teórico do encoder (``model.receptive_field()``, slides 35-38) com
o tamanho dos objetos do gabarito. Medido no checkpoint da Parte 2 (UNet base=16/depth=3,
RF=68 px): nenhum núcleo isolado do DSB2018 passa do RF (máximo observado, 63 px) — mas
quando vários núcleos se tocam, o BLOB conexo que eles formam no gabarito pode passar do RF
facilmente (vários exemplos medidos acima de 70-100 px), e é dentro desse blob que a cabeça
de offset perde a referência de qual centro pertence a cada pixel. ``_diagnostico`` testa as
duas hipóteses nessa ordem — objeto isolado, depois blob — antes de admitir que a falha não é
de campo receptivo (o caso típico sendo densidade extrema de núcleos minúsculos, que satura a
separação de picos no heatmap, não o alcance do encoder).

Nenhuma arquitetura deste projeto usa atrous/dilated convolution — a comparação "RF com/sem
atrous" do enunciado não se aplica aqui. O mecanismo equivalente já implementado para dar mais
contexto sem mudar a resolução de saída é o Pyramid Pooling Module (``pspnet``/``unet_ppm``),
usado como a correção desta parte.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import ndimage

from src.data.factory import DatasetFactoryRegistry
from src.data.modality import classify_samples
from src.metrics.instance import CountError, MeanAveragePrecision
from src.metrics.semantic import IoU
from src.models.checkpoint import load_checkpoint
from src.models.factory import ModelFactoryRegistry
from src.utils import to_binary

# nenhum backbone do projeto usa atrous/dilated conv (ver docstring do módulo) — mantido
# como constante para o item do enunciado que pede a comparação RF com/sem atrous.
ATROUS_EM_USO = False


def _diametros(labels: np.ndarray) -> list:
    """Diâmetro equivalente (círculo de mesma área) de cada instância do gabarito."""
    diametros = []
    for rotulo in np.unique(labels):
        if rotulo == 0:
            continue
        area = int((labels == rotulo).sum())
        diametros.append(2.0 * float(np.sqrt(area / np.pi)))
    return diametros


def _maior_blob(labels: np.ndarray) -> tuple:
    """Diâmetro equivalente e nº de instâncias do maior componente conexo do foreground.

    Dois núcleos vizinhos têm rótulos diferentes mas podem estar encostados: a máscara
    binária de ambos forma um componente conexo só. É esse blob, não o núcleo individual,
    que pode superar o campo receptivo mesmo quando nenhum núcleo isolado o faz.
    """
    fg = labels > 0
    if not fg.any():
        return 0.0, 0
    blobs, n_blobs = ndimage.label(fg)
    areas = ndimage.sum(fg, blobs, range(1, n_blobs + 1))
    maior_id = int(np.argmax(areas)) + 1
    diametro = 2.0 * float(np.sqrt(areas.max() / np.pi))
    rotulos_no_blob = labels[blobs == maior_id]
    n_instancias = len(np.unique(rotulos_no_blob[rotulos_no_blob != 0]))
    return diametro, int(n_instancias)


class FailGalleryRunner:
    """Roda o modelo na validação e monta a galeria das piores predições.

    Args:
        app_config: o ``AppConfig`` do modelo (dados, arquitetura e ``decode``).
        checkpoint: pesos a avaliar.
        n: quantas imagens piores mostrar (ignorado se ``indices`` for dado).
        indices: força estas imagens em vez de rankear — para reproduzir a mesma galeria
            com um checkpoint diferente (o "depois" de uma correção).
        saida: pasta de resultados e figuras.
    """

    def __init__(self, app_config, checkpoint, n: int = 5,
                 indices: list | None = None, saida="outputs/p5"):
        self.app_config = app_config
        self.checkpoint = Path(checkpoint)
        self.n = n
        self.indices = indices
        self.saida = Path(saida)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @torch.no_grad()
    def run(self) -> dict:
        cfg = self.app_config.get_eval_config()
        limiar = cfg.decode["threshold"]

        val = DatasetFactoryRegistry.get(self.app_config).build_val(cfg)
        amostras = getattr(val, "samples", None)
        modalidades = classify_samples(amostras) if amostras else None

        model = ModelFactoryRegistry.build(cfg).to(self.device)
        load_checkpoint(self.checkpoint, model, self.device)
        model.eval()
        rf = model.receptive_field()

        map_metric = MeanAveragePrecision()
        count_metric = CountError()
        iou_metric = IoU()

        registros, diametros_dataset = [], []
        for idx in range(len(val)):
            imagem, gt = val[idx]
            gt_np = gt.numpy()
            pred_labels_np, prob, _ = self._inferir(model, imagem, cfg)
            pred_labels = torch.tensor(pred_labels_np)
            gt_tensor = gt.long()

            diams = _diametros(gt_np)
            diametros_dataset.extend(diams)
            blob_diam, blob_n = _maior_blob(gt_np)

            m_ap, _ = map_metric(pred_labels, gt_tensor)
            iou_val = float(iou_metric(prob > limiar, to_binary(gt_tensor)))
            count_err = int(count_metric(pred_labels, gt_tensor))

            amostra = amostras[idx].name if amostras else str(idx)
            registros.append({
                "idx": idx, "amostra": amostra,
                "modalidade": modalidades[Path(amostras[idx])] if modalidades else None,
                "n_gt": int(len(np.unique(gt_np)) - 1),
                "n_pred": int(len(np.unique(pred_labels_np)) - 1),
                "map": float(m_ap), "iou": iou_val, "count_error": count_err,
                "diametro_max_objeto": max(diams) if diams else 0.0,
                "diametro_medio_objeto": float(np.mean(diams)) if diams else 0.0,
                "diametro_maior_blob": blob_diam, "instancias_no_maior_blob": blob_n,
            })

        selecionados = self._selecionar(registros)
        for r in selecionados:
            r["rf"] = rf
            r["diagnostico"] = self._diagnostico(r, rf)

        self.saida.mkdir(parents=True, exist_ok=True)
        for rank, r in enumerate(selecionados):
            imagem, gt = val[r["idx"]]
            pred_labels_np, _, logits = self._inferir(model, imagem, cfg)
            self._figura_falha(rank, r, imagem, gt.numpy(), pred_labels_np, logits)

        diametros_dataset = np.array(diametros_dataset)
        self._figura_distribuicao(diametros_dataset, rf)

        resumo = self._gravar(selecionados, diametros_dataset, rf)
        self._imprimir(resumo, selecionados)
        return resumo

    def _inferir(self, model, imagem, cfg):
        """Roda uma imagem sozinha e devolve (label map, prob. de foreground, logits crus)."""
        saida = model(imagem.unsqueeze(0).to(self.device))
        logits = (tuple(s[0].cpu() for s in saida) if isinstance(saida, tuple)
                  else saida[0].cpu())
        prob = model.foreground_prob(saida)[0].cpu()
        pred_labels_np = np.asarray(model.decode(logits, cfg.decode))
        return pred_labels_np, prob, logits

    # ------------------------------------------------------------------- seleção e diagnóstico

    def _selecionar(self, registros: list) -> list:
        if self.indices is not None:
            por_idx = {r["idx"]: r for r in registros}
            return [por_idx[i] for i in self.indices]
        piores = sorted(registros, key=lambda r: (r["map"], -r["count_error"]))
        return piores[: self.n]

    def _diagnostico(self, r: dict, rf: int) -> str:
        if r["diametro_max_objeto"] > rf:
            return (f"objeto de {r['diametro_max_objeto']:.0f} px de diâmetro > campo "
                    f"receptivo teórico de {rf} px — o pixel no centro do objeto nunca "
                    f"enxerga as duas bordas ao mesmo tempo.")
        if r["diametro_maior_blob"] > rf:
            return (f"nenhum núcleo isolado passa do RF ({rf} px), mas "
                    f"{r['instancias_no_maior_blob']} núcleos encostados formam um blob de "
                    f"{r['diametro_maior_blob']:.0f} px — maior que o RF. Um pixel no meio do "
                    f"blob não enxerga nenhum dos centros vizinhos, e o offset não tem como "
                    f"escolher entre eles.")
        return (f"campo receptivo ({rf} px) cobre até o maior objeto/blob desta imagem "
                f"({r['diametro_maior_blob']:.0f} px) — a falha não é de campo receptivo. "
                f"Causa mais provável: {r['n_gt']} núcleos numa imagem só (densidade extrema) "
                f"derrota a separação de picos no heatmap, não o alcance do encoder.")

    # -------------------------------------------------------------------------------- figuras

    @staticmethod
    def _colorir(rotulos: np.ndarray, seed: int = 0) -> np.ndarray:
        _, compacto = np.unique(rotulos, return_inverse=True)
        compacto = compacto.reshape(rotulos.shape)
        rng = np.random.default_rng(seed)
        paleta = np.vstack([[0, 0, 0], rng.random((int(compacto.max()) + 1, 3)) * 0.8 + 0.2])
        return paleta[compacto]

    def _mapa_intermediario(self, logits) -> tuple:
        """Mapa intermediário relevante: heatmap de centros (Trilha C) ou prob. de foreground."""
        if isinstance(logits, tuple):
            _, heatmap_logits, _ = logits
            return torch.sigmoid(heatmap_logits)[0].numpy(), "heatmap de centros"
        return torch.sigmoid(logits)[0].numpy(), "probabilidade de foreground"

    def _figura_falha(self, rank: int, r: dict, imagem, gt: np.ndarray,
                       pred: np.ndarray, logits) -> None:
        mapa, titulo_mapa = self._mapa_intermediario(logits)
        fig, eixos = plt.subplots(1, 4, figsize=(11, 3.2))
        eixos[0].imshow(imagem[0], cmap="gray")
        eixos[0].set_title("imagem", fontsize=9)
        eixos[1].imshow(self._colorir(gt))
        eixos[1].set_title(f"gabarito: {r['n_gt']} núcleos", fontsize=9)
        eixos[2].imshow(self._colorir(pred))
        eixos[2].set_title(f"predição: {r['n_pred']} núcleos", fontsize=9)
        eixos[3].imshow(mapa, cmap="magma", vmin=0, vmax=1)
        eixos[3].set_title(titulo_mapa, fontsize=9)
        for e in eixos:
            e.axis("off")
        fig.suptitle(
            f"falha {rank + 1}: mAP {r['map']:.3f} | maior objeto "
            f"{r['diametro_max_objeto']:.0f}px, maior blob {r['diametro_maior_blob']:.0f}px | "
            f"RF do encoder {r['rf']}px", fontsize=9)
        plt.tight_layout()
        plt.savefig(self.saida / f"falha_{rank + 1}_idx{r['idx']}.png", dpi=120)
        plt.close(fig)

    def _figura_distribuicao(self, diametros: np.ndarray, rf: int) -> None:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(diametros, bins=40, color="tab:blue", alpha=0.75)
        ax.axvline(rf, color="tab:red", lw=2,
                   label=f"campo receptivo teórico do encoder ({rf} px)")
        frac = float((diametros > rf).mean()) if len(diametros) else 0.0
        ax.set_xlabel("diâmetro equivalente do objeto (px)")
        ax.set_ylabel("nº de núcleos")
        ax.set_title(f"distribuição de tamanho dos núcleos vs. campo receptivo\n"
                     f"{frac * 100:.2f}% dos núcleos isolados passam do RF", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.saida / "p5_distribuicao.png", dpi=110)
        plt.close(fig)

    # ------------------------------------------------------------------------------ relatório

    def _gravar(self, selecionados: list, diametros: np.ndarray, rf: int) -> dict:
        resumo = {
            "checkpoint": str(self.checkpoint),
            "campo_receptivo": rf,
            "atrous_em_uso": ATROUS_EM_USO,
            "nota_atrous": (
                "nenhuma arquitetura deste projeto usa atrous/dilated convolution; a "
                "comparação RF com/sem atrous não se aplica. O mecanismo equivalente já "
                "implementado para mais contexto sem perder resolução de saída é o Pyramid "
                "Pooling Module (pspnet/unet_ppm)."),
            "objetos_no_dataset": int(len(diametros)),
            "diametro_p50": float(np.percentile(diametros, 50)) if len(diametros) else None,
            "diametro_p95": float(np.percentile(diametros, 95)) if len(diametros) else None,
            "diametro_max": float(diametros.max()) if len(diametros) else None,
            "fracao_objetos_isolados_acima_do_rf":
                float((diametros > rf).mean()) if len(diametros) else None,
            "piores": selecionados,
        }
        (self.saida / "piores.json").write_text(
            json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")
        return resumo

    def _imprimir(self, resumo: dict, selecionados: list) -> None:
        print(f"\ncampo receptivo teórico do encoder: {resumo['campo_receptivo']} px")
        print(f"diâmetro de objeto no dataset — p50 {resumo['diametro_p50']:.1f} px, "
              f"p95 {resumo['diametro_p95']:.1f} px, máx {resumo['diametro_max']:.1f} px")
        print(f"fração de núcleos isolados acima do RF: "
              f"{resumo['fracao_objetos_isolados_acima_do_rf'] * 100:.2f}%\n")
        for rank, r in enumerate(selecionados):
            print(f"  falha {rank + 1} (idx {r['idx']}, {r['amostra'][:12]}…, "
                  f"modalidade {r['modalidade']}): mAP {r['map']:.4f}, "
                  f"{r['n_gt']} núcleos, {r['count_error']} de erro de contagem")
            print(f"    diagnóstico: {r['diagnostico']}")
        print(f"\n  índices selecionados (para reusar com --fails-indices): "
              f"{[r['idx'] for r in selecionados]}")
        print(f"  resultados: {self.saida / 'piores.json'}")
        print(f"  figuras:    {self.saida}/falha_*.png, {self.saida}/p5_distribuicao.png")
