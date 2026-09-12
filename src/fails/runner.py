"""Parte 5 — galeria de falhas: onde o modelo final erra feio, e por quê.

O item obrigatório é comparar o campo receptivo teórico do encoder
(``model.receptive_field()``, slides 35-38) com a distribuição de tamanhos dos objetos. Duas
decisões de medida importam aqui:

**Qual "tamanho".** O exemplo do enunciado — *"o objeto tem 180 px de diâmetro e o campo
receptivo é 140 px, então o pixel central nunca enxerga as duas bordas"* — fala da **maior
extensão** do objeto. O diâmetro do círculo de mesma área subestima qualquer objeto alongado:
no DSB2018 a 256², 11 núcleos (0,26%) passam dos 68 px de campo receptivo em extensão, e
**nenhum** passa em diâmetro equivalente. As duas medidas são reportadas, sempre nomeadas.

**Qual falha o campo receptivo explica.** Campo receptivo curto faz o modelo **fundir**:
um pixel no meio de um objeto grande não enxerga as duas bordas, nem os centros vizinhos.
Ele não explica o contrário. Por isso ``_diagnostico`` classifica primeiro o modo de falha
(``_modo_de_falha``, contando rótulos previstos dentro do maior blob do gabarito) e só invoca
o campo receptivo quando o modo é compatível. Medido no checkpoint da Parte 2 (UNet
base=16/depth=3, RF=68 px), as piores falhas em histologia **fragmentam** — 2 núcleos viram
13 rótulos —, o que aponta para picos espúrios no heatmap, não para alcance do encoder.

Nenhuma arquitetura deste projeto usa atrous/dilated convolution — a comparação "RF com/sem
atrous" do enunciado não se aplica aqui. O mecanismo equivalente já implementado para dar mais
contexto sem mudar a resolução de saída é o Pyramid Pooling Module (``pspnet``/``unet_ppm``).
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

# razão nº de rótulos previstos / nº de núcleos do gabarito a partir da qual a falha conta
# como fragmentação ou fusão. Fora dessa faixa a contagem está próxima e o erro é de borda.
RAZAO_FRAGMENTA = 1.2
RAZAO_FUNDE = 0.8

# quantas vezes a mediana de núcleos por imagem uma imagem precisa ter para que "densidade"
# seja uma explicação honesta — sem isso o texto culpa a densidade até em imagens com 13 núcleos
FATOR_DENSIDADE = 2.0


def _medidas(mascara: np.ndarray) -> tuple:
    """(diâmetro equivalente, maior extensão) de uma máscara booleana, em px.

    O diâmetro equivalente é o do círculo de mesma área; a extensão é o maior lado da caixa
    que envolve o objeto. Para um disco as duas coincidem; para um núcleo alongado a extensão
    é bem maior, e é ela que o campo receptivo precisa cobrir.
    """
    if not mascara.any():
        return 0.0, 0.0
    ys, xs = np.nonzero(mascara)
    equivalente = 2.0 * float(np.sqrt(int(mascara.sum()) / np.pi))
    extensao = float(max(np.ptp(ys) + 1, np.ptp(xs) + 1))
    return equivalente, extensao


def medidas_instancias(labels: np.ndarray) -> tuple:
    """Diâmetro equivalente e maior extensão de cada instância do gabarito.

    Returns:
        Dois arrays de mesmo comprimento: ``(equivalentes, extensoes)``.
    """
    equivalentes, extensoes = [], []
    for rotulo in np.unique(labels):
        if rotulo == 0:
            continue
        eq, ext = _medidas(labels == rotulo)
        equivalentes.append(eq)
        extensoes.append(ext)
    return np.array(equivalentes), np.array(extensoes)


def maior_blob(labels: np.ndarray) -> dict:
    """Maior componente conexo do foreground: medidas, instâncias e máscara.

    Dois núcleos vizinhos têm rótulos diferentes mas podem estar encostados: a máscara
    binária de ambos forma um componente conexo só. É esse blob, não o núcleo individual,
    que pode superar o campo receptivo mesmo quando nenhum núcleo isolado o faz.
    """
    fg = labels > 0
    if not fg.any():
        return {"diametro": 0.0, "extensao": 0.0, "instancias": 0, "mascara": fg}

    blobs, n_blobs = ndimage.label(fg)
    areas = ndimage.sum(fg, blobs, range(1, n_blobs + 1))
    mascara = blobs == int(np.argmax(areas)) + 1
    equivalente, extensao = _medidas(mascara)
    rotulos = labels[mascara]
    return {
        "diametro": equivalente,
        "extensao": extensao,
        "instancias": int(len(np.unique(rotulos[rotulos != 0]))),
        "mascara": mascara,
    }


def _modo_de_falha(n_gt: int, n_pred: int) -> str:
    """Classifica a falha pela contagem: ``fragmentou``, ``fundiu`` ou ``contagem_proxima``.

    É o que decide se o campo receptivo pode ser a causa. Campo receptivo curto funde
    objetos; nunca divide um objeto em vários.
    """
    if n_gt == 0:
        return "contagem_proxima"
    razao = n_pred / n_gt
    if razao > RAZAO_FRAGMENTA:
        return "fragmentou"
    if razao < RAZAO_FUNDE:
        return "fundiu"
    return "contagem_proxima"


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

        registros, equivalentes_dataset, extensoes_dataset = [], [], []
        for idx in range(len(val)):
            imagem, gt = val[idx]
            gt_np = gt.numpy()
            pred_labels_np, prob, _ = self._inferir(model, imagem, cfg)
            pred_labels = torch.tensor(pred_labels_np)
            gt_tensor = gt.long()

            eq, ext = medidas_instancias(gt_np)
            equivalentes_dataset.extend(eq.tolist())
            extensoes_dataset.extend(ext.tolist())
            blob = maior_blob(gt_np)

            # quantos rótulos a predição colocou dentro do maior blob do gabarito: é a
            # medida direta de "fragmentou este aglomerado" ou "fundiu"
            dentro = pred_labels_np[blob["mascara"]]
            n_pred_no_blob = int(len(np.unique(dentro[dentro != 0])))

            n_gt = int(len(np.unique(gt_np)) - 1)
            n_pred = int(len(np.unique(pred_labels_np)) - 1)
            m_ap, _ = map_metric(pred_labels, gt_tensor)

            registros.append({
                "idx": idx,
                "amostra": amostras[idx].name if amostras else str(idx),
                "modalidade": modalidades[Path(amostras[idx])] if modalidades else None,
                "n_gt": n_gt, "n_pred": n_pred,
                "map": float(m_ap),
                "iou": float(iou_metric(prob > limiar, to_binary(gt_tensor))),
                "count_error": int(count_metric(pred_labels, gt_tensor)),
                "modo_falha": _modo_de_falha(n_gt, n_pred),
                "diametro_max_objeto": float(eq.max()) if len(eq) else 0.0,
                "extensao_max_objeto": float(ext.max()) if len(ext) else 0.0,
                "diametro_maior_blob": blob["diametro"],
                "extensao_maior_blob": blob["extensao"],
                "instancias_no_maior_blob": blob["instancias"],
                "pred_no_maior_blob": n_pred_no_blob,
            })

        mediana_n_gt = float(np.median([r["n_gt"] for r in registros]))
        selecionados = self._selecionar(registros)
        for r in selecionados:
            r["rf"] = rf
            r["diagnostico"] = self._diagnostico(r, rf, mediana_n_gt)

        self.saida.mkdir(parents=True, exist_ok=True)
        for rank, r in enumerate(selecionados):
            imagem, gt = val[r["idx"]]
            pred_labels_np, _, logits = self._inferir(model, imagem, cfg)
            self._figura_falha(rank, r, imagem, gt.numpy(), pred_labels_np, logits)

        equivalentes = np.array(equivalentes_dataset)
        extensoes = np.array(extensoes_dataset)
        self._figura_distribuicao(equivalentes, extensoes, rf)

        resumo = self._gravar(selecionados, equivalentes, extensoes, rf, mediana_n_gt)
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

    # ------------------------------------------------------------- seleção e diagnóstico

    def _selecionar(self, registros: list) -> list:
        if self.indices is not None:
            por_idx = {r["idx"]: r for r in registros}
            return [por_idx[i] for i in self.indices]
        piores = sorted(registros, key=lambda r: (r["map"], -r["count_error"]))
        return piores[: self.n]

    def _diagnostico(self, r: dict, rf: int, mediana_n_gt: float) -> str:
        """Texto do diagnóstico, guiado pelo modo de falha.

        O campo receptivo só é apontado como causa quando o modelo **funde** — que é o que
        um campo receptivo curto produz. Quando ele fragmenta, o texto diz explicitamente
        que o RF não explica, mesmo que o objeto passe do RF.
        """
        blob_ext = r["extensao_maior_blob"]
        cabe = f"o maior blob desta imagem tem {blob_ext:.0f} px de extensão e o campo " \
               f"receptivo teórico do encoder é {rf} px"

        if r["modo_falha"] == "fragmentou":
            return (
                f"super-segmentação: {r['n_gt']} núcleos viraram {r['n_pred']} rótulos "
                f"({r['n_pred'] / r['n_gt']:.1f}×). Dentro do maior blob, "
                f"{r['instancias_no_maior_blob']} núcleo(s) viraram "
                f"{r['pred_no_maior_blob']} rótulo(s). **Não é campo receptivo** — "
                f"{cabe}, mas campo receptivo curto funde objetos, nunca os divide. "
                f"Causa provável: picos espúrios no heatmap sobre a textura de "
                f"{r['modalidade']}, e cada pico vira um objeto na decodificação.")

        if r["modo_falha"] == "fundiu":
            if r["extensao_max_objeto"] > rf:
                return (
                    f"fusão: {r['n_gt']} núcleos viraram {r['n_pred']} rótulos. Há objeto de "
                    f"{r['extensao_max_objeto']:.0f} px de extensão (diâmetro equivalente "
                    f"{r['diametro_max_objeto']:.0f} px) contra campo receptivo de {rf} px — "
                    f"o pixel no centro dele nunca enxerga as duas bordas ao mesmo tempo.")
            if blob_ext > rf:
                return (
                    f"fusão: {r['n_gt']} núcleos viraram {r['n_pred']} rótulos. Nenhum núcleo "
                    f"isolado passa do RF ({rf} px), mas {r['instancias_no_maior_blob']} "
                    f"núcleos encostados formam um blob de {blob_ext:.0f} px de extensão. Um "
                    f"pixel no meio do blob não enxerga nenhum dos centros vizinhos, e o "
                    f"offset não tem como escolher entre eles — o blob saiu com "
                    f"{r['pred_no_maior_blob']} rótulo(s) para "
                    f"{r['instancias_no_maior_blob']} núcleos.")
            causa = (f"densidade: {r['n_gt']} núcleos numa imagem só, contra a mediana de "
                     f"{mediana_n_gt:.0f} do conjunto"
                     if r["n_gt"] > FATOR_DENSIDADE * mediana_n_gt else
                     f"núcleos pequenos demais para o heatmap separar (maior objeto: "
                     f"{r['extensao_max_objeto']:.0f} px)")
            return (
                f"fusão: {r['n_gt']} núcleos viraram {r['n_pred']} rótulos, mas {cabe} — "
                f"a falha **não é de campo receptivo**. Causa provável: {causa}; picos "
                f"vizinhos se fundem num só e a decodificação perde os centros.")

        return (
            f"a contagem está próxima ({r['n_gt']} núcleos, {r['n_pred']} rótulos) e o mAP "
            f"ainda é {r['map']:.3f}: o erro está nas **bordas**, não na separação. {cabe}, "
            f"então o campo receptivo não é a causa.")

    # ---------------------------------------------------------------------------- figuras

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
        fig, eixos = plt.subplots(1, 4, figsize=(11, 3.4))
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
        rotulo_modo = {"fragmentou": "SUPER-SEGMENTOU", "fundiu": "FUNDIU",
                       "contagem_proxima": "contagem próxima"}[r["modo_falha"]]
        fig.suptitle(
            f"falha {rank + 1} ({r['modalidade']}): mAP {r['map']:.3f} | {rotulo_modo} | "
            f"maior objeto {r['extensao_max_objeto']:.0f}px de extensão, maior blob "
            f"{r['extensao_maior_blob']:.0f}px | RF do encoder {r['rf']}px", fontsize=9)
        plt.tight_layout()
        plt.savefig(self.saida / f"falha_{rank + 1}_idx{r['idx']}.png", dpi=120)
        plt.close(fig)

    def _figura_distribuicao(self, equivalentes, extensoes, rf: int) -> None:
        """Histograma dos tamanhos contra o campo receptivo, nas duas definições.

        A comparação obrigatória do enunciado. As duas curvas aparecem juntas porque a
        conclusão muda entre elas: pelo diâmetro equivalente nenhum núcleo alcança o RF;
        pela extensão, alguns alcançam.
        """
        fig, ax = plt.subplots(figsize=(8.5, 5))
        bins = np.linspace(0, max(extensoes.max(), rf) * 1.05, 45)
        ax.hist(extensoes, bins=bins, color="tab:orange", alpha=0.65,
                label="maior extensão (o que o RF precisa cobrir)")
        ax.hist(equivalentes, bins=bins, color="tab:blue", alpha=0.65,
                label="diâmetro equivalente (círculo de mesma área)")
        ax.axvline(rf, color="tab:red", lw=2,
                   label=f"campo receptivo teórico do encoder ({rf} px)")
        f_ext = float((extensoes > rf).mean()) if len(extensoes) else 0.0
        f_eq = float((equivalentes > rf).mean()) if len(equivalentes) else 0.0
        ax.set_xlabel("tamanho do núcleo (px)")
        ax.set_ylabel("nº de núcleos")
        ax.set_title(
            f"tamanho dos núcleos vs. campo receptivo ({len(extensoes)} núcleos)\n"
            f"passam do RF: {f_ext * 100:.2f}% pela extensão, "
            f"{f_eq * 100:.2f}% pelo diâmetro equivalente", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.saida / "p5_distribuicao.png", dpi=110)
        plt.close(fig)

    # --------------------------------------------------------------------------- relatório

    def _gravar(self, selecionados, equivalentes, extensoes, rf: int,
                mediana_n_gt: float) -> dict:
        def percentis(v):
            if not len(v):
                return {"p50": None, "p95": None, "max": None, "acima_do_rf": None}
            return {"p50": float(np.percentile(v, 50)), "p95": float(np.percentile(v, 95)),
                    "max": float(v.max()), "acima_do_rf": float((v > rf).mean())}

        resumo = {
            "checkpoint": str(self.checkpoint),
            "campo_receptivo": rf,
            "atrous_em_uso": ATROUS_EM_USO,
            "nota_atrous": (
                "nenhuma arquitetura deste projeto usa atrous/dilated convolution; a "
                "comparação RF com/sem atrous não se aplica. O mecanismo equivalente já "
                "implementado para mais contexto sem perder resolução de saída é o Pyramid "
                "Pooling Module (pspnet/unet_ppm)."),
            "objetos_no_dataset": int(len(extensoes)),
            "mediana_nucleos_por_imagem": mediana_n_gt,
            # as duas definições de tamanho; a extensão é a que o campo receptivo precisa cobrir
            "extensao": percentis(extensoes),
            "diametro_equivalente": percentis(equivalentes),
            "piores": [{k: v for k, v in r.items() if k != "mascara"} for r in selecionados],
        }
        (self.saida / "piores.json").write_text(
            json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")
        return resumo

    def _imprimir(self, resumo: dict, selecionados: list) -> None:
        ext, eq = resumo["extensao"], resumo["diametro_equivalente"]
        print(f"\ncampo receptivo teórico do encoder: {resumo['campo_receptivo']} px "
              f"| {resumo['objetos_no_dataset']} núcleos")
        print(f"  maior extensão     — p50 {ext['p50']:.1f} p95 {ext['p95']:.1f} "
              f"máx {ext['max']:.1f} px | acima do RF: {ext['acima_do_rf'] * 100:.2f}%")
        print(f"  diâm. equivalente  — p50 {eq['p50']:.1f} p95 {eq['p95']:.1f} "
              f"máx {eq['max']:.1f} px | acima do RF: {eq['acima_do_rf'] * 100:.2f}%\n")
        for rank, r in enumerate(selecionados):
            print(f"  falha {rank + 1} (idx {r['idx']}, {r['amostra'][:12]}…, "
                  f"{r['modalidade']}): mAP {r['map']:.4f}, {r['n_gt']} núcleos → "
                  f"{r['n_pred']} rótulos [{r['modo_falha']}]")
            print(f"    diagnóstico: {r['diagnostico']}")
        print(f"\n  índices selecionados (para reusar com --fails-indices): "
              f"{[r['idx'] for r in selecionados]}")
        print(f"  resultados: {self.saida / 'piores.json'}")
        print(f"  figuras:    {self.saida}/falha_*.png, {self.saida}/p5_distribuicao.png")
