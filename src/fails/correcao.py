"""Parte 5 — a correção que o diagnóstico sugere, com antes e depois.

O diagnóstico corrigido (ver ``src/fails/runner.py``) diz que a pior falha do modelo final
**não** é campo receptivo: em histologia ele **fragmenta** — 9 núcleos viram 39 rótulos —
porque a textura do tecido produz muitos máximos locais fracos no heatmap, e a decodificação
transforma cada máximo num objeto.

A mudança que esse diagnóstico sugere não é treinar de novo: é exigir mais de um pixel para
ele contar como centro. Três parâmetros já existentes controlam isso, todos na decodificação
(``src/utils/decode_center_offset.py``):

    peak_threshold   altura mínima do máximo local para virar centro
    nms_kernel       vizinhança do máximo local — kernel maior funde picos próximos
    min_area         área mínima de uma instância (``src/utils/remove_small_objects``)

Este módulo roda a rede **uma vez** por imagem, guarda os mapas e varre as combinações em
cima deles — o que torna a varredura barata e garante que a única coisa que muda entre o
antes e o depois é a decodificação, nunca os pesos.

Honestidade obrigatória no relato: a varredura escolhe os parâmetros **no mesmo conjunto de
validação** em que o resultado é reportado. Não há conjunto de teste separado neste trabalho,
então o ganho medido é otimista, e isso está dito no JSON e na impressão.
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
from src.fails.runner import FailGalleryRunner, _modo_de_falha
from src.metrics.instance import CountError, MeanAveragePrecision
from src.models.checkpoint import load_checkpoint
from src.models.factory import ModelFactoryRegistry
from src.utils import remove_small_objects

# a grade da varredura. O primeiro valor de cada eixo é o que o config usa hoje, de modo que
# a combinação inicial da grade é exatamente o "antes".
PEAK_THRESHOLDS = (0.5, 0.6, 0.7, 0.8)
NMS_KERNELS = (3, 5, 7)
MIN_AREAS = (0, 10, 25)


class CorrecaoRunner:
    """Varre os parâmetros de decodificação e mede o antes/depois da Parte 5.

    Args:
        app_config: o ``AppConfig`` do modelo.
        checkpoint: pesos a avaliar — os mesmos no antes e no depois.
        saida: pasta de resultados e figuras.
        n: quantas imagens da galeria mostrar no antes/depois.
    """

    def __init__(self, app_config, checkpoint, saida="outputs/p5", n: int = 5):
        self.app_config = app_config
        self.checkpoint = Path(checkpoint)
        self.saida = Path(saida)
        self.n = n
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @torch.no_grad()
    def run(self) -> dict:
        cfg = self.app_config.get_eval_config()
        val = DatasetFactoryRegistry.get(self.app_config).build_val(cfg)
        amostras = getattr(val, "samples", None)
        modalidades = ([classify_samples(amostras)[Path(s)] for s in amostras]
                       if amostras else [None] * len(val))

        model = ModelFactoryRegistry.build(cfg).to(self.device)
        load_checkpoint(self.checkpoint, model, self.device)
        model.eval()

        # roda a rede uma vez só; a varredura acontece em cima dos mapas guardados
        print(f"inferindo {len(val)} imagens de validação uma vez...")
        mapas, gabaritos = [], []
        for idx in range(len(val)):
            imagem, gt = val[idx]
            saida = model(imagem.unsqueeze(0).to(self.device))
            mapas.append(tuple(s[0].cpu() for s in saida) if isinstance(saida, tuple)
                         else saida[0].cpu())
            gabaritos.append(gt.numpy())

        tupla = isinstance(mapas[0], tuple)
        base = dict(cfg.decode)
        combinacoes = self._grade(base, tupla)

        print(f"varrendo {len(combinacoes)} combinações de decodificação...")
        resultados = []
        for combo in combinacoes:
            resultados.append(self._avaliar(model, mapas, gabaritos, modalidades, combo))

        antes = resultados[0]
        depois = max(resultados, key=lambda r: r["map"])
        mudou = depois["decode"] != antes["decode"]
        # o mAP máximo sai do filtro de área, que apaga também núcleos pequenos verdadeiros
        # e piora a contagem. Esta é a escolha que melhora as duas métricas.
        conservadoras = [r for r in resultados if r["map"] > antes["map"]
                         and r["count_error"] <= antes["count_error"]]
        conservadora = max(conservadoras, key=lambda r: r["map"]) if conservadoras else None

        indices = self._piores(gabaritos, antes)
        self.saida.mkdir(parents=True, exist_ok=True)
        self._figura_antes_depois(val, model, mapas, gabaritos, modalidades,
                                  indices, antes, depois)
        self._figura_varredura(resultados, tupla)
        resumo = self._gravar(antes, depois, resultados, indices, mudou, conservadora)
        self._imprimir(resumo)
        return resumo

    # ------------------------------------------------------------------------- varredura

    def _grade(self, base: dict, tupla: bool) -> list:
        """Combinações a testar. A primeira é sempre a do config — o "antes"."""
        picos = PEAK_THRESHOLDS if tupla else (base.get("peak_threshold", 0.5),)
        kernels = NMS_KERNELS if tupla else (base.get("nms_kernel", 3),)
        combinacoes = []
        for area in MIN_AREAS:
            for kernel in kernels:
                for pico in picos:
                    combinacoes.append({**base, "peak_threshold": pico,
                                        "nms_kernel": kernel, "min_area": area})
        # garante que a combinação do config venha primeiro
        atual = {**base, "peak_threshold": base.get("peak_threshold", 0.5),
                 "nms_kernel": base.get("nms_kernel", 3), "min_area": 0}
        combinacoes.remove(atual)
        return [atual] + combinacoes

    def _decodificar(self, model, mapa, decode: dict) -> np.ndarray:
        rotulos = np.asarray(model.decode(mapa, decode))
        if decode.get("min_area", 0) > 0:
            rotulos = remove_small_objects(rotulos, decode["min_area"])
        return rotulos

    def _avaliar(self, model, mapas, gabaritos, modalidades, decode: dict) -> dict:
        """mAP, erro de contagem e razão predição/gabarito para uma combinação."""
        map_metric, count_metric = MeanAveragePrecision(), CountError()
        maps, erros, razoes_por_mod, modos = [], [], {}, {}
        for mapa, gt, modalidade in zip(mapas, gabaritos, modalidades):
            pred = self._decodificar(model, mapa, decode)
            pred_t, gt_t = torch.from_numpy(pred), torch.from_numpy(gt).long()
            m_ap, _ = map_metric(pred_t, gt_t)
            n_gt = int(len(np.unique(gt)) - 1)
            n_pred = int(len(np.unique(pred[pred > 0])))
            maps.append(float(m_ap))
            erros.append(int(count_metric(pred_t, gt_t)))
            razoes_por_mod.setdefault(modalidade, []).append(n_pred / max(n_gt, 1))
            modo = _modo_de_falha(n_gt, n_pred)
            modos[modo] = modos.get(modo, 0) + 1
        return {
            "decode": {k: decode[k] for k in ("peak_threshold", "nms_kernel", "min_area")
                       if k in decode},
            "map": float(np.mean(maps)),
            "count_error": float(np.mean(erros)),
            "map_por_imagem": maps,
            "razao_pred_gt": {m: float(np.median(v)) for m, v in razoes_por_mod.items()},
            "imagens_por_modo": modos,
        }

    def _piores(self, gabaritos, antes: dict) -> list:
        """As n piores imagens do ANTES, pela mesma regra da galeria."""
        registros = [{"idx": i, "map": m, "count_error": 0}
                     for i, m in enumerate(antes["map_por_imagem"])]
        runner = FailGalleryRunner(self.app_config, self.checkpoint, n=self.n)
        return [r["idx"] for r in runner._selecionar(registros)]

    # --------------------------------------------------------------------------- figuras

    def _figura_antes_depois(self, val, model, mapas, gabaritos, modalidades,
                             indices, antes, depois) -> None:
        fig, eixos = plt.subplots(len(indices), 3, figsize=(9, 3 * len(indices)))
        for linha, idx in enumerate(indices):
            gt = gabaritos[idx]
            pred_antes = self._decodificar(model, mapas[idx], {**antes["decode"]})
            pred_depois = self._decodificar(model, mapas[idx], {**depois["decode"]})
            for coluna, (rot, titulo) in enumerate([
                    (gt, f"idx {idx} ({modalidades[idx]})\ngabarito: {len(np.unique(gt)) - 1} núcleos"),
                    (pred_antes, f"antes\n{len(np.unique(pred_antes[pred_antes > 0]))} rótulos"),
                    (pred_depois, f"depois\n{len(np.unique(pred_depois[pred_depois > 0]))} rótulos")]):
                eixos[linha, coluna].imshow(FailGalleryRunner._colorir(rot))
                eixos[linha, coluna].set_title(titulo, fontsize=8)
                eixos[linha, coluna].axis("off")
        fig.suptitle(
            f"Parte 5 — correção na decodificação: {self._rotulo(antes['decode'])} → "
            f"{self._rotulo(depois['decode'])}", fontsize=10)
        plt.tight_layout()
        plt.savefig(self.saida / "p5_antes_depois.png", dpi=120)
        plt.close(fig)

    def _figura_varredura(self, resultados, tupla: bool) -> None:
        fig, eixos = plt.subplots(1, 2, figsize=(12, 4.5))
        for area in sorted({r["decode"].get("min_area", 0) for r in resultados}):
            for kernel in sorted({r["decode"].get("nms_kernel", 3) for r in resultados}):
                pontos = [r for r in resultados
                          if r["decode"].get("min_area", 0) == area
                          and r["decode"].get("nms_kernel", 3) == kernel]
                pontos.sort(key=lambda r: r["decode"].get("peak_threshold", 0.5))
                x = [r["decode"].get("peak_threshold", 0.5) for r in pontos]
                eixos[0].plot(x, [r["map"] for r in pontos], marker="o",
                              label=f"nms {kernel}, área ≥ {area}")
                eixos[1].plot(x, [r["count_error"] for r in pontos], marker="o",
                              label=f"nms {kernel}, área ≥ {area}")
        for eixo, titulo in zip(eixos, ("mAP (maior é melhor)",
                                        "erro de contagem (menor é melhor)")):
            eixo.set_xlabel("peak_threshold")
            eixo.set_title(titulo, fontsize=10)
            eixo.grid(alpha=0.3)
        eixos[1].legend(fontsize=7, ncol=2)
        plt.tight_layout()
        plt.savefig(self.saida / "p5_varredura.png", dpi=110)
        plt.close(fig)

    @staticmethod
    def _rotulo(decode: dict) -> str:
        return (f"pico {decode.get('peak_threshold')}, nms {decode.get('nms_kernel')}, "
                f"área ≥ {decode.get('min_area', 0)}")

    # ------------------------------------------------------------------------- relatório

    def _gravar(self, antes, depois, resultados, indices, mudou, conservadora) -> dict:
        def limpo(r):
            return {k: v for k, v in r.items() if k != "map_por_imagem"}

        resumo = {
            "checkpoint": str(self.checkpoint),
            "diagnostico": (
                "a pior falha do modelo é super-segmentação em histologia: a textura do "
                "tecido gera máximos locais fracos no heatmap e cada um vira um objeto. A "
                "correção aperta a seleção de picos na decodificação; os pesos não mudam."),
            "ressalva": (
                "os parâmetros foram escolhidos no mesmo conjunto de validação em que o "
                "resultado é reportado (o trabalho não tem conjunto de teste separado), "
                "então o ganho medido é otimista."),
            "antes": limpo(antes),
            "depois": limpo(depois),
            # o depois maximiza mAP; esta alternativa melhora mAP sem piorar a contagem
            "alternativa_sem_piorar_contagem": limpo(conservadora) if conservadora else None,
            "houve_mudanca": mudou,
            "ganho_map": depois["map"] - antes["map"],
            "indices_da_galeria": indices,
            "varredura": [limpo(r) for r in resultados],
        }
        (self.saida / "correcao.json").write_text(
            json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")
        return resumo

    def _imprimir(self, resumo) -> None:
        antes, depois = resumo["antes"], resumo["depois"]
        print(f"\n  {'':34s} {'mAP':>7s} {'erro cont.':>11s}")
        print(f"  antes  ({self._rotulo(antes['decode']):24s}) {antes['map']:7.4f} "
              f"{antes['count_error']:11.2f}")
        print(f"  depois ({self._rotulo(depois['decode']):24s}) {depois['map']:7.4f} "
              f"{depois['count_error']:11.2f}")
        alternativa = resumo["alternativa_sem_piorar_contagem"]
        if alternativa:
            print(f"  altern.({self._rotulo(alternativa['decode']):24s}) "
                  f"{alternativa['map']:7.4f} {alternativa['count_error']:11.2f}   "
                  f"<- melhora as duas métricas")
        print(f"\n  ganho de mAP: {resumo['ganho_map']:+.4f} "
              f"(erro de contagem: {antes['count_error']:.2f} -> {depois['count_error']:.2f})")
        print("\n  razão mediana predição/gabarito (1,0 = contagem certa):")
        for modalidade in antes["razao_pred_gt"]:
            print(f"    {str(modalidade):14s} antes {antes['razao_pred_gt'][modalidade]:.2f} "
                  f"→ depois {depois['razao_pred_gt'][modalidade]:.2f}")
        print(f"\n  ressalva: {resumo['ressalva']}")
        print(f"  resultados: {self.saida / 'correcao.json'}")
        print(f"  figuras:    {self.saida}/p5_antes_depois.png, {self.saida}/p5_varredura.png")
