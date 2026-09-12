"""Parte 6 — roda o modelo sob corrupções e monta a curva de degradação.

São 10 condições: a imagem limpa mais três famílias de corrupção em três intensidades cada
(`src/stress/corrupcoes.py`). Em todas, o modelo e os pesos são os mesmos — muda só a imagem
que entra. Os rótulos nunca passam por corrupção alguma.

Duas coisas que o relato precisa dizer junto com a curva:

- **o treino não usa augmentation.** Não há transform, flip nem rotação em nenhum lugar do
  `src/`. O modelo nunca viu borrão, ruído ou mudança de brilho, então uma curva íngreme é o
  resultado esperado, não uma descoberta.
- **a severidade 0 é a imagem limpa**, e o mAP nesse ponto tem que reproduzir o da Parte 2
  nesta máquina. É a checagem embutida de que o caminho novo não mexeu em nada.
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
from src.fails.runner import FailGalleryRunner
from src.metrics.instance import CountError, MeanAveragePrecision
from src.metrics.semantic import IoU
from src.models.checkpoint import load_checkpoint
from src.models.factory import ModelFactoryRegistry
from src.stress.corrupcoes import CORRUPCOES, NEUTRO
from src.utils import to_binary

# mAP da Parte 2 nesta máquina (outputs/p5/correcao.json -> antes). A condição limpa tem que
# bater com isto; se não bater, algo no caminho de avaliação mudou.
MAP_LIMPO_ESPERADO = 0.498414496670558


class StressRunner:
    """Avalia o modelo em cada corrupção e intensidade, e grava a curva de degradação.

    Args:
        app_config: o ``AppConfig`` do modelo (dados, arquitetura e ``decode``).
        checkpoint: pesos a avaliar — os mesmos em todas as condições.
        saida: pasta de resultados e figuras.
    """

    def __init__(self, app_config, checkpoint, saida="outputs/p6"):
        self.app_config = app_config
        self.checkpoint = Path(checkpoint)
        self.saida = Path(saida)
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

        limpa = self._condicao(model, cfg, val, modalidades, None, NEUTRO)
        print(f"limpa: mAP {limpa['map']:.4f} | IoU {limpa['iou']:.4f} | "
              f"erro de contagem {limpa['count_error']:.2f}")
        self._conferir_limpa(limpa["map"])

        por_imagem_limpa = limpa.pop("por_imagem")
        familias, por_imagem = {}, list(por_imagem_limpa)
        for nome, corrupcao in CORRUPCOES.items():
            niveis = []
            for severidade, intensidade in enumerate(corrupcao.intensidades, start=1):
                r = self._condicao(model, cfg, val, modalidades, corrupcao, intensidade)
                por_imagem.extend(r.pop("por_imagem"))
                niveis.append({"severidade": severidade, "intensidade": intensidade,
                               "rotulo": corrupcao.rotulo(intensidade), **r})
                print(f"{nome} {corrupcao.rotulo(intensidade):>16s}: mAP {r['map']:.4f} "
                      f"({r['map'] - limpa['map']:+.4f}) | IoU {r['iou']:.4f} "
                      f"({r['iou'] - limpa['iou']:+.4f})")
            familias[nome] = {"unidade": corrupcao.unidade, "niveis": niveis}

        self.saida.mkdir(parents=True, exist_ok=True)
        resumo = self._gravar(limpa, familias, por_imagem, len(val))
        self._figura_degradacao(limpa, familias)
        self._figura_exemplos(model, cfg, val, limpa, por_imagem_limpa)
        self._imprimir(resumo)
        return resumo

    # ------------------------------------------------------------------------ avaliação

    def _condicao(self, model, cfg, val, modalidades, corrupcao, intensidade) -> dict:
        """Roda a validação inteira sob uma condição e agrega as métricas."""
        map_metric, count_metric, iou_metric = MeanAveragePrecision(), CountError(), IoU()
        limiar = cfg.decode["threshold"]
        nome = f"{corrupcao.nome}:{intensidade}" if corrupcao else "limpa"

        maps, ious, erros, por_modalidade, por_imagem = [], [], [], {}, []
        for idx in range(len(val)):
            imagem, gt = val[idx]
            if corrupcao is not None:
                imagem = corrupcao.aplicar(imagem, intensidade, idx)

            saida = model(imagem.unsqueeze(0).to(self.device))
            logits = (tuple(s[0].cpu() for s in saida) if isinstance(saida, tuple)
                      else saida[0].cpu())
            prob = model.foreground_prob(saida)[0].cpu()
            pred = torch.tensor(np.asarray(model.decode(logits, cfg.decode)))
            gt_t = gt.long()

            m_ap, _ = map_metric(pred, gt_t)
            iou = float(iou_metric(prob > limiar, to_binary(gt_t)))
            erro = int(count_metric(pred, gt_t))

            maps.append(float(m_ap))
            ious.append(iou)
            erros.append(erro)
            por_modalidade.setdefault(modalidades[idx], []).append(float(m_ap))
            por_imagem.append({"condicao": nome, "idx": idx, "modalidade": modalidades[idx],
                               "map": float(m_ap), "iou": iou, "count_error": erro})

        return {
            "map": float(np.mean(maps)),
            "iou": float(np.mean(ious)),
            "count_error": float(np.mean(erros)),
            "map_por_modalidade": {m: float(np.mean(v)) for m, v in por_modalidade.items()},
            "por_imagem": por_imagem,
        }

    def _conferir_limpa(self, map_limpo: float) -> None:
        """A condição limpa tem que reproduzir o mAP da Parte 2 nesta máquina."""
        if abs(map_limpo - MAP_LIMPO_ESPERADO) < 1e-9:
            print(f"  confere com a Parte 2 ({MAP_LIMPO_ESPERADO:.6f})\n")
        else:
            print(f"  ATENÇÃO: esperado {MAP_LIMPO_ESPERADO:.6f}, obtido {map_limpo:.6f} — "
                  f"o caminho de avaliação ou os dados mudaram desde a Parte 5\n")

    # -------------------------------------------------------------------------- figuras

    def _figura_degradacao(self, limpa, familias) -> None:
        """A curva que o enunciado pede: mAP (e IoU) contra a severidade.

        O eixo x é a severidade 1–3, e não o parâmetro, porque σ de borrão em pixels e σ de
        ruído em fração de intensidade não são comparáveis entre si. O parâmetro de cada ponto
        está na legenda e no JSON.
        """
        fig, eixos = plt.subplots(1, 2, figsize=(12, 4.8))
        for eixo, chave, titulo in ((eixos[0], "map", "mAP — instâncias"),
                                    (eixos[1], "iou", "IoU — máscara semântica")):
            for nome, familia in familias.items():
                niveis = familia["niveis"]
                x = [0] + [n["severidade"] for n in niveis]
                y = [limpa[chave]] + [n[chave] for n in niveis]
                rotulos = " · ".join(n["rotulo"] for n in niveis)
                eixo.plot(x, y, marker="o", label=f"{nome} ({rotulos})")
            eixo.axhline(limpa[chave], color="gray", ls=":", lw=1)
            eixo.set_xlabel("severidade (0 = imagem limpa)")
            eixo.set_xticks([0, 1, 2, 3])
            eixo.set_ylim(bottom=0)
            eixo.set_title(titulo, fontsize=10)
            eixo.grid(alpha=0.3)
        eixos[0].legend(fontsize=7)
        fig.suptitle("Parte 6 — degradação sob corrupção (modelo treinado sem augmentation)",
                     fontsize=10)
        plt.tight_layout()
        plt.savefig(self.saida / "p6_degradacao.png", dpi=120)
        plt.close(fig)

    @torch.no_grad()
    def _figura_exemplos(self, model, cfg, val, limpa, por_imagem_limpa) -> None:
        """Uma imagem típica em cada família e severidade, com a predição embaixo.

        "Típica" = a imagem cujo mAP sem corrupção é o mais próximo da média — evita ilustrar
        a Parte 6 com um caso extremo.
        """
        alvo = int(np.argmin([abs(p["map"] - limpa["map"]) for p in por_imagem_limpa]))
        imagem_limpa, gt = val[alvo]

        familias = list(CORRUPCOES.values())
        fig, eixos = plt.subplots(2 * len(familias), 4,
                                  figsize=(10, 5.2 * len(familias)))
        for linha, corrupcao in enumerate(familias):
            for coluna, intensidade in enumerate((NEUTRO,) + tuple(corrupcao.intensidades)):
                imagem = corrupcao.aplicar(imagem_limpa, intensidade, alvo)
                saida = model(imagem.unsqueeze(0).to(self.device))
                logits = (tuple(s[0].cpu() for s in saida) if isinstance(saida, tuple)
                          else saida[0].cpu())
                pred = np.asarray(model.decode(logits, cfg.decode))

                topo, baixo = eixos[2 * linha, coluna], eixos[2 * linha + 1, coluna]
                topo.imshow(imagem[0], cmap="gray", vmin=0, vmax=1)
                topo.set_title(
                    f"{corrupcao.nome} — "
                    f"{'limpa' if intensidade == NEUTRO else corrupcao.rotulo(intensidade)}",
                    fontsize=8)
                baixo.imshow(FailGalleryRunner._colorir(pred))
                baixo.set_title(f"{len(np.unique(pred[pred > 0]))} núcleos previstos "
                                f"(gabarito: {len(np.unique(gt.numpy())) - 1})", fontsize=8)
                topo.axis("off")
                baixo.axis("off")
        fig.suptitle(f"Parte 6 — imagem {alvo} da validação sob cada corrupção", fontsize=10)
        plt.tight_layout()
        plt.savefig(self.saida / "p6_exemplos.png", dpi=110)
        plt.close(fig)

    # ------------------------------------------------------------------------ relatório

    def _gravar(self, limpa, familias, por_imagem, n_imagens) -> dict:
        resumo = {
            "checkpoint": str(self.checkpoint),
            "imagens": n_imagens,
            "nota_augmentation": (
                "o treino não usa augmentation (não há transform/flip/rotação em src/), então "
                "o modelo nunca viu estas corrupções; uma curva íngreme é o esperado."),
            "map_limpo_esperado": MAP_LIMPO_ESPERADO,
            "limpa": limpa,
            "familias": familias,
        }
        (self.saida / "summary.json").write_text(
            json.dumps(resumo, indent=2, ensure_ascii=False), encoding="utf-8")
        (self.saida / "per_image.json").write_text(
            json.dumps(por_imagem, indent=2, ensure_ascii=False), encoding="utf-8")
        return resumo

    def _imprimir(self, resumo) -> None:
        limpa = resumo["limpa"]
        print(f"\n  {'condição':34s} {'mAP':>7s} {'IoU':>7s} {'erro cont.':>11s}")
        print(f"  {'limpa':34s} {limpa['map']:7.4f} {limpa['iou']:7.4f} "
              f"{limpa['count_error']:11.2f}")
        for nome, familia in resumo["familias"].items():
            for n in familia["niveis"]:
                print(f"  {nome + ' ' + n['rotulo']:34s} {n['map']:7.4f} {n['iou']:7.4f} "
                      f"{n['count_error']:11.2f}")
        print(f"\n  resultados: {self.saida / 'summary.json'}")
        print(f"  figuras:    {self.saida}/p6_degradacao.png, {self.saida}/p6_exemplos.png")
