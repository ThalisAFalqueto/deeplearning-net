"""Runner de ablação: executa treino+avaliação para múltiplas configs e seeds,
coleta métricas e gera relatório com média e desvio padrão."""

import json
import yaml
from datetime import datetime
from pathlib import Path

import numpy as np

from src.training.config import TrainConfig
from src.evaluation.config import EvalConfig
from src.training.engine import TrainEngine
from src.evaluation.engine import EvalEngine


class StandaloneConfig:
    """Wrapper não-singleton para rodar múltiplas configs na mesma sessão."""

    def __init__(self, raw: dict):
        self._raw = raw
        self.train_config = TrainConfig.from_dict(raw)
        self.eval_config = EvalConfig.from_dict(raw)

    def get_train_config(self):
        return self.train_config

    def get_eval_config(self):
        return self.eval_config


class AblationRunner:
    """Executa ablação comparando duas (ou mais) configurações em múltiplas seeds."""

    def __init__(
        self,
        config_paths: list[str],
        seeds: list[int] | None = None,
        base_output_dir: str = "outputs/ablation",
    ):
        self.config_paths = config_paths
        self.seeds = seeds or [0, 42]
        self.base_output_dir = Path(base_output_dir)

    def run(self) -> dict:
        """Executa todas as runs e retorna o dicionário de resultados."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.base_output_dir / f"run_{timestamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        resultados = {}
        for config_path in self.config_paths:
            nome_config = Path(config_path).stem
            resultados[nome_config] = self._run_config(config_path, nome_config)

        self._gerar_relatorio(resultados)
        return resultados

    def _run_config(self, config_path: str, nome_config: str) -> dict:
        """Roda uma config em todas as seeds e coleta métricas."""
        metricas_por_seed = {}
        for seed in self.seeds:
            run_dir = self.run_dir / nome_config / f"seed_{seed}"
            raw = self._carregar_config_com_overrides(config_path, seed, run_dir)
            app_config = StandaloneConfig(raw)

            print(f"\n{'=' * 60}")
            print(f"Ablação: {nome_config} | seed={seed}")
            print(f"{'=' * 60}\n")

            # Treino
            TrainEngine(app_config, resume=False).run()

            # Avaliação
            checkpoint = run_dir / "best.pth"
            if not checkpoint.exists():
                print(f"AVISO: checkpoint não encontrado em {checkpoint}, pulando avaliação.")
                continue
            EvalEngine(app_config, checkpoint).run()

            # Ler summary
            summary_path = run_dir / "summary.json"
            if summary_path.exists():
                metricas_por_seed[seed] = json.loads(summary_path.read_text())
            else:
                print(f"AVISO: summary não encontrado em {summary_path}")

        return self._calcular_estatisticas(metricas_por_seed)

    def _carregar_config_com_overrides(self, config_path: str, seed: int, output_dir: Path) -> dict:
        """Carrega YAML, injeta seed e output_dir, retorna dict."""
        raw = yaml.safe_load(Path(config_path).read_text())
        raw["seed"] = seed
        raw["output_dir"] = str(output_dir)
        return raw

    def _calcular_estatisticas(self, metricas_por_seed: dict) -> dict:
        """Calcula média e desvio padrão das métricas across seeds."""
        estatisticas = {}
        metricas = ["iou", "dice", "map", "count_error"]
        for metrica in metricas:
            valores = []
            for seed in self.seeds:
                if seed in metricas_por_seed and metrica in metricas_por_seed[seed]:
                    valores.append(metricas_por_seed[seed][metrica])
            if valores:
                estatisticas[metrica] = {
                    "mean": float(np.mean(valores)),
                    "std": float(np.std(valores)),
                    "values": {str(s): metricas_por_seed[s][metrica] for s in metricas_por_seed if metrica in metricas_por_seed[s]},
                }
        return estatisticas

    def _gerar_relatorio(self, resultados: dict) -> None:
        """Gera relatório em Markdown e JSON."""
        # JSON
        json_path = self.run_dir / "report.json"
        json_path.write_text(json.dumps(resultados, indent=2))

        # Markdown
        linhas = ["# Ablation Report\n", "\n"]
        for nome_config, stats in resultados.items():
            linhas.append(f"## {nome_config}\n\n")
            cabecalho = "| Métrica | " + " | ".join(str(s) for s in self.seeds) + " | Média | Std |\n"
            linhas.append(cabecalho)
            linhas.append("|" + "---|" * (len(self.seeds) + 3) + "\n")
            for metrica, data in stats.items():
                linha = f"| {metrica} | "
                for s in self.seeds:
                    valor = data["values"].get(str(s), "-")
                    linha += f"{valor:.4f} | "
                linha += f"{data['mean']:.4f} | {data['std']:.4f} |\n"
                linhas.append(linha)
            linhas.append("\n")

        md_path = self.run_dir / "report.md"
        md_path.write_text("".join(linhas))

        # Gráficos de ablação
        from src.ablation.reporter import plot_ablation_bars, plot_ablation_table
        for metrica in ["iou", "dice", "map", "count_error"]:
            try:
                plot_ablation_bars(resultados, metrica, self.run_dir / f"{metrica}_bars.png")
            except Exception as e:
                print(f"AVISO: não foi possível gerar gráfico de {metrica}: {e}")
        try:
            plot_ablation_table(resultados, self.run_dir / "table.png")
        except Exception as e:
            print(f"AVISO: não foi possível gerar tabela: {e}")

        print(f"\nRelatório de ablação:")
        print(f"  Markdown: {md_path}")
        print(f"  JSON:     {json_path}")
        print(f"  Gráficos: {self.run_dir}")
