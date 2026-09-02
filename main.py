"""Entry point único do projeto.

Uso:
    python -m main                  # treino + avaliação
    python -m main --train          # apenas treino
    python -m main --eval           # apenas avaliação
    python -m main --config path    # config YAML customizado
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml

from src.training.config import TrainConfig
from src.training.engine import TrainEngine
from src.evaluation.config import EvalConfig
from src.evaluation.engine import EvalEngine


def main():
    parser = argparse.ArgumentParser(description="Deep Learning Net")
    parser.add_argument(
        "--mode", choices=["train", "eval", "both"], default="both",
        help="executa treino, avaliação ou ambos (padrão: both)",
    )
    parser.add_argument(
        "--config", default="configs/default.yaml",
        help="caminho do YAML de configuração (padrão: configs/default.yaml)",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="caminho do checkpoint para avaliação (padrão: <output_dir>/best.pth)",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    if args.mode in ("train", "both"):
        train_cfg = TrainConfig.from_dict(cfg)
        TrainEngine(train_cfg).run()

    if args.mode in ("eval", "both"):
        checkpoint = Path(args.checkpoint) if args.checkpoint else Path(cfg["output_dir"]) / "best.pth"
        eval_cfg = EvalConfig.from_dict(cfg)
        EvalEngine(eval_cfg, checkpoint).run()


if __name__ == "__main__":
    main()
