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

from src.core.config import AppConfig
from src.training.engine import TrainEngine
from src.evaluation.engine import EvalEngine


def parse_arguments() -> argparse.Namespace:
    """Parse automático dos argumentos inline

    Returns:
        argparse.Namespace: namespace com os argumentos e seus valores
    """
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
        "--synthetic", action="store_true",
        help="atalho para --config configs/synthetic.yaml",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="retoma o treino de <output_dir>/last.pth, se existir",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="caminho do checkpoint para avaliação (padrão: <output_dir>/best.pth)",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    if args.synthetic:
        args.config = "configs/synthetic.yaml"

    app_config = AppConfig(args.config)
    Path(app_config.train_config.output_dir).mkdir(parents=True, exist_ok=True)

    if args.mode in ("train", "both"):
        Path(app_config.get_train_config().output_dir).mkdir(parents=True, exist_ok=True)
        TrainEngine(app_config, resume=args.resume).run()

    if args.mode in ("eval", "both"):
        checkpoint = Path(args.checkpoint) if args.checkpoint else app_config.get_eval_config().output_dir / "best.pth"
        EvalEngine(app_config, checkpoint).run()


if __name__ == "__main__":
    main()
