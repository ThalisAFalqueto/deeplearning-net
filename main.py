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
        "--mode", choices=["train", "eval", "both", "mosaic", "fails"], default="both",
        help="executa treino, avaliação, ambos (padrão: both), a inferência em mosaico "
             "da Parte 4 (mosaic) ou a galeria de falhas da Parte 5 (fails)",
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
    parser.add_argument(
        "--ablation", nargs="+", metavar="CONFIG",
        help="executa ablação: treino+eval de múltiplas configs em múltiplas seeds",
    )
    parser.add_argument(
        "--ablation-seeds", nargs="+", type=int, default=[0, 42],
        help="seeds para a ablação (padrão: 0 42)",
    )
    parser.add_argument(
        "--ablation-output", default="outputs/ablation",
        help="diretório base para relatórios de ablação",
    )
    parser.add_argument(
        "--tile", type=int, default=256,
        help="--mode mosaic: lado do tile em pixels (padrão: 256, o tamanho de treino)",
    )
    parser.add_argument(
        "--stride", type=int, default=192,
        help="--mode mosaic: passo entre tiles; sobreposição = tile - stride (padrão: 192)",
    )
    parser.add_argument(
        "--mosaic-output", default="outputs/p4",
        help="--mode mosaic: pasta de resultados e figuras (padrão: outputs/p4)",
    )
    parser.add_argument(
        "--fails-n", type=int, default=5,
        help="--mode fails: quantas imagens piores mostrar (padrão: 5)",
    )
    parser.add_argument(
        "--fails-indices", nargs="+", type=int, default=None,
        help="--mode fails: força estes índices da validação em vez de rankear — para "
             "reproduzir a mesma galeria com outro checkpoint (antes/depois de uma correção)",
    )
    parser.add_argument(
        "--fails-output", default="outputs/p5",
        help="--mode fails: pasta de resultados e figuras (padrão: outputs/p5)",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    if args.synthetic:
        args.config = "configs/synthetic.yaml"

    # ===== MODO ABLAÇÃO =====
    if args.ablation:
        from src.ablation.runner import AblationRunner
        runner = AblationRunner(
            config_paths=args.ablation,
            seeds=args.ablation_seeds,
            base_output_dir=args.ablation_output,
        )
        runner.run()
        return

    app_config = AppConfig(args.config)

    # ===== MODO MOSAICO (Parte 4) =====
    # só avalia: usa o modelo do config e o checkpoint, e grava em --mosaic-output
    if args.mode == "mosaic":
        from src.mosaic.runner import MosaicRunner
        checkpoint = Path(args.checkpoint) if args.checkpoint else app_config.get_eval_config().output_dir / "best.pth"
        MosaicRunner(app_config, checkpoint, tile=args.tile, passo=args.stride,
                     saida=args.mosaic_output).run()
        return

    # ===== MODO GALERIA DE FALHAS (Parte 5) =====
    # só avalia: roda o checkpoint na validação e monta a galeria das piores predições
    if args.mode == "fails":
        from src.fails.runner import FailGalleryRunner
        checkpoint = Path(args.checkpoint) if args.checkpoint else app_config.get_eval_config().output_dir / "best.pth"
        FailGalleryRunner(app_config, checkpoint, n=args.fails_n,
                           indices=args.fails_indices, saida=args.fails_output).run()
        return

    # ===== MODO NORMAL =====
    Path(app_config.get_train_config().output_dir).mkdir(parents=True, exist_ok=True)

    if args.mode in ("train", "both"):
        Path(app_config.get_train_config().output_dir).mkdir(parents=True, exist_ok=True)
        TrainEngine(app_config, resume=args.resume).run()

    if args.mode in ("eval", "both"):
        checkpoint = Path(args.checkpoint) if args.checkpoint else app_config.get_eval_config().output_dir / "best.pth"
        EvalEngine(app_config, checkpoint).run()


if __name__ == "__main__":
    main()
