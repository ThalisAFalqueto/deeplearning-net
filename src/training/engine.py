"""Motor de treino: loop, validação e checkpoint."""

import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from src.metrics.semantic import IoU, Dice
from src.models.factory import ModelFactoryRegistry
from src.models.checkpoint import load_checkpoint
from src.losses.factory import LossFactoryRegistry
from src.utils import to_binary
from src.data import DataPipeline
from src.core.config import AppConfig


class TrainEngine:
    def __init__(self, app_config: AppConfig, resume: bool = False):
        self.app_config = app_config
        self.cfg = app_config.get_train_config()
        self.resume = resume

        # Detecto o device utilizado (cuda, rocm ou cpu)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _fixed_seeds(self):
        """Ajusta a seed aleatória para manter os resultados reprodutíveis.

        A seed sozinha não basta em GPU: com ``cudnn.benchmark`` ligado, o cuDNN escolhe o
        algoritmo de convolução medindo tempo, e a escolha pode mudar de uma execução para
        outra; alguns desses algoritmos também somam em ordem variável. As duas flags fixam
        algoritmos determinísticos. Isso garante repetir o resultado **na mesma máquina** —
        GPU, driver e versão do PyTorch diferentes ainda dão números diferentes, então só
        se comparam números produzidos na mesma máquina.
        """
        random.seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)
        torch.cuda.manual_seed_all(self.cfg.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def _save_checkpoint(self, path, model, optimizer, epoch, best_iou, history):
        """Salva o estado COMPLETO do treino.

        Só os pesos não bastam para retomar: o Adam mantém médias móveis dos gradientes
        (os momentos), e recomeçar sem elas faz as primeiras épocas depois do resume
        saírem instáveis. O histórico vai junto para o log não perder as épocas antigas.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_iou": best_iou,
            "history": history,
            "config": {**self.cfg.__dict__, "output_dir": str(self.cfg.output_dir)},
        }, path)

    def _load_checkpoint(self, path, model, optimizer):
        """Restaura o estado de um treino interrompido.

        Returns:
            Tupla (primeira época a rodar, melhor IoU até aqui, histórico).
        """
        # aceita também checkpoints salvos antes da separação backbone/cabeça
        estado = load_checkpoint(path, model, self.device)
        if "optimizer" in estado:
            optimizer.load_state_dict(estado["optimizer"])
        epoca = estado.get("epoch", 0)
        print(f"retomando de {path} — época {epoca}, melhor IoU {estado.get('best_iou', -1):.4f}\n")
        return epoca + 1, estado.get("best_iou", -1.0), estado.get("history", [])

    def run(self) -> None:
        cfg = self.cfg
        t = cfg.train

        self._fixed_seeds()

        # Crio a pipeline que gera os dataloaders
        data_pipeline = DataPipeline(self.app_config)

        # Carrego os loaders via pipeline (treino e validação)
        train_loader, val_loader = data_pipeline.build_dataloaders()

        # A factory escolhe a arquitetura (backbone + cabeça) pelo config. A perda vem de
        # uma factory separada: o modelo só extrai features e decodifica, não conhece a perda.
        model = ModelFactoryRegistry.build(cfg).to(self.device)
        self.loss = LossFactoryRegistry.build(cfg)


        # Instancio o método de otimização (Adam, SGD, etc)
        optimizer = torch.optim.Adam(model.parameters(), lr=t["lr"])

        # Conto o número de parâmetros do modelo e imprimo algumas informações
        n_params = sum(p.numel() for p in model.parameters())
        print(f"modelo: {type(model.backbone).__name__} | dispositivo: {self.device} | parâmetros: {n_params/1e3:.1f}k "
              f"| campo receptivo: {model.receptive_field()} px")
        print(f"treino: {len(train_loader.dataset)} imagens | validação: {len(val_loader.dataset)} imagens\n")

        # =================== LOOP DE TREINO ===================
        history, best_iou = [], -1.0  # Inicio o histórico de treino e o melhor IoU de validação
        primeira_epoca = 1

        # Retomar de um treino interrompido. `last.pth` guarda o estado completo e é
        # gravado a cada `save_every` épocas — a rede de segurança para queda de energia,
        # sessão do Colab que expira, ou os 12 treinos da Parte 3.
        ultimo = cfg.output_dir / "last.pth"
        if self.resume and ultimo.exists():
            primeira_epoca, best_iou, history = self._load_checkpoint(ultimo, model, optimizer)
        elif self.resume:
            print(f"--resume pedido, mas {ultimo} não existe: começando do zero\n")

        save_every = t.get("save_every", 5)
        t_start = time.perf_counter()  # Inicio a contagem do tempo total de treino

        for epoch in range(primeira_epoca, t["epochs"] + 1):
            model.train()
            epoch_loss, t_epoch = 0.0, time.perf_counter()

            componentes_epoca = {}
            for images, labels in train_loader:
                images = images.to(self.device)
                target = self.loss.build_targets(labels, self.device)

                optimizer.zero_grad()
                loss, componentes = self.loss(model(images), target)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * images.size(0)
                for k, v in componentes.items():
                    componentes_epoca[k] = componentes_epoca.get(k, 0.0) + v * images.size(0)

            n_train = len(train_loader.dataset)
            train_loss = epoch_loss / n_train
            componentes_epoca = {k: v / n_train for k, v in componentes_epoca.items()}
            val_loss, val_iou, val_dice = self._evaluate(model, val_loader)
            dt = time.perf_counter() - t_epoch
            history.append({
                "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                "val_iou": val_iou, "val_dice": val_dice, "seconds": dt,
                **{f"loss_{k}": v for k, v in componentes_epoca.items()},
            })
            # com várias perdas somadas, o total sozinho esconde qual delas estagnou
            detalhe = ""
            if len(componentes_epoca) > 1:
                detalhe = " (" + " ".join(f"{k} {v:.4f}" for k, v in componentes_epoca.items()) + ")"
            print(f"época {epoch:3d}/{t['epochs']} | treino {train_loss:.4f}{detalhe} "
                  f"| val {val_loss:.4f} | IoU {val_iou:.4f} | Dice {val_dice:.4f} "
                  f"| {dt:.1f}s")

            if val_iou > best_iou:
                best_iou = val_iou
                self._save_checkpoint(cfg.output_dir / "best.pth", model, optimizer,
                                      epoch, best_iou, history)

            # `last.pth` é o ponto de retomada; `best.pth` é o modelo que vai para a
            # avaliação. São arquivos diferentes de propósito: o melhor modelo pode ser
            # de uma época bem anterior à última.
            if epoch % save_every == 0 or epoch == t["epochs"]:
                self._save_checkpoint(cfg.output_dir / "last.pth", model, optimizer,
                                      epoch, best_iou, history)

        total = time.perf_counter() - t_start
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        (cfg.output_dir / "history.json").write_text(json.dumps(history, indent=2))
        print(f"\ntempo total: {total/60:.1f} min ({total/t['epochs']:.1f}s por época)")
        print(f"melhor IoU de validação: {best_iou:.4f}")
        print(f"checkpoint: {cfg.output_dir/'best.pth'}")
        print(f"retomável:  {cfg.output_dir/'last.pth'} (--resume)")

    @torch.no_grad()
    def _evaluate(self, model, loader):
        model.eval()
        iou_metric = IoU()
        dice_metric = Dice()

        total_loss, ious, dices = 0.0, [], []
        for images, labels in loader:
            images = images.to(self.device)
            target = self.loss.build_targets(labels, self.device)

            logits = model(images)
            loss, _ = self.loss(logits, target)
            total_loss += loss.item() * images.size(0)

            prob = model.foreground_prob(logits).cpu()
            for p, g in zip(prob, labels):
                pred_mask = p > self.cfg.decode["threshold"]
                binary_gt = to_binary(g)
                ious.append(iou_metric(pred_mask, binary_gt))
                dices.append(dice_metric(pred_mask, binary_gt))

        n = len(loader.dataset)
        return total_loss / n, float(torch.tensor(ious).mean()), float(torch.tensor(dices).mean())
