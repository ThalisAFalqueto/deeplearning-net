"""Constroi dataloaders de treino e validação a partir da configuração."""

from abc import ABC, abstractmethod

from src.training.config import TrainConfig
from src.core.config import AppConfig


class DatasetFactory(ABC):
    #: chaves que esta factory realmente lê de ``data``. Serve para o config recusar
    #: chaves que ninguém consome — um ``n_val`` no dataset errado, ou um typo como
    #: ``val_fration``, passaria em silêncio e daria a impressão de ter mudado algo.
    CHAVES: set[str] = set()

    @abstractmethod
    def build_train(self, cfg: TrainConfig):
        pass

    @abstractmethod
    def build_val(self, cfg: TrainConfig):
        pass

    def validate(self, cfg: TrainConfig) -> None:
        """Recusa chaves de ``data`` que esta factory não consome.

        Falhar aqui é barato; descobrir depois de treinar meia hora que o parâmetro
        estava sendo ignorado, não.
        """
        desconhecidas = set(cfg.data) - self.CHAVES - {"kind"}
        if desconhecidas:
            raise ValueError(
                f"chaves de 'data' que {type(self).__name__} não usa: "
                f"{sorted(desconhecidas)}. Aceitas: {sorted(self.CHAVES | {'kind'})}"
            )


class SyntheticDatasetFactory(DatasetFactory):
    CHAVES = {"size", "n_train", "n_val", "min_obj", "max_obj", "p_touch"}

    def build_train(self, cfg: TrainConfig):
        from .synthetic import SyntheticEllipses
        d = cfg.data
        return SyntheticEllipses(
            n_samples=d["n_train"], size=d["size"], seed=cfg.seed,
            min_obj=d["min_obj"], max_obj=d["max_obj"],
        )

    def build_val(self, cfg: TrainConfig):
        from .synthetic import SyntheticEllipses
        d = cfg.data
        return SyntheticEllipses(
            n_samples=d["n_val"], size=d["size"], seed=cfg.seed + 777,
            min_obj=d["min_obj"], max_obj=d["max_obj"],
        )


class DSB2018DatasetFactory(DatasetFactory):
    """Constrói os conjuntos do DSB2018 com split estratificado por modalidade.

    O config informa apenas **onde estão os dados** (``data_dir``) e **qual fração vai
    para validação** (``val_fraction``). A divisão em si é sempre recalculada por
    modalidade (ver ``src/data/split.py``), nunca lida do disco.

    Isso é deliberado: a organização em pastas do DSB2018 não é estratificada, e uma
    divisão herdada dela deixaria a proporção de modalidades ao acaso — o que estragaria
    as ablações da Parte 3, onde o desvio entre seeds precisa medir o efeito do
    experimento, e não a variação de composição do conjunto.

    Os arquivos nunca são movidos: a divisão é lógica, reproduzível pela seed e barata
    de mudar.
    """

    CHAVES = {"size", "data_dir", "val_fraction"}

    def __init__(self):
        self._cache = {}

    def _all_sample_dirs(self, cfg: TrainConfig) -> list:
        """Todas as amostras sob ``data_dir``, incluindo as de subpastas.

        Uma amostra é um diretório que contém ``images/``. A busca é recursiva porque o
        DSB2018 costuma vir organizado em ``train/`` e ``validation/`` — subpastas que
        aqui são apenas onde os arquivos calharam de ficar, sem significado para a
        divisão.
        """
        from pathlib import Path
        import os

        raiz = Path(os.getcwd()) / cfg.data["data_dir"]
        if not raiz.exists():
            raise FileNotFoundError(
                f"data.data_dir não existe: {raiz}\n"
                f"Baixe o DSB2018 e aponte data_dir para a pasta que contém as amostras."
            )

        amostras = [d for d in raiz.rglob("*") if d.is_dir() and (d / "images").is_dir()]
        if not amostras:
            raise FileNotFoundError(
                f"nenhuma amostra encontrada em {raiz}. Cada amostra deve ser um "
                f"diretório contendo images/ e masks/."
            )
        return sorted(amostras)

    def _split(self, cfg: TrainConfig):
        """Calcula (ou reaproveita) o split estratificado desta configuração."""
        from .modality import classify_samples
        from .split import stratified_split

        chave = (cfg.data["data_dir"], cfg.data.get("val_fraction", 0.2), cfg.seed)
        if chave in self._cache:
            return self._cache[chave]

        amostras = self._all_sample_dirs(cfg)
        # a classificação lê as 670 imagens uma vez; guardada junto com o split para
        # que build_train e build_val não repitam o trabalho
        modalidades = classify_samples(amostras)
        train, val = stratified_split(
            amostras,
            val_fraction=cfg.data.get("val_fraction", 0.2),
            seed=cfg.seed,
            modalities=modalidades,
        )
        self._cache[chave] = (train, val, modalidades)
        return self._cache[chave]

    def build_train(self, cfg: TrainConfig):
        from .dsb2018 import DSB2018
        train, _, _ = self._split(cfg)
        return DSB2018(size=cfg.data["size"], sample_dirs=train)

    def build_val(self, cfg: TrainConfig):
        from .dsb2018 import DSB2018
        _, val, _ = self._split(cfg)
        return DSB2018(size=cfg.data["size"], sample_dirs=val)

    def split_report(self, cfg: TrainConfig) -> str:
        """Tabela da composição do split — para conferir e para a apresentação."""
        from .split import split_summary
        train, val, modalidades = self._split(cfg)
        return split_summary(train, val, modalidades)


class DatasetFactoryRegistry:
    _factories = {
        "synthetic": SyntheticDatasetFactory(),
        "dsb2018": DSB2018DatasetFactory(),
    }

    @classmethod
    def get(cls, app_config: AppConfig) -> DatasetFactory:
        cfg = app_config.get_train_config()
        kind = cfg.data["kind"]
        if kind not in cls._factories:
            raise ValueError(
                f"data.kind desconhecido: {kind!r}. Disponíveis: {sorted(cls._factories)}"
            )
        factory = cls._factories[kind]
        factory.validate(cfg)
        return factory