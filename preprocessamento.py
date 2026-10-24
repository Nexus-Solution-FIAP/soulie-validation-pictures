"""
Pré-processamento e regras de decisão da validação de fotos do Soulie.

Usado pelo treinamento.ipynb e pelo app.py, para garantir que a foto
seja tratada exatamente igual no treino e em produção.
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

TAMANHO_IMAGEM = 224
DIMENSAO_EMBEDDING = 1280
EXTENSOES = {".jpg", ".jpeg", ".png", ".webp"}

_extrator = None


# ---------------------------------------------------------------------------
# Carregamento
# ---------------------------------------------------------------------------
def carregar_extrator():
    """Carrega a MobileNetV2 pré-treinada no ImageNet, sem a camada final.

    Ela não é treinada: só transforma a foto num vetor de 1280 números.
    Carrega uma única vez e reaproveita nas chamadas seguintes.
    """
    global _extrator
    if _extrator is None:
        from tensorflow.keras.applications import MobileNetV2

        _extrator = MobileNetV2(
            include_top=False,
            weights="imagenet",
            pooling="avg",
            input_shape=(TAMANHO_IMAGEM, TAMANHO_IMAGEM, 3),
        )
    return _extrator


def abrir_imagem(origem: str | Path | bytes) -> Image.Image:
    """Abre uma foto a partir de um caminho ou dos bytes recebidos pela API."""
    if isinstance(origem, (bytes, bytearray)):
        origem = BytesIO(origem)
    img = Image.open(origem)
    img = ImageOps.exif_transpose(img)  # corrige fotos de celular que vêm giradas
    return img.convert("RGB")


def listar_fotos(pasta: str | Path) -> list[Path]:
    """Lista as fotos de uma pasta do dataset (ignora arquivos que não são imagem)."""
    pasta = Path(pasta)
    if not pasta.exists():
        return []
    return sorted(p for p in pasta.iterdir() if p.suffix.lower() in EXTENSOES)


# ---------------------------------------------------------------------------
# Pré-processamento e embeddings
# ---------------------------------------------------------------------------
def recortar_quadrado(img: Image.Image) -> Image.Image:
    """Recorta o centro em quadrado e redimensiona para 224x224, sem distorcer."""
    return ImageOps.fit(
        img,
        (TAMANHO_IMAGEM, TAMANHO_IMAGEM),
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.5),
    )


def normalizar(embeddings: np.ndarray) -> np.ndarray:
    """Deixa cada vetor com tamanho 1, para as distâncias ficarem comparáveis."""
    normas = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.clip(normas, 1e-12, None)


def gerar_embeddings(imagens: list[Image.Image], batch_size: int = 32) -> np.ndarray:
    """Transforma uma lista de fotos numa matriz (n_fotos, 1280)."""
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    if not imagens:
        return np.empty((0, DIMENSAO_EMBEDDING), dtype=np.float32)

    lote = np.stack([np.asarray(recortar_quadrado(i), dtype=np.float32) for i in imagens])
    lote = preprocess_input(lote)  # escala os pixels para [-1, 1]
    embeddings = carregar_extrator().predict(lote, batch_size=batch_size, verbose=0)
    return normalizar(embeddings)


def aumentar(img: Image.Image) -> list[Image.Image]:
    """Variações da foto usadas só no treino, para compensar o dataset pequeno."""
    largura, altura = img.size
    margem = 0.1
    corte = img.crop(
        (
            int(largura * margem),
            int(altura * margem),
            int(largura * (1 - margem)),
            int(altura * (1 - margem)),
        )
    )
    return [
        ImageOps.mirror(img),                       # espelhada
        ImageEnhance.Brightness(img).enhance(0.7),  # mais escura
        ImageEnhance.Brightness(img).enhance(1.3),  # mais clara
        corte,                                      # zoom de 10%
    ]


# ---------------------------------------------------------------------------
# Anomalia e decisão
# ---------------------------------------------------------------------------
def distancia_anomalia(detector, embeddings: np.ndarray) -> np.ndarray:
    """Distância média até as k fotos válidas mais parecidas.

    Quanto maior, mais a foto foge do que o modelo conhece como válido.
    """
    distancias, _ = detector.kneighbors(embeddings)
    return distancias.mean(axis=1)


def decidir(probs: np.ndarray, classes: list[str], distancia: float, config: dict) -> dict:
    """Aplica as regras de aprovação. A ordem das regras importa."""
    indice = int(np.argmax(probs))
    classe = classes[indice]
    confianca = float(probs[indice])
    base = {
        "classe": classe,
        "confianca": round(confianca, 4),
        "distancia": round(float(distancia), 4),
    }

    if classe == config["classe_negativa"]:
        motivo = config.get("motivo_negativa", "CARRO_PARTICULAR")
        return {"status": "REPROVADO", **base, "motivo": motivo}
    if distancia > config["limite_distancia"]:
        return {"status": "REPROVADO", **base, "motivo": "CENA_INVALIDA"}
    if confianca < config["limite_confianca"]:
        return {"status": "REPROVADO", **base, "motivo": "INCONCLUSIVO"}
    return {"status": "APROVADO", **base, "motivo": None}
