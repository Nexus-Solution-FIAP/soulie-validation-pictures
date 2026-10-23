"""
Escolhe a cabeça de validação pela família da missão.

Se existir classificador treinado em modelos/<familia>/ (ou, em mobilidade,
na raiz de modelos/), usa o mesmo caminho do notebook. Senão, cai no
rotulador ImageNet da MobileNetV2 — um paliativo até haver fotos de treino.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

import joblib
import numpy as np
from PIL import Image

import preprocessamento as pp
from familias import FAMILIAS, resolver_familia

PASTA_MODELOS = Path(__file__).parent / "modelos"
LIMITE_IMAGENET_VALIDA = 0.18
LIMITE_IMAGENET_CENA = 0.12
LIMITE_IMAGENET_NEGATIVA = 0.15

_trava = threading.Lock()
_familias_carregadas: dict[str, dict] = {}
_rotulador = None


def pasta_modelo(familia: str) -> Path | None:
    especifico = PASTA_MODELOS / familia
    if (especifico / "classificador.joblib").exists():
        return especifico
    if familia == "mobilidade" and (PASTA_MODELOS / "classificador.joblib").exists():
        return PASTA_MODELOS
    return None


def familias_com_modelo() -> list[str]:
    return [f for f in FAMILIAS if pasta_modelo(f) is not None]


def carregar_familia(familia: str) -> dict | None:
    if familia in _familias_carregadas:
        return _familias_carregadas[familia]
    pasta = pasta_modelo(familia)
    if pasta is None:
        return None
    dados = {
        "classificador": joblib.load(pasta / "classificador.joblib"),
        "detector": joblib.load(pasta / "detector_anomalia.joblib"),
        "config": json.loads((pasta / "config.json").read_text(encoding="utf-8")),
    }
    _familias_carregadas[familia] = dados
    return dados


def carregar_rotulador():
    """MobileNetV2 com a cabeça ImageNet. Só entra no fallback."""
    global _rotulador
    if _rotulador is None:
        from tensorflow.keras.applications import MobileNetV2

        _rotulador = MobileNetV2(
            include_top=True,
            weights="imagenet",
            input_shape=(pp.TAMANHO_IMAGEM, pp.TAMANHO_IMAGEM, 3),
        )
    return _rotulador


def preparar_extrator() -> None:
    pp.carregar_extrator()
    for familia in FAMILIAS:
        carregar_familia(familia)


def _nome_imagenet(nome: str) -> str:
    return nome.lower().replace(" ", "_").replace("-", "_")


def _via_modelo(img: Image.Image, familia: str, modelos: dict) -> dict:
    embedding = pp.gerar_embeddings([img])
    distancia = float(pp.distancia_anomalia(modelos["detector"], embedding)[0])
    classificador = modelos["classificador"]
    probs = classificador.predict_proba(embedding)[0]
    cfg_modelo = modelos["config"]
    # Garante que a classe negativa da família atual vale, mesmo se o JSON
    # antigo só falar de carro (caso da mobilidade já treinada).
    cfg = {
        **cfg_modelo,
        "classe_negativa": FAMILIAS[familia]["negativa"],
        "motivo_negativa": FAMILIAS[familia]["motivo_negativa"],
        "limite_confianca": cfg_modelo.get("limite_confianca", 0.5),
        "limite_distancia": cfg_modelo.get("limite_distancia", 0.9),
    }
    resultado = pp.decidir(probs, list(classificador.classes_), distancia, cfg)
    if resultado["status"] == "REPROVADO" and resultado["motivo"] == "CARRO_PARTICULAR":
        resultado["motivo"] = FAMILIAS[familia]["motivo_negativa"]
    return resultado


def _via_imagenet(img: Image.Image, familia: str) -> dict:
    from tensorflow.keras.applications.mobilenet_v2 import decode_predictions, preprocess_input

    cfg = FAMILIAS[familia]
    lote = np.stack([np.asarray(pp.recortar_quadrado(img), dtype=np.float32)])
    lote = preprocess_input(lote)
    preds = carregar_rotulador().predict(lote, verbose=0)
    top = decode_predictions(preds, top=8)[0]

    melhor_valida = None
    melhor_negativa = None
    for _codigo, nome, prob in top:
        chave = _nome_imagenet(nome)
        entrada = (cfg["imagenet_classe"].get(chave), float(prob), chave)
        if chave in cfg["imagenet_negativas"] and (melhor_negativa is None or prob > melhor_negativa[1]):
            melhor_negativa = entrada
        if chave in cfg["imagenet_validas"] and (melhor_valida is None or prob > melhor_valida[1]):
            melhor_valida = entrada

    limite_valida = LIMITE_IMAGENET_CENA if cfg["modo"] == "cena" else LIMITE_IMAGENET_VALIDA

    if melhor_negativa and melhor_negativa[1] >= LIMITE_IMAGENET_NEGATIVA:
        if not melhor_valida or melhor_negativa[1] >= melhor_valida[1]:
            classe = melhor_negativa[0] or cfg["negativa"]
            return {
                "status": "REPROVADO",
                "classe": classe,
                "confianca": round(melhor_negativa[1], 4),
                "distancia": None,
                "motivo": cfg["motivo_negativa"],
            }

    if melhor_valida and melhor_valida[1] >= limite_valida:
        return {
            "status": "APROVADO",
            "classe": melhor_valida[0],
            "confianca": round(melhor_valida[1], 4),
            "distancia": None,
            "motivo": None,
        }

    if melhor_valida:
        return {
            "status": "REPROVADO",
            "classe": melhor_valida[0],
            "confianca": round(melhor_valida[1], 4),
            "distancia": None,
            "motivo": "INCONCLUSIVO",
        }

    top1 = top[0] if top else None
    return {
        "status": "REPROVADO",
        "classe": None,
        "confianca": round(float(top1[2]), 4) if top1 else 0.0,
        "distancia": None,
        "motivo": "CENA_INVALIDA",
    }


def analisar_imagem(img: Image.Image, mission: dict | None) -> tuple[dict, str, str, str]:
    """
    Devolve (veredito interno, familia, origem, versao).
    origem: "modelo" | "imagenet"
    """
    familia = resolver_familia(mission)
    if familia not in FAMILIAS:
        familia = "mobilidade"

    with _trava:
        modelos = carregar_familia(familia)
        if modelos is not None:
            interno = _via_modelo(img, familia, modelos)
            versao = str(modelos["config"].get("versao", "1.0"))
            return interno, familia, "modelo", versao
        interno = _via_imagenet(img, familia)
        return interno, familia, "imagenet", "imagenet-1.0"
