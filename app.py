"""
API de validação de fotos do Soulie.

Rodar:  uvicorn app:app --port 5000
Docs:   http://localhost:5000/docs
"""
from __future__ import annotations

import base64
import hmac
import os
import threading
from contextlib import asynccontextmanager

import imagehash
import httpx
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from PIL import UnidentifiedImageError

import inferencia
import preprocessamento as pp
from contrato import AnalisarRequest, AnalisarResponse
from familias import FAMILIAS, hint, resolver_familia
from resposta import para_contrato

TAMANHO_MAXIMO_BYTES = 10 * 1024 * 1024
API_KEY = os.getenv("AI_VALIDATION_API_KEY", "")
_trava_inferencia = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    inferencia.preparar_extrator()
    yield


app = FastAPI(
    title="Soulie - Validação de Fotos",
    version="1.1",
    description="Valida a foto de evidência de uma missão. O Java chama POST /v1/analisar.",
    lifespan=lifespan,
)


def _conferir_chave(authorization: str | None) -> None:
    if not API_KEY:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Token ausente.")
    enviado = authorization.removeprefix("Bearer ").encode("utf-8")
    esperado = API_KEY.encode("utf-8")
    if len(enviado) != len(esperado) or not hmac.compare_digest(enviado, esperado):
        raise HTTPException(status_code=401, detail="Token inválido.")


def _abrir(conteudo: bytes):
    if not conteudo:
        raise HTTPException(status_code=400, detail="Arquivo vazio.")
    if len(conteudo) > TAMANHO_MAXIMO_BYTES:
        raise HTTPException(status_code=413, detail="Foto maior que 10 MB.")
    try:
        return pp.abrir_imagem(conteudo)
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(status_code=400, detail="O arquivo enviado não é uma imagem válida.")


def _baixar(url: str) -> bytes:
    if not url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="imageUrl inválida.")
    try:
        with httpx.Client(timeout=20.0, follow_redirects=True) as cliente:
            resposta = cliente.get(url)
            resposta.raise_for_status()
            return resposta.content
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=502, detail="Não foi possível baixar a imagem.")


def _bytes_da_requisicao(pedido: AnalisarRequest) -> bytes:
    if pedido.image_base64:
        dado = pedido.image_base64.strip()
        if dado.startswith("data:") and "," in dado:
            dado = dado.split(",", 1)[1]
        try:
            return base64.b64decode(dado, validate=False)
        except Exception:
            raise HTTPException(status_code=400, detail="imageBase64 inválido.")
    if pedido.image_url:
        return _baixar(pedido.image_url)
    raise HTTPException(status_code=400, detail="Envie imageUrl ou imageBase64.")


def _analisar(img, mission: dict | None) -> AnalisarResponse:
    with _trava_inferencia:
        interno, familia, origem, versao = inferencia.analisar_imagem(img, mission)
    corpo = para_contrato(interno, familia, origem, versao)
    corpo["hash_imagem"] = str(imagehash.phash(img))
    return AnalisarResponse(**corpo)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "familias": list(FAMILIAS),
        "modelos_treinados": inferencia.familias_com_modelo(),
        "hint_por_familia": {nome: hint(nome) for nome in FAMILIAS},
    }


@app.post("/v1/analisar", response_model=AnalisarResponse)
def analisar(
    pedido: AnalisarRequest,
    authorization: str | None = Header(default=None),
):
    """Contrato do java-soulie: JSON com a missão e a imagem (URL ou base64)."""
    _conferir_chave(authorization)
    img = _abrir(_bytes_da_requisicao(pedido))
    mission = pedido.mission.model_dump() if pedido.mission else None
    return _analisar(img, mission)


@app.post("/validar", response_model=AnalisarResponse)
def validar(
    foto: UploadFile = File(...),
    authorization: str | None = Header(default=None),
):
    """Atalho manual (multipart). Sem missão, assume a família mobilidade."""
    _conferir_chave(authorization)
    img = _abrir(foto.file.read())
    return _analisar(img, {"tag": "Mobilidade", "name": "Transporte Público"})


@app.get("/v1/familias/{tag}")
def familia_da_tag(tag: str, name: str = ""):
    familia = resolver_familia({"tag": tag, "name": name})
    return {
        "familia": familia,
        "hint": hint(familia),
        "modelo_treinado": pasta_existe(familia),
    }


def pasta_existe(familia: str) -> bool:
    return inferencia.pasta_modelo(familia) is not None
