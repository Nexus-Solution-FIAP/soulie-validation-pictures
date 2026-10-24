"""
API de validação de fotos do Soulie.

Rodar:  uvicorn app:app --port 5000
Docs:   http://localhost:5000/docs
"""
from __future__ import annotations

import base64
import hmac
import logging
import os
import threading
import time
from contextlib import asynccontextmanager

import imagehash
import httpx
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from PIL import UnidentifiedImageError

import visao_llm
from contrato import AnalisarRequest, AnalisarResponse
from familias import FAMILIAS, hint, resolver_familia

log = logging.getLogger("uvicorn.error")

TAMANHO_MAXIMO_BYTES = 10 * 1024 * 1024
API_KEY = os.getenv("AI_VALIDATION_API_KEY", "")
_trava_inferencia = threading.Lock()


def _preparar_em_fundo() -> None:
    import inferencia

    inicio = time.perf_counter()
    try:
        inferencia.preparar_extrator()
        log.info("Modelos prontos em %.1fs", time.perf_counter() - inicio)
    except Exception:
        log.exception("Startup dos modelos falhou após %.1fs", time.perf_counter() - inicio)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Não carrega TensorFlow no bind: no Render 512 MB isso estoura a RAM.
    # Com GEMINI_API_KEY o processo fica leve; TF só sobe se a flag pedir.
    if visao_llm.carregar_tensorflow_no_startup():
        threading.Thread(
            target=_preparar_em_fundo,
            daemon=True,
            name="carregar-modelos",
        ).start()
    elif visao_llm.configurado():
        log.info("Visão via Gemini (%s). TensorFlow não será carregado.", visao_llm.modelo())
    else:
        log.warning("Sem GEMINI_API_KEY e sem TensorFlow: POST /v1/analisar devolve 503.")
    yield


app = FastAPI(
    title="Soulie - Validação de Fotos",
    version="1.2",
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
    import preprocessamento as pp

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
    inicio = time.perf_counter()
    try:
        with httpx.Client(timeout=20.0, follow_redirects=True) as cliente:
            resposta = cliente.get(url)
            resposta.raise_for_status()
            log.info("Imagem baixada em %.1fs (%s bytes)", time.perf_counter() - inicio, len(resposta.content))
            return resposta.content
    except HTTPException:
        raise
    except Exception:
        log.warning("Falha ao baixar a imagem após %.1fs", time.perf_counter() - inicio)
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


def _via_tensorflow(img, mission: dict | None) -> AnalisarResponse:
    import inferencia
    from resposta import para_contrato

    try:
        inferencia.aguardar_pronto()
    except inferencia.ModelosNaoProntos as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    inicio = time.perf_counter()
    with _trava_inferencia:
        interno, familia, origem, versao = inferencia.analisar_imagem(img, mission)
    corpo = para_contrato(interno, familia, origem, versao)
    corpo["hash_imagem"] = str(imagehash.phash(img))
    log.info(
        "Analisou familia=%s origem=%s status=%s em %.1fs",
        familia,
        origem,
        corpo["status"],
        time.perf_counter() - inicio,
    )
    return AnalisarResponse(**corpo)


def _analisar(img, bruto: bytes, mission: dict | None, content_type: str | None) -> AnalisarResponse:
    familia = resolver_familia(mission)
    usar_gemini = visao_llm.configurado() and os.getenv("SOULIE_VISION_BACKEND", "auto").strip().lower() != "tensorflow"

    if usar_gemini:
        try:
            corpo = visao_llm.analisar_com_gemini(img, bruto, mission, content_type)
            if img is not None:
                corpo["hash_imagem"] = str(imagehash.phash(img))
            log.info(
                "Analisou familia=%s origem=%s status=%s",
                corpo.get("familia") or familia,
                corpo.get("origem"),
                corpo.get("status"),
            )
            return AnalisarResponse(**corpo)
        except visao_llm.LlmIndisponivel as exc:
            log.warning("Gemini indisponível (%s); tentando reserva local.", exc)
            if img is None or not visao_llm.carregar_tensorflow_no_startup():
                raise HTTPException(
                    status_code=503,
                    detail="A análise por visão está temporariamente indisponível.",
                ) from exc

    if img is None:
        raise HTTPException(status_code=400, detail="O arquivo enviado não é uma imagem válida.")
    return _via_tensorflow(img, mission)


@app.get("/health")
def health():
    tf_no_ar = visao_llm.carregar_tensorflow_no_startup()
    tf_pronto = False
    tf_erro = None
    treinados: list[str] = []
    if tf_no_ar:
        import inferencia

        tf_pronto = inferencia.modelos_prontos()
        tf_erro = inferencia.erro_preparo()
        treinados = inferencia.familias_com_modelo()

    gemini_ok = visao_llm.configurado()
    ready = gemini_ok or tf_pronto
    backend = "gemini" if gemini_ok and os.getenv("SOULIE_VISION_BACKEND", "auto").strip().lower() != "tensorflow" else (
        "tensorflow" if tf_no_ar else "nenhum"
    )
    return {
        "status": "ok",
        "ready": ready,
        "backend": backend,
        "gemini": gemini_ok,
        "gemini_model": visao_llm.modelo() if gemini_ok else None,
        "tensorflow": tf_no_ar,
        "tensorflow_ready": tf_pronto,
        "familias": list(FAMILIAS),
        "modelos_treinados": treinados,
        "hint_por_familia": {nome: hint(nome) for nome in FAMILIAS},
        "erro_preparo": tf_erro,
    }


@app.post("/v1/analisar", response_model=AnalisarResponse)
def analisar(
    pedido: AnalisarRequest,
    authorization: str | None = Header(default=None),
):
    """Contrato do java-soulie: JSON com a missão e a imagem (URL ou base64)."""
    log.info(
        "POST /v1/analisar validationId=%s temUrl=%s temBase64=%s",
        pedido.validation_id,
        bool(pedido.image_url),
        bool(pedido.image_base64),
    )
    _conferir_chave(authorization)
    bruto = _bytes_da_requisicao(pedido)
    if len(bruto) > TAMANHO_MAXIMO_BYTES:
        raise HTTPException(status_code=413, detail="Foto maior que 10 MB.")
    img = None
    try:
        img = _abrir(bruto)
    except HTTPException:
        if not (visao_llm.configurado() and visao_llm.e_heic(bruto)):
            raise
        log.info("Foto HEIC encaminhada ao Gemini sem decode local.")
    mission = pedido.mission.model_dump() if pedido.mission else None
    return _analisar(img, bruto, mission, pedido.content_type)


@app.post("/validar", response_model=AnalisarResponse)
def validar(
    foto: UploadFile = File(...),
    authorization: str | None = Header(default=None),
):
    """Atalho manual (multipart). Sem missão, assume a família mobilidade."""
    _conferir_chave(authorization)
    bruto = foto.file.read()
    img = _abrir(bruto)
    return _analisar(img, bruto, {"tag": "Mobilidade", "name": "Transporte Público"}, foto.content_type)


@app.get("/v1/familias/{tag}")
def familia_da_tag(tag: str, name: str = ""):
    familia = resolver_familia({"tag": tag, "name": name})
    return {
        "familia": familia,
        "hint": hint(familia),
        "modelo_treinado": pasta_existe(familia),
    }


def pasta_existe(familia: str) -> bool:
    from inferencia import pasta_modelo

    return pasta_modelo(familia) is not None
