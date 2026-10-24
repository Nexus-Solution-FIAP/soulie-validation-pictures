"""
Valida a evidência com um LLM de visão (Gemini), no contrato do Java.

O FastAPI continua em POST /v1/analisar. O modelo local (TensorFlow) só entra
se a API cair — e no Render de 512 MB ele NÃO deve ser carregado.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import re
import time
from io import BytesIO
from typing import Any

import httpx
from PIL import Image

from familias import FAMILIAS, hint, resolver_familia

log = logging.getLogger("uvicorn.error")

URL_PADRAO = "https://generativelanguage.googleapis.com/v1beta/models/{modelo}:generateContent"
MODELO_PADRAO = "gemini-3.5-flash-lite"
LADO_MAXIMO = 1280
JPEG_QUALIDADE = 75
TIMEOUT = httpx.Timeout(20.0, read=45.0)

_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_HEIC_MARCAS = (b"ftypheic", b"ftypheif", b"ftypmif1", b"ftypmsf1")


class LlmIndisponivel(Exception):
    """429, timeout, 5xx ou JSON ilegível — NÃO é reprovação da foto."""


def _flag(nome: str) -> bool:
    return os.getenv(nome, "").strip().lower() in {"1", "true", "yes", "on"}


def chave() -> str:
    return (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()


def modelo() -> str:
    return (os.getenv("GEMINI_MODEL") or MODELO_PADRAO).strip()


def configurado() -> bool:
    return bool(chave())


def carregar_tensorflow_no_startup() -> bool:
    """True só quando o processo deve importar MobileNet (estoura 512 MB)."""
    backend = os.getenv("SOULIE_VISION_BACKEND", "auto").strip().lower()
    fallback = _flag("SOULIE_VISION_FALLBACK_TF")
    if backend == "tensorflow":
        return True
    if backend == "gemini":
        return fallback
    return fallback or not configurado()


def e_heic(conteudo: bytes) -> bool:
    if len(conteudo) < 12:
        return False
    cabeca = conteudo[:32].lower()
    return any(marca in cabeca for marca in _HEIC_MARCAS)


def jpeg_compacto(img: Image.Image) -> bytes:
    """Reduz a foto de celular antes de mandar ao Gemini (RAM e tokens)."""
    rgb = img.convert("RGB")
    largura, altura = rgb.size
    maior = max(largura, altura)
    if maior > LADO_MAXIMO:
        escala = LADO_MAXIMO / maior
        rgb = rgb.resize(
            (max(1, int(largura * escala)), max(1, int(altura * escala))),
            Image.Resampling.LANCZOS,
        )
    buf = BytesIO()
    rgb.save(buf, format="JPEG", quality=JPEG_QUALIDADE, optimize=True)
    return buf.getvalue()


def montar_prompt(mission: dict | None) -> str:
    familia = resolver_familia(mission)
    cfg = FAMILIAS[familia]
    missao = mission or {}
    nome = (missao.get("name") or "").strip() or "esta missão"
    tag = (missao.get("tag") or "").strip() or familia
    descricao = (missao.get("description") or "").strip() or cfg["hint"]
    return (
        "Você valida evidências do app Soulie. A pergunta não é o que há na foto: "
        "é se a imagem comprova ESTA missão.\n"
        f"Missão: {nome}\n"
        f"Tag: {tag}\n"
        f"Descrição: {descricao}\n"
        f"Dica visual: {hint(familia)}\n"
        "Regras:\n"
        "- A evidência vem de computador: arquivo da pasta, print de tela, foto baixada, "
        "imagem da web salva no PC ou webcam. Isso é válido.\n"
        "- NÃO reprove só porque parece print, print do Windows, foto de banco de imagens, "
        "janela do navegador, watermark, ou porque não foi tirada no celular.\n"
        "- APPROVED se o conteúdo da imagem razoavelmente mostra a ação da missão "
        "(ônibus, bike, lixeiras, prato, planta, etc.). Não exija o usuário na foto.\n"
        "- REJECTED só se o conteúdo não tem relação com a missão "
        "(meme sem evidência, tela preta, documento em branco, outra ação).\n"
        "- Água e energia: a imagem não prova duração de banho nem standby. "
        "Se mostrar a cena (chuveiro, torneira, lâmpada, tomada, cômodo com luz natural), aprove.\n"
        "- Reciclagem: material separado ou lixeiras seletivas. Saco misturado reprova.\n"
        "- Mobilidade: bicicleta, ônibus, metrô, trem. Carro particular reprova.\n"
        "- Responda APENAS JSON: "
        '{"status":"APPROVED"|"REJECTED","confidence":0a1,"reason":"frase curta em português",'
        '"detections":["o que você viu"]}\n'
        "- reason é mostrada ao usuário. Seja concreto, sem acusar fraude."
    )


def extrair_json(texto: str) -> dict[str, Any]:
    if not texto or not texto.strip():
        raise LlmIndisponivel("Resposta vazia do Gemini.")
    limpo = _FENCE.sub("", texto.strip()).strip()
    try:
        dados = json.loads(limpo)
    except json.JSONDecodeError as exc:
        inicio = limpo.find("{")
        fim = limpo.rfind("}")
        if inicio < 0 or fim <= inicio:
            raise LlmIndisponivel("Gemini não devolveu JSON.") from exc
        dados = json.loads(limpo[inicio : fim + 1])
    if not isinstance(dados, dict):
        raise LlmIndisponivel("JSON do Gemini não é um objeto.")
    return dados


def normalizar_veredito(dados: dict[str, Any], familia: str) -> dict[str, Any]:
    bruto = str(dados.get("status") or "").strip().upper()
    if bruto in {"APPROVED", "APROVED", "APROVADO"}:
        status = "APPROVED"
    elif bruto in {"REJECTED", "REJECT", "REPROVADO"}:
        status = "REJECTED"
    else:
        raise LlmIndisponivel(f"Status inesperado do Gemini: {bruto!r}")

    try:
        confianca = float(dados.get("confidence") if dados.get("confidence") is not None else 0.7)
    except (TypeError, ValueError):
        confianca = 0.7
    confianca = max(0.0, min(1.0, confianca))

    reason = str(dados.get("reason") or "").strip()
    if not reason:
        reason = (
            "A foto comprova a missão."
            if status == "APPROVED"
            else "Não foi possível comprovar esta missão com esta foto."
        )

    detections: list[str] = []
    bruto_det = dados.get("detections") or []
    if isinstance(bruto_det, str) and bruto_det.strip():
        bruto_det = [bruto_det]
    if isinstance(bruto_det, list):
        for item in bruto_det:
            if item is None:
                continue
            texto = str(item).strip()
            if texto:
                detections.append(texto[:180])
            if len(detections) >= 8:
                break

    return {
        "status": status,
        "detections": detections,
        "confidence": confianca,
        "reason": reason[:500],
        "provider": f"soulie-vision-gemini-{modelo()}",
        "familia": familia,
        "origem": "gemini",
        "hint": hint(familia),
        "classe": None,
        "distancia": None,
        "motivo": None,
    }


def _texto_da_resposta(corpo: dict[str, Any]) -> str:
    prompt = corpo.get("promptFeedback") or {}
    if str(prompt.get("blockReason") or ""):
        return json.dumps(
            {
                "status": "REJECTED",
                "confidence": 0.0,
                "reason": "Não foi possível visualizar esta foto. Tente outra, sem filtro pesado.",
                "detections": [],
            }
        )
    candidatos = corpo.get("candidates") or []
    if not candidatos:
        raise LlmIndisponivel("Gemini respondeu sem candidatos.")
    primeiro = candidatos[0] or {}
    finish = str(primeiro.get("finishReason") or "")
    if finish in {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST"}:
        return json.dumps(
            {
                "status": "REJECTED",
                "confidence": 0.0,
                "reason": "Não foi possível visualizar esta foto. Tente outra.",
                "detections": [],
            }
        )
    partes = ((primeiro.get("content") or {}).get("parts")) or []
    textos = [str(p.get("text") or "") for p in partes if p.get("text")]
    return "\n".join(textos).strip()


def _payload(prompt: str, imagem: bytes, mime: str, com_thinking: bool = True) -> dict[str, Any]:
    config: dict[str, Any] = {
        "temperature": 0.1,
        "maxOutputTokens": 256,
        "responseMimeType": "application/json",
    }
    if com_thinking:
        config["thinkingConfig"] = {"thinkingBudget": 0}
    return {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"inlineData": {"mimeType": mime, "data": base64.b64encode(imagem).decode("ascii")}},
                    {"text": prompt},
                ],
            }
        ],
        "generationConfig": config,
    }


def analisar_com_gemini(
    img: Image.Image | None,
    bruto: bytes,
    mission: dict | None,
    content_type: str | None = None,
) -> dict[str, Any]:
    if not configurado():
        raise LlmIndisponivel("GEMINI_API_KEY ausente.")

    familia = resolver_familia(mission)
    prompt = montar_prompt(mission)

    if img is not None:
        imagem = jpeg_compacto(img)
        mime = "image/jpeg"
    elif e_heic(bruto):
        imagem = bruto
        mime = "image/heic"
    else:
        imagem = bruto
        mime = (content_type or "image/jpeg").split(";")[0].strip() or "image/jpeg"

    url = URL_PADRAO.format(modelo=modelo())
    headers = {
        "x-goog-api-key": chave(),
        "Content-Type": "application/json",
    }
    inicio = time.perf_counter()
    try:
        with httpx.Client(timeout=TIMEOUT) as cliente:
            resposta = cliente.post(url, headers=headers, json=_payload(prompt, imagem, mime))
            if resposta.status_code == 400:
                resposta = cliente.post(
                    url,
                    headers=headers,
                    json=_payload(prompt, imagem, mime, com_thinking=False),
                )
    except httpx.TimeoutException as exc:
        raise LlmIndisponivel("Timeout no Gemini.") from exc
    except httpx.HTTPError as exc:
        raise LlmIndisponivel("Falha de rede no Gemini.") from exc

    if resposta.status_code == 429:
        raise LlmIndisponivel("Cota do Gemini esgotada (429).")
    if resposta.status_code >= 500:
        raise LlmIndisponivel(f"Gemini {resposta.status_code}.")
    if resposta.status_code >= 400:
        log.warning("Gemini recusou a chamada: HTTP %s", resposta.status_code)
        raise LlmIndisponivel(f"Gemini {resposta.status_code}.")

    try:
        corpo = resposta.json()
    except ValueError as exc:
        raise LlmIndisponivel("Gemini não devolveu JSON HTTP.") from exc

    veredito = normalizar_veredito(extrair_json(_texto_da_resposta(corpo)), familia)
    log.info(
        "Gemini familia=%s status=%s em %.1fs",
        familia,
        veredito["status"],
        time.perf_counter() - inicio,
    )
    return veredito
