import json
from io import BytesIO

import httpx
import pytest
from PIL import Image

import visao_llm


def _jpeg_bytes(cor=(10, 80, 200), tamanho=(32, 32)) -> bytes:
    img = Image.new("RGB", tamanho, cor)
    buf = BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_sem_chave_nao_carrega_tf_se_backend_gemini(monkeypatch):
    monkeypatch.setenv("SOULIE_VISION_BACKEND", "gemini")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("SOULIE_VISION_FALLBACK_TF", raising=False)
    assert visao_llm.configurado() is False
    assert visao_llm.carregar_tensorflow_no_startup() is False


def test_com_chave_auto_nao_sobe_tensorflow(monkeypatch):
    monkeypatch.setenv("SOULIE_VISION_BACKEND", "auto")
    monkeypatch.setenv("GEMINI_API_KEY", "teste")
    monkeypatch.delenv("SOULIE_VISION_FALLBACK_TF", raising=False)
    assert visao_llm.carregar_tensorflow_no_startup() is False


def test_fallback_tf_sobe_mesmo_com_gemini(monkeypatch):
    monkeypatch.setenv("SOULIE_VISION_BACKEND", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "teste")
    monkeypatch.setenv("SOULIE_VISION_FALLBACK_TF", "true")
    assert visao_llm.carregar_tensorflow_no_startup() is True


def test_extrair_json_com_fence():
    dados = visao_llm.extrair_json("```json\n{\"status\": \"APPROVED\"}\n```")
    assert dados["status"] == "APPROVED"


def test_normalizar_aceita_aprovado(monkeypatch):
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    corpo = visao_llm.normalizar_veredito(
        {
            "status": "aprovado",
            "confidence": 1.4,
            "reason": "Tem uma bicicleta na ciclofaixa.",
            "detections": ["bicicleta", "  ", None],
        },
        "mobilidade",
    )
    assert corpo["status"] == "APPROVED"
    assert corpo["confidence"] == 1.0
    assert corpo["origem"] == "gemini"
    assert corpo["detections"] == ["bicicleta"]


def test_status_desconhecido_nao_vira_rejected():
    with pytest.raises(visao_llm.LlmIndisponivel):
        visao_llm.normalizar_veredito({"status": "PENDING"}, "mobilidade")


def test_prompt_fala_da_missao():
    texto = visao_llm.montar_prompt(
        {
            "name": "Transporte Público",
            "tag": "Mobilidade",
            "description": "Use ônibus, metrô ou bicicleta",
        }
    )
    assert "Transporte Público" in texto
    assert "APPROVED" in texto
    assert "REJECTED" in texto


def test_e_heic_reconhece_ftyp():
    cabeca = b"\x00\x00\x00\x18ftypheic" + b"\x00" * 8
    assert visao_llm.e_heic(cabeca) is True
    assert visao_llm.e_heic(_jpeg_bytes()) is False


def test_jpeg_compacto_reduz_foto_grande():
    img = Image.new("RGB", (4000, 3000), (20, 20, 20))
    compacto = visao_llm.jpeg_compacto(img)
    reaberto = Image.open(BytesIO(compacto))
    assert max(reaberto.size) <= visao_llm.LADO_MAXIMO
    assert reaberto.format == "JPEG"


class _FakeResposta:
    def __init__(self, status_code: int, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class _FakeCliente:
    def __init__(self, resposta: _FakeResposta):
        self.resposta = resposta

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def post(self, *args, **kwargs):
        return self.resposta


def test_429_vira_indisponivel_nao_reprovacao(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "teste")
    monkeypatch.setattr(
        visao_llm.httpx,
        "Client",
        lambda **kwargs: _FakeCliente(_FakeResposta(429, {})),
    )
    img = Image.open(BytesIO(_jpeg_bytes()))
    with pytest.raises(visao_llm.LlmIndisponivel, match="429"):
        visao_llm.analisar_com_gemini(img, _jpeg_bytes(), {"tag": "Mobilidade", "name": "Transporte Público"})


def test_resposta_gemini_aprovada(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "teste")
    payload = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": json.dumps(
                                {
                                    "status": "APPROVED",
                                    "confidence": 0.91,
                                    "reason": "A foto mostra um ônibus urbano.",
                                    "detections": ["ônibus"],
                                }
                            )
                        }
                    ]
                }
            }
        ]
    }
    monkeypatch.setattr(
        visao_llm.httpx,
        "Client",
        lambda **kwargs: _FakeCliente(_FakeResposta(200, payload)),
    )
    img = Image.open(BytesIO(_jpeg_bytes()))
    corpo = visao_llm.analisar_com_gemini(
        img,
        _jpeg_bytes(),
        {"tag": "Mobilidade", "name": "Transporte Público"},
    )
    assert corpo["status"] == "APPROVED"
    assert corpo["origem"] == "gemini"
    assert "ônibus" in corpo["reason"].lower() or corpo["detections"] == ["ônibus"]


def test_timeout_vira_indisponivel(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "teste")

    class _TimeoutCliente(_FakeCliente):
        def post(self, *args, **kwargs):
            raise httpx.TimeoutException("slow")

    monkeypatch.setattr(visao_llm.httpx, "Client", lambda **kwargs: _TimeoutCliente(_FakeResposta(200, {})))
    img = Image.open(BytesIO(_jpeg_bytes()))
    with pytest.raises(visao_llm.LlmIndisponivel, match="Timeout"):
        visao_llm.analisar_com_gemini(img, _jpeg_bytes(), {"tag": "Reciclagem", "name": "Reciclar Resíduos"})
