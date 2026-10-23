"""
API de validação de fotos do Soulie.

Rodar:  uvicorn app:app --port 5000
Docs:   http://localhost:5000/docs
"""
import json
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import imagehash
import joblib
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import UnidentifiedImageError

import preprocessamento as pp

PASTA_MODELOS = Path(__file__).parent / "modelos"
TAMANHO_MAXIMO_BYTES = 10 * 1024 * 1024  # 10 MB

modelos: dict = {}
_trava_inferencia = threading.Lock()  # uma inferência por vez no TensorFlow


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carrega os modelos uma vez, quando a API sobe."""
    modelos["classificador"] = joblib.load(PASTA_MODELOS / "classificador.joblib")
    modelos["detector"] = joblib.load(PASTA_MODELOS / "detector_anomalia.joblib")
    modelos["config"] = json.loads((PASTA_MODELOS / "config.json").read_text(encoding="utf-8"))
    pp.carregar_extrator()
    yield
    modelos.clear()


app = FastAPI(title="Soulie - Validação de Fotos", version="1.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "versao_modelo": modelos["config"]["versao"]}


@app.post("/validar")
def validar(foto: UploadFile = File(...)):
    conteudo = foto.file.read()
    if not conteudo:
        raise HTTPException(status_code=400, detail="Arquivo vazio.")
    if len(conteudo) > TAMANHO_MAXIMO_BYTES:
        raise HTTPException(status_code=413, detail="Foto maior que 10 MB.")

    try:
        img = pp.abrir_imagem(conteudo)
    except (UnidentifiedImageError, OSError):
        raise HTTPException(status_code=400, detail="O arquivo enviado não é uma imagem válida.")

    config = modelos["config"]
    classificador = modelos["classificador"]

    with _trava_inferencia:
        embedding = pp.gerar_embeddings([img])

    distancia = float(pp.distancia_anomalia(modelos["detector"], embedding)[0])
    probs = classificador.predict_proba(embedding)[0]
    resultado = pp.decidir(probs, list(classificador.classes_), distancia, config)

    # Hash perceptual: o backend guarda e compara com fotos anteriores
    # para barrar a mesma foto enviada duas vezes.
    resultado["hash_imagem"] = str(imagehash.phash(img))
    resultado["versao_modelo"] = config["versao"]
    return resultado
