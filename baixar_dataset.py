"""
Baixa fotos de domínio público para o dataset. Não treina.

    python baixar_dataset.py
    python baixar_dataset.py --familia reciclagem
    python baixar_dataset.py --familia coleta --quantidade 30

Depois abra as pastas, apague o que não comprova a missão, e só então:

    python treinar.py --familia reciclagem
"""
from __future__ import annotations

import argparse
import time
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import httpx
from PIL import Image, UnidentifiedImageError

from familias import FAMILIAS, pasta_dataset
from preprocessamento import EXTENSOES, listar_fotos

PASTA_DATASET = Path("dataset")
USER_AGENT = "SoulieDataset/1.0 (https://github.com; academic FIAP project)"
TAMANHO_MAXIMO_BYTES = 8 * 1024 * 1024
TIMEOUT = httpx.Timeout(8.0, connect=5.0)
PAUSA_S = 0.2

# Várias queries por classe: a busca devolve lixo, então o filtro final é seu.
BUSCAS: dict[str, dict[str, list[str]]] = {
    "mobilidade": {
        "bicicleta": [
            "bicycle in city street",
            "person riding bicycle urban",
            "bike commuting",
        ],
        "transporte_publico": [
            "city bus public transport",
            "subway metro train platform",
            "tram streetcar",
        ],
        "carro_particular": [
            "car parked street",
            "sedan driving city",
            "private car traffic",
        ],
    },
    "reciclagem": {
        "reciclavel": [
            "recycling bins plastic bottles",
            "sorted recyclables paper plastic",
            "selective waste collection bins",
        ],
        "lixo_misturado": [
            "mixed garbage bag",
            "household trash pile",
            "overflowing dumpster",
        ],
    },
    "compostagem": {
        "composteira": [
            "home compost bin",
            "compost pile kitchen scraps",
            "composting organic waste",
        ],
        "outro": [
            "kitchen counter food",
            "plastic packaging on table",
        ],
    },
    "coleta": {
        "coleta": [
            "community cleanup litter pickup",
            "volunteers collecting trash street",
            "bags of collected street waste",
        ],
        "outro": [
            "empty clean sidewalk",
            "city park no litter",
        ],
    },
    "agua": {
        "banho": [
            "bathroom shower head",
            "bathroom faucet sink",
            "shower timer",
        ],
        "outro": [
            "living room interior",
            "office desk",
        ],
    },
    "consumo": {
        "reutilizavel": [
            "reusable water bottle",
            "cloth shopping bag",
            "reusable coffee cup",
        ],
        "descartavel": [
            "plastic shopping bag",
            "disposable plastic cup",
            "single use packaging",
        ],
    },
    "energia": {
        "aparelho": [
            "unplugged power cord outlet",
            "room with natural daylight lamps off",
            "electrical outlet switch",
        ],
        "outro": [
            "busy neon city night",
            "television living room",
        ],
    },
    "alimentacao": {
        "prato_vegetal": [
            "vegetarian meal top view",
            "salad bowl overhead",
            "vegan plate vegetables",
        ],
        "prato_com_carne": [
            "steak dinner plate",
            "hamburger meal",
            "grilled meat plate",
        ],
    },
    "natureza": {
        "muda": [
            "seedling planted in soil",
            "hands planting a tree sapling",
            "home vegetable garden seedlings",
        ],
        "outro": [
            "cut flowers in vase",
            "plastic plant decoration",
        ],
    },
}

BUSCAS_INVALIDAS = [
    "selfie portrait smartphone",
    "blank wall interior",
    "computer screenshot",
    "orange color background",
]


def _extensao(url: str, content_type: str | None) -> str | None:
    caminho = urlparse(url).path.lower()
    for ext in EXTENSOES:
        if caminho.endswith(ext):
            return ext
    tipos = {
        "image/jpeg": ".jpg",
        "image/jpg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
    }
    if content_type:
        tipo = content_type.split(";")[0].strip().lower()
        if tipo in tipos:
            return tipos[tipo]
    return None


def _proximo_nome(pasta: Path, extensao: str) -> Path:
    usados = {p.name.lower() for p in pasta.iterdir()}
    n = 1
    while True:
        nome = f"web_{n:03d}{extensao}"
        if nome not in usados:
            return pasta / nome
        n += 1


def _e_imagem(conteudo: bytes) -> bool:
    try:
        with Image.open(BytesIO(conteudo)) as img:
            img.verify()
        return True
    except (UnidentifiedImageError, OSError, ValueError):
        return False


def _cliente() -> httpx.Client:
    return httpx.Client(
        timeout=TIMEOUT,
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT, "Accept": "application/json,image/*"},
    )


def buscar_commons(cliente: httpx.Client, query: str, limite: int) -> list[str]:
    try:
        resposta = cliente.get(
            "https://commons.wikimedia.org/w/api.php",
            params={
                "action": "query",
                "format": "json",
                "generator": "search",
                "gsrsearch": query,
                "gsrnamespace": 6,
                "gsrlimit": min(40, max(limite, 10)),
                "prop": "imageinfo",
                "iiprop": "url|mime|size",
                "iiurlwidth": 1280,
            },
        )
        resposta.raise_for_status()
        paginas = (resposta.json().get("query") or {}).get("pages") or {}
    except Exception:
        return []

    urls: list[str] = []
    for pagina in paginas.values():
        infos = pagina.get("imageinfo") or []
        if not infos:
            continue
        info = infos[0]
        mime = str(info.get("mime") or "")
        if mime not in {"image/jpeg", "image/png", "image/webp"}:
            continue
        url = info.get("thumburl") or info.get("url")
        if url:
            urls.append(url)
        if len(urls) >= limite:
            break
    return urls


def coletar_urls(cliente: httpx.Client, queries: list[str], limite: int) -> list[str]:
    vistos: set[str] = set()
    escolhidas: list[str] = []
    por_query = max(8, limite)
    for query in queries:
        if len(escolhidas) >= limite:
            break
        # Commons é a fonte estável. Openverse costuma estourar timeout daqui.
        for url in buscar_commons(cliente, query, por_query):
            if url in vistos:
                continue
            vistos.add(url)
            escolhidas.append(url)
            if len(escolhidas) >= limite:
                break
        time.sleep(PAUSA_S)
    return escolhidas


def baixar_para(cliente: httpx.Client, pasta: Path, urls: list[str], faltam: int) -> int:
    pasta.mkdir(parents=True, exist_ok=True)
    salvas = 0
    for url in urls:
        if salvas >= faltam:
            break
        try:
            resposta = cliente.get(url)
            resposta.raise_for_status()
        except Exception:
            continue
        if len(resposta.content) > TAMANHO_MAXIMO_BYTES:
            continue
        if not _e_imagem(resposta.content):
            continue
        ext = _extensao(url, resposta.headers.get("content-type"))
        if ext is None:
            ext = ".jpg"
        destino = _proximo_nome(pasta, ext)
        destino.write_bytes(resposta.content)
        salvas += 1
        time.sleep(PAUSA_S)
    return salvas


def encher_pasta(cliente: httpx.Client, pasta: Path, queries: list[str], quantidade: int) -> tuple[int, int]:
    ja_tem = len(listar_fotos(pasta))
    if ja_tem >= quantidade:
        return 0, ja_tem
    faltam = quantidade - ja_tem
    urls = coletar_urls(cliente, queries, faltam * 3)
    novas = baixar_para(cliente, pasta, urls, faltam)
    return novas, ja_tem + novas


def baixar(familias: list[str], quantidade: int, incluir_invalidas: bool) -> None:
    with _cliente() as cliente:
        print(
            "Só baixo fotos. O treino continua na sua mão: apague as ruins e rode treinar.py.\n",
            flush=True,
        )
        for familia in familias:
            if familia == "agua":
                print(
                    "água: a foto não prova banho de 5 min. Baixo a cena mesmo assim; "
                    "vale mais reescrever a missão do que treinar isto.",
                    flush=True,
                )
            cfg = FAMILIAS[familia]
            for classe in cfg["classes"]:
                pasta = PASTA_DATASET / pasta_dataset(familia, classe)
                print(f"buscando {pasta.as_posix()} …", flush=True)
                queries = BUSCAS[familia][classe]
                novas, total = encher_pasta(cliente, pasta, queries, quantidade)
                print(f"{pasta.as_posix():42s} +{novas:2d}  (total {total})", flush=True)

        if incluir_invalidas:
            pasta = PASTA_DATASET / "invalidas"
            print(f"buscando {pasta.as_posix()} …", flush=True)
            novas, total = encher_pasta(cliente, pasta, BUSCAS_INVALIDAS, quantidade)
            print(f"{pasta.as_posix():42s} +{novas:2d}  (total {total})", flush=True)

    print("\nPróximo passo: abra dataset/, apague o que não serve, depois:", flush=True)
    if len(familias) == 1:
        print(f"  python treinar.py --familia {familias[0]}", flush=True)
    else:
        print("  python treinar.py --familia <nome>", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Baixa fotos para o dataset. Não treina.")
    parser.add_argument(
        "--familia",
        choices=sorted(FAMILIAS),
        help="Uma família só. Sem este flag, baixa todas.",
    )
    parser.add_argument("--quantidade", type=int, default=25, help="Alvo de fotos por pasta (já existentes contam).")
    parser.add_argument("--sem-invalidas", action="store_true", help="Não preenche dataset/invalidas/.")
    args = parser.parse_args()
    if args.quantidade < 2:
        raise SystemExit("--quantidade precisa ser pelo menos 2.")
    familias = [args.familia] if args.familia else sorted(FAMILIAS)
    baixar(familias, args.quantidade, incluir_invalidas=not args.sem_invalidas)


if __name__ == "__main__":
    main()
