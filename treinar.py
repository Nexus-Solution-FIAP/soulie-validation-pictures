"""
Treina a cabeça de uma família e grava em modelos/<familia>/.

    python treinar.py --familia mobilidade
    python treinar.py --familia reciclagem
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from collections import Counter

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors

import preprocessamento as pp
from familias import FAMILIAS

PASTA_DATASET = Path("dataset")
PASTA_MODELOS = Path("modelos")
K_VIZINHOS = 3
N_FOLDS = 5
TAXA_REJEICAO_VALIDAS = 0.05
SEED = 42
VERSAO = "1.1"


def _carregar(familia: str):
    cfg = FAMILIAS[familia]
    fotos, rotulos = [], []
    for classe in cfg["classes"]:
        caminhos = pp.listar_fotos(PASTA_DATASET / cfg["pastas"][classe])
        print(f"{classe:22s} {len(caminhos)} fotos")
        if not caminhos:
            raise SystemExit(
                f"Sem fotos em dataset/{cfg['pastas'][classe]}. "
                f"Coloque exemplos e rode de novo. Dica: {cfg['hint']}"
            )
        for caminho in caminhos:
            fotos.append(pp.abrir_imagem(caminho))
            rotulos.append(classe)
    invalidas = [pp.abrir_imagem(c) for c in pp.listar_fotos(PASTA_DATASET / "invalidas")]
    print(f"{'invalidas':22s} {len(invalidas)} fotos (só calibração)")
    return cfg, fotos, np.array(rotulos), invalidas


def _novo_clf():
    return LogisticRegression(max_iter=2000, class_weight="balanced")


def treinar(familia: str) -> None:
    if familia not in FAMILIAS:
        raise SystemExit(f"Família desconhecida: {familia}. Opções: {', '.join(FAMILIAS)}")

    cfg, fotos, rotulos, invalidas = _carregar(familia)
    validas = cfg["validas"]
    negativa = cfg["negativa"]

    print("Gerando embeddings…")
    emb = pp.gerar_embeddings(fotos)
    variacoes = [pp.aumentar(f) for f in fotos]
    n_var = len(variacoes[0])
    emb_aug = pp.gerar_embeddings([v for vs in variacoes for v in vs]).reshape(len(fotos), n_var, -1)
    emb_invalidas = pp.gerar_embeddings(invalidas) if invalidas else np.empty((0, emb.shape[1]))

    def montar(indices):
        x = np.concatenate([emb[indices], emb_aug[indices].reshape(-1, emb.shape[1])])
        y = np.concatenate([rotulos[indices], np.repeat(rotulos[indices], n_var)])
        return x, y

    ordem = sorted(set(rotulos.tolist()))
    n_splits = min(N_FOLDS, min(Counter(rotulos).values()))
    if n_splits < 2:
        raise SystemExit("Precisa de pelo menos 2 fotos por classe para a validação cruzada.")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    probs_oof = np.zeros((len(fotos), len(ordem)))
    dist_oof = np.zeros(len(fotos))

    for treino, teste in skf.split(emb, rotulos):
        x, y = montar(treino)
        clf = _novo_clf().fit(x, y)
        # previsões alinhadas à ordem alfabética do classificador
        probs = clf.predict_proba(emb[teste])
        for i, classe in enumerate(clf.classes_):
            probs_oof[teste, ordem.index(classe)] += probs[:, i]
        det = NearestNeighbors(n_neighbors=min(K_VIZINHOS, len(x))).fit(x[np.isin(y, validas)])
        dist_oof[teste] = pp.distancia_anomalia(det, emb[teste])

    pred = np.array(ordem)[probs_oof.argmax(axis=1)]
    print(classification_report(rotulos, pred, digits=3))

    x_todas, y_todas = montar(np.arange(len(fotos)))
    clf_final = _novo_clf().fit(x_todas, y_todas)
    det_final = NearestNeighbors(n_neighbors=min(K_VIZINHOS, len(x_todas))).fit(
        x_todas[np.isin(y_todas, validas)]
    )
    probs_inv = clf_final.predict_proba(emb_invalidas) if len(invalidas) else np.empty((0, len(ordem)))
    dist_inv = pp.distancia_anomalia(det_final, emb_invalidas) if len(invalidas) else np.empty(0)

    eh_valida = np.isin(rotulos, validas)
    eh_neg = rotulos == negativa
    limite_distancia = float(np.quantile(dist_oof[eh_valida], 1 - TAXA_REJEICAO_VALIDAS))

    def simular(limite_confianca):
        cfg_dec = {
            "limite_distancia": limite_distancia,
            "limite_confianca": limite_confianca,
            "classe_negativa": negativa,
            "motivo_negativa": cfg["motivo_negativa"],
        }
        status = np.array([
            pp.decidir(p, ordem, d, cfg_dec)["status"]
            for p, d in zip(probs_oof, dist_oof)
        ])
        status_inv = np.array([
            pp.decidir(p, list(clf_final.classes_), d, cfg_dec)["status"]
            for p, d in zip(probs_inv, dist_inv)
        ]) if len(invalidas) else np.array([])
        aprov = (status[eh_valida] == "APROVADO").mean() if eh_valida.any() else 0
        fraudes = np.concatenate([status[eh_neg], status_inv]) if eh_neg.any() else status_inv
        bloq = (fraudes == "REPROVADO").mean() if len(fraudes) else 1.0
        return aprov, bloq

    resultados = []
    for limite in np.round(np.arange(0.40, 0.96, 0.05), 2):
        aprov, bloq = simular(limite)
        resultados.append(((aprov + bloq) / 2, limite, aprov, bloq))
        print(f"limite {limite:.2f} | válidas {aprov:.0%} | fraudes barradas {bloq:.0%}")
    _, limite_confianca, aprov, bloq = max(resultados)
    limite_confianca = float(limite_confianca)

    destino = PASTA_MODELOS / familia
    destino.mkdir(parents=True, exist_ok=True)
    config = {
        "versao": VERSAO,
        "familia": familia,
        "tamanho_imagem": pp.TAMANHO_IMAGEM,
        "limite_confianca": round(limite_confianca, 4),
        "limite_distancia": round(limite_distancia, 4),
        "k_vizinhos": K_VIZINHOS,
        "classes_modelo": list(clf_final.classes_),
        "classes_validas": validas,
        "classe_negativa": negativa,
        "motivo_negativa": cfg["motivo_negativa"],
        "metricas_validacao": {
            "validas_aprovadas": round(float(aprov), 4),
            "fraudes_barradas": round(float(bloq), 4),
            "fotos_treino": int(len(fotos)),
            "fotos_invalidas_calibracao": int(len(invalidas)),
        },
    }
    joblib.dump(clf_final, destino / "classificador.joblib")
    joblib.dump(det_final, destino / "detector_anomalia.joblib")
    (destino / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(config, indent=2, ensure_ascii=False))
    print(f"Salvo em {destino}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--familia", required=True, choices=sorted(FAMILIAS))
    treinar(parser.parse_args().familia)
