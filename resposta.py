"""Traduz o veredito interno (APROVADO/REPROVADO) para o contrato do Java."""
from __future__ import annotations

from familias import FAMILIAS, hint


def para_contrato(interno: dict, familia: str, origem: str, versao: str) -> dict:
    cfg = FAMILIAS[familia]
    classe = interno.get("classe")
    motivo = interno.get("motivo")
    status_interno = interno.get("status")
    aprovado = status_interno == "APROVADO"

    if aprovado:
        chave_razao = f"APROVADO_{str(classe).upper()}" if classe else None
        reason = cfg["razoes"].get(chave_razao) or cfg["deteccoes"].get(classe)
        detections = [cfg["deteccoes"][classe]] if classe in cfg["deteccoes"] else []
        if not detections and reason:
            detections = [reason]
    else:
        reason = cfg["razoes"].get(motivo, "Não foi possível comprovar esta missão com esta foto.")
        detections = [cfg["deteccoes"][classe]] if classe in cfg["deteccoes"] else []

    return {
        "status": "APPROVED" if aprovado else "REJECTED",
        "detections": detections,
        "confidence": float(interno.get("confianca") or 0),
        "reason": reason,
        "provider": f"soulie-vision-{familia}-{versao}",
        "classe": classe,
        "distancia": interno.get("distancia"),
        "motivo": motivo,
        "familia": familia,
        "origem": origem,
        "hint": hint(familia),
    }
