"""Modelos do contrato com o java-soulie (docs/04-contrato-ia.md)."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Mission(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: int | None = None
    name: str | None = None
    tag: str | None = None
    description: str | None = None
    emoji: str | None = None


class AnalisarRequest(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    validation_id: int | None = Field(default=None, alias="validationId")
    image_url: str | None = Field(default=None, alias="imageUrl")
    image_base64: str | None = Field(default=None, alias="imageBase64")
    content_type: str | None = Field(default=None, alias="contentType")
    mission: Mission | None = None
    callback_url: str | None = Field(default=None, alias="callbackUrl")


class AnalisarResponse(BaseModel):
    """O Java ignora campos extras; os de cá além do contrato ficam para depuração."""

    status: str
    detections: list[str]
    confidence: float
    reason: str | None
    provider: str
    classe: str | None = None
    distancia: float | None = None
    motivo: str | None = None
    hash_imagem: str | None = None
    familia: str | None = None
    origem: str | None = None
    hint: str | None = None
