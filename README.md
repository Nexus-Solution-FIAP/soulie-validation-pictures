# Soulie: validação de fotos

Confere se a foto do "Regar Soulie" comprova **a missão que o usuário está fazendo**.

O Java (`java-soulie`) chama `POST /v1/analisar`. As 21 missões do app caem em 9 famílias visuais. Mobilidade já tem modelo treinado; as outras usam um fallback ImageNet até você colocar fotos e rodar `treinar.py`.

```
foto + missão.tag → família → classificador treinado (ou ImageNet) → APPROVED | REJECTED
```

## 1. Instalar

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Subir a API

```bash
uvicorn app:app --port 5000
```

- Docs: http://localhost:5000/docs
- Saúde: http://localhost:5000/health

No `java-soulie` (perfil `dev`):

```
AI_VALIDATION_URL=http://localhost:5000/v1/analisar
AI_DELIVERY_MODE=BYTES
```

Se `AI_VALIDATION_API_KEY` estiver definida aqui e no Java, a chamada leva `Authorization: Bearer …`.

## 3. Contrato (o que o Java envia)

`POST /v1/analisar`

```json
{
  "validationId": 42,
  "imageUrl": null,
  "imageBase64": "...",
  "contentType": "image/jpeg",
  "mission": {
    "id": 2,
    "name": "Transporte Público",
    "tag": "Mobilidade",
    "description": "Use ônibus, metrô ou bicicleta",
    "emoji": "🚌"
  }
}
```

Resposta (campos que o Java lê):

```json
{
  "status": "APPROVED",
  "detections": ["Bicicleta reconhecida"],
  "confidence": 0.96,
  "reason": "A foto mostra uma bicicleta em uso.",
  "provider": "soulie-vision-mobilidade-1.0"
}
```

`status` é `APPROVED` ou `REJECTED`. Qualquer outro valor o Java trata como falha de análise, não como reprovação.

Atalho manual: `curl -F "foto=@minha.jpg" http://localhost:5000/validar` (assume mobilidade).

## 4. Famílias

| Família | Tags / missões | Modelo hoje |
|---|---|---|
| `mobilidade` | Transporte Público, Semana Sem Carro | treinado |
| `reciclagem` | Reciclar Resíduos, Reciclagem da Semana | fallback ImageNet |
| `compostagem` | Compostagem | fallback |
| `coleta` | Limpeza Comunitária, Mutirão Anual | fallback |
| `agua` | Economia de Água, Meta Semanal de Água | fallback (só a cena; não mede os 5 min) |
| `consumo` | Compra Consciente, Zero Descartáveis, Mês Zero Plástico, Hábito Sustentável | fallback |
| `energia` | Energia Limpa, Luz Natural, Auditoria de Energia | fallback (cena) |
| `alimentacao` | Refeição Sem Carne, Semana Leve | fallback |
| `natureza` | Plantar uma Muda, Horta do Mês, Árvore do Ano | fallback |

O que fotografar em cada pasta: `dataset/LEIA-ME.md`.

## 5. Encher o dataset (opcional)

Baixa fotos do Wikimedia Commons para as pastas. **Não treina.** Você olha, apaga o que não comprova a missão, e só então treina.

```bash
python baixar_dataset.py
python baixar_dataset.py --familia reciclagem
```

## 6. Treinar uma família nova

Coloque umas 25 fotos por classe e:

```bash
python treinar.py --familia reciclagem
```

Isso grava `modelos/reciclagem/classificador.joblib` (e o detector + `config.json`). Na próxima subida da API, essa família deixa o ImageNet.

Mobilidade já treinada continua em `modelos/` na raiz, para não quebrar o que já está no git.

## 7. Testes (sem TensorFlow)

```bash
pytest
```
