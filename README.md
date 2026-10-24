# Soulie: validação de fotos

Confere se a foto do "Regar Soulie" comprova **a missão que o usuário está fazendo**.

O Java (`java-soulie`) chama `POST /v1/analisar`. O FastAPI não aponta o Java direto no Gemini: recebe a foto + nome/tag/descrição e devolve o mesmo contrato de sempre.

```
foto + missão → Gemini Flash-Lite → APPROVED | REJECTED
                 ↘ 429/timeout → 503 (FAILED no Java), ou TensorFlow se a flag estiver ligada
```

No Render de **512 MB** o TensorFlow estoura a RAM no primeiro POST (`Instance failed: used over 512MB`). O caminho do evento é **Gemini ligado, TensorFlow desligado**.

## 1. Instalar (API)

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Para treinar famílias locais (MobileNet + joblib):

```bash
pip install -r requirements-treino.txt
```

## 2. Subir a API

Crie um `.env` a partir do `.env.example`. No Render / na demo:

```
GEMINI_API_KEY=...          # https://aistudio.google.com/app/apikey
SOULIE_VISION_BACKEND=gemini
SOULIE_VISION_FALLBACK_TF=false
```

```bash
uvicorn app:app --port 5000
```

- Docs: http://localhost:5000/docs
- Saúde: http://localhost:5000/health → `ready: true`, `backend: gemini`, `tensorflow: false`

No `java-soulie` nada muda:

```
AI_VALIDATION_URL=http://localhost:5000/v1/analisar
AI_DELIVERY_MODE=BYTES
```

Se `AI_VALIDATION_API_KEY` estiver definida aqui e no Java, a chamada leva `Authorization: Bearer …`.

`GET /health` responde 200 mesmo sem o TensorFlow. Não configure o health check do Render exigindo modelo local.

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
  "detections": ["Ônibus urbano"],
  "confidence": 0.91,
  "reason": "A foto mostra um ônibus em uso.",
  "provider": "soulie-vision-gemini-gemini-3.5-flash-lite"
}
```

`status` é `APPROVED` ou `REJECTED`. Qualquer outro valor — ou HTTP 5xx — o Java trata como **falha de análise (`FAILED`)**, não como reprovação. Cota 429 e timeout seguem esse caminho de propósito: recusar a foto por falta de cota puniria quem fez a missão.

Atalho manual: `curl -F "foto=@minha.jpg" http://localhost:5000/validar` (assume mobilidade).

## 4. Gemini vs TensorFlow

| | Gemini (`gemini-3.5-flash-lite`) | TensorFlow local |
|---|---|---|
| RAM no Render | cabe em 512 MB | OOM (>512 MB) |
| Entende a missão? | sim (lê nome/tag/descrição) | classifica a cena (ImageNet / joblib) |
| Cold start | segundos | ~60 s + download da MobileNet |
| Sem chave / 429 | HTTP 503 | só se `SOULIE_VISION_FALLBACK_TF=true` |

Água e energia: a foto não prova duração de banho nem standby. O prompt já manda aprovar a **cena**. O texto da missão no app é que precisa mudar.

O Gemini no plano Free pode usar o conteúdo para treinar. Para um dia de estande costuma ser aceitável; para produção, não.

Chat da Alma continua no **Groq**. Não compartilhe a mesma cota com as fotos.

## 5. Famílias (reserva local)

Usadas pelo TensorFlow e pelo texto de dica. Com Gemini, a família só contextualiza o prompt.

| Família | Tags / missões |
|---|---|
| `mobilidade` | Transporte Público, Semana Sem Carro |
| `reciclagem` | Reciclar Resíduos, Reciclagem da Semana |
| `compostagem` | Compostagem |
| `coleta` | Limpeza Comunitária, Mutirão Anual |
| `agua` | Economia de Água, Meta Semanal de Água |
| `consumo` | Compra Consciente, Zero Descartáveis, Mês Zero Plástico, Hábito Sustentável |
| `energia` | Energia Limpa, Luz Natural, Auditoria de Energia |
| `alimentacao` | Refeição Sem Carne, Semana Leve |
| `natureza` | Plantar uma Muda, Horta do Mês, Árvore do Ano |

O que fotografar em cada pasta (treino local): `dataset/LEIA-ME.md`.

## 6. Encher o dataset (opcional)

Baixa fotos do Wikimedia Commons para as pastas. **Não treina.** Você olha, apaga o que não comprova a missão, e só então treina.

```bash
python baixar_dataset.py
python baixar_dataset.py --familia reciclagem
```

## 7. Treinar uma família nova

Coloque umas 25 fotos por classe, com `requirements-treino.txt` instalado, e:

```bash
python treinar.py --familia reciclagem
```

Isso grava `modelos/<familia>/classificador.joblib`. Só entra em cena com `SOULIE_VISION_FALLBACK_TF=true` (e RAM de sobra).

## 8. Testes (sem TensorFlow e sem chamar o Gemini)

```bash
pytest
```
