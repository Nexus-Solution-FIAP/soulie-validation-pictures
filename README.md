# Soulie: validação de fotos

Confere se a foto enviada no "Regar Soulie" é mesmo de mobilidade sustentável (bicicleta ou transporte público).

```
foto → recorte 224x224 → MobileNetV2 (embedding) → classificador + detector de anomalia → decisão
```

## 1. Instalar

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Montar o dataset

Coloque as fotos (`.jpg`, `.jpeg`, `.png` ou `.webp`) nas pastas:

| Pasta | O que vai nela | Uso |
|---|---|---|
| `dataset/bicicleta/` | as 29 fotos de bicicleta | treino |
| `dataset/transporte_publico/` | as 29 fotos de ônibus/metrô | treino |
| `dataset/carro_particular/` | as 29 fotos de carro | treino (classe negativa) |
| `dataset/invalidas/` | 20 a 30 fotos aleatórias: parede, mesa, comida, selfie, tela | só calibração |

## 3. Treinar

Abra `treinamento.ipynb` no VS Code (ou Colab) e rode todas as células. No fim, a pasta `modelos/` terá:

- `classificador.joblib`: diz se a foto é bicicleta, transporte público ou carro
- `detector_anomalia.joblib`: mede o quanto a foto foge das fotos válidas
- `config.json`: limites de confiança e distância, classes e métricas

Na primeira execução o Keras baixa os pesos da MobileNetV2 (~9 MB).

## 4. Subir a API

```bash
uvicorn app:app --port 5000
```

Documentação interativa: http://localhost:5000/docs

## 5. Usar

```bash
curl -F "foto=@minha_foto.jpg" http://localhost:5000/validar
```

Resposta:

```json
{
  "status": "APROVADO",
  "classe": "bicicleta",
  "confianca": 0.9691,
  "distancia": 0.1535,
  "motivo": null,
  "hash_imagem": "ca956bf4b97481a8",
  "versao_modelo": "1.0"
}
```

### Regras de decisão (nessa ordem)

| Condição | status | motivo |
|---|---|---|
| Classe prevista = carro | REPROVADO | `CARRO_PARTICULAR` |
| Distância acima do limite | REPROVADO | `CENA_INVALIDA` |
| Confiança abaixo do limite | REPROVADO | `INCONCLUSIVO` |
| Bicicleta ou transporte, confiante e perto das válidas | APROVADO | `null` |

### Erros

- `400`: arquivo vazio ou que não é imagem
- `413`: foto maior que 10 MB

### Foto repetida

O campo `hash_imagem` é um hash perceptual: fotos iguais ou quase iguais geram hashes parecidos. Quem chama a API guarda esse hash e compara com os anteriores do usuário. Se a diferença for de 5 bits ou menos, é a mesma foto.

### Outros endpoints

- `GET /health`: responde se a API está de pé e a versão do modelo
