# O que fotografar em cada pasta

A API escolhe a família pela **tag da missão**. Fotos em `invalidas/` servem para todas as famílias (selfie, tela, parede, meme).

| Pasta | Missões | Foto positiva | Foto negativa |
|---|---|---|---|
| `bicicleta/`, `transporte_publico/`, `carro_particular/` | Transporte Público, Semana Sem Carro | bike / ônibus / metrô | carro particular |
| `reciclagem/reciclavel/` | Reciclar Resíduos, Reciclagem da Semana | reciclável separado ou lixeiras seletivas | `lixo_misturado/` saco misturado |
| `compostagem/composteira/` | Compostagem | composteira, orgânico compostando | `outro/` |
| `coleta/residuos_recolhidos/` | Limpeza Comunitária, Mutirão Anual | lixo recolhido em rua/condomínio | `outro/` |
| `agua/banho/` | Economia de Água, Meta Semanal de Água | chuveiro, timer, torneira | `outro/` |
| `consumo/reutilizavel/` | Compra Consciente, Zero Descartáveis, Mês Zero Plástico, Hábito Sustentável | copo, garrafa, ecobag | `descartavel/` |
| `energia/aparelho/` | Energia Limpa, Luz Natural, Auditoria de Energia | tomada desligada, cômodo com sol e luz apagada | `outro/` |
| `alimentacao/prato_vegetal/` | Refeição Sem Carne, Semana Leve | prato vegetariano de cima | `prato_com_carne/` |
| `natureza/muda/` | Plantar uma Muda, Horta do Mês, Árvore do Ano | muda, horta, árvore plantada | `outro/` |

Atalho para encher as pastas (você ainda precisa filtrar o que ficou ruim):

```bash
python baixar_dataset.py --familia reciclagem
```

Depois de encher uma família (umas 25 fotos por classe):

```bash
python treinar.py --familia reciclagem
```
