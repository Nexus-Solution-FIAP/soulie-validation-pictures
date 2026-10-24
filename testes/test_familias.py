from familias import FAMILIAS, resolver_familia
from preprocessamento import decidir
from resposta import para_contrato

MISSOES = [
    ("Reciclar Resíduos", "Reciclagem", "reciclagem"),
    ("Transporte Público", "Mobilidade", "mobilidade"),
    ("Economia de Água", "Água", "agua"),
    ("Compra Consciente", "Consumo", "consumo"),
    ("Energia Limpa", "Energia", "energia"),
    ("Compostagem", "Resíduos", "compostagem"),
    ("Refeição Sem Carne", "Alimentação", "alimentacao"),
    ("Luz Natural", "Energia", "energia"),
    ("Plantar uma Muda", "Natureza", "natureza"),
    ("Zero Descartáveis", "Consumo", "consumo"),
    ("Reciclagem da Semana", "Reciclagem", "reciclagem"),
    ("Semana Sem Carro", "Mobilidade", "mobilidade"),
    ("Meta Semanal de Água", "Água", "agua"),
    ("Semana Leve", "Alimentação", "alimentacao"),
    ("Horta do Mês", "Natureza", "natureza"),
    ("Mês Zero Plástico", "Consumo", "consumo"),
    ("Auditoria de Energia", "Energia", "energia"),
    ("Limpeza Comunitária", "Resíduos", "coleta"),
    ("Árvore do Ano", "Natureza", "natureza"),
    ("Mutirão Anual", "Resíduos", "coleta"),
    ("Hábito Sustentável", "Consumo", "consumo"),
]


def test_todas_as_missoes_do_app_tem_familia():
    for nome, tag, esperada in MISSOES:
        assert resolver_familia({"name": nome, "tag": tag}) == esperada, (nome, tag)


def test_sem_missao_assume_mobilidade():
    assert resolver_familia(None) == "mobilidade"


def test_cada_familia_tem_pastas_e_mensagens():
    for nome, cfg in FAMILIAS.items():
        assert cfg["validas"]
        assert cfg["hint"]
        for classe in cfg["classes"]:
            assert classe in cfg["pastas"]
            assert classe in cfg["deteccoes"]
        assert "CENA_INVALIDA" in cfg["razoes"]
        assert "INCONCLUSIVO" in cfg["razoes"]


def test_decidir_reprova_classe_negativa():
    cfg = {
        "classe_negativa": "carro_particular",
        "motivo_negativa": "CARRO_PARTICULAR",
        "limite_distancia": 0.9,
        "limite_confianca": 0.5,
    }
    probs = [0.1, 0.8, 0.1]
    classes = ["bicicleta", "carro_particular", "transporte_publico"]
    r = decidir(probs, classes, 0.2, cfg)
    assert r["status"] == "REPROVADO"
    assert r["motivo"] == "CARRO_PARTICULAR"


def test_contrato_java_usa_approved_rejected():
    interno = {
        "status": "APROVADO",
        "classe": "bicicleta",
        "confianca": 0.96,
        "distancia": 0.15,
        "motivo": None,
    }
    corpo = para_contrato(interno, "mobilidade", "modelo", "1.0")
    assert corpo["status"] == "APPROVED"
    assert corpo["detections"] == ["Bicicleta reconhecida"]
    assert corpo["provider"] == "soulie-vision-mobilidade-1.0"

    reprovado = para_contrato(
        {**interno, "status": "REPROVADO", "classe": "carro_particular", "motivo": "CARRO_PARTICULAR"},
        "mobilidade",
        "modelo",
        "1.0",
    )
    assert reprovado["status"] == "REJECTED"
    assert "carro" in reprovado["reason"].lower()
