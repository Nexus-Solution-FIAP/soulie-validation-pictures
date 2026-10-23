"""
Famílias visuais das missões do Soulie.

O app tem 21 missões, mas várias pedem a mesma cena (diária/semanal/mensal).
A validação escolhe a família pela tag (e, em Resíduos, pelo nome).
"""
from __future__ import annotations

import unicodedata
from typing import Any


def _norm(texto: str | None) -> str:
    if not texto:
        return ""
    decomposto = unicodedata.normalize("NFD", texto)
    return "".join(c for c in decomposto if unicodedata.category(c) != "Mn").lower().strip()


def _rotulos(*nomes: str) -> set[str]:
    return {n.replace(" ", "_").replace("-", "_") for n in nomes}


# ImageNet (MobileNetV2, decode_predictions): nomes normalizados com underscore.
IMAGENET_TRANSPORTE = _rotulos(
    "trolleybus", "minibus", "school_bus", "streetcar", "bullet_train",
)
IMAGENET_BICICLETA = _rotulos(
    "mountain_bike", "bicycle_built_for_two", "tricycle", "unicycle", "moped", "motor_scooter",
)
# "passenger_car" no ImageNet é carro de passeio, não ônibus.
IMAGENET_CARRO = _rotulos(
    "passenger_car", "sports_car", "beach_wagon", "convertible", "minivan", "jeep",
    "pickup", "cab", "limousine", "model_t", "racer", "recreational_vehicle",
    "minivan", "moving_van", "police_van", "golfcart", "go_kart",
)

IMAGENET_RECICLAVEL = _rotulos(
    "water_bottle", "pop_bottle", "beer_bottle", "plastic_bag", "crate", "barrel",
    "ashcan", "carton", "packet", "envelope", "pill_bottle", "water_jug",
)
IMAGENET_LIXO_MISTURADO = _rotulos("garbage_truck")

IMAGENET_COMPOSTAGEM = _rotulos(
    "mushroom", "potted_plant", "head_cabbage", "broccoli", "ear", "corn",
)
IMAGENET_COLETA = _rotulos(
    "ashcan", "garbage_truck", "broom", "park_bench", "street_sign",
)
IMAGENET_AGUA = _rotulos(
    "bathtub", "washbasin", "toilet_seat", "shower_curtain", "soap_dispenser",
    "toweling", "bath_towel", "water_bottle",
)
IMAGENET_REUTILIZAVEL = _rotulos(
    "water_bottle", "water_jug", "coffee_mug", "cup", "canteen", "shopping_basket",
    "picnic_basket", "wallet", "purse", "backpack", "mailbag",
)
IMAGENET_DESCARTAVEL = _rotulos("plastic_bag", "packet", "carton")
IMAGENET_ENERGIA = _rotulos(
    "switch", "table_lamp", "lampshade", "notebook", "desktop_computer",
    "monitor", "screen", "television", "toaster", "microwave", "printer",
    "space_heater", "washer", "refrigerator", "dishwasher",
)
IMAGENET_VEGETAL = _rotulos(
    "broccoli", "cucumber", "zucchini", "spaghetti_squash", "acorn_squash",
    "butternut_squash", "bell_pepper", "cauliflower", "head_cabbage", "artichoke",
    "mushroom", "granny_smith", "orange", "lemon", "banana", "pomegranate",
    "fig", "pineapple", "strawberry", "guacamole", "salad", "soup", "hot_pot",
    "vegetarian", "ear", "corn", "cardoon",
)
IMAGENET_CARNE = _rotulos(
    "cheeseburger", "hotdog", "meat_loaf", "carbonara", "pizza", "potpie",
)
IMAGENET_NATUREZA = _rotulos(
    "potted_plant", "daisy", "pot", "vase", "greenhouse", "park_bench",
    "rapeseed", "valley", "lakeside",
)

FAMILIAS: dict[str, dict[str, Any]] = {
    "mobilidade": {
        "tags": ["Mobilidade"],
        "modo": "visual",
        "classes": ["bicicleta", "transporte_publico", "carro_particular"],
        "validas": ["bicicleta", "transporte_publico"],
        "negativa": "carro_particular",
        "motivo_negativa": "CARRO_PARTICULAR",
        "pastas": {
            "bicicleta": "bicicleta",
            "transporte_publico": "transporte_publico",
            "carro_particular": "carro_particular",
        },
        "hint": "Foto da bicicleta, ônibus ou metrô. Não envie o carro.",
        "deteccoes": {
            "bicicleta": "Bicicleta reconhecida",
            "transporte_publico": "Transporte público reconhecido",
            "carro_particular": "Carro particular identificado",
        },
        "razoes": {
            "CARRO_PARTICULAR": "A foto parece um carro particular, não um deslocamento sustentável.",
            "CENA_INVALIDA": "Não reconhecemos bicicleta nem transporte público nesta foto.",
            "INCONCLUSIVO": "Não foi possível confirmar o deslocamento. Tente outra foto, mais nítida e de perto.",
            "APROVADO_BICICLETA": "A foto mostra uma bicicleta em uso.",
            "APROVADO_TRANSPORTE_PUBLICO": "A foto mostra transporte público.",
        },
        "imagenet_validas": IMAGENET_BICICLETA | IMAGENET_TRANSPORTE,
        "imagenet_negativas": IMAGENET_CARRO,
        "imagenet_classe": {
            **{n: "bicicleta" for n in IMAGENET_BICICLETA},
            **{n: "transporte_publico" for n in IMAGENET_TRANSPORTE},
            **{n: "carro_particular" for n in IMAGENET_CARRO},
        },
    },
    "reciclagem": {
        "tags": ["Reciclagem"],
        "modo": "visual",
        "classes": ["reciclavel", "lixo_misturado"],
        "validas": ["reciclavel"],
        "negativa": "lixo_misturado",
        "motivo_negativa": "LIXO_MISTURADO",
        "pastas": {
            "reciclavel": "reciclagem/reciclavel",
            "lixo_misturado": "reciclagem/lixo_misturado",
        },
        "hint": "Foto do reciclável já separado (plástico, papel, metal) ou das lixeiras seletivas.",
        "deteccoes": {
            "reciclavel": "Material reciclável separado reconhecido",
            "lixo_misturado": "Lixo misturado identificado",
        },
        "razoes": {
            "LIXO_MISTURADO": "A foto não mostra o reciclável separado.",
            "CENA_INVALIDA": "Não reconhecemos coleta seletiva nesta foto. Mostre as embalagens ou as lixeiras.",
            "INCONCLUSIVO": "Não deu para confirmar a reciclagem. Aproxime a câmera dos materiais separados.",
            "APROVADO_RECICLAVEL": "A foto mostra material reciclável separado.",
        },
        "imagenet_validas": IMAGENET_RECICLAVEL,
        "imagenet_negativas": IMAGENET_LIXO_MISTURADO,
        "imagenet_classe": {
            **{n: "reciclavel" for n in IMAGENET_RECICLAVEL},
            **{n: "lixo_misturado" for n in IMAGENET_LIXO_MISTURADO},
        },
    },
    "compostagem": {
        "tags": ["Resíduos"],
        "modo": "visual",
        "classes": ["composteira", "outro"],
        "validas": ["composteira"],
        "negativa": "outro",
        "motivo_negativa": "CENA_INVALIDA",
        "pastas": {
            "composteira": "compostagem/composteira",
            "outro": "compostagem/outro",
        },
        "hint": "Foto da composteira ou do orgânico sendo compostado.",
        "deteccoes": {
            "composteira": "Compostagem reconhecida",
            "outro": "Cena sem compostagem",
        },
        "razoes": {
            "CENA_INVALIDA": "Não reconhecemos uma composteira ou orgânico em compostagem.",
            "INCONCLUSIVO": "Não deu para confirmar a compostagem. Mostre a composteira de perto.",
            "APROVADO_COMPOSTEIRA": "A foto mostra compostagem.",
        },
        "imagenet_validas": IMAGENET_COMPOSTAGEM,
        "imagenet_negativas": set(),
        "imagenet_classe": {n: "composteira" for n in IMAGENET_COMPOSTAGEM},
    },
    "coleta": {
        "tags": ["Resíduos"],
        "modo": "visual",
        "classes": ["coleta", "outro"],
        "validas": ["coleta"],
        "negativa": "outro",
        "motivo_negativa": "CENA_INVALIDA",
        "pastas": {
            "coleta": "coleta/residuos_recolhidos",
            "outro": "coleta/outro",
        },
        "hint": "Foto do lixo recolhido em espaço público, condomínio ou mutirão.",
        "deteccoes": {
            "coleta": "Coleta de resíduos reconhecida",
            "outro": "Cena sem coleta",
        },
        "razoes": {
            "CENA_INVALIDA": "Não reconhecemos uma ação de coleta nesta foto.",
            "INCONCLUSIVO": "Não deu para confirmar a coleta. Mostre os resíduos recolhidos no local.",
            "APROVADO_COLETA": "A foto mostra coleta de resíduos.",
        },
        "imagenet_validas": IMAGENET_COLETA,
        "imagenet_negativas": set(),
        "imagenet_classe": {n: "coleta" for n in IMAGENET_COLETA},
    },
    "agua": {
        "tags": ["Água"],
        "modo": "cena",
        "classes": ["banho", "outro"],
        "validas": ["banho"],
        "negativa": "outro",
        "motivo_negativa": "CENA_INVALIDA",
        "pastas": {
            "banho": "agua/banho",
            "outro": "agua/outro",
        },
        "hint": "Foto do chuveiro, do timer do banho ou da torneira fechada. A foto não prova a duração.",
        "deteccoes": {
            "banho": "Ambiente de banho ou uso de água reconhecido",
            "outro": "Cena sem relação com água",
        },
        "razoes": {
            "CENA_INVALIDA": "Não reconhecemos um ambiente de banho ou economia de água nesta foto.",
            "INCONCLUSIVO": "Não deu para confirmar. Mostre o chuveiro, o timer ou a torneira.",
            "APROVADO_BANHO": "A foto mostra um ambiente de uso de água. A duração do banho não dá para medir só pela imagem.",
        },
        "imagenet_validas": IMAGENET_AGUA,
        "imagenet_negativas": set(),
        "imagenet_classe": {n: "banho" for n in IMAGENET_AGUA},
    },
    "consumo": {
        "tags": ["Consumo"],
        "modo": "visual",
        "classes": ["reutilizavel", "descartavel"],
        "validas": ["reutilizavel"],
        "negativa": "descartavel",
        "motivo_negativa": "DESCARTAVEL",
        "pastas": {
            "reutilizavel": "consumo/reutilizavel",
            "descartavel": "consumo/descartavel",
        },
        "hint": "Foto do copo, garrafa ou ecobag reutilizável em uso.",
        "deteccoes": {
            "reutilizavel": "Item reutilizável reconhecido",
            "descartavel": "Descartável identificado",
        },
        "razoes": {
            "DESCARTAVEL": "A foto parece um descartável, não um item reutilizável.",
            "CENA_INVALIDA": "Não reconhecemos copo, garrafa ou sacola reutilizável nesta foto.",
            "INCONCLUSIVO": "Não deu para confirmar o item reutilizável. Aproxime a câmera.",
            "APROVADO_REUTILIZAVEL": "A foto mostra um item reutilizável.",
        },
        "imagenet_validas": IMAGENET_REUTILIZAVEL,
        "imagenet_negativas": IMAGENET_DESCARTAVEL,
        "imagenet_classe": {
            **{n: "reutilizavel" for n in IMAGENET_REUTILIZAVEL},
            **{n: "descartavel" for n in IMAGENET_DESCARTAVEL},
        },
    },
    "energia": {
        "tags": ["Energia"],
        "modo": "cena",
        "classes": ["aparelho", "outro"],
        "validas": ["aparelho"],
        "negativa": "outro",
        "motivo_negativa": "CENA_INVALIDA",
        "pastas": {
            "aparelho": "energia/aparelho",
            "outro": "energia/outro",
        },
        "hint": "Foto do aparelho fora da tomada ou do cômodo com luz natural e lâmpadas apagadas.",
        "deteccoes": {
            "aparelho": "Aparelho ou iluminação reconhecidos",
            "outro": "Cena sem relação com energia",
        },
        "razoes": {
            "CENA_INVALIDA": "Não reconhecemos aparelhos, tomadas ou iluminação nesta foto.",
            "INCONCLUSIVO": "Não deu para confirmar a economia de energia. Mostre a tomada ou as luzes apagadas.",
            "APROVADO_APARELHO": "A foto mostra um ambiente de uso de energia. O standby em si não dá para medir só pela imagem.",
        },
        "imagenet_validas": IMAGENET_ENERGIA,
        "imagenet_negativas": set(),
        "imagenet_classe": {n: "aparelho" for n in IMAGENET_ENERGIA},
    },
    "alimentacao": {
        "tags": ["Alimentação"],
        "modo": "visual",
        "classes": ["prato_vegetal", "prato_com_carne"],
        "validas": ["prato_vegetal"],
        "negativa": "prato_com_carne",
        "motivo_negativa": "PRATO_COM_CARNE",
        "pastas": {
            "prato_vegetal": "alimentacao/prato_vegetal",
            "prato_com_carne": "alimentacao/prato_com_carne",
        },
        "hint": "Foto do prato vegetariano, de cima e bem iluminado.",
        "deteccoes": {
            "prato_vegetal": "Refeição vegetariana reconhecida",
            "prato_com_carne": "Prato com carne identificado",
        },
        "razoes": {
            "PRATO_COM_CARNE": "A foto parece um prato com carne.",
            "CENA_INVALIDA": "Não reconhecemos uma refeição vegetariana nesta foto.",
            "INCONCLUSIVO": "Não deu para confirmar o prato. Fotografe de cima, com boa luz.",
            "APROVADO_PRATO_VEGETAL": "A foto mostra uma refeição que parece vegetariana.",
        },
        "imagenet_validas": IMAGENET_VEGETAL,
        "imagenet_negativas": IMAGENET_CARNE,
        "imagenet_classe": {
            **{n: "prato_vegetal" for n in IMAGENET_VEGETAL},
            **{n: "prato_com_carne" for n in IMAGENET_CARNE},
        },
    },
    "natureza": {
        "tags": ["Natureza"],
        "modo": "visual",
        "classes": ["muda", "outro"],
        "validas": ["muda"],
        "negativa": "outro",
        "motivo_negativa": "CENA_INVALIDA",
        "pastas": {
            "muda": "natureza/muda",
            "outro": "natureza/outro",
        },
        "hint": "Foto da muda, da horta ou da árvore recém-plantada.",
        "deteccoes": {
            "muda": "Muda, horta ou plantio reconhecido",
            "outro": "Cena sem plantio",
        },
        "razoes": {
            "CENA_INVALIDA": "Não reconhecemos uma muda, horta ou árvore plantada nesta foto.",
            "INCONCLUSIVO": "Não deu para confirmar o plantio. Mostre a muda de perto.",
            "APROVADO_MUDA": "A foto mostra uma planta, muda ou horta.",
        },
        "imagenet_validas": IMAGENET_NATUREZA,
        "imagenet_negativas": set(),
        "imagenet_classe": {n: "muda" for n in IMAGENET_NATUREZA},
    },
}

TAG_PARA_FAMILIA = {
    "mobilidade": "mobilidade",
    "reciclagem": "reciclagem",
    "agua": "agua",
    "consumo": "consumo",
    "energia": "energia",
    "alimentacao": "alimentacao",
    "natureza": "natureza",
}


def resolver_familia(mission: dict | None, padrao: str = "mobilidade") -> str:
    """Escolhe a família visual a partir da missão enviada pelo Java."""
    if not mission:
        return padrao
    nome = _norm(str(mission.get("name") or ""))
    tag = _norm(str(mission.get("tag") or ""))

    if "compost" in nome:
        return "compostagem"
    if tag == "residuos" or any(p in nome for p in ("limpeza", "mutirao", "recolha")):
        return "coleta"

    if tag in TAG_PARA_FAMILIA:
        return TAG_PARA_FAMILIA[tag]

    for familia, cfg in FAMILIAS.items():
        if any(_norm(t) == tag for t in cfg["tags"]):
            return familia
    return padrao


def hint(familia: str) -> str:
    return FAMILIAS[familia]["hint"]


def pasta_dataset(familia: str, classe: str) -> str:
    return FAMILIAS[familia]["pastas"][classe]
