from baixar_dataset import BUSCAS, BUSCAS_INVALIDAS, _extensao
from familias import FAMILIAS


def test_toda_classe_tem_busca():
    for familia, cfg in FAMILIAS.items():
        assert familia in BUSCAS, familia
        for classe in cfg["classes"]:
            queries = BUSCAS[familia][classe]
            assert queries, (familia, classe)


def test_invalidas_tem_buscas():
    assert len(BUSCAS_INVALIDAS) >= 3


def test_extensao_pela_url_e_content_type():
    assert _extensao("https://exemplo.com/foto.JPEG", None) == ".jpeg"
    assert _extensao("https://exemplo.com/sem-ext", "image/png") == ".png"
    assert _extensao("https://exemplo.com/x.gif", "image/gif") is None
