import pytest

import inferencia


def test_aguardar_pronto_estoura_enquanto_nao_carregou():
    if inferencia.preparo_terminou():
        pytest.skip("extrator já estava carregado neste processo")
    with pytest.raises(inferencia.ModelosNaoProntos, match="ainda carregando"):
        inferencia.aguardar_pronto(timeout=0.05)
    assert inferencia.modelos_prontos() is False
