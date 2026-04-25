import pytest

from phone_similarity.g2p.charsiu.generator import (
    CharsiuGraphemeToPhonemeGenerator,
    GraphemeToPhonemeResourceType,
    ResearchUseOnlyError,
)


def test_generate_requires_explicit_research_opt_in():
    g2p = CharsiuGraphemeToPhonemeGenerator("eng-us")
    with pytest.raises(ResearchUseOnlyError):
        g2p.generate(("hello",))


def test_g2p_resource_path_requires_explicit_research_opt_in():
    g2p = CharsiuGraphemeToPhonemeGenerator("eng-us")
    with pytest.raises(ResearchUseOnlyError):
        g2p.get_phones_for_word(
            "hello",
            limit_resource=GraphemeToPhonemeResourceType.G2P_GENERATOR,
        )


def test_env_var_enables_research_opt_in(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PHONE_SIM_ALLOW_RESEARCH_G2P", "1")
    g2p = CharsiuGraphemeToPhonemeGenerator("eng-us")
    assert g2p._research_g2p_enabled() is True
    monkeypatch.setenv("PHONE_SIM_ALLOW_RESEARCH_G2P", "0")
    assert g2p._research_g2p_enabled() is False
