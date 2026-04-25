from phone_similarity.g2p.charsiu.generator import CharsiuGraphemeToPhonemeGenerator


def test_generate_batched_empty_input():
    g2p = CharsiuGraphemeToPhonemeGenerator("eng-us")
    phones, probs = g2p.generate_batched([])
    assert phones == []
    assert probs == []


def test_generate_batched_invalid_batch_size():
    g2p = CharsiuGraphemeToPhonemeGenerator("eng-us")
    try:
        g2p.generate_batched(["hello"], batch_size=0)
    except ValueError:
        return
    raise AssertionError("Expected ValueError for batch_size=0")


def test_generate_batched_reuses_generate(monkeypatch):
    g2p = CharsiuGraphemeToPhonemeGenerator("eng-us")

    calls = []

    def fake_generate(words: tuple[str], **kwargs):
        calls.append((words, kwargs))
        phones = [f"/{w}/" for w in words]
        probs = [1.0 for _ in words]
        return phones, probs

    monkeypatch.setattr(g2p, "generate", fake_generate)

    phones, probs = g2p.generate_batched(
        ["alpha", "beta", "gamma", "delta", "epsilon"],
        batch_size=2,
        num_beams=2,
    )

    assert calls == [
        (("alpha", "beta"), {"num_beams": 2}),
        (("gamma", "delta"), {"num_beams": 2}),
        (("epsilon",), {"num_beams": 2}),
    ]
    assert phones == ["/alpha/", "/beta/", "/gamma/", "/delta/", "/epsilon/"]
    assert probs == [1.0, 1.0, 1.0, 1.0, 1.0]
