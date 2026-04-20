"""
Grapheme-to-phoneme (G2P) conversion backends.

Currently provides the Charsiu G2P backend
(:mod:`phone_similarity.g2p.charsiu`) which supports 100+ languages via
a ByT5-based ONNX model and per-language pronunciation dictionaries.
"""

# TODO: Add espeak-ng
