---
language:
  - multilingual
  - af
  - am
  - sq
  - ar
  - hy
  - az
  - eu
  - be
  - bs
  - bg
  - my
  - ca
  - zh
  - hr
  - cs
  - da
  - nl
  - en
  - eo
  - et
  - fi
  - fr
  - ka
  - de
  - el
  - hi
  - hu
  - is
  - id
  - it
  - ja
  - kk
  - km
  - ko
  - ku
  - la
  - lt
  - lb
  - mk
  - ms
  - mt
  - mi
  - nb
  - or
  - fa
  - pl
  - pt
  - ro
  - ru
  - sa
  - sr
  - sk
  - sl
  - es
  - sw
  - sv
  - tl
  - ta
  - tt
  - th
  - tr
  - tk
  - uk
  - ur
  - ug
  - uz
  - vi
  - cy
  - sd
license: cc-by-4.0
library_name: optimum
tags:
  - onnx
  - t5
  - g2p
  - grapheme-to-phoneme
  - phonetics
  - phonology
  - ipa
  - byt5
  - text-generation
  - charsiu
  - multilingual
pipeline_tag: text-generation
base_model: charsiu/g2p_multilingual_byT5_tiny_16_layers_100
model-index:
  - name: g2p_multilingual_byT5_tiny_onnx
    results:
      - task:
          type: text-generation
          name: Grapheme-to-Phoneme
        dataset:
          type: custom
          name: CharsiuG2P Test Set (100 languages, 500 words each)
          args:
            link: https://github.com/lingjzhu/CharsiuG2P/tree/main/data/test
          config: multilingual
          split: test
        metrics:
          - type: phoneme_error_rate
            value: 0.0812
            name: PER (ONNX FP32, greedy)
          - type: phoneme_error_rate
            value: 0.0817
            name: PER (ONNX INT8, greedy)
          - type: word_error_rate
            value: 0.2529
            name: WER (ONNX FP32, greedy)
          - type: word_error_rate
            value: 0.2537
            name: WER (ONNX INT8, greedy)
---

# Multilingual G2P ByT5 Tiny — ONNX

ONNX export of [charsiu/g2p_multilingual_byT5_tiny_16_layers_100](https://huggingface.co/charsiu/g2p_multilingual_byT5_tiny_16_layers_100). Converts written words to IPA transcriptions across ~100 languages.

| | |
| --- | --- |
| **Architecture** | ByT5-tiny (T5ForConditionalGeneration), 20.6M params |
| **PER / WER** | 8.1% / 25.3% (100 langs, 500 words each, greedy) |
| **ONNX FP32 size** | 106 MB (3 graphs: encoder + decoder + decoder_with_past) |
| **ONNX INT8 size** | 27 MB (74% reduction, +0.0005 PER) |
| **Best latency** | 5.30 ms/word (INT8, threads=8) — 1.5x faster than PyTorch CPU |
| **License** | [CC BY 4.0](http://creativecommons.org/licenses/by/4.0/) |

## Quick Start

```python
import onnxruntime as ort
from optimum.onnxruntime import ORTModelForSeq2SeqLM
from transformers import AutoTokenizer

so = ort.SessionOptions()
so.intra_op_num_threads = 8   # ~cores/4
so.inter_op_num_threads = 1

model = ORTModelForSeq2SeqLM.from_pretrained(
    "klebster/g2p_multilingual_byT5_tiny_onnx",
    provider="CPUExecutionProvider",
    session_options=so,
)
tokenizer = AutoTokenizer.from_pretrained("klebster/g2p_multilingual_byT5_tiny_onnx")

inputs = tokenizer("<eng-us>: hello", padding=True, add_special_tokens=False, return_tensors="pt")
preds = model.generate(**inputs, num_beams=1, max_length=50)
print(tokenizer.decode(preds[0], skip_special_tokens=True))
# Output: ˈhɛɫoʊ
```

Input format: `<language_code>: word` (e.g. `<fra>: bonjour`, `<ger>: Straße`). See [CharsiuG2P](https://github.com/lingjzhu/CharsiuG2P/tree/main/dicts) for all ~100 language codes.

## Benchmark Summary

Tested on 10 languages, 300 words, greedy decoding. Hardware: Intel i9-13900KS, 128 GB DDR5.

| Configuration | ms/word | vs PyTorch CPU |
| --- | ---: | ---: |
| **ONNX INT8 + threads=8** | **5.30** | **1.48x faster** |
| ONNX FP32 + threads=8 | 7.47 | 1.05x faster |
| PyTorch CPU (baseline) | 7.83 | 1.00x |
| ONNX default settings | 25.09 | 0.31x |

**Key findings:** Thread tuning is critical (default ONNX is 3x slower than PyTorch). INT8 quantization + threads=8 is the best config. GPU provides no benefit for this model size. Beam search provides negligible improvement (PER 0.0812 → 0.0808) at 3–8x latency cost.

**Correctness:** ONNX FP32 output is bit-identical to PyTorch. INT8 degrades PER by only 0.0005.

## Evaluation

100 languages, 500 words each (50,000 total), [CharsiuG2P test set](https://github.com/lingjzhu/CharsiuG2P/tree/main/data/test):

| Metric | PyTorch CPU | ONNX FP32 | ONNX INT8 |
| --- | ---: | ---: | ---: |
| **PER** | 0.0812 | 0.0812 | 0.0817 |
| **WER** | 0.2529 | 0.2529 | 0.2537 |

<details>
<summary>PER by language (click to expand)</summary>

| Language | PyTorch CPU | ONNX FP32 | ONNX INT8 |
| --- | ---: | ---: | ---: |
| ady | 0.0521 | 0.0521 | 0.0518 |
| afr | 0.0251 | 0.0251 | 0.0243 |
| amh | 0.3347 | 0.3347 | 0.3356 |
| ang | 0.0559 | 0.0559 | 0.0559 |
| ara | 0.6710 | 0.6710 | 0.6716 |
| arg | 0.0181 | 0.0181 | 0.0183 |
| arm-e | 0.0215 | 0.0215 | 0.0220 |
| arm-w | 0.0075 | 0.0075 | 0.0070 |
| aze | 0.0008 | 0.0008 | 0.0008 |
| bak | 0.0124 | 0.0124 | 0.0138 |
| bel | 0.0048 | 0.0048 | 0.0040 |
| bos | 0.0184 | 0.0184 | 0.0171 |
| bul | 0.0158 | 0.0158 | 0.0161 |
| bur | 0.1718 | 0.1718 | 0.1712 |
| cat | 0.1566 | 0.1566 | 0.1598 |
| cze | 0.0077 | 0.0077 | 0.0079 |
| dan | 0.2118 | 0.2118 | 0.2157 |
| dut | 0.0347 | 0.0347 | 0.0345 |
| egy | 0.3752 | 0.3752 | 0.3781 |
| eng-uk | 0.1542 | 0.1542 | 0.1534 |
| eng-us | 0.2182 | 0.2182 | 0.2193 |
| enm | 0.1988 | 0.1988 | 0.1979 |
| epo | 0.0002 | 0.0002 | 0.0000 |
| est | 0.0072 | 0.0072 | 0.0070 |
| eus | 0.0056 | 0.0056 | 0.0058 |
| fas | 0.1503 | 0.1503 | 0.1507 |
| fin | 0.0011 | 0.0011 | 0.0011 |
| fra | 0.0075 | 0.0075 | 0.0075 |
| fra-qu | 0.0041 | 0.0041 | 0.0041 |
| geo | 0.0254 | 0.0254 | 0.0258 |
| ger | 0.0454 | 0.0454 | 0.0442 |
| gle | 0.1775 | 0.1775 | 0.1802 |
| glg | 0.0690 | 0.0690 | 0.0687 |
| grc | 0.2944 | 0.2944 | 0.2942 |
| gre | 0.0232 | 0.0232 | 0.0229 |
| hbs-cyrl | 0.0967 | 0.0967 | 0.0972 |
| hbs-latn | 0.0900 | 0.0900 | 0.0913 |
| hin | 0.0421 | 0.0421 | 0.0417 |
| hun | 0.0227 | 0.0227 | 0.0224 |
| ice | 0.0281 | 0.0281 | 0.0273 |
| ido | 0.0484 | 0.0484 | 0.0479 |
| ina | 0.0529 | 0.0529 | 0.0532 |
| ind | 0.0219 | 0.0219 | 0.0231 |
| isl | 0.0341 | 0.0341 | 0.0341 |
| ita | 0.0273 | 0.0273 | 0.0282 |
| jpn | 0.1050 | 0.1050 | 0.1062 |
| kaz | 0.0055 | 0.0055 | 0.0061 |
| khm | 0.2787 | 0.2787 | 0.2850 |
| kor | 0.0776 | 0.0776 | 0.0776 |
| kur | 0.0093 | 0.0093 | 0.0099 |
| lat-clas | 0.0094 | 0.0094 | 0.0092 |
| lat-eccl | 0.0070 | 0.0070 | 0.0074 |
| lit | 0.0316 | 0.0316 | 0.0316 |
| ltz | 0.1288 | 0.1288 | 0.1307 |
| mac | 0.0113 | 0.0113 | 0.0110 |
| mlt | 0.0226 | 0.0226 | 0.0228 |
| mri | 0.2214 | 0.2214 | 0.2198 |
| msa | 0.0008 | 0.0008 | 0.0008 |
| nan | 0.1069 | 0.1069 | 0.1082 |
| nob | 0.0874 | 0.0874 | 0.0888 |
| ori | 0.0028 | 0.0028 | 0.0026 |
| pap | 0.0019 | 0.0019 | 0.0019 |
| pol | 0.0057 | 0.0057 | 0.0057 |
| por-bz | 0.0866 | 0.0866 | 0.0875 |
| por-po | 0.0872 | 0.0872 | 0.0877 |
| ron | 0.0072 | 0.0072 | 0.0072 |
| rus | 0.0211 | 0.0211 | 0.0202 |
| san | 0.0762 | 0.0762 | 0.0755 |
| slk | 0.0289 | 0.0289 | 0.0283 |
| slo | 0.0415 | 0.0415 | 0.0418 |
| slv | 0.1829 | 0.1829 | 0.1838 |
| sme | 0.0207 | 0.0207 | 0.0214 |
| snd | 0.2636 | 0.2636 | 0.2685 |
| spa | 0.0017 | 0.0017 | 0.0019 |
| spa-latin | 0.0005 | 0.0005 | 0.0007 |
| spa-me | 0.0043 | 0.0043 | 0.0047 |
| sqi | 0.0148 | 0.0148 | 0.0192 |
| srp | 0.0620 | 0.0620 | 0.0638 |
| swa | 0.0025 | 0.0025 | 0.0025 |
| swe | 0.0084 | 0.0084 | 0.0079 |
| syc | 0.3765 | 0.3765 | 0.3776 |
| tam | 0.0099 | 0.0099 | 0.0112 |
| tat | 0.0026 | 0.0026 | 0.0033 |
| tgl | 0.1066 | 0.1066 | 0.1066 |
| tha | 0.0921 | 0.0921 | 0.0929 |
| tts | 0.0196 | 0.0196 | 0.0199 |
| tuk | 0.0119 | 0.0119 | 0.0123 |
| tur | 0.0006 | 0.0006 | 0.0004 |
| uig | 0.0335 | 0.0335 | 0.0340 |
| ukr | 0.0338 | 0.0338 | 0.0332 |
| urd | 0.2352 | 0.2352 | 0.2374 |
| uzb | 0.0043 | 0.0043 | 0.0045 |
| vie-c | 0.0144 | 0.0144 | 0.0144 |
| vie-n | 0.0075 | 0.0075 | 0.0076 |
| vie-s | 0.0112 | 0.0112 | 0.0107 |
| wel-nw | 0.0967 | 0.0967 | 0.0990 |
| wel-sw | 0.1291 | 0.1291 | 0.1321 |
| yue | 0.2238 | 0.2238 | 0.2242 |
| zho-s | 0.2986 | 0.2986 | 0.2992 |
| zho-t | 0.3459 | 0.3459 | 0.3478 |
| **AVERAGE** | **0.0812** | **0.0812** | **0.0817** |

</details>

<details>
<summary>WER by language (click to expand)</summary>

| Language | PyTorch CPU | ONNX FP32 | ONNX INT8 |
| --- | ---: | ---: | ---: |
| ady | 0.2700 | 0.2700 | 0.2680 |
| afr | 0.1240 | 0.1240 | 0.1220 |
| amh | 0.9940 | 0.9940 | 0.9940 |
| ang | 0.3260 | 0.3260 | 0.3200 |
| ara | 1.0000 | 1.0000 | 1.0000 |
| arg | 0.0960 | 0.0960 | 0.1000 |
| arm-e | 0.1120 | 0.1120 | 0.1140 |
| arm-w | 0.0520 | 0.0520 | 0.0480 |
| aze | 0.0060 | 0.0060 | 0.0060 |
| bak | 0.0920 | 0.0920 | 0.1020 |
| bel | 0.0540 | 0.0540 | 0.0440 |
| bos | 0.0620 | 0.0620 | 0.0600 |
| bul | 0.1080 | 0.1080 | 0.1140 |
| bur | 0.5080 | 0.5080 | 0.5000 |
| cat | 0.6000 | 0.6000 | 0.5980 |
| cze | 0.0420 | 0.0420 | 0.0440 |
| dan | 0.6740 | 0.6740 | 0.6700 |
| dut | 0.1940 | 0.1940 | 0.1920 |
| egy | 0.5820 | 0.5820 | 0.5920 |
| eng-uk | 0.3180 | 0.3180 | 0.3260 |
| eng-us | 0.4840 | 0.4840 | 0.4860 |
| enm | 0.7160 | 0.7160 | 0.7120 |
| epo | 0.0020 | 0.0020 | 0.0000 |
| est | 0.0640 | 0.0640 | 0.0620 |
| eus | 0.0220 | 0.0220 | 0.0220 |
| fas | 0.5640 | 0.5640 | 0.5660 |
| fin | 0.0080 | 0.0080 | 0.0080 |
| fra | 0.0200 | 0.0200 | 0.0200 |
| fra-qu | 0.0200 | 0.0200 | 0.0200 |
| geo | 0.2360 | 0.2360 | 0.2400 |
| ger | 0.1720 | 0.1720 | 0.1700 |
| gle | 0.6660 | 0.6660 | 0.6720 |
| glg | 0.3580 | 0.3580 | 0.3560 |
| grc | 0.7340 | 0.7340 | 0.7360 |
| gre | 0.1620 | 0.1620 | 0.1600 |
| hbs-cyrl | 0.5180 | 0.5180 | 0.5200 |
| hbs-latn | 0.4620 | 0.4620 | 0.4680 |
| hin | 0.2140 | 0.2140 | 0.2060 |
| hun | 0.1140 | 0.1140 | 0.1140 |
| ice | 0.1700 | 0.1700 | 0.1680 |
| ido | 0.2300 | 0.2300 | 0.2260 |
| ina | 0.2040 | 0.2040 | 0.2020 |
| ind | 0.1020 | 0.1020 | 0.1040 |
| isl | 0.2260 | 0.2260 | 0.2260 |
| ita | 0.2000 | 0.2000 | 0.2040 |
| jpn | 0.2380 | 0.2380 | 0.2400 |
| kaz | 0.0420 | 0.0420 | 0.0420 |
| khm | 0.7320 | 0.7320 | 0.7220 |
| kor | 0.0940 | 0.0940 | 0.0940 |
| kur | 0.0520 | 0.0520 | 0.0480 |
| lat-clas | 0.0580 | 0.0580 | 0.0560 |
| lat-eccl | 0.0380 | 0.0380 | 0.0400 |
| lit | 0.1860 | 0.1860 | 0.1900 |
| ltz | 0.3800 | 0.3800 | 0.3840 |
| mac | 0.0880 | 0.0880 | 0.0860 |
| mlt | 0.0880 | 0.0880 | 0.0900 |
| mri | 0.6740 | 0.6740 | 0.6740 |
| msa | 0.0040 | 0.0040 | 0.0040 |
| nan | 0.4100 | 0.4100 | 0.4140 |
| nob | 0.3400 | 0.3400 | 0.3340 |
| ori | 0.0220 | 0.0220 | 0.0200 |
| pap | 0.0120 | 0.0120 | 0.0120 |
| pol | 0.0240 | 0.0240 | 0.0240 |
| por-bz | 0.4320 | 0.4320 | 0.4320 |
| por-po | 0.3920 | 0.3920 | 0.3980 |
| ron | 0.0500 | 0.0500 | 0.0500 |
| rus | 0.1260 | 0.1260 | 0.1220 |
| san | 0.4400 | 0.4400 | 0.4360 |
| slk | 0.1140 | 0.1140 | 0.1160 |
| slo | 0.2180 | 0.2180 | 0.2200 |
| slv | 0.5860 | 0.5860 | 0.5880 |
| sme | 0.1360 | 0.1360 | 0.1380 |
| snd | 0.6000 | 0.6000 | 0.6120 |
| spa | 0.0160 | 0.0160 | 0.0180 |
| spa-latin | 0.0040 | 0.0040 | 0.0060 |
| spa-me | 0.0380 | 0.0380 | 0.0400 |
| sqi | 0.0740 | 0.0740 | 0.0900 |
| srp | 0.4480 | 0.4480 | 0.4440 |
| swa | 0.0180 | 0.0180 | 0.0180 |
| swe | 0.0600 | 0.0600 | 0.0560 |
| syc | 0.8980 | 0.8980 | 0.9080 |
| tam | 0.0500 | 0.0500 | 0.0560 |
| tat | 0.0140 | 0.0140 | 0.0180 |
| tgl | 0.4460 | 0.4460 | 0.4500 |
| tha | 0.2140 | 0.2140 | 0.2160 |
| tts | 0.0780 | 0.0780 | 0.0760 |
| tuk | 0.1100 | 0.1100 | 0.1140 |
| tur | 0.0040 | 0.0040 | 0.0020 |
| uig | 0.1480 | 0.1480 | 0.1540 |
| ukr | 0.2240 | 0.2240 | 0.2180 |
| urd | 0.5860 | 0.5860 | 0.5900 |
| uzb | 0.0220 | 0.0220 | 0.0240 |
| vie-c | 0.0400 | 0.0400 | 0.0400 |
| vie-n | 0.0300 | 0.0300 | 0.0300 |
| vie-s | 0.0400 | 0.0400 | 0.0380 |
| wel-nw | 0.3660 | 0.3660 | 0.3720 |
| wel-sw | 0.4120 | 0.4120 | 0.4260 |
| yue | 0.4040 | 0.4040 | 0.4080 |
| zho-s | 0.5320 | 0.5320 | 0.5300 |
| zho-t | 0.5600 | 0.5600 | 0.5620 |
| **AVERAGE** | **0.2529** | **0.2529** | **0.2537** |

</details>

## Links

- **Paper:** [Zhu et al. (2022), Interspeech](https://arxiv.org/abs/2204.03067)
- **Original repo:** [CharsiuG2P](https://github.com/lingjzhu/CharsiuG2P)
- **Base model:** [charsiu/g2p_multilingual_byT5_tiny_16_layers_100](https://huggingface.co/charsiu/g2p_multilingual_byT5_tiny_16_layers_100)
- **ONNX export by:** [klebster](https://huggingface.co/klebster) for [phone-similarity](https://github.com/klebster/phone-similarity)

## Citation

If you use this ONNX model, please cite both the original paper and the ONNX export:

```bibtex
@misc{zhu2022byt5modelmassivelymultilingual,
      title={ByT5 model for massively multilingual grapheme-to-phoneme conversion},
      author={Jian Zhu and Cong Zhang and David Jurgens},
      year={2022},
      eprint={2204.03067},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2204.03067},
}
```

```bibtex
@misc{noel2025g2pmultilingualbyT5tinyonnx,
      title={Multilingual G2P ByT5 Tiny — ONNX export},
      author={Kleber Noel},
      year={2026},
      month={apr},
      note={Published 2026-04-07},
      url={https://huggingface.co/klebster/g2p_multilingual_byT5_tiny_onnx},
}
```
