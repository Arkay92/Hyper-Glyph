# Hyper Glyph Benchmark

| Representation | Bytes | Ratio vs FP32 | MSE | MAE | Max abs error |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP32 | 24576 | 1.00x | 0 | 0 | 0 |
| FP16 estimate | 12288 | 2.00x | - | - | - |
| INT8 estimate | 6144 | 4.00x | - | - | - |
| Hyper Glyph | 22032 | 1.12x | 0.00266153 | 0.0405458 | 0.197096 |

## Tensor Summary

- Tensors compressed: 2
- Tensors skipped: 0
