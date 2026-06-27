# Algorithm overview

Hyper Glyph works in four stages:

1. Block splitting: tensors are flattened and split into fixed-size blocks.
2. Hyperdimensional encoding: each block is encoded into a deterministic hypervector using role vectors and symbolic metadata.
3. Prototype learning: blocks are clustered into reusable prototypes.
4. Sparse residual repair: small residual corrections are stored for reconstruction fidelity.

The reconstruction is:

$$W \approx \text{Decode}(\text{prototype}) + \text{sparse residual}$$

In v0.2, prototype scales can be calculated per block, per tensor, or per
channel. Sparse residual values can be stored as float32 or quantized to int8
with a residual scale for decoding.

In v0.3, compact mode stores large payloads as binary streams instead of JSON.
The default compact tensor path uses packed int4 affine values with calibrated
scale metadata because it gives quantization-class archive ratios on GPT-style
weight matrices. Prototype codebook and residual-budget helpers are available
for symbolic codec experiments.

In v0.4, compact mode defaults to a learned global codebook. Blocks are assigned
to a shared prototype bank, assignments are packed as uint4 when possible, and
RLE is selected when it is smaller. This greatly reduces assignment bytes, but
the packed-int4 path remains better for reconstruction quality on random
GPT-style synthetic weights.

The implementation is intentionally simple and leaves room for learned decoders
and richer codecs later.
