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

The implementation is intentionally simple and leaves room for learned decoders
and richer codecs later.
