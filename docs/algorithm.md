# Algorithm overview

Hyper Glyph works in four stages:

1. Block splitting: tensors are flattened and split into fixed-size blocks.
2. Hyperdimensional encoding: each block is encoded into a deterministic hypervector using role vectors and symbolic metadata.
3. Prototype learning: blocks are clustered into reusable prototypes.
4. Sparse residual repair: small residual corrections are stored for reconstruction fidelity.

The reconstruction is:

$$W \approx \text{Decode}(\text{prototype}) + \text{sparse residual}$$

The implementation is intentionally simple for v0.1 and leaves room for learned decoders and richer codecs later.
