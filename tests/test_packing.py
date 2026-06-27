import numpy as np

from hyperglyph.packing import (
    choose_min_uint_dtype,
    delta_decode,
    delta_encode,
    pack_int4,
    pack_uint4,
    unpack_int4,
    unpack_uint4,
    varint_decode,
    varint_encode,
)


def test_uint4_roundtrip_even_length() -> None:
    values = np.array([0, 1, 2, 15], dtype=np.uint8)
    assert np.array_equal(unpack_uint4(pack_uint4(values), len(values)), values)


def test_uint4_roundtrip_odd_length() -> None:
    values = np.array([0, 7, 15], dtype=np.uint8)
    assert np.array_equal(unpack_uint4(pack_uint4(values), len(values)), values)


def test_int4_signed_roundtrip() -> None:
    values = np.array([-8, -1, 0, 7], dtype=np.int8)
    assert np.array_equal(unpack_int4(pack_int4(values), len(values)), values)


def test_varint_roundtrip() -> None:
    values = [0, 1, 127, 128, 16384]
    assert varint_decode(varint_encode(values)) == values


def test_delta_roundtrip() -> None:
    values = [3, 9, 10, 100]
    assert delta_decode(delta_encode(values)) == values


def test_choose_min_uint_dtype() -> None:
    assert choose_min_uint_dtype(15) == "uint4"
    assert str(choose_min_uint_dtype(255)) == "uint8"
    assert str(choose_min_uint_dtype(65535)) == "uint16"
