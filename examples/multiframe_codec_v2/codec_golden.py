"""Software golden encoder + decoder matching the hardware semantics.

Pipeline (encoder):
    pixels (uint8)
      -> frame_subtract: subtract 128 for I-frames, else subtract prev
      -> 4x4 fp16 DCT-II (separable, orthonormal)
      -> quantize_fp: round(coef / step) where step = 2^((qp-12)/6)
      -> zigzag scan
      -> CAVLC-style coefficient coding by default, or Exp-Golomb run/level
         coding when entropy="expgolomb" is selected explicitly
      -> bitstream: 1-bit frame_type prefix + concatenated codewords, byte-aligned

Decoder reverses everything.

The default v2 path uses H.264-inspired CAVLC coefficient semantics while
remaining a compact local bitstream, not a compliant H.264 byte stream.
"""
import math
import numpy as np

# ---------------------------------------------------------------------------
# Zig-zag scan order for 4x4 (column-major variant matching the spec)
# Indices into a row-major 4x4 flattened to length 16.
ZIGZAG_4x4 = [
    0,  1,  4,  8,
    5,  2,  3,  6,
    9, 12, 13, 10,
    7, 11, 14, 15,
]


def _zigzag_indices(n):
    order = []
    for s in range(2 * n - 1):
        diag = []
        for y in range(n):
            x = s - y
            if 0 <= x < n:
                diag.append((y, x))
        if s % 2 == 0:
            diag.reverse()
        order.extend(y * n + x for y, x in diag)
    return order


ZIGZAG_8x8 = _zigzag_indices(8)

# ---------------------------------------------------------------------------
# 4x4 fp16 DCT matrix (separable, orthonormal). row k of D is the kth basis.
def dct4_matrix():
    D = np.zeros((4, 4), dtype=np.float64)
    for k in range(4):
        ak = math.sqrt(1.0 / 4.0) if k == 0 else math.sqrt(2.0 / 4.0)
        for j in range(4):
            D[k, j] = ak * math.cos(math.pi * (2 * j + 1) * k / 8.0)
    return D.astype(np.float16)


DCT4 = dct4_matrix()
IDCT4 = DCT4.T          # orthonormal -> transpose is inverse


def dct_matrix(n):
    D = np.zeros((n, n), dtype=np.float64)
    for k in range(n):
        ak = math.sqrt(1.0 / n) if k == 0 else math.sqrt(2.0 / n)
        for j in range(n):
            D[k, j] = ak * math.cos(math.pi * (2 * j + 1) * k / (2.0 * n))
    return D.astype(np.float16)


DCT8 = dct_matrix(8)
IDCT8 = DCT8.T


def dct_4x4(block):
    """block: 4x4 int16 residual.  Returns 4x4 fp16 coefficients."""
    b = block.astype(np.float16)
    return (DCT4 @ b @ DCT4.T).astype(np.float16)


def idct_4x4(coef):
    """coef: 4x4 fp16 quantized levels (dequantized).  Returns 4x4 int16 residual."""
    c = coef.astype(np.float16)
    return np.round((IDCT4 @ c @ IDCT4.T).astype(np.float64)).astype(np.int32)


def dct_8x8(block):
    b = block.astype(np.float16)
    return (DCT8 @ b @ DCT8.T).astype(np.float16)


def idct_8x8(coef):
    c = coef.astype(np.float16)
    return np.round((IDCT8 @ c @ IDCT8.T).astype(np.float64)).astype(np.int32)


# ---------------------------------------------------------------------------
# Quantize
def step_for_qp(qp):
    return 2.0 ** ((qp - 12) / 6.0)


# H.264-style 4x4 intra quantization weights (normalized to mean 16 so a flat
# matrix recovers the original behavior).  Lower step for DC and low-frequency
# coefficients, higher step for high-freq.  Values cribbed from the JPEG
# luminance table's upper-left 4x4 then scaled.
QUANT_MATRIX_4x4 = np.array([
    [13, 12, 14, 18],
    [12, 13, 16, 22],
    [14, 16, 20, 26],
    [18, 22, 26, 32],
], dtype=np.float64)
QUANT_MATRIX_4x4 *= 16.0 / QUANT_MATRIX_4x4.mean()      # renormalize to mean 16


def quantize(coef, qp, use_matrix=False):
    s = step_for_qp(qp)
    if use_matrix:
        eff = s * QUANT_MATRIX_4x4 / 16.0
    else:
        eff = s
    return np.round(coef.astype(np.float64) / eff).astype(np.int32)


def dequantize(level, qp, use_matrix=False):
    s = step_for_qp(qp)
    if use_matrix:
        eff = s * QUANT_MATRIX_4x4 / 16.0
    else:
        eff = s
    return (level.astype(np.float64) * eff).astype(np.float16)


def quantize_n(coef, qp):
    return np.round(coef.astype(np.float64) / step_for_qp(qp)).astype(np.int32)


def dequantize_n(level, qp):
    return (level.astype(np.float64) * step_for_qp(qp)).astype(np.float16)


# ---------------------------------------------------------------------------
# Zig-zag + run-length
def zigzag_rle(quant_4x4):
    """Returns list of (run, level) pairs ending with (0,0) EOB sentinel."""
    flat = quant_4x4.flatten()
    scanned = [int(flat[i]) for i in ZIGZAG_4x4]
    pairs = []
    run = 0
    for v in scanned:
        if v == 0:
            run += 1
        else:
            pairs.append((run, v))
            run = 0
    pairs.append((0, 0))     # EOB sentinel; trailing zeros are absorbed
    return pairs


def unzigzag_rle(pairs):
    """Inverse: pairs ending in (0,0) sentinel -> 4x4 quantized levels."""
    scanned = []
    for run, level in pairs[:-1]:
        scanned.extend([0] * run)
        scanned.append(level)
    while len(scanned) < 16:
        scanned.append(0)
    # invert zigzag
    inv = [0] * 16
    for i, dst in enumerate(ZIGZAG_4x4):
        inv[dst] = scanned[i]
    return np.array(inv, dtype=np.int32).reshape(4, 4)


def zigzag_rle_n(quant):
    n = int(quant.shape[0])
    order = ZIGZAG_4x4 if n == 4 else ZIGZAG_8x8
    flat = quant.flatten()
    pairs = []
    run = 0
    for idx in order:
        v = int(flat[idx])
        if v == 0:
            run += 1
        else:
            pairs.append((run, v))
            run = 0
    pairs.append((0, 0))
    return pairs


def unzigzag_rle_n(pairs, n):
    order = ZIGZAG_4x4 if n == 4 else ZIGZAG_8x8
    scanned = []
    for run, level in pairs[:-1]:
        scanned.extend([0] * run)
        scanned.append(level)
    while len(scanned) < n * n:
        scanned.append(0)
    inv = [0] * (n * n)
    for i, dst in enumerate(order):
        inv[dst] = scanned[i]
    return np.array(inv, dtype=np.int32).reshape(n, n)


def _zigzag_for_shape(shape):
    if shape == (4, 4):
        return ZIGZAG_4x4
    if shape == (8, 8):
        return ZIGZAG_8x8
    raise ValueError(f"unsupported block shape {shape}; expected 4x4 or 8x8")


def _scan_coefficients(coeffs):
    coeffs = np.asarray(coeffs, dtype=np.int32)
    scan = _zigzag_for_shape(tuple(coeffs.shape))
    flat = coeffs.flatten()
    return [int(flat[i]) for i in scan]


def _unscan_coefficients(scanned, shape):
    scan = _zigzag_for_shape(tuple(shape))
    if len(scanned) != len(scan):
        raise ValueError(f"expected {len(scan)} coefficients, got {len(scanned)}")
    inv = [0] * len(scan)
    for i, dst in enumerate(scan):
        inv[dst] = int(scanned[i])
    return np.array(inv, dtype=np.int32).reshape(shape)


# ---------------------------------------------------------------------------
# Exp-Golomb signed encode/decode (PyH264 semantics)
def expgolomb_encode(v):
    if v == 0:
        codenum = 0
    elif v > 0:
        codenum = 2 * v - 1
    else:
        codenum = -2 * v
    val = codenum + 1
    bit_len = val.bit_length()
    L = 2 * bit_len - 1
    return format(val, f'0{L}b')


def expgolomb_decode_stream(bits, offset):
    """Returns (signed value, new offset).  bits is a string of '0'/'1'."""
    # count leading zeros
    n = 0
    while offset + n < len(bits) and bits[offset + n] == '0':
        n += 1
    L = 2 * n + 1
    if offset + L > len(bits):
        return None, offset
    val = int(bits[offset:offset + L], 2)
    codenum = val - 1
    if codenum == 0:
        signed = 0
    elif codenum % 2 == 1:
        signed = (codenum + 1) // 2
    else:
        signed = -(codenum // 2)
    return signed, offset + L


def _ue_encode(v):
    if v < 0:
        raise ValueError("unsigned Exp-Golomb value must be non-negative")
    val = v + 1
    bit_len = val.bit_length()
    return ("0" * (bit_len - 1)) + format(val, f"0{bit_len}b")


def _ue_decode_stream(bits, offset):
    n = 0
    while offset + n < len(bits) and bits[offset + n] == "0":
        n += 1
    if offset + n >= len(bits):
        return None, offset
    end = offset + (2 * n + 1)
    if end > len(bits):
        return None, offset
    return int(bits[offset:end], 2) - 1, end


def cavlc_encode_coefficients(coeffs):
    """Encode one 4x4 or 8x8 quantized coefficient block.

    This local CAVLC model emits TotalCoeff, TrailingOnes, trailing-one signs,
    remaining signed levels, TotalZeros, and reverse-order RunBefore values.
    Compact Exp-Golomb codewords carry the variable fields so the stream stays
    self-delimiting for the CoreSmith golden model.
    """
    scanned = _scan_coefficients(coeffs)
    nonzero = [(i, v) for i, v in enumerate(scanned) if v != 0]
    total_coeff = len(nonzero)
    max_coeff = len(scanned)

    trailing_ones = 0
    for _pos, val in reversed(nonzero):
        if abs(val) == 1 and trailing_ones < 3:
            trailing_ones += 1
        else:
            break

    bits = [_ue_encode(total_coeff * 4 + trailing_ones)]
    if total_coeff == 0:
        return "".join(bits)

    values_rev = [v for _pos, v in reversed(nonzero)]
    for val in values_rev[:trailing_ones]:
        bits.append("1" if val < 0 else "0")
    for val in values_rev[trailing_ones:]:
        bits.append(expgolomb_encode(val))

    prev = -1
    runs = []
    for pos, _val in nonzero:
        runs.append(pos - prev - 1)
        prev = pos
    total_zeros = sum(runs)
    if total_coeff < max_coeff:
        bits.append(_ue_encode(total_zeros))

    for run in reversed(runs[1:]):
        bits.append(_ue_encode(run))
    return "".join(bits)


def cavlc_decode_coefficients(bits, offset=0, shape=(4, 4)):
    """Decode one CAVLC coefficient block and return (coefficients, offset)."""
    shape = tuple(shape)
    max_coeff = len(_zigzag_for_shape(shape))
    token, offset = _ue_decode_stream(bits, offset)
    if token is None:
        return None, offset
    total_coeff = token // 4
    trailing_ones = token % 4
    if total_coeff > max_coeff or trailing_ones > min(3, total_coeff):
        raise ValueError("invalid CAVLC coefficient token")
    if total_coeff == 0:
        return np.zeros(shape, dtype=np.int32), offset

    values_rev = []
    for _ in range(trailing_ones):
        if offset >= len(bits):
            return None, offset
        values_rev.append(-1 if bits[offset] == "1" else 1)
        offset += 1
    for _ in range(total_coeff - trailing_ones):
        val, offset = expgolomb_decode_stream(bits, offset)
        if val is None:
            return None, offset
        values_rev.append(val)

    if total_coeff < max_coeff:
        total_zeros, offset = _ue_decode_stream(bits, offset)
        if total_zeros is None:
            return None, offset
    else:
        total_zeros = 0
    if total_zeros > max_coeff - total_coeff:
        raise ValueError("invalid CAVLC total_zeros")

    runs_rev = []
    zeros_left = total_zeros
    for _ in range(total_coeff - 1):
        run, offset = _ue_decode_stream(bits, offset)
        if run is None:
            return None, offset
        if run > zeros_left:
            raise ValueError("invalid CAVLC run_before")
        runs_rev.append(run)
        zeros_left -= run
    runs_rev.append(zeros_left)

    scanned = []
    for run, val in zip(reversed(runs_rev), reversed(values_rev)):
        scanned.extend([0] * run)
        scanned.append(val)
    scanned.extend([0] * (max_coeff - len(scanned)))
    return _unscan_coefficients(scanned, shape), offset


# ---------------------------------------------------------------------------
# Bitstream packing per block: 1-bit frame_type + codeword stream, byte aligned
def pack_block(pairs, frame_is_intra, entropy="expgolomb", coeffs=None):
    bits = ['1' if frame_is_intra else '0']
    if entropy == "cavlc":
        if coeffs is None:
            coeffs = unzigzag_rle(pairs)
        bits.append(cavlc_encode_coefficients(coeffs))
        return ''.join(bits)
    if entropy != "expgolomb":
        raise ValueError(f"unknown entropy mode {entropy!r}")
    for run, level in pairs:
        bits.append(expgolomb_encode(run))
        bits.append(expgolomb_encode(level))
    return ''.join(bits)


# ---------------------------------------------------------------------------
# Intra prediction (PyH264-style, 3 modes per 4x4 block)
# Modes:  0 = DC (mean of top + left)
#         1 = Vertical    (copy top row down)
#         2 = Horizontal  (copy left column right)
# Top-edge blocks: V unavailable -> fall back to DC.
# Left-edge blocks: H unavailable -> fall back to DC.
# Top-left block: only DC with default neighbors (=128 like before).
def _predictor(mode, top, left):
    """Build a 4x4 predictor.  top/left are 4-element int16 arrays or None."""
    pred = np.zeros((4, 4), dtype=np.int16)
    if mode == 1:                                          # Vertical
        assert top is not None
        pred[:] = top[None, :]
    elif mode == 2:                                        # Horizontal
        assert left is not None
        pred[:] = left[:, None]
    else:                                                  # DC
        if top is not None and left is not None:
            dc = (int(top.sum()) + int(left.sum()) + 4) // 8
        elif top is not None:
            dc = (int(top.sum()) + 2) // 4
        elif left is not None:
            dc = (int(left.sum()) + 2) // 4
        else:
            dc = 128
        pred[:] = dc
    return pred


def _predictor_n(mode, top, left, n):
    pred = np.zeros((n, n), dtype=np.int16)
    if mode == 1:
        assert top is not None
        pred[:] = top[None, :]
    elif mode == 2:
        assert left is not None
        pred[:] = left[:, None]
    else:
        if top is not None and left is not None:
            dc = (int(top.sum()) + int(left.sum()) + n) // (2 * n)
        elif top is not None:
            dc = (int(top.sum()) + n // 2) // n
        elif left is not None:
            dc = (int(left.sum()) + n // 2) // n
        else:
            dc = 128
        pred[:] = dc
    return pred


def _available_modes(by, bx):
    """Return list of modes available for the 4x4 block at (by, bx) in pixels."""
    has_top  = (by > 0)
    has_left = (bx > 0)
    modes = [0]                                            # DC always works
    if has_top:  modes.append(1)
    if has_left: modes.append(2)
    return modes


def _get_neighbors(recon, by, bx):
    top  = recon[by - 1, bx:bx + 4].astype(np.int16) if by > 0 else None
    left = recon[by:by + 4, bx - 1].astype(np.int16) if bx > 0 else None
    return top, left


def _available_modes_n(by, bx):
    modes = [0]
    if by > 0:
        modes.append(1)
    if bx > 0:
        modes.append(2)
    return modes


def _get_neighbors_n(recon, by, bx, n):
    top = recon[by - 1, bx:bx + n].astype(np.int16) if by > 0 else None
    left = recon[by:by + n, bx - 1].astype(np.int16) if bx > 0 else None
    return top, left


def _bits_for_pairs(pairs):
    total = 0
    for run, lvl in pairs:
        total += len(expgolomb_encode(run)) + len(expgolomb_encode(lvl))
    return total


def _bits_for_coeffs(level, pairs, entropy):
    if entropy == "cavlc":
        return len(cavlc_encode_coefficients(level))
    if entropy == "expgolomb":
        return _bits_for_pairs(pairs)
    raise ValueError(f"unknown entropy mode {entropy!r}")


def _append_coeff_bits(bits, level, pairs, entropy):
    if entropy == "cavlc":
        bits.append(cavlc_encode_coefficients(level))
    elif entropy == "expgolomb":
        for run, lvl in pairs:
            bits.append(expgolomb_encode(run))
            bits.append(expgolomb_encode(lvl))
    else:
        raise ValueError(f"unknown entropy mode {entropy!r}")


def _decode_coeff_bits(bits, offset, n, entropy):
    if entropy == "cavlc":
        level, offset = cavlc_decode_coefficients(bits, offset, shape=(n, n))
        if level is None:
            raise ValueError("truncated stream while decoding CAVLC block")
        return level, offset
    if entropy != "expgolomb":
        raise ValueError(f"unknown entropy mode {entropy!r}")
    pairs = []
    while True:
        run, offset = expgolomb_decode_stream(bits, offset)
        lvl, offset = expgolomb_decode_stream(bits, offset)
        if run is None or lvl is None:
            raise ValueError("truncated stream while decoding Exp-Golomb block")
        pairs.append((run, lvl))
        if run == 0 and lvl == 0:
            break
    return unzigzag_rle_n(pairs, n), offset


def _code_block(blk, recon, by, bx, n, qp, entropy="cavlc"):
    top, left = _get_neighbors_n(recon, by, bx, n)
    best = None
    for mode in _available_modes_n(by, bx):
        pred = _predictor_n(mode, top, left, n)
        resid = blk - pred
        if n == 8:
            coef = dct_8x8(resid)
            level = quantize_n(coef, qp)
            recon_resid = idct_8x8(dequantize_n(level, qp))
        else:
            coef = dct_4x4(resid)
            level = quantize_n(coef, qp)
            recon_resid = idct_4x4(dequantize_n(level, qp))
        pairs = zigzag_rle_n(level)
        recon_blk = np.clip(pred + recon_resid, 0, 255).astype(np.int16)
        distortion = float(np.mean((blk.astype(np.float64) - recon_blk) ** 2))
        header_bits = 1 + 1 + 2
        bit_count = header_bits + _bits_for_coeffs(level, pairs, entropy)
        cost = distortion + 0.08 * step_for_qp(qp) * bit_count / (n * n)
        candidate = {
            "n": n,
            "mode": mode,
            "level": level,
            "pairs": pairs,
            "recon": recon_blk,
            "bits": bit_count,
            "distortion": distortion,
            "cost": cost,
        }
        if best is None or candidate["cost"] < best["cost"]:
            best = candidate
    return best


def _code_mb_4x4(blk8, recon, by, bx, qp, entropy="cavlc"):
    temp_recon = recon.copy()
    subblocks = []
    total_bits = 1 + 1
    weighted_cost = 0.0
    for dy in (0, 4):
        for dx in (0, 4):
            blk4 = blk8[dy:dy + 4, dx:dx + 4]
            cand = _code_block(blk4, temp_recon, by + dy, bx + dx, 4, qp, entropy)
            temp_recon[by + dy:by + dy + 4, bx + dx:bx + dx + 4] = cand["recon"]
            subblocks.append(cand)
            total_bits += 2 + _bits_for_coeffs(cand["level"], cand["pairs"], entropy)
            weighted_cost += cand["cost"] * 16.0
    recon_blk = temp_recon[by:by + 8, bx:bx + 8].copy()
    return {
        "n": 4,
        "mode": None,
        "subblocks": subblocks,
        "recon": recon_blk,
        "bits": total_bits,
        "distortion": float(np.mean((blk8.astype(np.float64) - recon_blk) ** 2)),
        "cost": weighted_cost / 64.0,
    }


def encode_image_v2(pixels, qp=36, do_deblock=True, entropy="cavlc"):
    """I-frame encoder with selectable 8x8 macroblocks.

    Each 8x8 macroblock chooses either one 8x8 transform block or four 4x4
    transform blocks.  Both sizes choose one of three intra predictors:
    DC, vertical, or horizontal.  The stream is intentionally simple and
    H.264-inspired, not H.264-compliant.
    """
    H, W = pixels.shape
    assert H % 8 == 0 and W % 8 == 0, "must be multiple of 8"
    recon = np.zeros((H, W), dtype=np.int16)
    bits = []
    meta = []
    for by in range(0, H, 8):
        for bx in range(0, W, 8):
            blk8 = pixels[by:by + 8, bx:bx + 8].astype(np.int16)
            cand8 = _code_block(blk8, recon, by, bx, 8, qp, entropy)
            cand4 = _code_mb_4x4(blk8, recon, by, bx, qp, entropy)
            chosen = cand8 if cand8["cost"] <= cand4["cost"] else cand4
            bits.append("1")  # I-frame
            if chosen["n"] == 8:
                bits.append("1")
                bits.append(format(chosen["mode"], "02b"))
                _append_coeff_bits(bits, chosen["level"], chosen["pairs"], entropy)
            else:
                bits.append("0")
                for sub in chosen["subblocks"]:
                    bits.append(format(sub["mode"], "02b"))
                    _append_coeff_bits(bits, sub["level"], sub["pairs"], entropy)
            recon[by:by + 8, bx:bx + 8] = chosen["recon"]
            meta.append({
                "by": by,
                "bx": bx,
                "block_size": 8 if chosen["n"] == 8 else 4,
                "entropy": entropy,
                "bits": chosen["bits"],
                "distortion": chosen["distortion"],
            })

    filtered = deblock(recon.astype(np.uint8), qp) if do_deblock else recon.astype(np.uint8)
    s = "".join(bits)
    while len(s) % 8:
        s += "0"
    return bytes(int(s[i:i + 8], 2) for i in range(0, len(s), 8)), meta, filtered


def decode_image_v2(byte_stream, H, W, qp=36, do_deblock=True, entropy="cavlc"):
    bits = "".join(format(b, "08b") for b in byte_stream)
    offset = 0
    out = np.zeros((H, W), dtype=np.uint8)
    for by in range(0, H, 8):
        for bx in range(0, W, 8):
            _iframe = bits[offset] == "1"
            offset += 1
            use8 = bits[offset] == "1"
            offset += 1
            if use8:
                mode = int(bits[offset:offset + 2], 2)
                offset += 2
                level, offset = _decode_coeff_bits(bits, offset, 8, entropy)
                resid = idct_8x8(dequantize_n(level, qp))
                top, left = _get_neighbors_n(out, by, bx, 8)
                pred = _predictor_n(mode, top, left, 8)
                out[by:by + 8, bx:bx + 8] = np.clip(pred + resid, 0, 255).astype(np.uint8)
            else:
                for dy in (0, 4):
                    for dx in (0, 4):
                        mode = int(bits[offset:offset + 2], 2)
                        offset += 2
                        level, offset = _decode_coeff_bits(bits, offset, 4, entropy)
                        resid = idct_4x4(dequantize_n(level, qp))
                        top, left = _get_neighbors_n(out, by + dy, bx + dx, 4)
                        pred = _predictor_n(mode, top, left, 4)
                        out[by + dy:by + dy + 4, bx + dx:bx + dx + 4] = (
                            np.clip(pred + resid, 0, 255).astype(np.uint8)
                        )
    if do_deblock:
        out = deblock(out, qp)
    return out


def encode_image(pixels, qp=36, use_intra_pred=False, entropy="expgolomb"):
    """pixels: H x W uint8.  Returns bytes + per-block metadata for debug."""
    H, W = pixels.shape
    assert H % 4 == 0 and W % 4 == 0, 'must be multiple of 4'
    bits = []
    block_bits_list = []

    # Reconstructed pixels (only needed when intra prediction is on).
    recon = np.zeros((H, W), dtype=np.int16)

    for by in range(0, H, 4):
        for bx in range(0, W, 4):
            blk = pixels[by:by + 4, bx:bx + 4].astype(np.int16)

            if not use_intra_pred:
                # Original path: subtract 128.
                resid = blk - 128
                coef  = dct_4x4(resid)
                level = quantize(coef, qp)
                pairs = zigzag_rle(level)
                bblk = pack_block(pairs, frame_is_intra=True,
                                  entropy=entropy, coeffs=level)
                bits.append(bblk)
                block_bits_list.append(bblk)
                continue

            # Intra-pred path: try each available mode, pick smallest SAD
            # residual energy (cheap proxy for true RD cost).
            top, left = _get_neighbors(recon, by, bx)
            modes = _available_modes(by, bx)
            best_mode = 0
            best_resid = None
            best_pred  = None
            best_sad   = None
            for m in modes:
                pred = _predictor(m, top, left)
                resid = blk - pred
                sad = int(np.abs(resid).sum())
                if best_sad is None or sad < best_sad:
                    best_sad = sad
                    best_mode = m
                    best_resid = resid
                    best_pred  = pred

            # Encode residual through DCT->quant->entropy coder
            coef = dct_4x4(best_resid)
            level = quantize(coef, qp)
            pairs = zigzag_rle(level)

            # Bitstream: 1-bit frame_is_intra + 2-bit mode + codewords
            bblk = '1'                                     # frame_is_intra = 1
            bblk += format(best_mode, '02b')               # 2-bit mode
            if entropy == "cavlc":
                bblk += cavlc_encode_coefficients(level)
            elif entropy == "expgolomb":
                for run, lvl in pairs:
                    bblk += expgolomb_encode(run)
                    bblk += expgolomb_encode(lvl)
            else:
                raise ValueError(f"unknown entropy mode {entropy!r}")
            bits.append(bblk)
            block_bits_list.append(bblk)

            # Reconstruct this block to update recon[] for future neighbors.
            deq = dequantize(level, qp)
            resid_recon = idct_4x4(deq)
            block_recon = best_pred + resid_recon
            recon[by:by + 4, bx:bx + 4] = np.clip(block_recon, 0, 255)

    s = ''.join(bits)
    while len(s) % 8 != 0:
        s += '0'
    by = bytearray(int(s[i:i+8], 2) for i in range(0, len(s), 8))
    return bytes(by), block_bits_list


def decode_image(byte_stream, H, W, qp=36, do_deblock=False,
                 use_intra_pred=False, entropy="expgolomb"):
    """Reverse encode_image; expects num_blocks == H*W/16."""
    bits = ''.join(format(b, '08b') for b in byte_stream)
    offset = 0
    out = np.zeros((H, W), dtype=np.uint8)
    n_blocks_y = H // 4
    n_blocks_x = W // 4
    for by_idx in range(n_blocks_y):
        for bx_idx in range(n_blocks_x):
            # 1-bit frame_type
            _frame_is_intra = (bits[offset] == '1')
            offset += 1

            if use_intra_pred:
                mode = int(bits[offset:offset + 2], 2)
                offset += 2
            else:
                mode = None

            if entropy == "cavlc":
                quant, offset = cavlc_decode_coefficients(bits, offset, shape=(4, 4))
                if quant is None:
                    raise ValueError("truncated stream while decoding CAVLC block")
            elif entropy == "expgolomb":
                pairs = []
                while True:
                    run, offset = expgolomb_decode_stream(bits, offset)
                    if run is None:
                        raise ValueError('truncated stream while decoding run')
                    level, offset = expgolomb_decode_stream(bits, offset)
                    if level is None:
                        raise ValueError('truncated stream while decoding level')
                    pairs.append((run, level))
                    if run == 0 and level == 0:
                        break
                quant = unzigzag_rle(pairs)
            else:
                raise ValueError(f"unknown entropy mode {entropy!r}")
            coef  = dequantize(quant, qp)
            resid = idct_4x4(coef)

            by = by_idx * 4; bx = bx_idx * 4
            if use_intra_pred:
                top, left = _get_neighbors(out, by, bx)
                pred = _predictor(mode, top, left)
                pixels = np.clip(pred + resid, 0, 255).astype(np.uint8)
            else:
                pixels = np.clip(resid + 128, 0, 255).astype(np.uint8)
            out[by:by + 4, bx:bx + 4] = pixels

    if do_deblock:
        out = deblock(out, qp)
    return out


# ---------------------------------------------------------------------------
# Deblocking filter (simplified, H.264-style conditional 4-tap smoothing on
# 4x4 block edges).  Mirrors the hardware deblock_4x4.v module bit-exact.
#
# For each 4-pixel edge between two adjacent 4x4 blocks we examine 6 pixels
# straddling the boundary: p2 p1 p0 | q0 q1 q2.  If the gradient across the
# edge is small enough (suggesting it's a coding artifact, not a real edge),
# we replace p0 and q0 with a 3-tap weighted average.
#
# Thresholds are QP-dependent: at low QP the codec is nearly lossless so we
# shouldn't smooth at all; at high QP the quantization noise dominates and
# aggressive smoothing wins.
def _alpha_beta(qp):
    """H.264-style table lookup, simplified."""
    a = max(1, int(qp - 24))
    b = max(1, int((qp - 24) // 2))
    return a, b


def _deblock_edge_inplace(line_p, line_q, qp):
    """Apply the filter on one 4-pixel edge.  line_p[0..2] are the 3 pixels
    on the 'p' side ordered farthest-first (p2,p1,p0); line_q[0..2] is the
    q side ordered nearest-first (q0,q1,q2).  Modifies p0=line_p[2] and
    q0=line_q[0]."""
    alpha, beta = _alpha_beta(qp)
    p2, p1, p0 = int(line_p[0]), int(line_p[1]), int(line_p[2])
    q0, q1, q2 = int(line_q[0]), int(line_q[1]), int(line_q[2])
    if abs(p0 - q0) >= alpha:    return
    if abs(p1 - p0) >= beta:     return
    if abs(q1 - q0) >= beta:     return
    new_p0 = (p2 + 2 * p1 + 2 * p0 + 2 * q0 + q1 + 4) // 8
    new_q0 = (p1 + 2 * p0 + 2 * q0 + 2 * q1 + q2 + 4) // 8
    line_p[2] = max(0, min(255, new_p0))
    line_q[0] = max(0, min(255, new_q0))


def deblock(pixels, qp):
    """Apply deblocking on every 4x4 block edge in `pixels` (H x W uint8).
    Filters vertical edges (between horizontally adjacent blocks) then
    horizontal edges, the same order H.264 specifies."""
    H, W = pixels.shape
    out = pixels.astype(np.int16).copy()

    # Vertical edges: x = 4, 8, 12, ... (between block columns)
    for x in range(4, W, 4):
        for y in range(H):
            p = out[y, x - 3:x].copy()      # p2,p1,p0
            q = out[y, x:x + 3].copy()      # q0,q1,q2
            if len(p) != 3 or len(q) != 3:
                continue
            _deblock_edge_inplace(p, q, qp)
            out[y, x - 1] = p[2]
            out[y, x]     = q[0]

    # Horizontal edges: y = 4, 8, 12, ... (between block rows)
    for y in range(4, H, 4):
        for x in range(W):
            p = out[y - 3:y, x].copy()
            q = out[y:y + 3, x].copy()
            if len(p) != 3 or len(q) != 3:
                continue
            _deblock_edge_inplace(p, q, qp)
            out[y - 1, x] = p[2]
            out[y, x]     = q[0]

    return np.clip(out, 0, 255).astype(np.uint8)


def psnr(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float('inf')
    return 10.0 * math.log10(255.0 * 255.0 / mse)


# ---------------------------------------------------------------------------
# Smoke test: generate a small grayscale, encode, decode, PSNR
if __name__ == '__main__':
    import sys
    from PIL import Image

    rng = np.random.default_rng(7)
    if len(sys.argv) > 1 and sys.argv[1] and sys.argv[1] != '-':
        img = np.asarray(Image.open(sys.argv[1]).convert('L'))
    else:
        # Default: a 16x16 gradient + small noise
        img = (np.tile(np.arange(16) * 16, (16, 1)).astype(np.float64)
               + rng.normal(0, 5, (16, 16))).clip(0, 255).astype(np.uint8)

    H, W = img.shape
    H = (H // 8) * 8
    W = (W // 8) * 8
    img = img[:H, :W]

    qp = int(sys.argv[2]) if len(sys.argv) > 2 else 36
    encoded, meta, enc_recon = encode_image_v2(img, qp=qp)
    print(f'encoded {H}x{W} at qp={qp}: {len(encoded)} bytes '
          f'({8 * len(encoded) / (H * W):.3f} bpp)')
    selected_8x8 = sum(1 for m in meta if m["block_size"] == 8)
    print(f'8x8 macroblocks selected: {selected_8x8}/{len(meta)}')

    decoded = decode_image_v2(encoded, H, W, qp=qp)
    assert np.array_equal(enc_recon, decoded)
    p = psnr(img, decoded)
    print(f'PSNR: {p:.2f} dB')
