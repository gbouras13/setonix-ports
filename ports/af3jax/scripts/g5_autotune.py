#!/usr/bin/env python
"""G5: copy an XLA autotune-results file (--xla_gpu_dump_autotune_results_to=X.textproto) with its hipBLASLt GEMM entries forced to
one algorithm index, for --xla_gpu_load_autotune_results_from. Text format only; standard library.
  g5_autotune.py show SRC.textproto              -> the entries: backend, algorithm, the start of the instruction
  g5_autotune.py force SRC.textproto ALG OUT     -> OUT with every hipBLASLt entry set to ALG (exit 2 if there is none)
Two layouts are handled. Older XLA writes 'gemm { algorithm: N }'. This XLA (jax 0.10.2; results version 45) writes
  other { name: "HIPBLASLT" config { type_url: ".../xla.autotuner.BackendConfig" value: "<serialised bytes>" } }
where the bytes are field 1 = { field 1 = algorithm (varint), field 2 = workspace bytes (varint) } (G5 job 50409332: the dumps held
53 / 12 / 18 there, the algorithms XLA compiled).
"""
import codecs
import re
import sys

GEMM = re.compile(r"gemm\s*\{\s*algorithm:\s*(-?\d+)\s*\}")


def rv(b, i):
    x = s = 0
    while True:
        c = b[i]
        i += 1
        x |= (c & 0x7F) << s
        s += 7
        if not c & 0x80:
            return x, i


def wv(x):
    out = bytearray()
    while True:
        c, x = x & 0x7F, x >> 7
        out.append(c | 0x80 if x else c)
        if not x:
            return bytes(out)


def fields(b):
    i, out = 0, []
    while i < len(b):
        key, i = rv(b, i)
        f, wt = key >> 3, key & 7
        if wt == 0:
            v, i = rv(b, i)
        elif wt == 2:
            n, i = rv(b, i)
            v, i = b[i:i + n], i + n
        elif wt == 1:
            v, i = b[i:i + 8], i + 8
        elif wt == 5:
            v, i = b[i:i + 4], i + 4
        else:
            raise ValueError(f"wire type {wt}")
        out.append((f, wt, v))
    return out


def ser(fs):
    out = bytearray()
    for f, wt, v in fs:
        out += wv((f << 3) | wt)
        out += wv(v) if wt == 0 else (wv(len(v)) + v if wt == 2 else v)
    return bytes(out)


def esc(b):
    m = {0x0A: "\\n", 0x0D: "\\r", 0x09: "\\t", 0x22: '\\"', 0x27: "\\'", 0x5C: "\\\\"}
    return "".join(m.get(c) or (chr(c) if 0x20 <= c < 0x7F else "\\%03o" % c) for c in b)


def hip_algorithm(raw):
    for f, wt, v in fields(raw):
        if f == 1 and wt == 2:
            for g, wt2, x in fields(v):
                if g == 1 and wt2 == 0:
                    return x
            return 0                                  # proto3 omits a zero
    return None


def hip_force(raw, alg):
    outer = []
    for f, wt, v in fields(raw):
        if f == 1 and wt == 2:
            inner = [(g, w2, x) for g, w2, x in fields(v) if g != 1]
            v = ser([(1, 0, alg)] + inner)
        outer.append((f, wt, v))
    return ser(outer)


def blocks(text):
    """(start, end) of each top-level 'results { ... }' block"""
    out, i = [], 0
    while True:
        j = text.find("results {", i)
        if j < 0:
            return out
        depth = 0
        for k in range(j, len(text)):
            if text[k] == "{":
                depth += 1
            elif text[k] == "}":
                depth -= 1
                if depth == 0:
                    break
        out.append((j, k + 1))
        i = k + 1


VALUE = re.compile(r'(value:\s*")((?:[^"\\]|\\.)*)(")')


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "show":
        text = open(sys.argv[2]).read()
        bs = blocks(text)
        print(f"{len(bs)} entries")
        for j, k in bs:
            b = text[j:k]
            hlo = re.search(r'hlo:\s*"((?:[^"\\]|\\.)*)"', b)
            name = re.search(r'name:\s*"([^"]+)"', b)
            g = GEMM.search(b)
            if g:
                kind, alg = "gemm", g.group(1)
            elif name and name.group(1) == "HIPBLASLT":
                kind, alg = "HIPBLASLT", hip_algorithm(codecs.escape_decode(VALUE.search(b).group(2).encode())[0])
            else:
                kind, alg = (name.group(1) if name else "other"), "-"
            if kind in ("gemm", "HIPBLASLT") or len(sys.argv) > 3:
                print(f"  {kind:<10} algorithm {alg!s:>4}  {(hlo.group(1) if hlo else '?')[:150]}")
        return 0
    if len(sys.argv) == 5 and sys.argv[1] == "force":
        src, alg, out = sys.argv[2], int(sys.argv[3]), sys.argv[4]
        text = open(src).read()
        text, n = GEMM.subn(f"gemm {{ algorithm: {alg} }}", text)
        parts, last = [], 0
        for j, k in blocks(text):
            b = text[j:k]
            if re.search(r'name:\s*"HIPBLASLT"', b) and "__cublas$lt$matmul" in b:
                m = VALUE.search(b)
                raw = codecs.escape_decode(m.group(2).encode())[0]
                new = hip_force(raw, alg)
                assert hip_algorithm(new) == alg
                b = b[:m.start(2)] + esc(new) + b[m.end(2):]
                n += 1
            parts.append(text[last:j] + b)
            last = k
        parts.append(text[last:])
        if n == 0:
            print(f"no hipBLASLt / gemm entry in {src}")
            return 2
        open(out, "w").write("".join(parts))
        print(f"{out}: {n} entr{'y' if n == 1 else 'ies'} forced to algorithm {alg}")
        return 0
    print(__doc__)
    return 1


if __name__ == "__main__":
    sys.exit(main())
