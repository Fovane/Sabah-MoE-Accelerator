"""Dump the GGUF header: architecture, tensor names of interest, per shard.

Reads only the metadata (no tensor data), so it is safe to run while the GPU is
busy. Used to settle where per_layer_token_embd lives before claiming where it
runs.
"""
import struct
import sys

T_U8, T_I8, T_U16, T_I16, T_U32, T_I32, T_F32, T_BOOL, T_STR, T_ARR, T_U64, T_I64, T_F64 = range(13)

SCALAR = {
    T_U8: ('<B', 1), T_I8: ('<b', 1), T_U16: ('<H', 2), T_I16: ('<h', 2),
    T_U32: ('<I', 4), T_I32: ('<i', 4), T_F32: ('<f', 4), T_BOOL: ('<?', 1),
    T_U64: ('<Q', 8), T_I64: ('<q', 8), T_F64: ('<d', 8),
}

TYPE_NAME = {
    0: 'F32', 1: 'F16', 2: 'Q4_0', 3: 'Q4_1', 6: 'Q5_0', 7: 'Q5_1', 8: 'Q8_0',
    10: 'Q2_K', 11: 'Q3_K', 12: 'Q4_K', 13: 'Q5_K', 14: 'Q6_K', 15: 'Q8_K',
    20: 'IQ4_NL', 21: 'IQ4_XS', 22: 'IQ3_S', 23: 'IQ3_XXS', 24: 'IQ2_S',
    25: 'IQ2_XS', 26: 'IQ1_S', 27: 'IQ1_M', 28: 'BF16', 30: 'TQ1_0', 31: 'TQ2_0',
    39: 'BF16', 40: 'Q4_0_4_4',
}


def rstr(f):
    n = struct.unpack('<Q', f.read(8))[0]
    return f.read(n).decode('utf-8', 'replace')


def rval(f, t):
    if t in SCALAR:
        fmt, size = SCALAR[t]
        return struct.unpack(fmt, f.read(size))[0]
    if t == T_STR:
        return rstr(f)
    if t == T_ARR:
        et = struct.unpack('<I', f.read(4))[0]
        n = struct.unpack('<Q', f.read(8))[0]
        return [rval(f, et) for _ in range(n)]
    raise ValueError('type %d' % t)


def main(path, pats):
    with open(path, 'rb') as f:
        magic = f.read(4)
        if magic != b'GGUF':
            raise SystemExit('%s: not a GGUF' % path)
        ver = struct.unpack('<I', f.read(4))[0]
        n_tensors = struct.unpack('<Q', f.read(8))[0]
        n_kv = struct.unpack('<Q', f.read(8))[0]
        kv = {}
        for _ in range(n_kv):
            k = rstr(f)
            t = struct.unpack('<I', f.read(4))[0]
            kv[k] = rval(f, t)
        print('== %s' % path)
        print('gguf=%d tensors=%d kv=%d' % (ver, n_tensors, n_kv))
        for key in ('general.architecture', 'general.name', 'general.size_label',
                    'general.file_type', 'split.count', 'split.no'):
            if key in kv:
                print('  %s = %s' % (key, kv[key]))
        for k in kv:
            if k.startswith('general.architecture') or k.endswith('.expert_count') or k.endswith('.block_count'):
                print('  %s = %s' % (k, kv[k]))
        if 'ALL' in pats:
            for k in kv:
                v = kv[k]
                if isinstance(v, list) and len(v) > 8:
                    v = v[:8] + ['...(%d)' % len(kv[k])]
                print('  %s = %s' % (k, v))
        hits = []
        for _ in range(n_tensors):
            name = rstr(f)
            nd = struct.unpack('<I', f.read(4))[0]
            dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(nd)]
            ttype = struct.unpack('<I', f.read(4))[0]
            off = struct.unpack('<Q', f.read(8))[0]
            for p in pats:
                if p in name:
                    hits.append((name, ttype, dims, off))
        print('  matching tensors: %d' % len(hits))
        for name, ttype, dims, off in hits:
            n_elem = 1
            for d in dims:
                n_elem *= d
            print('    %-52s %-7s %s (%d elems)' %
                  (name, TYPE_NAME.get(ttype, str(ttype)), 'x'.join(str(d) for d in dims), n_elem))


if __name__ == '__main__':
    pat = sys.argv[1:] or ['per_layer_token_embd']
    for p in [r'D:\Strata-data\models\IQ3_S\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf',
              r'D:\Strata-data\models\IQ3_S\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf']:
        try:
            main(p, pat)
        except FileNotFoundError:
            print('missing %s' % p)
