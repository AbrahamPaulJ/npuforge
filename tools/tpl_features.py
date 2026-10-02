#!/usr/bin/env python3
"""Feature omission for a pack-loading template model_tpl.cpp (Swap v3: LoRA, ControlNet, IP,
inpaint; a Swap v2 source has no inpaint inputs and works unchanged).

Usage: tpl_features.py model_tpl.cpp model_feat.cpp report.json

For every subset of dropped features (bit 1 = lora, 2 = cn, 4 = ip, 8 = inp) it walks the compose list
in order: a node with an input tainted by a dropped feature is skipped and taints its outputs,
EXCEPT an ElementWiseBinary ADD with exactly one tainted input whose clean side has the output's
shape and dtype: that is the branch's MERGE, so it is skipped and its output aliased to the clean
side (whose own encoding the consumer then reads). Static tensors and graph inputs with no kept
consumer are skipped. The patched lib reads $QNN_TPL_DROP ("lora,cn,ip,inp", any subset) at
compose time; unset or empty = the original graph, call for call.
"""
import json, re, sys

FEATURES = [("lora", 1), ("cn", 2), ("ip", 4), ("inp", 8)]
NSUB = 16


def feature_of(name):
    if name == "lora_S" or re.fullmatch(r"l[ab]_\d+", name):
        return 1
    if re.fullmatch(r"res_\d+", name):
        return 2
    if re.fullmatch(r"ip[kv]_\d+", name):
        return 4
    if name in ("mask", "masked_latent"):
        return 8
    return 0


FUNC_RE = re.compile(r"^static ModelError_t (add(?:Node|Tensor)_\w+)\(QnnModel& model\)\{", re.M)
DIMS_RE = re.compile(r"uint32_t (dimensions_\w+)\[\] = \{([^}]*)\};")
TLIT_RE = re.compile(r'\.name= "([^"]*)",\s*\.type= (\w+),.*?\.dataType= (\w+),.*?\.dimensions=(\w+),', re.S)
INPUTS_RE = re.compile(r"const char\*  inputs_\w+\[\] = \{(.*?)\};", re.S)
TYPE_RE = re.compile(r'"(\w+)", // Qnn Node Type')
OPER_RE = re.compile(r'\.name="operation",\s*\{\.scalarParam= \(Qnn_Scalar_t\) \{QNN_DATATYPE_UINT_32, \{\.uint32Value = (\d+)\}\}')
CALL_RE = re.compile(r"^(\s*)VALIDATE\((add(?:Node|Tensor)_\w+)\(model\), err\);$")


def main(src, dst, report_path):
    text = open(src).read()
    starts = [(m.start(), m.group(1)) for m in FUNC_RE.finditer(text)]
    comp_at = text.index("ModelError_t QnnModel_composeGraphs(")
    funcs = {}
    for i, (s, fname) in enumerate(starts):
        e = starts[i + 1][0] if i + 1 < len(starts) else comp_at
        funcs[fname] = (s, min(e, comp_at) if s < comp_at else e)

    info = {}   # fname -> dict
    shapes = {}  # tensor name -> (dims tuple, dtype)
    for fname, (s, e) in funcs.items():
        body = text[s:e]
        dims = {k: tuple(int(x) for x in v.split(",") if x.strip()) for k, v in DIMS_RE.findall(body)}
        lits = [(n, t, dt, dims.get(dv)) for n, t, dt, dv in TLIT_RE.findall(body)]
        if fname.startswith("addTensor_"):
            assert len(lits) == 1, (fname, len(lits))
            n, t, dt, d = lits[0]
            info[fname] = {"kind": "tensor", "name": n, "type": t}
            shapes[n] = (d, dt)
        else:
            im = INPUTS_RE.search(body)
            ins = re.findall(r'"([^"]*)"', im.group(1)) if im else []
            tm = TYPE_RE.search(body)
            om = OPER_RE.search(body)
            outs = [n for n, _, _, _ in lits]
            for n, t, dt, d in lits:
                shapes[n] = (d, dt)
            info[fname] = {"kind": "node", "type": tm.group(1), "op": int(om.group(1)) if om else None,
                           "ins": ins, "outs": outs}

    # compose call list
    comp_end = text.index("// Add all models to array to get graphsInfo")
    comp_lines = text[comp_at:comp_end].split("\n")
    calls = [CALL_RE.match(l).group(2) for l in comp_lines if CALL_RE.match(l)]
    print(f"functions {len(funcs)}, compose calls {len(calls)}")

    skip_bits = [0] * len(calls)
    aliases = {}
    report = {}
    for sub in range(1, NSUB):
        tainted, alias = set(), {}
        res = lambda n: alias.get(n, n)
        kept_node = [False] * len(calls)
        merges, dropped_nodes = 0, 0
        for i, fname in enumerate(calls):
            f = info[fname]
            if f["kind"] == "tensor":
                if feature_of(f["name"]) & sub:
                    tainted.add(f["name"])
                continue
            ins = [res(n) for n in f["ins"]]
            bad = [n in tainted for n in ins]
            if not any(bad):
                kept_node[i] = True
                continue
            out = f["outs"][0] if len(f["outs"]) == 1 else None
            if (f["type"] == "ElementWiseBinary" and f["op"] == 0 and len(ins) == 2 and sum(bad) == 1
                    and out is not None):
                clean = ins[bad.index(False)]
                if shapes.get(clean) == shapes.get(out):
                    alias[out] = clean
                    merges += 1
                    dropped_nodes += 1
                    continue
                print(f"  sub {sub}: merge-like Add {fname} shape {shapes.get(clean)} vs {shapes.get(out)} -> tainted")
            tainted.update(f["outs"])
            dropped_nodes += 1
        # consumers of each tensor among kept nodes (after aliasing)
        used = set()
        for i, fname in enumerate(calls):
            if kept_node[i]:
                used.update(res(n) for n in info[fname]["ins"])
        dropped_tensors = 0
        for i, fname in enumerate(calls):
            f = info[fname]
            drop = (f["kind"] == "node" and not kept_node[i]) or (f["kind"] == "tensor" and f["name"] not in used)
            if drop:
                skip_bits[i] |= 1 << sub
                dropped_tensors += f["kind"] == "tensor"
        assert "output" not in tainted, f"sub {sub}: the graph output is tainted"
        alias = {k: res(v) for k, v in alias.items()}
        aliases[sub] = alias
        kept_inputs = sorted({info[c]["name"] for i, c in enumerate(calls)
                              if info[c]["kind"] == "tensor" and info[c]["type"] == "QNN_TENSOR_TYPE_APP_WRITE"
                              and not (skip_bits[i] >> sub & 1)})
        report[sub] = {"merges": merges, "dropped_nodes": dropped_nodes, "dropped_tensors": dropped_tensors,
                       "inputs": len(kept_inputs)}
        print(f"sub {sub}: merges {merges}, dropped nodes {dropped_nodes}, dropped tensors {dropped_tensors}, "
              f"inputs left {len(kept_inputs)}")
    # sanity: no skipped call in sub 0, and every alias source exists
    alias_keys = set().union(*[set(a) for a in aliases.values()])

    # ---- emit
    out = []
    pos = 0
    first_func = starts[0][0]
    hdr = ["namespace tplf {",
           "inline int sub() {",
           "  static int s = -1;",
           "  if (s >= 0) return s;",
           "  s = 0;",
           "  const char* e = getenv(\"QNN_TPL_DROP\");",
           "  if (e) {",
           "    std::string v = std::string(\",\") + e + \",\";",
           "    if (v.find(\",lora,\") != std::string::npos) s |= 1;",
           "    if (v.find(\",cn,\") != std::string::npos) s |= 2;",
           "    if (v.find(\",ip,\") != std::string::npos) s |= 4;",
           "    if (v.find(\",inp,\") != std::string::npos) s |= 8;",
           "  }",
           "  fprintf(stderr, \"[tpl] feature drop mask %d (%s)\\n\", s, e ? e : \"\");",
           "  return s;",
           "}",
           "struct A { const char* from; const char* to; };"]
    for sub in range(1, NSUB):
        items = sorted(aliases[sub].items())
        body = ",\n".join(f'  {{"{k}", "{v}"}}' for k, v in items) or '  {nullptr, nullptr}'
        hdr.append(f"static const A ALIAS_{sub}[] = {{\n{body}\n}};")
    hdr.append(f"static const A* const ALIASES[{NSUB}] = {{nullptr, "
               + ", ".join(f"ALIAS_{s}" for s in range(1, NSUB)) + "};")
    hdr.append(f"static const unsigned NALIAS[{NSUB}] = {{0, "
               + ", ".join(str(len(aliases[s])) for s in range(1, NSUB)) + "};")
    hdr += ["inline const char* in(const char* n) {",
            "  static std::unordered_map<std::string, const char*>* m = nullptr;",
            "  if (!m) {",
            "    m = new std::unordered_map<std::string, const char*>();",
            "    int s = sub();",
            "    for (unsigned i = 0; i < NALIAS[s]; i++) (*m)[ALIASES[s][i].from] = ALIASES[s][i].to;",
            "  }",
            "  auto it = m->find(n);",
            "  return it == m->end() ? n : it->second;",
            "}",
            "static const unsigned short SKIP[] = {"]
    for i in range(0, len(skip_bits), 40):
        hdr.append("  " + ",".join(str(b) for b in skip_bits[i:i + 40]) + ",")
    hdr += ["};", "inline bool keep(unsigned i) { return !((SKIP[i] >> sub()) & 1); }", "}  // namespace tplf", ""]

    out.append(text[:first_func])
    out.append("\n".join(hdr))
    # function bodies: wrap aliased input names
    body_text = text[first_func:comp_at]

    def wrap_inputs(m):
        inner = re.sub(r'"([^"]*)"', lambda q: f'tplf::in("{q.group(1)}")' if q.group(1) in alias_keys else q.group(0),
                       m.group(1))
        return m.group(0).replace(m.group(1), inner)

    body_text, nwrap = INPUTS_RE.subn(wrap_inputs, body_text)
    out.append(body_text)
    # compose: guard every call
    k = 0
    new_lines = []
    for l in text[comp_at:comp_end].split("\n"):
        m = CALL_RE.match(l)
        if m:
            new_lines.append(f"{m.group(1)}if (tplf::keep({k})) VALIDATE({m.group(2)}(model), err);")
            k += 1
        else:
            new_lines.append(l)
    assert k == len(calls)
    out.append("\n".join(new_lines))
    out.append(text[comp_end:])
    open(dst, "w").write("".join(out))
    report["alias_keys"] = len(alias_keys)
    json.dump({"subsets": report, "aliases": {s: aliases[s] for s in aliases}}, open(report_path, "w"), indent=1)
    print(f"wrote {dst}: {len(calls)} guarded calls, {len(alias_keys)} aliased names")


if __name__ == "__main__":
    main(*sys.argv[1:4])
