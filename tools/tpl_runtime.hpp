// Runtime weight loading for a patched qnn-onnx-converter model.cpp.
// On-device converter MVP, Phase 2 (docs/ON-DEVICE-CONVERT.md).
//
// tpl_patch.py rewrites every STATIC tensor's BINVARSTART/BINLEN and its
// quantizeParams to call these functions, so the compiled libmodel.so carries
// the graph TOPOLOGY and activation encodings only. Weights and their
// encodings come from a pack file named by $QNN_TPL_PACK at compose time.
//
// Pack layout (little-endian), written by tpl_pack.py / the on-device tool:
//   "TPLPACK1" | u32 count | count x entry | data region
//   entry: u16 name_len, name, u32 npairs, npairs x (f32 scale, i32 offset),
//          u64 data_offset (absolute), u64 data_len
//
// ⚠ Strict by design: a name missing from the pack, or a per-axis count that
// differs from the graph's, aborts the compose. Silently falling back to anything
// would build a model with the wrong weights that still loads and runs.
#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "QnnTypes.h"

namespace tpl {

struct Entry {
  std::vector<Qnn_ScaleOffset_t> pairs;
  uint64_t off = 0, len = 0;
};

struct Pack {
  const uint8_t* base = nullptr;
  size_t size = 0;
  std::unordered_map<std::string, Entry> map;
  bool loaded = false;
};

inline Pack& pack() {
  static Pack p;
  if (p.loaded) return p;
  p.loaded = true;
  const char* path = getenv("QNN_TPL_PACK");
  if (!path) { fprintf(stderr, "[tpl] QNN_TPL_PACK not set\n"); abort(); }
  int fd = open(path, O_RDONLY);
  if (fd < 0) { fprintf(stderr, "[tpl] cannot open %s\n", path); abort(); }
  struct stat st;
  fstat(fd, &st);
  p.size = (size_t)st.st_size;
  void* m = mmap(nullptr, p.size, PROT_READ, MAP_PRIVATE, fd, 0);
  close(fd);
  if (m == MAP_FAILED) { fprintf(stderr, "[tpl] mmap failed\n"); abort(); }
  p.base = (const uint8_t*)m;
  if (p.size < 12 || memcmp(p.base, "TPLPACK1", 8) != 0) { fprintf(stderr, "[tpl] bad magic\n"); abort(); }
  const uint8_t* c = p.base + 8;
  auto rd = [&](void* dst, size_t n) { memcpy(dst, c, n); c += n; };
  uint32_t count; rd(&count, 4);
  for (uint32_t i = 0; i < count; i++) {
    uint16_t nl; rd(&nl, 2);
    std::string name((const char*)c, nl); c += nl;
    Entry e;
    uint32_t np; rd(&np, 4);
    e.pairs.resize(np);
    for (uint32_t k = 0; k < np; k++) {
      float s; int32_t o; rd(&s, 4); rd(&o, 4);
      e.pairs[k].scale = s; e.pairs[k].offset = o;
    }
    rd(&e.off, 8); rd(&e.len, 8);
    if (e.off + e.len > p.size) { fprintf(stderr, "[tpl] %s out of range\n", name.c_str()); abort(); }
    p.map.emplace(std::move(name), std::move(e));
  }
  fprintf(stderr, "[tpl] pack %s: %u tensors, %zu bytes\n", path, count, p.size);
  return p;
}

inline Entry& get(const char* name) {
  auto& m = pack().map;
  auto it = m.find(name);
  if (it == m.end()) { fprintf(stderr, "[tpl] missing tensor %s\n", name); abort(); }
  return it->second;
}

inline void* data(const char* name) { auto& e = get(name); return (void*)(pack().base + e.off); }
inline uint32_t len(const char* name) { return (uint32_t)get(name).len; }

inline Qnn_ScaleOffset_t* axis(const char* name, uint32_t n) {
  auto& e = get(name);
  if (e.pairs.size() != n) {
    fprintf(stderr, "[tpl] %s: pack has %zu axis encodings, graph wants %u\n", name, e.pairs.size(), n);
    abort();
  }
  return e.pairs.data();
}

inline Qnn_ScaleOffset_t scalar(const char* name) {
  auto& e = get(name);
  if (e.pairs.size() != 1) { fprintf(stderr, "[tpl] %s: expected 1 scalar encoding\n", name); abort(); }
  return e.pairs[0];
}

}  // namespace tpl
