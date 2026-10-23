#ifndef COMMON_H
#define COMMON_H
#include <stdio.h>
#include <time.h>
#include "ntuple.h"
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static int max_tile(board_t b) { int m = 0; for (int i = 0; i < 16; i++) if (cell(b, i) > m) m = cell(b, i); return m; }
static uint64_t splitmix64(uint64_t x) { x += 0x9E3779B97F4A7C15ULL; x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL; x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL; return (x ^ (x >> 31)) | 1; }
/* greedy 1-ply: argmax_a r + V(afterstate); ties -> lowest action index (0=up,1=right,2=down,3=left) */
static int choose(board_t s, board_t *af_out, uint32_t *r_out, float q[4], uint32_t rr[4], float vv[4], int legal[4]) {
  int best = -1; float bq = 0;
  for (int a = 0; a < 4; a++) {
    uint32_t r; int mv; board_t af = do_move(s, a, &r, &mv); legal[a] = mv; rr[a] = r;
    if (!mv) { vv[a] = 0; q[a] = 0; continue; }
    vv[a] = V(af); q[a] = (float)r + vv[a];
    if (best < 0 || q[a] > bq) { bq = q[a]; best = a; *af_out = af; *r_out = r; }
  }
  return best;
}
static int save_net(const char *path) {
  FILE *f = fopen(path, "wb"); if (!f) return -1;
  for (int t = 0; t < NT; t++) fwrite(LUT[t], sizeof(float), (size_t)1 << (4 * TUP_LEN[t]), f);
  return fclose(f);
}
static int load_net(const char *path) {
  FILE *f = fopen(path, "rb"); if (!f) return -1; size_t ok = 1;
  for (int t = 0; t < NT; t++) { size_t n = (size_t)1 << (4 * TUP_LEN[t]); ok &= fread(LUT[t], sizeof(float), n, f) == n; }
  fclose(f); return ok ? 0 : -2;
}
#endif
