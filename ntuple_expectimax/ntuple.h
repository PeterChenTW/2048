/* Minimal n-tuple afterstate TD(0) for 2048 (Szubert & Jaskowski 2014 style).
 * Uses ONLY game rules + game score as reward. No corner/monotonicity/snake
 * features, no shaping, no search, no imitation.
 * Board: uint64, 16 nibbles, cell i (row-major, i=0 top-left) = bits 4i..4i+3,
 * nibble value k means tile 2^k (0 = empty). */
#ifndef NTUPLE_H
#define NTUPLE_H
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
typedef uint64_t board_t;

static uint16_t row_left[65536], row_right[65536];
static uint32_t row_score[65536]; /* score from moving a row left (== right of reversed) */

static inline int cell(board_t b, int i) { return (b >> (4 * i)) & 0xF; }
static inline board_t set_cell(board_t b, int i, int v) {
  return (b & ~((board_t)0xF << (4 * i))) | ((board_t)v << (4 * i));
}
static inline uint16_t rev_row(uint16_t r) {
  return (uint16_t)(((r & 0xF) << 12) | ((r & 0xF0) << 4) | ((r & 0xF00) >> 4) | (r >> 12));
}
static void init_tables(void) {
  for (int r = 0; r < 65536; r++) {
    int t[4] = {r & 0xF, (r >> 4) & 0xF, (r >> 8) & 0xF, (r >> 12) & 0xF};
    int o[4] = {0, 0, 0, 0}, n = 0; uint32_t sc = 0; int last = 0;
    for (int i = 0; i < 4; i++) {
      if (!t[i]) continue;
      if (last && last == t[i] && last < 15) { o[n - 1] = last + 1; sc += 1u << (last + 1); last = 0; }
      else { o[n++] = t[i]; last = t[i]; }
    }
    uint16_t res = (uint16_t)(o[0] | (o[1] << 4) | (o[2] << 8) | (o[3] << 12));
    row_left[r] = res; row_score[r] = sc;
    row_right[rev_row((uint16_t)r)] = rev_row(res);
  }
}
/* transpose: cell (r,c) -> (c,r) */
static inline board_t transpose(board_t b) {
  board_t o = 0;
  for (int r = 0; r < 4; r++) for (int c = 0; c < 4; c++) o = set_cell(o, c * 4 + r, cell(b, r * 4 + c));
  return o;
}
static inline board_t flip_h(board_t b) { /* mirror columns */
  board_t o = 0;
  for (int r = 0; r < 4; r++) o |= (board_t)rev_row((uint16_t)(b >> (16 * r))) << (16 * r);
  return o;
}
/* actions: 0=up 1=right 2=down 3=left. Returns afterstate, sets *score; *moved=0 if illegal */
static inline board_t do_move(board_t b, int a, uint32_t *score, int *moved) {
  board_t t = (a == 0 || a == 2) ? transpose(b) : b, o = 0; uint32_t sc = 0;
  for (int r = 0; r < 4; r++) {
    uint16_t row = (uint16_t)(t >> (16 * r));
    uint16_t nr;
    if (a == 3 || a == 0) { nr = row_left[row]; sc += row_score[row]; }
    else { nr = row_right[row]; sc += row_score[rev_row(row)]; }
    o |= (board_t)nr << (16 * r);
  }
  if (a == 0 || a == 2) o = transpose(o);
  *score = sc; *moved = (o != b); return o;
}
static inline uint64_t rng_next(uint64_t *s) { /* xorshift64* */
  uint64_t x = *s; x ^= x >> 12; x ^= x << 25; x ^= x >> 27; *s = x; return x * 2685821657736338717ULL;
}
static inline board_t spawn(board_t b, uint64_t *rs) {
  int empt[16], n = 0;
  for (int i = 0; i < 16; i++) if (!cell(b, i)) empt[n++] = i;
  if (!n) return b;
  uint64_t r = rng_next(rs);
  int pos = empt[(r >> 32) % n];
  int v = ((r & 0xFFFF) % 10 == 0) ? 2 : 1; /* 10% -> 4, 90% -> 2 (game rule) */
  return set_cell(b, pos, v);
}

/* ---- n-tuple network: Szubert & Jaskowski 2014 Fig.10 ----
 * 2 straight 4-tuples (row 0, row 1) + 2 rectangular 2x3 6-tuples,
 * each applied to all 8 board symmetries (symmetric sampling, shared weights). */
#define NT 4
static const int TUP_LEN[NT] = {4, 4, 6, 6};
static const int TUP[NT][6] = {
  {0, 1, 2, 3, -1, -1}, {4, 5, 6, 7, -1, -1},
  {0, 1, 2, 4, 5, 6}, {4, 5, 6, 8, 9, 10}};
static int SYM_TUP[8][NT][6]; /* cell indices after each symmetry */
static float *LUT[NT];

static int sym_cell(int s, int i) { /* map cell i under symmetry s (4 rotations x optional mirror) */
  int r = i / 4, c = i % 4;
  if (s & 4) c = 3 - c;           /* mirror */
  for (int k = 0; k < (s & 3); k++) { int nr = c, nc = 3 - r; r = nr; c = nc; } /* rotate 90 cw */
  return r * 4 + c;
}
static void init_net(void) {
  for (int s = 0; s < 8; s++) for (int t = 0; t < NT; t++) for (int j = 0; j < TUP_LEN[t]; j++)
    SYM_TUP[s][t][j] = sym_cell(s, TUP[t][j]);
  for (int t = 0; t < NT; t++) {
    size_t n = (size_t)1 << (4 * TUP_LEN[t]);
    LUT[t] = (float *)calloc(n, sizeof(float));
  }
}
static inline uint32_t tidx(board_t b, int s, int t) {
  uint32_t idx = 0;
  for (int j = 0; j < TUP_LEN[t]; j++) idx = (idx << 4) | cell(b, SYM_TUP[s][t][j]);
  return idx;
}
static inline float V(board_t b) {
  float v = 0;
  for (int s = 0; s < 8; s++) for (int t = 0; t < NT; t++) v += LUT[t][tidx(b, s, t)];
  return v;
}
static inline void update(board_t b, float delta) {
  for (int s = 0; s < 8; s++) for (int t = 0; t < NT; t++) LUT[t][tidx(b, s, t)] += delta;
}
static size_t n_weights(void) { size_t w = 0; for (int t = 0; t < NT; t++) w += (size_t)1 << (4 * TUP_LEN[t]); return w; }
#endif
