/* Expectimax inference search over a trained n-tuple V(afterstate). NO training, NO hand-crafted
 * heuristics: leaf value = n-tuple V(afterstate) only; terminal (no legal move) = 0.
 * Player node: max_a [ r(s,a) + C(afterstate, k-1) ]; chance node C(af,k): uniform over empty cells,
 * tile 2 w.p. 0.9, 4 w.p. 0.1; C(af,0) = V(af). Depth d = number of player moves searched; d=1 == greedy
 * (uses the original choose() from common.h verbatim). Ties -> lowest action index (strict >), same as greedy.
 * Acceleration: exact transposition table (key = board, remaining depth, node type), cleared per root move.
 * No sampling -> exact expectation (no approximation at all).
 * usage: ./expectimax SNAPSHOT DEPTH SEED_START N OUT_CSV [DUMP_FILE K]
 * Corner metrics are DIAGNOSTIC ONLY (copied from eval.c), never used by the search. */
#include "common.h"

static int DEPTH = 1;
#define TT_BITS 20
typedef struct { uint64_t key; uint32_t gen; uint32_t tag; double val; } tte;
static tte *TT; static uint32_t GEN = 1; static long nodes = 0;
static inline int tt_get(board_t b, uint32_t tag, double *v) {
  uint64_t h = splitmix64(b ^ ((uint64_t)tag << 56)) & ((1u << TT_BITS) - 1);
  for (int p = 0; p < 8; p++) { tte *e = &TT[(h + p) & ((1u << TT_BITS) - 1)];
    if (e->gen != GEN) return 0; if (e->key == b && e->tag == tag) { *v = e->val; return 1; } }
  return 0;
}
static inline void tt_put(board_t b, uint32_t tag, double v) {
  uint64_t h = splitmix64(b ^ ((uint64_t)tag << 56)) & ((1u << TT_BITS) - 1);
  for (int p = 0; p < 8; p++) { tte *e = &TT[(h + p) & ((1u << TT_BITS) - 1)];
    if (e->gen != GEN || (e->key == b && e->tag == tag)) { e->key = b; e->tag = tag; e->gen = GEN; e->val = v; return; } }
  tte *e = &TT[h]; e->key = b; e->tag = tag; e->gen = GEN; e->val = v; /* overwrite: only loses speed, value stays exact */
}
static double chance(board_t af, int k);
static double maxnode(board_t s, int k) { /* k >= 1 moves remaining */
  double v; if (tt_get(s, 2 * k, &v)) return v;
  double best = 0; int any = 0;
  for (int a = 0; a < 4; a++) { uint32_t r; int mv; board_t af = do_move(s, a, &r, &mv); if (!mv) continue;
    double q = (double)r + chance(af, k - 1); if (!any || q > best) { best = q; any = 1; } }
  if (!any) best = 0; /* terminal */
  tt_put(s, 2 * k, best); return best;
}
static double chance(board_t af, int k) {
  if (k == 0) { nodes++; return (double)V(af); }
  double v; if (tt_get(af, 2 * k + 1, &v)) return v;
  double sum = 0; int n = 0;
  for (int i = 0; i < 16; i++) if (!cell(af, i)) { n++; sum += 0.9 * maxnode(set_cell(af, i, 1), k) + 0.1 * maxnode(set_cell(af, i, 2), k); }
  v = n ? sum / n : (double)V(af); /* n==0 impossible for a legal-move afterstate */
  tt_put(af, 2 * k + 1, v); return v;
}
static int search(board_t s, board_t *af_out, uint32_t *r_out) {
  if (DEPTH == 1 && !getenv("EXPECTIMAX_GENERIC")) { float q[4], vv[4]; uint32_t rr[4]; int legal[4]; return choose(s, af_out, r_out, q, rr, vv, legal); }
  GEN++; int best = -1; double bq = 0;
  for (int a = 0; a < 4; a++) { uint32_t r; int mv; board_t af = do_move(s, a, &r, &mv); if (!mv) continue;
    double q = (double)r + chance(af, DEPTH - 1);
    if (best < 0 || q > bq) { bq = q; best = a; *af_out = af; *r_out = r; } }
  return best;
}

static const int CORN[4] = {0, 3, 12, 15};
static int SNK[8][16];
static void mk_snakes(void) {
  int idx = 0;
  for (int ci = 0; ci < 4; ci++) for (int by_rows = 1; by_rows >= 0; by_rows--) {
    int r0 = CORN[ci] / 4, c0 = CORN[ci] % 4, rows[4], cols[4], n = 0;
    for (int i = 0; i < 4; i++) { rows[i] = r0 == 0 ? i : 3 - i; cols[i] = c0 == 0 ? i : 3 - i; }
    if (by_rows) { for (int k = 0; k < 4; k++) for (int j = 0; j < 4; j++) SNK[idx][n++] = rows[k] * 4 + (k % 2 == 0 ? cols[j] : cols[3 - j]); }
    else { for (int k = 0; k < 4; k++) for (int j = 0; j < 4; j++) SNK[idx][n++] = (k % 2 == 0 ? rows[j] : rows[3 - j]) * 4 + cols[k]; }
    idx++;
  }
}
static double mono_snake(board_t b) {
  double best = 0;
  for (int o = 0; o < 8; o++) { int ok = 0; for (int i = 0; i < 15; i++) ok += cell(b, SNK[o][i]) >= cell(b, SNK[o][i + 1]); if (ok / 15.0 > best) best = ok / 15.0; }
  return best;
}
static double mono_rowcol(board_t b) {
  double best = 0;
  for (int ci = 0; ci < 4; ci++) {
    int r0 = CORN[ci] / 4, c0 = CORN[ci] % 4, ok = 0;
    for (int r = 0; r < 4; r++) for (int k = 0; k < 3; k++) { int a = c0 == 0 ? r * 4 + k : r * 4 + 3 - k, bb = c0 == 0 ? r * 4 + k + 1 : r * 4 + 2 - k; ok += cell(b, a) >= cell(b, bb); }
    for (int col = 0; col < 4; col++) for (int k = 0; k < 3; k++) { int a = r0 == 0 ? k * 4 + col : (3 - k) * 4 + col, bb = r0 == 0 ? (k + 1) * 4 + col : (2 - k) * 4 + col; ok += cell(b, a) >= cell(b, bb); }
    if (ok / 24.0 > best) best = ok / 24.0;
  }
  return best;
}
static int hits(board_t b) { int m = max_tile(b), h = 0; if (m == 0) return 0; for (int k = 0; k < 4; k++) if (cell(b, CORN[k]) == m) h |= 1 << k; return h; }
static int region(board_t b) { /* 0 corner 1 edge 2 center */
  int m = max_tile(b), allc = 1, anyk = 0;
  for (int i = 0; i < 16; i++) if (cell(b, i) == m) { if (i == 0 || i == 3 || i == 12 || i == 15) anyk = 1; if (!(i == 5 || i == 6 || i == 9 || i == 10)) allc = 0; }
  return anyk ? 0 : (allc ? 2 : 1);
}
int main(int argc, char **argv) {
  const char *snap = argv[1]; DEPTH = atoi(argv[2]); argv++; argc--; int s0 = atoi(argv[2]), N = atoi(argv[3]); FILE *out = fopen(argv[4], "w");
  FILE *dump = argc > 6 ? fopen(argv[5], "w") : NULL; int K = argc > 6 ? atoi(argv[6]) : 0;
  TT = calloc((size_t)1 << TT_BITS, sizeof(tte));
  int rnd = !strcmp(snap, "random");
  init_tables(); init_net(); mk_snakes();
  if (!rnd && load_net(snap)) { fprintf(stderr, "load failed %s\n", snap); return 1; }
  fprintf(out, "seed,score,max_tile,steps,corner_rate,corner_rate_256,edge_rate_256,center_rate_256,leaves,leaves_256,leaves_per100,leaves_256_per100,boards_256,longest_streak,longest_streak_frac,final_mono_snake,final_mono_rowcol,mean_mono_snake,game_sec,ms_per_move\n");
  size_t cap = 1 << 16; board_t *B = malloc(cap * sizeof(board_t)); double t0 = now();
  for (int g = 0; g < N; g++) {
    int k = s0 + g; uint64_t rs = splitmix64(0xE7A1ULL * 1000003ULL + (uint64_t)k);
    board_t s = spawn(spawn(0, &rs), &rs); long score = 0; size_t n = 0; double tg = now();
    for (;;) {
      float q[4], vv[4]; uint32_t rr[4], r = 0; int legal[4]; board_t af = 0; int a;
      if (rnd) { int L[4], nl = 0; for (int x = 0; x < 4; x++) { int mv; uint32_t rx; do_move(s, x, &rx, &mv); if (mv) L[nl++] = x; }
        if (!nl) break; a = L[(rng_next(&rs) >> 33) % nl]; int mv; af = do_move(s, a, &r, &mv); }
      else { (void)q; (void)vv; (void)rr; (void)legal; a = search(s, &af, &r); if (a < 0) break; }
      score += r; s = spawn(af, &rs);
      if (n == cap) { cap *= 2; B = realloc(B, cap * sizeof(board_t)); }
      B[n++] = s;
    }
    if (dump && g < K) { fprintf(dump, "%d", k); for (size_t i = 0; i < n; i++) { fprintf(dump, " %016llx", (unsigned long long)B[i]); } fprintf(dump, "\n"); }
    double gsec = now() - tg;
    /* metrics */
    long inc_sum = 0, big = 0, inc_big = 0, leaves = 0, leaves256 = 0, longest = 0, st[4] = {0, 0, 0, 0}, reg[3] = {0, 0, 0}; double mm = 0; int prev_inc = 0;
    for (size_t i = 0; i < n; i++) {
      int h = hits(B[i]), inc = h != 0, m = max_tile(B[i]);
      inc_sum += inc; if (m >= 8) { big++; inc_big += inc; reg[region(B[i])]++; }
      for (int c = 0; c < 4; c++) { st[c] = (h >> c & 1) ? st[c] + 1 : 0; if (st[c] > longest) longest = st[c]; }
      if (i > 0 && prev_inc && !inc) { leaves++; if (m >= 8) leaves256++; }
      prev_inc = inc; mm += mono_snake(B[i]);
    }
    fprintf(out, "%d,%ld,%d,%zu,%.10f,", k, score, n ? 1 << max_tile(B[n - 1]) : 0, n, n ? (double)inc_sum / n : 0);
    if (big) fprintf(out, "%.10f,%.10f,%.10f,", (double)inc_big / big, (double)reg[1] / big, (double)reg[2] / big); else fprintf(out, ",,,");
    fprintf(out, "%ld,%ld,%.10f,", leaves, leaves256, n ? 100.0 * leaves / n : 0);
    if (big) fprintf(out, "%.10f,", 100.0 * leaves256 / big); else fprintf(out, ",");
    fprintf(out, "%ld,%ld,%.10f,%.10f,%.10f,%.10f\n", big, longest, n ? (double)longest / n : 0, mono_snake(B[n - 1]), mono_rowcol(B[n - 1]), mm / n);
    fseek(out, -1, SEEK_CUR); fprintf(out, ",%.4f,%.6f\n", gsec, n ? 1000.0 * gsec / n : 0); fflush(out);
  }
  fclose(out); if (dump) fclose(dump);
  fprintf(stderr, "EXPECTIMAX d=%d nodes=%ld ", DEPTH, nodes); fprintf(stderr, "EVAL %s games=%d secs=%.1f\n", snap, N, now() - t0); return 0;
}
