/* Greedy eval (no learning) + corner DIAGNOSTICS (same definitions as corner/corner_metrics.py).
 * Diagnostic only. usage: ./eval SNAPSHOT|random SEED_START N OUT_CSV [DUMP_BOARDS_FILE K]
 * Game with eval seed k uses rng = splitmix64(0xE7A1*1000003 + k). Boards s_1..s_n = post-spawn after each move. */
#include "common.h"
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
  const char *snap = argv[1]; int s0 = atoi(argv[2]), N = atoi(argv[3]); FILE *out = fopen(argv[4], "w");
  FILE *dump = argc > 6 ? fopen(argv[5], "w") : NULL; int K = argc > 6 ? atoi(argv[6]) : 0;
  int rnd = !strcmp(snap, "random");
  init_tables(); init_net(); mk_snakes();
  if (!rnd && load_net(snap)) { fprintf(stderr, "load failed %s\n", snap); return 1; }
  fprintf(out, "seed,score,max_tile,steps,corner_rate,corner_rate_256,edge_rate_256,center_rate_256,leaves,leaves_256,leaves_per100,leaves_256_per100,boards_256,longest_streak,longest_streak_frac,final_mono_snake,final_mono_rowcol,mean_mono_snake\n");
  size_t cap = 1 << 16; board_t *B = malloc(cap * sizeof(board_t)); double t0 = now();
  for (int g = 0; g < N; g++) {
    int k = s0 + g; uint64_t rs = splitmix64(0xE7A1ULL * 1000003ULL + (uint64_t)k);
    board_t s = spawn(spawn(0, &rs), &rs); long score = 0; size_t n = 0;
    for (;;) {
      float q[4], vv[4]; uint32_t rr[4], r = 0; int legal[4]; board_t af = 0; int a;
      if (rnd) { int L[4], nl = 0; for (int x = 0; x < 4; x++) { int mv; uint32_t rx; do_move(s, x, &rx, &mv); if (mv) L[nl++] = x; }
        if (!nl) break; a = L[(rng_next(&rs) >> 33) % nl]; int mv; af = do_move(s, a, &r, &mv); }
      else { a = choose(s, &af, &r, q, rr, vv, legal); if (a < 0) break; }
      score += r; s = spawn(af, &rs);
      if (n == cap) { cap *= 2; B = realloc(B, cap * sizeof(board_t)); }
      B[n++] = s;
    }
    if (dump && g < K) { fprintf(dump, "%d", k); for (size_t i = 0; i < n; i++) { fprintf(dump, " %016llx", (unsigned long long)B[i]); } fprintf(dump, "\n"); }
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
  }
  fclose(out); if (dump) fclose(dump);
  fprintf(stderr, "EVAL %s games=%d secs=%.1f\n", snap, N, now() - t0); return 0;
}
