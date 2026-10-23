#include <stdio.h>
#include <assert.h>
#include "ntuple.h"
static int fails = 0;
#define CHECK(c, msg) do { if (!(c)) { printf("FAIL: %s\n", msg); fails++; } else printf("ok: %s\n", msg); } while (0)
static board_t from_rows(const int v[16]) { board_t b = 0; for (int i = 0; i < 16; i++) b = set_cell(b, i, v[i]); return b; }
static uint16_t mkrow(int a, int b, int c, int d) { return (uint16_t)(a | b << 4 | c << 8 | d << 12); }
int main(void) {
  init_tables(); init_net();
  /* row rules (nibble exponents): 2 2 2 2 -> 4 4 . . score 8 */
  CHECK(row_left[mkrow(1,1,1,1)] == mkrow(2,2,0,0) && row_score[mkrow(1,1,1,1)] == 8, "2222 left -> 44.. score 8");
  CHECK(row_left[mkrow(1,1,2,0)] == mkrow(2,2,0,0) && row_score[mkrow(1,1,2,0)] == 4, "224. left -> 44.. (no chain merge) score 4");
  CHECK(row_left[mkrow(2,0,2,3)] == mkrow(3,3,0,0) && row_score[mkrow(2,0,2,3)] == 8, "4.48 left -> 88.. score 8");
  CHECK(row_left[mkrow(0,0,0,1)] == mkrow(1,0,0,0) && row_score[mkrow(0,0,0,1)] == 0, "...2 left -> 2... score 0");
  CHECK(row_left[mkrow(1,2,3,4)] == mkrow(1,2,3,4) && row_score[mkrow(1,2,3,4)] == 0, "2 4 8 16 no move");
  CHECK(row_left[mkrow(10,10,0,0)] == mkrow(11,0,0,0) && row_score[mkrow(10,10,0,0)] == 2048, "1024+1024 -> 2048 score 2048");
  CHECK(row_right[mkrow(1,1,1,0)] == mkrow(0,0,1,2), "222. right -> ..24 (merge from the right)");
  /* board moves */
  int g[16] = {1,0,0,1, 0,0,0,0, 1,0,0,0, 1,0,2,2};
  board_t b = from_rows(g); uint32_t sc; int mv;
  board_t l = do_move(b, 3, &sc, &mv);
  int gl[16] = {2,0,0,0, 0,0,0,0, 1,0,0,0, 1,3,0,0};
  CHECK(l == from_rows(gl) && sc == 4 + 8 && mv, "board LEFT + score 12");
  board_t u = do_move(b, 0, &sc, &mv);
  int gu[16] = {2,0,2,1, 1,0,0,2, 0,0,0,0, 0,0,0,0};
  CHECK(u == from_rows(gu) && sc == 4 && mv, "board UP + score 4");
  board_t d = do_move(b, 2, &sc, &mv);
  int gd[16] = {0,0,0,0, 0,0,0,0, 1,0,0,1, 2,0,2,2};
  CHECK(d == from_rows(gd) && sc == 4 && mv, "board DOWN + score 4");
  int full[16] = {1,2,1,2, 2,1,2,1, 1,2,1,2, 2,1,2,1};
  int any = 0; for (int a = 0; a < 4; a++) { do_move(from_rows(full), a, &sc, &mv); any |= mv; }
  CHECK(!any, "checkerboard: no legal move (terminal)");
  /* symmetry: 8 distinct maps, each a permutation; moves commute with symmetries */
  int perm_ok = 1, distinct = 1;
  for (int s = 0; s < 8; s++) { int seen[16] = {0}; for (int i = 0; i < 16; i++) seen[sym_cell(s, i)]++; for (int i = 0; i < 16; i++) perm_ok &= seen[i] == 1; }
  for (int s = 0; s < 8; s++) for (int s2 = s + 1; s2 < 8; s2++) { int same = 1; for (int i = 0; i < 16; i++) same &= sym_cell(s, i) == sym_cell(s2, i); distinct &= !same; }
  CHECK(perm_ok && distinct, "8 symmetries are distinct permutations");
  uint64_t rs = 12345; int comm = 1;
  for (int k = 0; k < 2000; k++) {
    board_t x = 0; for (int i = 0; i < 16; i++) x = set_cell(x, i, (rng_next(&rs) >> 40) % 6);
    /* mirror columns swaps left<->right, transpose swaps up<->left, down<->right */
    uint32_t s1, s2; int m1, m2;
    comm &= flip_h(do_move(x, 3, &s1, &m1)) == do_move(flip_h(x), 1, &s2, &m2) && s1 == s2;
    comm &= transpose(do_move(x, 0, &s1, &m1)) == do_move(transpose(x), 3, &s2, &m2) && s1 == s2;
    comm &= transpose(do_move(x, 2, &s1, &m1)) == do_move(transpose(x), 1, &s2, &m2) && s1 == s2;
  }
  CHECK(comm, "moves/scores commute with mirror & transpose (2000 random boards)");
  /* symmetric sampling => V invariant under all 8 symmetries (random weights) */
  for (int t = 0; t < NT; t++) { size_t n = (size_t)1 << (4 * TUP_LEN[t]); for (size_t i = 0; i < n; i++) LUT[t][i] = (float)((rng_next(&rs) >> 40) % 1000) / 100.f; }
  int inv = 1;
  for (int k = 0; k < 500; k++) {
    board_t x = 0; for (int i = 0; i < 16; i++) x = set_cell(x, i, (rng_next(&rs) >> 40) % 12);
    float v0 = V(x);
    for (int s = 0; s < 8; s++) { board_t y = 0; for (int i = 0; i < 16; i++) y = set_cell(y, sym_cell(s, i), cell(x, i)); inv &= fabsf(V(y) - v0) <= 1e-3f * fabsf(v0) + 1e-3f; }
  }
  CHECK(inv, "V(T(b)) == V(b) for all 8 symmetries T (500 random boards)");
  /* spawn: 90/10 two/four, only on empty cells */
  int c2 = 0, c4 = 0, bad = 0;
  for (int k = 0; k < 100000; k++) { board_t s = spawn(from_rows(g), &rs); int nd = 0; for (int i = 0; i < 16; i++) if (cell(s, i) != g[i]) { nd++; if (g[i]) bad++; else if (cell(s, i) == 1) c2++; else if (cell(s, i) == 2) c4++; else bad++; } bad += nd != 1; }
  CHECK(!bad && fabs(c4 / 100000.0 - 0.1) < 0.005, "spawn on empty only, P(4)~0.10");
  printf("weights=%zu (%.1f MB)\n", n_weights(), n_weights() * 4 / 1e6);
  printf(fails ? "TESTS FAILED (%d)\n" : "ALL TESTS PASSED\n", fails);
  return fails != 0;
}
