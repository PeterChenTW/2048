/* Official n-tuple run (see NTUPLE_PREREG.md). Afterstate TD(0), gamma=1, reward = game score,
 * alpha=0.0025, greedy (no exploration). NOTHING corner-related here.
 * Training clock = wall time minus snapshot-save time.
 * usage: ./train SEED TOTAL_SECONDS OUTDIR */
#include "common.h"
static const double SNAPS[] = {5, 15, 30, 60, 120, 240, 480, 900, 1800, 3600};
#define NSNAP (sizeof SNAPS / sizeof SNAPS[0])
int main(int argc, char **argv) {
  int seed = atoi(argv[1]); double total = atof(argv[2]); const char *out = argv[3];
  const float alpha = 0.0025f; uint64_t rs = splitmix64(0x7A11ULL * 1000003ULL + (uint64_t)seed);
  init_tables(); init_net();
  char path[512]; snprintf(path, sizeof path, "%s/train_log.csv", out); FILE *lg = fopen(path, "w");
  fprintf(lg, "train_s,wall_s,games,updates,block_games,block_mean_score,block_reach2048,block_maxtile,updates_per_s_block\n");
  double t0 = now(), paused = 0, tl = 0; long games = 0, updates = 0, bu = 0, bn = 0, b2048 = 0; double bs = 0; int bmax = 0; size_t si = 0;
#define TRAIN_T (now() - t0 - paused)
  while (si < NSNAP && SNAPS[si] <= total) {
    if (TRAIN_T >= SNAPS[si]) {
      double p0 = now(); snprintf(path, sizeof path, "%s/s%d_t%05.0f.bin", out, seed, SNAPS[si]);
      if (save_net(path)) { fprintf(stderr, "save failed %s\n", path); return 1; }
      paused += now() - p0;
      fprintf(stderr, "SNAP %s train_s=%.2f games=%ld updates=%ld\n", path, TRAIN_T, games, updates); si++; continue;
    }
    board_t s = spawn(spawn(0, &rs), &rs), prev = 0; int have = 0; long score = 0;
    for (;;) {
      float q[4], vv[4]; uint32_t rr[4], r = 0; int legal[4]; board_t af = 0;
      int a = choose(s, &af, &r, q, rr, vv, legal);
      if (a < 0) { if (have) { update(prev, alpha * (0.f - V(prev))); updates++; } break; }
      if (have) { update(prev, alpha * ((float)r + V(af) - V(prev))); updates++; }
      score += r; prev = af; have = 1; s = spawn(af, &rs);
    }
    games++; bn++; bs += score; int mt = max_tile(s); if (mt >= 11) b2048++; if (mt > bmax) bmax = mt;
    if (games % 5000 == 0) {
      double t = TRAIN_T; fprintf(lg, "%.2f,%.2f,%ld,%ld,%ld,%.1f,%.4f,%d,%.0f\n", t, now() - t0, games, updates, bn, bs / bn, (double)b2048 / bn, 1 << bmax, (updates - bu) / (t - tl));
      fflush(lg); tl = t; bu = updates; bn = 0; bs = 0; b2048 = 0; bmax = 0;
    }
  }
  double T = TRAIN_T; fprintf(lg, "# END train_s=%.2f wall_s=%.2f games=%ld updates=%ld updates_per_s=%.0f games_per_s=%.2f\n", T, now() - t0, games, updates, updates / T, games / T);
  fclose(lg); fprintf(stderr, "END train_s=%.2f games=%ld updates=%ld\n", T, games, updates); return 0;
}
