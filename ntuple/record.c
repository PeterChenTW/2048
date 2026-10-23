/* Record one greedy eval game (same rng/policy as eval.c) as JSON for the demo.
 * usage: ./record SNAPSHOT EVAL_SEED LABEL > out.json
 * frame: g=board hex (16 nibbles, cell0 first), s=score before move, a=chosen action (0 up,1 right,2 down,3 left),
 * q[a]=[r, V(afterstate), r+V] or null if illegal. final: board after last spawn. */
#include "common.h"
static void hexb(board_t b, char *o) { for (int i = 0; i < 16; i++) o[i] = "0123456789abcdef"[cell(b, i)]; o[16] = 0; }
int main(int argc, char **argv) {
  int k = atoi(argv[2]); init_tables(); init_net();
  if (load_net(argv[1])) { fprintf(stderr, "load failed\n"); return 1; }
  uint64_t rs = splitmix64(0xE7A1ULL * 1000003ULL + (uint64_t)k);
  board_t s = spawn(spawn(0, &rs), &rs); long score = 0, n = 0; char h[17];
  printf("{\"label\":\"%s\",\"seed\":%d,\"frames\":[", argv[3], k);
  for (;;) {
    float q[4], vv[4]; uint32_t rr[4], r = 0; int legal[4]; board_t af = 0;
    int a = choose(s, &af, &r, q, rr, vv, legal);
    if (a < 0) break;
    hexb(s, h); printf("%s{\"g\":\"%s\",\"s\":%ld,\"a\":%d,\"q\":[", n ? "," : "", h, score, a);
    for (int x = 0; x < 4; x++) { if (x) printf(","); if (legal[x]) printf("[%u,%.2f,%.2f]", rr[x], vv[x], q[x]); else printf("null"); }
    printf("]}"); score += r; s = spawn(af, &rs); n++;
  }
  hexb(s, h); printf("],\"final\":{\"g\":\"%s\",\"s\":%ld},\"score\":%ld,\"steps\":%ld,\"max_tile\":%d}\n", h, score, score, n, 1 << max_tile(s));
  return 0;
}
