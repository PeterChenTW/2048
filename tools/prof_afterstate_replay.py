import cProfile, pstats, sys, torch
sys.path.insert(0, '.'); sys.path.insert(0, 'tools')
torch.set_num_threads(1)
import bench_afterstate_expand as b
for mode in ('reference', 'vectorized'):
    agent = b.create_agent(mode, seed=3)
    b.fill_replay_buffer(agent, mode, agent.config.memory_capacity, seed=3)
    for _ in range(5): agent.replay()
    pr = cProfile.Profile(); pr.enable()
    for _ in range(150): agent.replay()
    pr.disable()
    st = pstats.Stats(pr); tot = st.total_tt
    print(f'== {mode}: 150 replays, total {tot:.2f}s ({1000*tot/150:.1f} ms/update)')
    rows = []
    for (f, l, n), (cc, nc, tt, ct, callers) in st.stats.items():
        rows.append((ct, n, f.split('/')[-1]))
    seen = set()
    for ct, n, f in sorted(rows, reverse=True):
        key = (n, f)
        if any(k in n for k in ('_targets','batch_afterstates','afterstate_scores','backward','forward','get','update','_score_expanded','step','clip_grad')) and key not in seen and ct/tot > 0.02:
            seen.add(key); print(f'  {100*ct/tot:5.1f}%  {n}  ({f})')
