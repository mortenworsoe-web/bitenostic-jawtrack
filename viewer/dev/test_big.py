#!/usr/bin/env python3
"""Load 2 x ~100 MB binary STLs (dev/big, from make_fixture.py --res 14) and measure
load time, longest main-thread stall and contact timings. Headless SwiftShader = CPU rendering,
so timings are pessimistic compared to a real Mac/iPad GPU."""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_viewer as T
T0 = time.time()
from playwright.sync_api import sync_playwright
T.make_test_html(); port = T.serve()
D = os.path.join(T.HERE, 'big')
with sync_playwright() as p:
    b = p.chromium.launch(args=['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'])
    pg = b.new_page(viewport={'width': 1200, 'height': 760})
    errs = []; pg.on('pageerror', lambda e: errs.append(str(e))); pg.on('console', lambda m: print(round(time.time()-T0,1), 'console:', m.text[:150], flush=True))
    pg.goto(f'http://127.0.0.1:{port}/dev/index.test.html'); pg.wait_for_function('window.JT && JT.ready')
    pg.evaluate('''() => { window.__gaps = []; let last = performance.now(); const f = (t) => { window.__gaps.push(t - last); last = t; requestAnimationFrame(f); }; requestAnimationFrame(f); }''')
    t0 = time.time()
    pg.set_input_files('#filePick', [os.path.join(D, 'upper.stl'), os.path.join(D, 'lower.stl'), os.path.join(D, 'motion.json')])
    pg.wait_for_timeout(1500)
    pg.screenshot(path=os.path.join(T.SHOTS, '14_big_loading_progress.png'))
    pg.wait_for_function('JT.S.upper && JT.S.lower && JT.S.motion && JT.S.axes', timeout=600000, polling=1000)
    t_load = time.time() - t0
    print('loaded', t_load, flush=True)
    gaps = pg.evaluate('() => { const g = window.__gaps.slice(); g.sort((a,b)=>b-a); return g.slice(0,6).map(Math.round); }')
    pg.evaluate('() => { JT.setPlaying(false); JT.setTime(0.5); JT.requestRender(); }')
    t1 = time.time()
    pg.wait_for_function('JT.CT.sig && !JT.CT.job && !JT.CT.dirty', timeout=600000, polling=1000)
    t_full = time.time() - t1
    info = pg.evaluate('() => ({nv: JT.S.upper.nv, tris: JT.S.upper.tris, cand: JT.CT.lower.cand.length, fast: JT.CT.lower.fast.reps.length, ms: JT.CT.lastFullMs, c: [0,1,2,3].map(k=>document.querySelector("#cnt"+k).textContent)})')
    pg.evaluate('() => { JT.setTime(20); JT.setPlaying(true); }'); pg.wait_for_timeout(4000)
    st = pg.evaluate('() => document.querySelector("#contactStatus").textContent'); pg.evaluate('() => JT.setPlaying(false)')
    pg.screenshot(path=os.path.join(T.SHOTS, '15_big_loaded.png'))
    res = dict(load_s=round(t_load, 1), longest_frame_gaps_ms=gaps, full_contacts_s=round(t_full, 1), info=info, playback_status=st, errors=errs)
    print(json.dumps(res, ensure_ascii=False, indent=1))
    json.dump(res, open(os.path.join(T.HERE, 'test_results_big.json'), 'w'), indent=1, ensure_ascii=False)
    b.close()
