#!/usr/bin/env python3
"""Headless test of viewer/index.html with Playwright (Chromium, SwiftShader WebGL).

The shipped index.html loads three.js from jsdelivr. The sandbox may block the CDN, so this
script writes dev/index.test.html with the importmap rewritten to dev/node_modules (test only),
serves viewer/ on localhost and runs logic + UI checks, saving screenshots to viewer/screenshots/.

Usage: python3 dev/test_viewer.py [--data DIR] [--prefix NAME] [--chrome]
"""
import argparse, functools, http.server, json, math, os, re, socketserver, struct, sys, threading
from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
VIEWER = os.path.dirname(HERE)
SHOTS = os.path.join(VIEWER, 'screenshots')
RESULTS = []

def check(name, ok, detail=''):
    RESULTS.append((name, bool(ok), detail))
    print(('PASS ' if ok else 'FAIL ') + name + ('  ' + str(detail) if detail else ''), flush=True)

def make_test_html():
    src = open(os.path.join(VIEWER, 'index.html'), encoding='utf-8').read()
    src = src.replace('https://cdn.jsdelivr.net/npm/three@0.186.0/', '/dev/node_modules/three/')
    src = src.replace('https://cdn.jsdelivr.net/npm/three-mesh-bvh@0.9.16/', '/dev/node_modules/three-mesh-bvh/')
    assert '/dev/node_modules/three/build' in src
    open(os.path.join(HERE, 'index.test.html'), 'w', encoding='utf-8').write(src.replace('<title>', '<title>[TEST] '))

def serve():
    h = functools.partial(http.server.SimpleHTTPRequestHandler, directory=VIEWER)
    class Q(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *a): pass
    h = functools.partial(Q, directory=VIEWER)
    socketserver.TCPServer.allow_reuse_address = True
    srv = socketserver.ThreadingTCPServer(('127.0.0.1', 0), h)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv.server_address[1]

def read_stl_bin(b):
    n = struct.unpack_from('<I', b, 80)[0]
    out = []
    for i in range(n):
        o = 84 + i * 50 + 12
        out.append(struct.unpack_from('<9f', b, o))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(HERE, 'fixture'))
    ap.add_argument('--prefix', default='')
    ap.add_argument('--chrome', action='store_true', help='use /usr/bin/google-chrome (H.264 support)')
    ap.add_argument('--motion', default='motion.json')
    a = ap.parse_args()
    os.makedirs(SHOTS, exist_ok=True)
    make_test_html()
    port = serve()
    D = a.data
    f = lambda n: os.path.join(D, n)
    pre = a.prefix
    is_fix = os.path.exists(f('lower_ascii.stl'))
    mj = json.load(open(f(a.motion)))
    segd = {sg['label']: sg for sg in mj['segments']}
    dur = mj['frames'][-1]['t']
    with sync_playwright() as p:
        kw = dict(args=['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--autoplay-policy=no-user-gesture-required'])
        if a.chrome: kw['executable_path'] = '/usr/bin/google-chrome'
        b = p.chromium.launch(**kw)
        ctx = b.new_context(viewport={'width': 1512, 'height': 900}, device_scale_factor=1, accept_downloads=True)
        pg = ctx.new_page()
        errs = []
        pg.on('pageerror', lambda e: errs.append(str(e)))
        pg.on('console', lambda m: errs.append('console.' + m.type + ': ' + m.text) if m.type == 'error' else None)
        pg.goto(f'http://127.0.0.1:{port}/dev/index.test.html')
        pg.wait_for_function('window.JT && window.JT.ready', timeout=30000)
        pg.wait_for_timeout(500)
        pg.screenshot(path=os.path.join(SHOTS, pre + '01_empty.png'))

        # ---------- pure logic ----------
        r = pg.evaluate('''() => {
          const {matFromRowMajor, matToRowMajor} = JT;
          // Rz(90°) + translation (1,2,3), row-major
          const T = [0,-1,0,1, 1,0,0,2, 0,0,1,3, 0,0,0,1];
          const m = matFromRowMajor(T);
          const THREE_V = m.elements; // column-major
          const p = [1,0,0]; // apply manually using row-major definition
          const ref = [T[0]*p[0]+T[1]*p[1]+T[2]*p[2]+T[3], T[4]*p[0]+T[5]*p[1]+T[6]*p[2]+T[7], T[8]*p[0]+T[9]*p[1]+T[10]*p[2]+T[11]];
          const e = THREE_V; const got = [e[0]*1+e[12], e[1]*1+e[13], e[2]*1+e[14]];
          return {ref, got, elements: Array.from(e), back: matToRowMajor(m), nested: Array.from(matFromRowMajor([[0,-1,0,1],[1,0,0,2],[0,0,1,3],[0,0,0,1]]).elements), bad: matFromRowMajor([1,2,3])};
        }''')
        check('row-major T -> Matrix4: translation lands in elements[12..14]', r['elements'][12:15] == [1, 2, 3], r['elements'])
        check('row-major T -> Matrix4: point transform matches row-major math', all(abs(x - y) < 1e-9 for x, y in zip(r['ref'], r['got'])) and r['got'] == [1, 3, 3], r['got'])
        check('Matrix4 -> row-major round trip', r['back'] == [0, -1, 0, 1, 1, 0, 0, 2, 0, 0, 1, 3, 0, 0, 0, 1])
        check('nested 4x4 accepted, malformed rejected', r['nested'] == r['elements'] and r['bad'] is None)

        roles = pg.evaluate('''() => {
          const d = JT.detectRole; return {
            a: d('upper.stl'), b: d('Patient_Overkaebe.stl'), c: d('maxillary_scan.STL'), d: d('lower.stl'), e: d('UnderKæbe.stl'), f: d('mandibular.stl'),
            g: d('Scan_OK.stl'), h: d('scan_UK.stl'), i: d('scan.stl'), j: d('motion.json', {frames:[{T:[]}]}), k: d('x.json', {points:[]}),
            l: d('ground_truth.json', {frames:[], parameters:{}}), m: d('IMG_0042.MOV'), n: d('reg_frame.png'), o: d('book.json', {foo:1}), p: d('video.mp4'),
            q: d('registration_landmarks.json', {version:1, points:[]}), r: d('broken.json'), s: d('raw_tracking.json', {frames:[{i:0, head:null, jaw:null}]}), u: d('motion_true.json', {frames:[{t:0,T:[1]}]})
          }}''')
        exp = dict(a='upper', b='upper', c='upper', d='lower', e='lower', f='lower', g='upper', h='lower', i='stl', j='motion', k='landmarks',
                   l='truth', m='video', n='image', o='ignore', p='video', q='landmarks', r='ignore', s='ignore', u='motion')
        bad = {k: (roles[k], v) for k, v in exp.items() if roles[k] != v}
        check('file-role detection (20 cases)', not bad, bad or '')

        th = pg.evaluate('() => [-0.2,-0.06,-0.05,0,0.05,0.051,0.3,0.31,1.0,1.01,Infinity,null].map(JT.distClass)')
        check('distance colour thresholds', th == [0, 0, 1, 1, 1, 2, 2, 3, 3, 4, 4, 4], th)

        # ---------- load fixture ----------
        files = [f('upper.stl'), f('lower.stl'), f(a.motion)]
        if os.path.exists(f('video.mp4')) and a.chrome: files.append(f('video.mp4'))
        elif os.path.exists(f('video.webm')): files.append(f('video.webm'))
        elif os.path.exists(f('video.mp4')): files.append(f('video.mp4'))
        pg.set_input_files('#filePick', files)
        pg.wait_for_function('JT.S.upper && JT.S.lower && JT.S.motion && JT.S.axes', timeout=120000)
        pg.wait_for_timeout(800)
        info = pg.evaluate('''() => ({roles: JT.S.files.map(f => [f.file.name, f.role, f.status]), ax: JT.S.axes && ['L','U','A'].map(k => JT.S.axes[k].toArray().map(x=>+x.toFixed(3))),
            nU: JT.S.upper.nv, nL: JT.S.lower.nv, tris: JT.S.upper.tris})''')
        check('files loaded with detected roles', all(r[2] == 'ok' for r in info['roles']), info['roles'])
        print('  PCA axes (L,U,A):', info['ax'])
        if not a.data.endswith('fixture'):
            pass
        else:
            L, U, A = info['ax']
            check('PCA orientation recovers fixture frame (L≈+x, U≈+y, A≈+z)', L[0] > 0.9 and U[1] > 0.9 and A[2] > 0.9, info['ax'])
        check('STL welded (vertices < 3*tris)', info['nU'] < info['tris'] * 3, f"{info['nU']} verts / {info['tris']} tris")

        # STL export round trip at a protrusion pose
        t_test = segd['protrusion']['t0'] + 0.35 * (segd['protrusion']['t1'] - segd['protrusion']['t0']) if 'protrusion' in segd else dur / 2
        rt = pg.evaluate(f'''() => {{ JT.setPlaying(false); JT.setTime({t_test}); const pose = JT.currentPose(); const dv = JT.exportLowerSTL();
            const bytes = Array.from(new Uint8Array(dv.buffer, dv.byteOffset, dv.byteLength));
            const parsed = JT.parseSTL(new Uint8Array(bytes).buffer); const g = JT.S.lower.geom; const p = g.attributes.position.array, ix = g.index.array;
            // first 3 triangles from the geometry, transformed
            const e = pose.M.elements; const ref = []; for (let t=0;t<3*3;t++) {{ const i = ix[t]*3; const x=p[i],y=p[i+1],z=p[i+2];
              ref.push(e[0]*x+e[4]*y+e[8]*z+e[12], e[1]*x+e[5]*y+e[9]*z+e[13], e[2]*x+e[6]*y+e[10]*z+e[14]); }}
            return {{count: parsed.count, tris: ix.length/3, first: Array.from(parsed.positions.slice(0,27)), ref, fmt: parsed.format, T: JT.matToRowMajor(pose.M), i: pose.i}}; }}''')
        maxerr = max(abs(x - y) for x, y in zip(rt['first'], rt['ref']))
        check('STL export round-trip (binary, transformed by current T)', rt['count'] == rt['tris'] and rt['fmt'] == 'binær' and maxerr < 1e-4, f"tris={rt['count']} maxerr={maxerr:.2e}")
        # compare with motion.json raw T (row-major) for that frame
        fr = mj['frames'][rt['i']]
        if fr.get('T'):
            check('pose used = motion.json row-major T of nearest frame', max(abs(x - y) for x, y in zip(fr['T'], rt['T'])) < 1e-5)

        # ASCII STL parse == binary parse
        if is_fix:
            asc = pg.evaluate('''async () => { const [a,b] = await Promise.all(['/dev/fixture/lower_ascii.stl','/dev/fixture/lower.stl'].map(u=>fetch(u).then(r=>r.arrayBuffer())));
                const pa = JT.parseSTL(a), pb = JT.parseSTL(b); let m = 0; for (let i=0;i<pa.positions.length;i++) m = Math.max(m, Math.abs(pa.positions[i]-pb.positions[i]));
                return {fa: pa.format, fb: pb.format, n: [pa.count,pb.count], m}; }''')
            check('ASCII STL parses identically to binary', asc['fa'] == 'ASCII' and asc['n'][0] == asc['n'][1] and asc['m'] < 1e-4, asc)

        # ---------- contacts in MI (paused, full res) ----------
        def wait_full():
            pg.wait_for_function('JT.CT.sig && !JT.CT.job && !JT.CT.dirty', timeout=120000)
        pg.evaluate('() => { JT.setTime(0.5); JT.requestRender(); }')
        wait_full()
        c_mi = pg.evaluate('() => ({c: [0,1,2,3].map(k => document.querySelector("#cnt"+k).textContent), st: document.querySelector("#contactStatus").textContent, ms: JT.CT.lastFullMs, n: JT.CT.lower.cand.length})')
        print('  contacts MI:', c_mi)
        nc = int(c_mi['c'][1].replace('.', '') or 0)
        check('contacts present in MI', nc > 0, c_mi)
        pg.evaluate('() => JT.setView("front", false)')
        pg.wait_for_timeout(300)
        pg.screenshot(path=os.path.join(SHOTS, pre + '03_contacts_MI.png'))
        pg.evaluate('() => { JT.setView("occLower", false); JT.requestRender(); }'); pg.wait_for_timeout(400)
        pg.screenshot(path=os.path.join(SHOTS, pre + '04_contacts_occlusal.png'))
        pg.evaluate('() => { JT.setView("front", false); }')
        # max opening -> no contacts
        seg = segd['open_close']
        t_open = seg['t0'] + (seg['t1'] - seg['t0']) / 6
        pg.evaluate(f'() => {{ JT.setTime({t_open}); JT.requestRender(); }}')
        wait_full()
        c_open = pg.evaluate('() => [0,1,2,3].map(k => document.querySelector("#cnt"+k).textContent)')
        check('no contacts at max opening', c_open[0] in ('0', '–') and c_open[1] in ('0', '–'), c_open)
        pg.screenshot(path=os.path.join(SHOTS, pre + '02_loaded_open.png'))

        # ---------- playback (fast contact mode) ----------
        pg.evaluate(f'() => {{ JT.setTime({dur * 0.6}); JT.setPlaying(true); }}')
        pg.wait_for_timeout(2500)
        pb = pg.evaluate('() => ({t: JT.S.t, playing: JT.S.playing, st: document.querySelector("#contactStatus").textContent})')
        check('playback advances time and uses fast contact mode', pb['t'] > dur * 0.6 + 0.1 and 'hurtig' in pb['st'], pb)
        pg.screenshot(path=os.path.join(SHOTS, pre + '05_playing_laterotrusion.png'))
        pg.evaluate('() => JT.setPlaying(false)')
        # segment chip loop
        pg.click('.chip:has-text("Protrusion")')
        pg.wait_for_timeout(1200)
        lp = pg.evaluate('() => ({loop: JT.S.loopSeg && JT.S.loopSeg.label, t: JT.S.t})')
        ps = segd['protrusion']
        check('segment chip jumps to and loops segment', lp['loop'] == 'protrusion' and ps['t0'] <= lp['t'] <= ps['t1'], lp)
        pg.evaluate('() => JT.setPlaying(false)')
        # invalid frame
        inv = next((i for i, fr in enumerate(mj['frames']) if not fr['valid']), None)
        if inv is not None:
            pg.evaluate(f'() => JT.setTime({mj["frames"][inv]["t"]})'); pg.wait_for_timeout(300)
            hud = pg.inner_text('#hud')
            check('invalid frame flagged in HUD and pose held', 'Ugyldigt' in hud, hud.replace('\n', ' | '))
        # video sync
        if len(files) > 3:
            tv = round(dur * 0.45, 2); pg.evaluate(f'() => JT.setTime({tv})'); pg.wait_for_timeout(1200)
            vt = pg.evaluate('() => { const v = document.querySelector("#video"); return {ct: v.currentTime, rs: v.readyState, off: JT.S.motion.offset, err: v.error && v.error.code}; }')
            check('video synced: currentTime = t + video_offset_s', vt['rs'] >= 1 and abs(vt['ct'] - (tv + vt['off'])) < 0.05, vt)
            pg.click('[data-off="0.1"]'); pg.wait_for_timeout(800)
            vt2 = pg.evaluate('() => document.querySelector("#video").currentTime')
            check('offset nudge +0,1 s moves video', abs(vt2 - (tv + vt['off'] + 0.1)) < 0.05, vt2)
            pg.click('[data-off="-0.1"]')
        pg.evaluate(f'() => {{ JT.setTime({ps["t0"] + 1.2}); JT.setView("right", false); }}');
        pg.wait_for_timeout(300); wait_full()
        pg.screenshot(path=os.path.join(SHOTS, pre + '06_protrusion_lateral_video.png'))
        pg.evaluate('() => JT.setView("sagittal", false)'); pg.wait_for_timeout(400)
        pg.screenshot(path=os.path.join(SHOTS, pre + '07_sagittal_section.png'))
        pg.evaluate('() => JT.setView("front", false)')

        # ---------- ground truth compare ----------
        if os.path.exists(f('ground_truth.json')):
            pg.set_input_files('#filePick', [f('ground_truth.json')])
            pg.wait_for_function('JT.S.truth', timeout=20000); pg.wait_for_timeout(500)
            txt = pg.inner_text('#paramTable')
            check('ground truth: estimated vs true table shown', 'Facit' in txt and 'Δ' in txt, txt[:120].replace('\n', ' | '))
            pg.check('#visGhost') if pg.is_visible('#visGhost') else None
            pg.evaluate(f'() => JT.setTime({t_open})'); pg.wait_for_timeout(500)
            pg.screenshot(path=os.path.join(SHOTS, pre + '08_compare_truth.png'))

        # dark theme
        pg.click('#btnTheme'); pg.evaluate('() => JT.setTime(3.2)'); pg.wait_for_timeout(300); wait_full()
        pg.screenshot(path=os.path.join(SHOTS, pre + '09_dark.png'))
        pg.click('#btnTheme')

        # summary + exports via UI download
        with pg.expect_download() as dl:
            pg.click('#btnSummary')
        sp = os.path.join(HERE, 'out_summary.txt'); dl.value.save_as(sp)
        check('summary text export', 'Maks. gabning' in open(sp, encoding='utf-8').read())
        with pg.expect_download() as dl:
            pg.click('#btnExportStl')
        stlp = os.path.join(HERE, 'out_lower.stl'); dl.value.save_as(stlp)
        sb = open(stlp, 'rb').read()
        check('STL download is valid binary STL', len(sb) == 84 + 50 * struct.unpack_from('<I', sb, 80)[0], f'{len(sb)} bytes')
        with pg.expect_download() as dl:
            pg.click('#btnPng')
        pngp = os.path.join(HERE, 'out_shot.png'); dl.value.save_as(pngp)
        check('PNG screenshot export', open(pngp, 'rb').read(8) == b'\x89PNG\r\n\x1a\n')

        # ---------- registration ----------
        pg.click('#tabReg'); pg.wait_for_timeout(300)
        if os.path.exists(f('reg_frame.png')) and is_fix:
            pg.set_input_files('#filePick', [f('reg_frame.png')])
            pg.wait_for_function('JT.S.reg.image', timeout=20000); pg.wait_for_timeout(300)
            # zoom to a non-trivial scale and pan so that display scale != 1
            pg.evaluate('() => { const iv = JT.iv; iv.s *= 2.3; iv.ox -= 700; iv.oy -= 420; }')
            pg.click('#imgFit'); pg.click('#imgZoomIn'); pg.click('#imgZoomIn')
            # find the red dots from the RENDERED canvas pixels (independent of our mapping)
            dots = pg.evaluate('''() => { const c = document.querySelector('#imgcanvas'); const ctx = c.getContext('2d'); const d = ctx.getImageData(0,0,c.width,c.height).data; const W=c.width,H=c.height;
                const dpr = c.width / c.clientWidth; const seen = new Uint8Array(W*H); const blobs=[];
                for (let y=0;y<H;y++) for (let x=0;x<W;x++) { const i=(y*W+x)*4; if (seen[y*W+x] || !(d[i]>200 && d[i+1]<60 && d[i+2]<60)) continue;
                  const st=[[x,y]]; seen[y*W+x]=1; let sx=0,sy=0,n=0; while(st.length){const [a,b]=st.pop(); sx+=a; sy+=b; n++; for (const [u,v] of [[a+1,b],[a-1,b],[a,b+1],[a,b-1]]) { if(u<0||v<0||u>=W||v>=H||seen[v*W+u]) continue; const j=(v*W+u)*4; if (d[j]>200&&d[j+1]<60&&d[j+2]<60){seen[v*W+u]=1; st.push([u,v]);} } }
                  if (n>20) blobs.push([(sx/n+0.5)/dpr, (sy/n+0.5)/dpr, n]); }
                return {blobs, s: JT.iv.s}; }''')
            reg = json.load(open(f('registration_landmarks.json')))
            print('  visible dots:', len(dots['blobs']), 'scale', dots['s'])
            box = pg.locator('#imgcanvas').bounding_box()
            clicked = []
            for bx, by, n in dots['blobs'][:4]:
                pg.mouse.click(box['x'] + bx, box['y'] + by)
                clicked.append((bx, by))
                # click on STL: use a fixed STL point via API (raycast checked separately)
                pg.wait_for_timeout(80)
                pg.evaluate('() => JT.regClickSTL(new (JT.S.upper.mesh.position.constructor)(1,2,3), "upper")')
            # raycast click on the 3D canvas centre
            pg.evaluate('() => JT.setView("front", false)'); pg.wait_for_timeout(300)
            cbox = pg.locator('#view3d canvas').bounding_box()
            # click the upper incisors region: project known STL point to screen
            sp3 = pg.evaluate('''() => { const v = new (JT.S.upper.mesh.position.constructor)(2.5, -3.0, 0.5); v.applyMatrix4(JT.S.upper.mesh.matrixWorld);
                 const cam = JT.S.upper.mesh.parent.parent.children.find(o => o.isCamera); v.project(cam); const c = document.querySelector('#view3d canvas'); return [(v.x+1)/2*c.clientWidth, (1-v.y)/2*c.clientHeight]; }''')
            pg.click('#regIncisal')
            pg.mouse.click(cbox['x'] + sp3[0], cbox['y'] + sp3[1])
            pg.wait_for_timeout(200)
            inc = pg.evaluate('() => JT.S.reg.incisal && JT.S.reg.incisal.toArray()')
            check('raycast click on STL sets lower incisal point (near the clicked incisal region)', inc is not None and abs(inc[0] - 2.5) < 2.5 and abs(inc[1] + 3) < 3, inc)
            out = pg.evaluate('() => JT.buildLandmarksJson()')
            # match each exported img_px to nearest true dot
            errs_px = []
            for pt in out['points']:
                u, v = pt['img_px']
                dmin = min(math.hypot(u - q['img_px'][0], v - q['img_px'][1]) for q in reg['points'])
                errs_px.append(round(dmin, 2))
            check('img_px exported in ORIGINAL image pixels at zoom != 1 (error vs drawn dot centres < 1 px)', len(errs_px) >= 3 and max(errs_px) < 1.0, f'scale={dots["s"]:.3f} errors={errs_px}')
            check('export schema per SPEC 4', set(out) == {'version', 'image', 'points', 'incisal_point_mm', 'occlusal_plane_mm'} and out['image']['width'] == reg['image']['width'] and set(out['points'][0]) == {'name', 'img_px', 'stl_mm', 'jaw'}, out['image'])
            pg.screenshot(path=os.path.join(SHOTS, pre + '10_registration_clicking.png'))
            # import existing landmarks for review
            pg.set_input_files('#filePick', [f('registration_landmarks.json')])
            pg.wait_for_timeout(600)
            pg.click('#imgFit')
            n_imp = pg.evaluate('() => JT.S.reg.points.length')
            check('import registration_landmarks.json for review', n_imp == len(reg['points']), n_imp)
            rt2 = pg.evaluate('() => JT.buildLandmarksJson()')
            check('landmarks import -> export round trip identical', rt2['points'] == reg['points'] and rt2['incisal_point_mm'] == reg['incisal_point_mm'] and rt2['image']['frame_index'] == reg['image']['frame_index'])
            pg.wait_for_timeout(300)
            pg.screenshot(path=os.path.join(SHOTS, pre + '11_registration_review.png'))
            # export via UI
            with pg.expect_download() as dl:
                pg.click('#regExport')
            lp2 = os.path.join(HERE, 'out_landmarks.json'); dl.value.save_as(lp2)
            check('registration_landmarks.json download', json.load(open(lp2))['version'] == 1)
        elif os.path.exists(f('reg_frame.png')):
            pg.set_input_files('#filePick', [f('reg_frame.png'), f('registration_landmarks.json')])
            pg.wait_for_function('JT.S.reg.image && JT.S.reg.points.length', timeout=30000); pg.wait_for_timeout(500)
            reg = json.load(open(f('registration_landmarks.json')))
            rt2 = pg.evaluate('() => JT.buildLandmarksJson()')
            check('synth landmarks import -> export round trip', [q['img_px'] for q in rt2['points']] == [q['img_px'] for q in reg['points']] and [q['stl_mm'] for q in rt2['points']] == [q['stl_mm'] for q in reg['points']])
            pg.click('#imgFit'); pg.wait_for_timeout(300)
            pg.screenshot(path=os.path.join(SHOTS, pre + '11_registration_review.png'))
        pg.click('#tabAna'); pg.wait_for_timeout(300)

        # ---------- mobile / iPad layouts ----------
        pg.set_viewport_size({'width': 820, 'height': 1180}); pg.wait_for_timeout(600)
        pg.screenshot(path=os.path.join(SHOTS, pre + '12_ipad_portrait.png'))
        pg.set_viewport_size({'width': 390, 'height': 844}); pg.wait_for_timeout(600)
        sw = pg.evaluate('() => document.documentElement.scrollWidth')
        check('phone width: no horizontal page scroll', sw <= 390, sw)
        pg.screenshot(path=os.path.join(SHOTS, pre + '13_phone.png'))

        check('no page errors', not errs, errs[:5])
        b.close()
    n_fail = sum(1 for r in RESULTS if not r[1])
    print(f'\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed')
    json.dump([{'check': r[0], 'ok': r[1], 'detail': str(r[2])} for r in RESULTS], open(os.path.join(HERE, f'test_results{("_" + pre.strip("_")) if pre else ""}.json'), 'w'), indent=1, ensure_ascii=False)
    sys.exit(1 if n_fail else 0)

if __name__ == '__main__':
    main()
