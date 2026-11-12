# JawTrack

Prototype: kæbebevægelse fra iPhone-video overført til intraorale STL-scanninger, som forberedelse til en virtuel artikulator i Bitedesign (Bitenostic).

Status: alt er kun kørt på **syntetiske data**. Intet er afprøvet på rigtige optagelser endnu. Læs `KENDTE_PROBLEMER.md`, før du stoler på et tal, og `BESLUTNINGER.md` for baggrunden.

## Formål

En tandlæge optager patientens kæbebevægelser med en iPhone og har over- og underkæbe som STL, eksporteret i okklusion. JawTrack beregner underkæbens 6-DoF-bevægelse i forhold til overkæben og leverer:

- `motion.json`: underkæbens stilling pr. billede, i overkæbe-STL'ens koordinater.
- Artikulatorparametre: sagittal kondylbanehældning, Bennett-vinkel, incisalføring, maks. gabning, protrusion og laterotrusion.
- Underkæben som STL i nøglepositioner, til import i CAD.
- En 3D-viewer, der animerer STL'erne synkront med videoen og viser kontakter og Posselt-baner.

Der er to måder at optage på:

| | Version 1: markørplader | Version 2: hudmarkører (aktuel retning) |
|---|---|---|
| På patienten | Clutch limet på 33–43 med markørplade, plus pandeplade | Seks 12 mm papirmarkører på huden, lille læbeholder |
| Optagelse | Ét klip med hele protokollen | Fire korte klip: gab, højre, venstre, frem/tilbage |
| Kommando | `track` | `track-skin` |
| Styrke | Stiv kobling til tænderne | Intet på tænderne, intet 3D-print |
| Svaghed | Ligner eksisterende produkter, duer ikke ved dybt bid | Hud følger ikke knoglen |

Begge skriver samme `raw_tracking.json`, så `solve`, `report` og vieweren er fælles.

## Mappestruktur

| Sti | Indhold |
|---|---|
| `SPEC.md` | Fælles specifikation: koordinatsystemer, matematikken bag `T_S(t)`, registrering, filformater. Læs den først. |
| `pipeline/` | Python-pakken `jawtrack` og CLI'et `jawtrack_cli.py`. Egen `README.md` med detaljer. |
| `pipeline/jawtrack/skin.py` | Version 2: sporing af hudmarkører (`track-skin`). |
| `pipeline/jawtrack/skin_synth.py` | Version 2: syntetiske klip med simuleret hudefterslæb og ende-til-ende-test. |
| `pipeline/tests/` | 12 tests: matematik, konventioner og en indbygget syntetisk ende-til-ende-test. |
| `testdata/` | Generator til syntetiske datasæt med facit (`make_synth.py`, `synthlib/`) og `validate.py`, som måler en `motion.json` mod facit. |
| `viewer/` | `index.html`: 3D-viewer i én fil. `dev/`: automatiske tests med Playwright. |
| `capture_kit/` | Version 1: markørark, kalibreringsark, 3D-printbare holdere og scripts, der genererer og kontrollerer dem. |
| `v2_hudmarkoerer/` | Version 2: markørark (frit ark og etiketark), generatorscripts og optageguide. |
| `docs/` | Optageguide til version 1 (dansk). |
| `run_synthetic_test.sh` | Kører hele den syntetiske test. |
| `KENDTE_PROBLEMER.md` | Åbne fejl og næste forsøg. |
| `BESLUTNINGER.md` | De vigtigste valg og hvorfor. |

## Opsætning

Krav: Python 3.13, `ffmpeg` i PATH. Udviklet på Linux. `pdftoppm` (poppler) bruges kun af enkelte kontrolscripts.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Installer kun `opencv-contrib-python`. Ligger `opencv-python` eller `opencv-python-headless` i samme miljø, kan `cv2.aruco` mangle.

Viewer-tests (valgfrit) kræver desuden Node 22:

```bash
pip install -r requirements-dev.txt
cd viewer/dev && npm ci && npx playwright install chromium
```

## Kør den syntetiske test fra start til slut

```bash
./run_synthetic_test.sh            # resultater i ./_out
```

Scriptet gør fire ting:

1. **Tests:** `pytest` i `pipeline/tests` (12 tests).
2. **Datasæt:** `testdata/make_synth.py --variant quick` genererer `synth_quick`: kunstige kæber som STL, en 10 sekunders 1080p-video med markørplader og bevægeligt hoved, et kalibreringsklip og facit.
3. **Version 1:** `calibrate` → `track` → `solve` → `report` → `validate.py` på `synth_quick` med 6 klikpunkter og 1,5 px klikstøj.
4. **Version 2:** `skin-selftest` genererer fire klip med seks hudmarkører, sporer dem, løser bevægelsen og sammenligner med facit. Her følger huden knoglen 100 % (`--lag 0.0`).

På 2 CPU-kerner tager det ca. 10 minutter (målt: 9 min 32 s) og fylder ca. 60 MB i `_out`.

### Hvad resultatet skal vise

Scriptet skal slutte med `FÆRDIG` og exit-kode 0. Tallene herunder er fra en kørsel 12. november 2026 i en ren mappe med et nyt miljø installeret fra `requirements.txt`.

**Trin 1.** `12 passed`.

**Trin 3, version 1 på `synth_quick`** (`_out/v1_quick/validation/validation_report.md`):

| Mål | Resultat | Facit / mål |
|---|---|---|
| Kamerakalibrering, RMS | 0,07 px | fx ≈ 1391 px |
| Registrering, RMS | 1,71 px | 6 klik med 1,5 px støj |
| Incisalpunkt, medianfejl | 0,63 mm | mål ≤ 0,3 mm |
| Incisalpunkt, 95 %-fejl | 1,45 mm | mål ≤ 0,8 mm |
| Maks. gabning | 44,4 mm | 45,0 mm |
| Maks. protrusion | 8,7 mm | 9,0 mm |
| Sagittal kondylbanehældning V / H | 30,7° / 35,3° | 35° / 38° |
| Incisalføring | 45,8° | 36,6° |

`validate.py` melder **OVERALL: FAIL**, fordi medianfejlen overskrider målet. Det er forventet og er den kendte registreringsfejl (`KENDTE_PROBLEMER.md` punkt 2). Scriptet fortsætter alligevel. `synth_quick` har ingen laterotrusion, så Bennett-vinkler og laterotrusion er tomme, og pipelinen advarer om, at de segmenter ikke blev fundet.

**Trin 4, version 2 med hudmarkører** (`_out/v2_skin/skin_selftest_result.json`). Sidste linje skal ligne:

```
SKIN SELFTEST lag=0.0: incisal error median 0.23 mm, p95 1.24 mm, tracked 100%
```

| Mål | Resultat | Facit |
|---|---|---|
| Incisalpunkt, medianfejl | 0,23 mm | |
| Incisalpunkt, 95 %-fejl (gab-klippet) | 1,53 mm | |
| Maks. gabning | 45,8 mm | 44,9 mm |
| Maks. protrusion | 7,1 mm | 7,1 mm |
| Maks. laterotrusion H / V | 7,9 / 7,8 mm | 7,7 mm |
| Sagittal kondylbanehældning V / H | 39,5° / 39,9° | 35° |
| Bennett-vinkel V / H | 14,1° / 13,2° | 10° |
| Markørernes pasform, kæbe (median) | 0,04 mm | |

Markørerne findes i alle billeder, og bevægelsesudslagene rammer inden for ca. 1 mm. Vinklerne ligger 3–5° for højt, selv når huden følger knoglen perfekt.

Tallene er deterministiske (faste seeds), men kan afvige lidt mellem maskiner og ffmpeg-versioner, fordi videoerne komprimeres.

### Flere kørsler

```bash
# Fuldt datasæt (31 s, alle bevægelser). Ca. 10 min at generere.
python3 testdata/make_synth.py --variant full --outdir _out/testdata

# Hudmarkører med 15 % efterslæb
python3 pipeline/jawtrack_cli.py skin-selftest --outdir _out/v2_lag15 --lag 0.15
```

Det fulde datasæt `synth_01` er referencen for tallene i `KENDTE_PROBLEMER.md`. Kør kæden på det ved at skifte `synth_quick` ud med `synth_01` i trin 3 i `run_synthetic_test.sh`.

## Kørsel på en rigtig optagelse

Optageguiderne: `v2_hudmarkoerer/optageguide_v2_DA.md` (hudmarkører) og `docs/optageguide_DA.md` (markørplader).

```bash
cd pipeline
# Kamera: kalibreringsklip, eller --preset iphone_main_4k uden klip
python3 jawtrack_cli.py calibrate --video charuco.mov --out work/camera.json

# Version 2: fire klip
python3 jawtrack_cli.py track-skin --clip open=gab.mov --clip right=hoejre.mov \
    --clip left=venstre.mov --clip protrusion=frem.mov \
    --camera work/camera.json --out work/raw_tracking.json

# Åbn viewer/index.html, vælg "Registrering", klik 8-10 tandpunkter på work/reg_frame.png
# og på STL'en, eksportér work/registration_landmarks.json

python3 jawtrack_cli.py solve --raw work/raw_tracking.json --segments work/segments.json \
    --upper upper.stl --lower lower.stl --landmarks work/registration_landmarks.json \
    --out work/motion.json --out-positions work/positions
python3 jawtrack_cli.py report --motion work/motion.json --outdir work/report
```

Version 1 bruger `track --video rec.mov --camera ... --out ... --reg-image reg.mov` i stedet for `track-skin` og har ikke brug for `--segments`.

Træk derefter STL'erne, `motion.json` og videoen (`work/combined_1080p.mp4` for version 2) ind i `viewer/index.html`.

## Data, der ikke hører hjemme i repoet

Repoet indeholder kun kode, dokumentation, printark og holdere. Der må ikke lægges rigtige patientscanninger, videoer eller billeder af personer i det. `.gitignore` udelukker videoer, de genererede testdata og en mappe `patientdata/`, men STL-filer og billeder fanges ikke automatisk, så hold dem uden for repoet.

Disse genererede filer er udeladt og genskabes sådan:

| Udeladt | Genskabes med |
|---|---|
| `testdata/synth_quick/`, `testdata/synth_01/` (STL, video, facit) | `python3 testdata/make_synth.py` |
| Demo-output (`motion.json`, positioner, rapport, validering) | `./run_synthetic_test.sh` |
| Syntetiske hudklip | `jawtrack_cli.py skin-selftest --outdir ...` |
| `viewer/dev/fixture/`, de fleste screenshots, testresultater | `python3 viewer/dev/make_fixture.py` og `python3 viewer/dev/test_viewer.py` |
| `viewer/dev/node_modules/` | `npm ci` i `viewer/dev/` |
| `capture_kit/verify/*.stl` (selvtest af specialtilpasset clutch) | `python3 capture_kit/make_custom_clutch.py --selftest` |
