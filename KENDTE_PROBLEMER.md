# Kendte problemer

Status 12. november 2026. Alt er kun kørt på syntetiske data. Intet er afprøvet på rigtige optagelser, rigtige scanner-STL'er eller i en mund.

## 1. Bennett-vinkel venstre er forkert (version 1, synth_01)

**Symptom.** På det fulde syntetiske sæt `synth_01` (1080p, 6 klik med 1,5 px støj, registreringsfinjustering slået til) måler pipelinen Bennett-vinkel venstre til 19,7° mod facit 10,0°. Højre side er 9,6° mod 8,0°. Uden finjustering er venstre ca. 6° forkert.

**Hvad vi ved.**
- Der findes to definitioner af Bennett-vinklen i koden:
  - `pipeline/jawtrack/params.py` (`_bennett`): vinkel i horisontalplanet for den orbiterende kondyls bane, tilpasset over et stykke af banen.
  - `testdata/synthlib/gtdefs.py`: vinklen aflæst ved præcis 5 mm lateral forskydning af incisalpunktet (`LATERAL_RECORD_MM`), median over gentagelser.
- Når `validate.py` genberegner vinklen fra pipelinens bevægelse med sin egen definition, lander begge sider inden for 0,4° af facit efter finjustering. Selve bevægelsen er altså nogenlunde rigtig, og fejlen ligger i, hvordan `params.py` udleder vinklen af den.
- Forskellen mellem de to definitioner er ikke forklaret.

**Næste forsøg.**
1. Lad `params.py` og `gtdefs.py` bruge samme funktion, så der kun er én definition.
2. Kør `params.py` direkte på `motion_true.json`. Giver det 10°/8°, er definitionen i orden og fejlen kommer fra støj. Giver det noget andet, er det en regulær fejl i `_bennett`.
3. Tjek, hvilke billeder der indgår i tilpasningen: de få billeder med store fejl (problem 3) kan trække linjen skæv.

## 2. Registreringen er den største fejlkilde (begge versioner)

**Symptom.** Koblingen mellem STL og markører (X i `SPEC.md` afsnit 3–4) laves ud fra klik på tandpunkter i ét billede. Med 6 klik er X dårligt bestemt i én retning (dybde kombineret med tilt): ca. 5–6 mm usikkerhed dér, under 0,4 mm i de andre retninger. Resultat på syntetiske data med 1,5 px klikstøj:

| Datasæt | Finjustering | Incisal median | Incisal 95 % | Gabning 95 % | Maks. gabning (facit 45,0) |
|---|---|---|---|---|---|
| synth_quick | fra | 0,63 mm | 4,52 mm | 4,67 mm | 49,2 mm |
| synth_quick | til | 0,63 mm | 1,45 mm | 1,68 mm | 44,4 mm |
| synth_01 | fra | 0,53 mm | 1,28 mm | 3,64 mm | 43,8 mm |
| synth_01 | til | 0,50 mm | 1,12 mm | 2,17 mm | 46,2 mm |

**Hvad vi ved.**
- `solve --refine-registration` (i `registration_refine.py`) søger langs den svage retning og sammenligner STL'ens tandkanter med kanter i billedet. Den hjælper på de store fejl, men fjerner ikke fejlen på maks. gabning i `synth_01`.
- De syntetiske tænder er fladt skraverede med rene kanter. På rigtig emalje, gingiva og spyt må man forvente mindre gevinst.
- En lokal kant-tilpasning blev prøvet og droppet: omkostningen var flad langs den svage retning, og minimum lå ikke ved facit.

**Næste forsøg.**
1. 8–10 klik spredt over begge kæber inkl. molarer, og 4K i stedet for 1080p.
2. Flere billeder med forskellig hovedstilling i samme registrering (parallakse bestemmer dybden).
3. Dybdedata fra iPhonens TrueDepth-kamera, som netop måler den svage retning.
4. Sporing direkte på de synlige tænder i hvert billede (se problem 5).

## 3. Enkelte billeder med op til 8 mm fejl (version 1, synth_01)

**Symptom.** I `synth_01` har nogle få billeder under gabning fejl på op til ca. 8 mm og 10° på incisalpunktet. De ses som punkter over MI i Posselt-plottet.

**Hvad vi ved.** `synth_01` simulerer korte delvise tildækninger af en markør (fx en læbe over et hjørne). Det er den sandsynlige årsag, men det er ikke undersøgt. Tracking meldte begge plader fundet i alle 1861 billeder, så fejlen bliver ikke fanget som ugyldigt billede.

**Næste forsøg.**
1. Find de konkrete billeder via `validate.py`'s fejl pr. billede, og se hvilke markørhjørner der indgår.
2. Kassér enkeltmarkører med høj reprojektionsfejl før PnP (RANSAC over markører).
3. Afvis spring i tid: et billede, hvis stilling afviger markant fra naboerne, markeres ugyldigt og interpoleres.

## 4. Hudmodellens antagelser (version 2)

**Symptom/begrænsning.** `skin_synth.py` simulerer hud ved, at hver kæbemarkør "halter" en fast andel (`lag`) bagefter knoglens bevægelse, med forskellig vægt pr. markør (`LAG_WEIGHT`). Det er en antagelse, ikke en måling. Rigtig hud strækkes, trækkes af læber og mentalis, og opfører sig anderledes med læbeholder og i skæg.

Syntetiske resultater (1080p, 30 fps, fire klip, incisalpunktets fejl):

| Hud følger knoglen | Klikstøj | Median | 95 % ved gabning | Maks. gabning (facit 44,9) |
|---|---|---|---|---|
| 100 % | 0 px | 0,22 mm | 1,35 mm | 46,2 mm |
| 95 % | 1 px | 0,23 mm | 0,79 mm | 44,2 mm |
| 85 % | 1 px | 0,31 mm | 4,55 mm | 41,1 mm |

**Hvad vi ved.**
- Ved 15 % efterslæb måles gabningen knap 4 mm for kort, og Bennett-vinklen falder fra 10° til 2–3°.
- Kvalitetsmålet `skin_deformation` (hvor dårligt de tre markører passer som stift legeme) reagerer næsten ikke på efterslæbet, fordi alle tre markører flytter sig samlet. Samlet glid kan altså ikke ses på markørerne selv.
- Hovedmarkørerne er modelleret som helt stive mod kraniet. Pandehud bevæger sig også.

**Næste forsøg.**
1. Én rigtig optagelse med både hudmarkører og clutch, så hudens faktiske bevægelse kan måles (clutchens plade er facit).
2. Kør samme optagelse med 6, 4 og 2 markører i beregningen for at se, hvor få der er nok.
3. Brug tænderne som facit (problem 5).

## 5. Ikke bygget endnu

- **Sporing direkte på tænderne.** Match af STL mod de synlige tænder i hvert billede (med læbeholder). Det er den del, der skal gøre hudmarkører præcise nok. Udvikles på første rigtige optagelse.
- **Tandkontakt som betingelse.** Under protrusion og laterotrusion glider tænderne på hinanden. Kravet "kontakt uden gennemtrængning" mellem de to STL'er er ikke implementeret.
- **Retrusion.** Klippet "frem/tilbage" mærkes som protrusion i `segments.json`. Der beregnes ingen retrusionsværdi.

## 6. Incisalføringen afviger mellem de to definitioner (version 1, synth_quick)

**Symptom.** I kørslen på `synth_quick` rapporterer `motion.json` incisalføring 45,8° mod facit 36,6°. Når `validate.py` genberegner den fra samme bevægelse med sin egen definition, giver den 37,2°.

**Hvad vi ved.** Samme mønster som Bennett-vinklen (punkt 1): bevægelsen er nogenlunde rigtig, men `params.py` og `gtdefs.py` udleder parameteren forskelligt. `params.py` bruger incisalbanen for 0,3–3,5 mm fremføring.

**Næste forsøg.** Løs det sammen med punkt 1 ved at samle parameterdefinitionerne ét sted og teste dem på `motion_true.json`.

## 7. Mindre ting

- **4K er ikke kørt på de fulde syntetiske sæt.** Kun pipelinens egen test er kørt i 4K (0,12 mm median mod 0,33 mm i 1080p ved præcis registrering).
- **Version 2 kører med `refine_registration="off"`** i selvtesten, fordi de syntetiske hudklip ikke har tænder i billedet.
- **Kamera uden kalibrering.** `calibrate --preset` bruger antagne iPhone-værdier (26 mm-ækvivalent, ingen forvrængning). Effekten på præcisionen er ikke målt.
- **Vieweren** er kun testet i headless Chromium/Chrome. Ikke i Safari, på iPad/iPhone eller med rigtigt grafikkort. Automatisk orientering (PCA) og gennemtrængningsfarver er kun prøvet på genererede STL'er.
- **Vieweren som publiceret side:** eksportknapperne (STL, landmarks-JSON) virker kun, når `viewer/index.html` åbnes lokalt.
- **Vieweren og fire klip:** `track-skin` skriver en samlet `combined_1080p.mp4`, men det er ikke testet i vieweren.
- **iPhone-indstillinger i guiderne:** "Macro Control", "Preserve Settings" og de danske menunavne er ikke bekræftet mod en telefon.
- **HERMA 10107-arket:** layoutet (6 × 11, pitch 27,94 × 25,4 mm) er et gæt. Det skal tjekkes mod et rigtigt ark med `testprint_med_rammer.pdf`.
- **Clutch og pandeplade** er aldrig printet. Den specialtilpassede clutch (`make_custom_clutch.py`) er kun prøvet på en genereret kæbe.
- **Kondyler** er gennemsnitsværdier (Bonwill). Vinkler er målt mod okklusionsplanet, ikke Frankfort.
- **Regulatorisk (MDR)** er ikke afklaret. Software, der måler kæbebevægelse til diagnostik, kan falde under regel 11.
- **Version 2, vinkler ved perfekt hud:** i selvtesten ligger kondylbanehældningen 4–5° og Bennett-vinklen 3–4° for højt, selv uden hudefterslæb (1080p, 30 fps, 1 px klikstøj). Årsagen er ikke skilt ad i tracking og registrering.
