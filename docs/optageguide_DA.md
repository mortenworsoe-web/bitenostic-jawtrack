# JawTrack – optageguide (selvtest, prototype)

Formål: én iPhone-video af dine kæbebevægelser med to printede markørplader (pande + underkæbe), et kalibreringsklip og en registreringsframe, plus intraorale STL'er i okklusion. Ud af det beregner pipelinen underkæbens 6-DoF-bevægelse relativt til overkæben og animerer dine egne STL'er.

**Ærligt om teknikken:** iPhones LiDAR/TrueDepth-dybde er alt for grov (cm-skala, få tusinde punkter) til at opløse tænder eller sub-mm bevægelser. Prototypen bruger derfor kun 4K-RGB-kameraet + printede ArUco-markører; markørernes kendte størrelse (15,0 mm) giver metrisk skala. Positionsnøjagtigheden afhænger direkte af printskala, plan montering og en stiv clutch. TrueDepth/LiDAR-ansigtsscan er et senere trin til at lokalisere kondylerne (virtuel ansigtsbue) – i denne test bruges gennemsnitsværdier (Bonwill) eller en kinematisk hængselakse fra åbne/lukke-sekvensen.

Filer i `capture_kit/`: `markers_head_jaw.pdf`, `charuco_calibration_A4.pdf`, `jaw_clutch_paddle.stl`, `head_marker_plate.stl`.

---

## 1. Print og kontrol af skala

1. Print `markers_head_jaw.pdf` og `charuco_calibration_A4.pdf` på laserprinter, **100 % / "Faktisk størrelse"** (ikke "Tilpas til side"/"Fit"). Mat papir, gerne 120–160 g.
2. Mål den vandrette **og** lodrette skalabjælke med skydelære/stållineal: begge skal være **100,0 mm** (tolerance ±0,5 mm). Mål også én sort markør: **15,0 mm**; ét ChArUco-felt: **25,0 mm**.
3. Afviger målet: notér det. Pipelinen tager `--print-scale = målt/100` (fx 99,4 mm → 0,994). Afviger vandret og lodret forskelligt med >0,3 mm: skift printer/driver – anisotrop skalering kan ikke korrigeres.
4. Tonerkvalitet: sorte felter helt dækkende, skarpe kanter. Ingen blankt laminat (refleks). Mat laminering er OK; ellers ingen laminering.

## 2. Klargøring af plader

1. 3D-print `jaw_clutch_paddle.stl` og `head_marker_plate.stl` (se `capture_kit/README.md`: orientering, materiale). Vask og efterhærd resin fuldt ud. Fjern supportrester, så lommerne er plane.
2. Klip mærkaterne ud langs den **stiplede linje** (4 mm hvid kant er en del af markøren – må ikke klippes væk):
   - JAW (ID 10–13): 41 × 41 mm → lomme på clutch-pladen.
   - HEAD (ID 0–5): 59 × 41 mm → lomme på pandepladen.
3. Lim mærkaten **helt plant** i lommen (spraylim eller tyndt dobbeltklæbende tape over hele fladen; ingen bobler, ingen bøjede hjørner). Teksten "^ op/top" over mærkaten angiver op; pladen har et lille hak i overkanten. Orienteringen er ikke kritisk for sporingen, men gør det ens på begge.
4. Klæb ChArUco-arket plant på en stiv plade (forex, karton >2 mm, glasplade). Det må ikke bue.
5. Pandepladen: før et 20–25 mm elastikbånd igennem de to slidser i hver vinge (ind–ud–ind, som en rygsæk-spænde), så båndet holdes af friktion. Læg evt. et stykke tyndt skum bag rammen.

## 3. Fiksering af clutch (paraokklusal, labialt 33–43)

Princip: clutchen sidder kun på den **cervikale halvdel** af de labiale flader 33–43 og må aldrig røre overtænderne i MI, protrusion eller laterotrusion. Enhver fleksion eller glidning mellem clutch og tænder er direkte målefejl.

1. **Prøv uden materiale.** Baren (r ≈ 22 mm, 1,5 × 6 mm) skal ligge cervikalt, med overkant **mindst (overbid + 1,5 mm) under de nedre incisalkanter**. Ved dybt bid: slib barens overkant ned (den tåler ned til ca. 4 mm højde). Stammen skal gå ud mellem læberne uden at presse på overlæben; pladen hænger foran hagen.
2. **Isolering/tørlægning:** kindholder/bomuldsruller, rengør labialfladerne (pimpsten uden fluor), tør grundigt. Isolér gingiva let med vaseline hvis ønsket (ikke på emaljen).
3. **Fiksering, standard (let at fjerne):** ingen ætsning. Bis-akryl (fx Luxatemp/Protemp) på barens inderside (rillerne giver retention), sæt på plads med stammen i midtlinjen, pres materialet ind i de labiale embrasurer (mekanisk retention i underskæringerne). Fjern overskud mod incisalt og fra gingivalranden før afbinding. Hold stille til fuld afbinding.
4. **Hvis retentionen er utilstrækkelig:** spot-ætsning 2–3 punkter (fx 33, 31/41, 43) 15 s, bonding, flowable composite som punkt-luting mellem bar og tand. Kræver efterfølgende fjernelse med scaler/finerbor og polering.
5. **Kontrol af stivhed:** tryk let på pladen i alle retninger – der må ikke ses bevægelse af clutch relativt til tænderne.
6. **Kontrol af interferens** (artikulationsfolie / shimstock): MI, protrusion til kant-mod-kant og videre, højre og venstre laterotrusion. Ingen mærker på baren, ingen kontakt stamme–overtænder/overlæbe. Læberne skal kunne hvile let åbne omkring stammen.
7. **Fjernelse efter optagelse:** løsn med scaler/plastinstrument ved barens kant; fjern rester af bis-akryl/composite, polér.

## 4. Pandeplade

1. Båndet over øjenbrynene, pladen **højt** på panden (over os frontale, under hårgrænsen), centreret. Stramt nok til at pladen ikke vipper, ikke så stramt at det generer.
2. **Undgå mimik** under optagelsen: ingen rynken, ingen løftede øjenbryn – panden huden flytter sig flere mm og det tolkes som overkæbebevægelse. Hold ansigtet neutralt og blikket roligt på et punkt over kameraet.
3. Pladen skal være synlig for kameraet og ikke dækket af hår.

## 5. iPhone-indstillinger

Indstillingsnavne er fra Apples iPhone-brugerhåndbog til iOS 26 (engelsk UI). Danske navne er min oversættelse og kan afvige lidt – tjek på telefonen.

1. **Indstillinger › Kamera › Optag video (Record Video):** **4K ved 60 fps.**
2. Samme menu: **HDR-video (HDR Video): fra.**
3. **Forbedret stabilisering (Enhanced Stabilization): fra** (den beskærer og flytter billedet dynamisk → ødelægger kamerakalibreringen).
4. **Lås kamera (Lock Camera): til** (forhindrer objektivskift under optagelse).
5. **Auto-FPS / Auto Low Light FPS: fra** (konstant 60 fps). **Lås hvidbalance (Lock White Balance): til.**
6. **Makrostyring (Macro Control)** i Indstillinger › Kamera (kun Pro-modeller): slå til, så makro-knappen vises, og sørg for at makro er **slået fra** i kamera-appen; alternativt slå automatisk makro fra. (Ikke bekræftet i iOS 26-håndbogen – navnet kan afvige.)
7. **Bevar indstillinger (Preserve Settings):** slå til for Kreativ kontrol/Makro, så indstillingerne ikke nulstilles mellem klip.
8. I kamera-appen, Video: **Action-tilstand fra** (løbende-mand-ikonet ikke gult), **1×** (hovedkamera, ingen zoom), **liggende format**.
9. **AE/AF-lås:** placer dig (eller ChArUco-pladen) på ~35 cm, tryk og hold på ansigtet/pladen til **"AE/AF LÅS"** vises. Træk eksponering lidt ned (½ trin), så det hvide papir ikke brænder ud. Skal låses igen hver gang kamera-appen har været lukket.
10. Format: "Høj effektivitet" (HEVC) er fint. Frigør lagerplads (4K60 ≈ 400 MB/min).

## 6. Opstilling og lys

1. Telefon på stativ, **30–40 cm** foran ansigtet, linsen lidt under mundhøjde, let skrå opad. Både pandeplade og kæbeplade skal være i billedet ved maks. gab (tjek før optagelse: gab maks. og se at JAW-pladen stadig er inde).
2. Sid med ryg-/nakkestøtte, så hovedet ikke bevæger sig ud af billedet (hovedbevægelse i sig selv er OK – den trækkes fra – men skarphed og billedudsnit skal holdes).
3. **Diffust, jævnt lys forfra/skråt forfra** (vindue med gardin, softbox, to lamper). Ingen modlys, ingen spotlys der giver refleks i markørerne. Tjek på skærmen at der ikke er blanke pletter på markørerne ved hovedbevægelse.
4. Start/stop optagelse: Apple Watch-fjernbetjening, kablede høretelefoners volumenknap, eller en hjælper. Undgå at røre telefonen mellem klip (ændrer fokus/position).

## 7. Kalibreringsklip (ChArUco) – samme indstillinger

1. Samme telefon, samme indstillinger, 1×, 4K60, AF/AE-lås på ChArUco-pladen ved ~35 cm (telefonen bliver på stativet).
2. Bevæg pladen **langsomt** gennem hele billedet på **30–40 cm**: centrum og alle fire hjørner, vip den ±30–45° om begge akser, lidt nærmere/fjernere. **20–30 s.**
3. Ingen bevægelsesuskarphed (langsomt!), pladen skal være plan, fingre væk fra markørerne.
4. Optag gerne kalibreringen **lige før** selve optagelsen, uden at ændre noget derimellem.

## 8. Optagelse af bevægelser (SPEC afsnit 2)

Én sammenhængende video. Hver sekvens starter og slutter i MI. Tæl stille, bevæg dig langsomt og jævnt.

1. **MI 3 s** – tænderne fast sammen.
2. **Åbne/lukke ×3** – langsomt, maksimal gabning.
3. **Protrusion ×3** – til kant-mod-kant og videre, tilbage til MI.
4. **Højre laterotrusion ×3** – tilbage til MI hver gang.
5. **Venstre laterotrusion ×3.**
6. **Fri tygning 5 s** (tom tygning, evt. paraffin-/tyggegummi-imitation uden at røre clutchen).
7. **MI 3 s.**

Undgå at læberne eller tungen rører kæbepladen. Hold panden neutral.

## 9. Registreringsframe

1. Kindholder i, **tænderne i MI**, begge plader synlige, tænderne skarpe og godt belyste (incisiver, hjørnetænder, gerne præmolarer/første molar synlig).
2. Optag som **kort video (3–5 s) med de samme indstillinger** – ikke fotoapp: fotos har anden opløsning/beskæring/objektivkorrektion end videoen, og kalibreringen gælder kun videoen. Pipelinen udvælger den skarpeste frame.
3. Hold hovedet stille og telefonen urørt. Kindholderen må ikke trykke på clutch eller pandeplade.

## 10. Intraoral scanning

1. Scan overkæbe, underkæbe (med clutchen **fjernet** – eller scan før fiksering) og bukkal bidregistrering i MI.
2. Eksportér **overkæbe og underkæbe hver for sig, i okklusion** (samme koordinatsystem, "aligned to bite"), STL, mm. Ingen "flyt til origo"/separat placering.
3. Valgfrit, nyttigt til senere automatisk registrering: en ekstra underkæbe-scan **med clutchen siddende** (anteriort område + stamme).

## 11. Filer der skal sendes

| Fil | Indhold |
|---|---|
| `calib_charuco.mov` | Kalibreringsklip 20–30 s |
| `jawmotion.mov` | Bevægelsessekvens (pkt. 8) |
| `registration.mov` | Registrerings-klip med kindholder (pkt. 9) |
| `upper.stl`, `lower.stl` | Overkæbe/underkæbe i okklusion, mm |
| `lower_with_clutch.stl` | Valgfrit (pkt. 10.3) |
| `notes.txt` | Målt skalabjælke (vandret/lodret, mm), iPhone-model, iOS-version, overbid/overjet, evt. afvigelser |

Overfør originalfiler (AirDrop / "Indstillinger: Alle fotodata" / kabel) – **ikke** via Beskeder/mail, der komprimerer video.

## 12. Fejlfinding

| Problem | Løsning |
|---|---|
| Markører detekteres ikke / flimrer | Mere diffust lys, eksponering ned, fjern refleks (mat), tjek skarphed (AF-lås på ~35 cm), større nærhed. |
| JAW-plade ude af billedet ved maks. gab | Telefon længere væk (40 cm) eller lavere; zoom aldrig. |
| Uskarpe markører under bevægelse | Mere lys (kortere lukketid), langsommere bevægelser. |
| Clutch løsner sig/vipper | Fjern, rens, gentag med spot-ætsning + flowable; tjek at baren ikke rammer overtænder. |
| Kontakt mellem bar og overtænder | Slib barens overkant ned, eller flyt baren mere cervikalt; gentag foliekontrol. |
| Pandepladen flytter sig | Strammere bånd, højere placering, neutralt ansigt; alternativt fastgør pladen til en brille-/visirstel. |
| Skalabjælke ≠ 100,0 mm | Print igen i 100 %, eller angiv `--print-scale`. |
| Telefonen skifter objektiv/zoomer | Lås kamera til, makro fra, 1×, afstand ≥ 25 cm. |

---

## Tjekliste (1 side)

**Før**
- [ ] Printet 100 %; skalabjælke vandret ___ mm, lodret ___ mm; markør 15,0 mm
- [ ] Mærkater klippet langs stiplet linje (4 mm hvid kant), limet plant i lommerne
- [ ] ChArUco limet plant på stiv plade
- [ ] Pandeplade på bånd, høj placering, neutral pande
- [ ] Clutch: cervikalt 33–43, fikseret, stiv, ingen interferens i MI/protrusion/laterotrusion (folie)
- [ ] iPhone: 4K/60, HDR-video fra, Forbedret stabilisering fra, Lås kamera til, Auto-FPS fra, Lås hvidbalance til, Action-tilstand fra, makro fra, 1×, liggende
- [ ] Stativ 30–40 cm, lidt under mundhøjde; begge plader synlige ved maks. gab
- [ ] Diffust lys, ingen refleks i markørerne, AE/AF-LÅS, eksponering −½

**Optagelse**
- [ ] Kalibreringsklip ChArUco 20–30 s, langsomt, hjørner + vip
- [ ] Bevægelser: MI 3 s → åbne/lukke ×3 → protrusion ×3 → højre ×3 → venstre ×3 → tygning 5 s → MI 3 s
- [ ] Registreringsklip: kindholder, MI, begge plader, skarpt, 3–5 s video

**Efter**
- [ ] Clutch fjernet, rester fjernet, poleret
- [ ] Intraoral scan: OK + UK + bid; eksport i okklusion, STL, mm
- [ ] Send: `calib_charuco.mov`, `jawmotion.mov`, `registration.mov`, `upper.stl`, `lower.stl`, `notes.txt`
