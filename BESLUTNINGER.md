# Beslutninger

De vigtigste valg i JawTrack og begrundelsen for dem. Nyeste øverst.

## 1. Fra clutch på tænderne til hudmarkører (12. november 2026)

**Valg.** Version 2 bruger seks små papirmarkører (12 mm) på huden: tre på kraniet (næseryg, pande) og tre på underkæben (hage, kæberand). Intet sidder på tænderne. Det eneste udstyr er en lille læbeholder. Version 1 (clutch på 33–43 og pandeplade) bliver liggende i repoet som reference og som facit ved validering.

**Hvorfor.** Mortens vurdering som kliniker og produktejer:
- En clutch ligner de produkter, der allerede er på markedet (Modjaw, Zebris). Det gør det svært at vinde nye kunder.
- Ved dybt bid kan en markør ikke sidde fast på tænderne, eller den bliver bidt af i sammenbid.
- Noget på tænderne ændrer patientens bevægelsesmønster.
- Kunden skal kunne nøjes med en almindelig printer og en læbeholder.

**Pris.** Hud følger ikke knoglen. Den syntetiske test viser, at 15 % efterslæb giver knap 4 mm fejl på gabningen, og at glid ikke kan ses på markørerne selv (se `KENDTE_PROBLEMER.md`, punkt 4). Hudmarkører alene forventes derfor ikke at være præcise nok til kondylbanehældning og kontakter.

**Konsekvens.** Planen er at lade tænderne være den egentlige reference: STL'en matches mod de synlige tænder i hvert billede, og tandkontakt bruges som betingelse. Hudmarkørerne giver så startgæt og robusthed. Den del er ikke bygget endnu.

## 2. Markørernes geometri beregnes ud fra videoen (version 2)

**Valg.** Hudmarkørernes indbyrdes placering er ukendt og estimeres med bundle adjustment over billeder fra klippene (`skin.py`). Kun gruppen har betydning: ID 20–22 på kraniet, 30–32 på underkæben.

**Hvorfor.** Brugeren skal kunne sætte markørerne på uden skabelon eller opmåling. Kæbekonstellationens form fastlægges ud fra MIP-holdet i starten af hvert klip, fordi registreringen sker i MIP.

## 3. Fire korte klip i stedet for én lang optagelse (version 2)

**Valg.** Gab, højre, venstre og frem/tilbage optages som hver sit klip, hvert startende med 2 sekunder i MIP. `track-skin` lægger dem på én tidslinje og skriver `segments.json` ud fra klipnavnene.

**Hvorfor.** Det er Mortens forslag til protokol og nemmere for patienten. Det fjerner også behovet for automatisk segmentering, som var en fejlkilde i version 1. Hvert klip har sin egen MIP-reference.

## 4. MIP-hold i starten af hvert klip i stedet for separat foto

**Valg.** Registreringsbilledet tages fra videoen (skarpeste billede i første MIP-hold), ikke fra et separat foto.

**Hvorfor.** Et iPhone-foto har et andet billedudsnit og andre kameraparametre end video. Kamerakalibreringen gælder kun videotilstanden. Det blev opdaget under arbejdet med optagekittet til version 1.

## 5. RGB-kamera og printede markører, ikke LiDAR

**Valg.** Prototypen bruger iPhonens 4K-bagkamera og ArUco-markører af kendt størrelse. Markørstørrelsen giver den metriske skala.

**Hvorfor.** LiDAR er for grov til at opløse tænder. TrueDepth-kameraet på forsiden er bedre på 30 cm og måler netop den retning, registreringen er svagest i, men det kræver en lille app til optagelsen. Det er derfor udskudt til et senere trin.

## 6. Registrering med klik på tandpunkter

**Valg.** STL og markører kobles ved, at brugeren klikker de samme 6–10 tandpunkter i et billede og på STL'en i vieweren (`SPEC.md` afsnit 4). `solve` kan derefter finjustere mod tandkanter i billedet.

**Hvorfor.** Det er den enkleste metode uden ekstra udstyr (ingen scanbody, ingen ansigtsscanning). Det viste sig at være den største fejlkilde, se `KENDTE_PROBLEMER.md` punkt 2.

## 7. Artikulatorparametre frem for kontakter billede for billede

**Valg.** Målet er parametre til en virtuel artikulator (kondylbanehældning, Bennett-vinkel, incisalføring, maks. udslag) og nøglepositioner som STL. Kontakter i vieweren er en visualisering, ikke et måleresultat.

**Hvorfor.** Parametre er robuste over for støj på ca. 1 mm. Kontakter kræver ca. 0,1 mm, som ingen af versionerne leverer.

## 8. Kondyler fra gennemsnitsværdier

**Valg.** Kondylpunkterne placeres ud fra Bonwill-trekanten (ca. 105 mm) i okklusionsplanets koordinatsystem. `solve --condyles kinematic` kan i stedet lægge dem på en beregnet hængselakse. Vinkler angives mod okklusionsplanet.

**Hvorfor.** Videoen viser ikke kondylerne, og Frankfort-planet er ikke tilgængeligt uden ansigtsscanning. En virtuel ansigtsbue via TrueDepth er et muligt næste trin.

## 9. Syntetiske data med facit før rigtige optagelser

**Valg.** Hver del valideres først mod genererede data med kendt facit (`testdata/`, `pipeline/jawtrack/synthetic.py`, `skin_synth.py`). Højre og venstre side har bevidst forskellige værdier (35°/38° og 10°/8°).

**Hvorfor.** Så kan fejl måles i millimeter og grader, og fejlkilder kan skilles ad (tracking, registrering, hud). Begrænsningen er, at syntetiske billeder er renere end rigtige.

## 10. Alt kører lokalt

**Valg.** Vieweren er én HTML-fil, hvor filer behandles i browseren. Pipelinen er et Python-CLI.

**Hvorfor.** Patientdata (video af ansigt, scanninger) skal ikke forlade klinikkens computer i prototypen. Repoet indeholder derfor kun syntetiske data, og `.gitignore` udelukker videoer.

## 11. Fælles specifikation og filformater

**Valg.** `SPEC.md` fastlægger koordinatsystemer, matematik og filformaterne `raw_tracking.json`, `motion.json` og `registration_landmarks.json`. Version 2 skriver samme `raw_tracking.json` som version 1, så `solve`, `report` og vieweren er uændrede.

**Hvorfor.** Delene blev bygget parallelt og skulle passe sammen uden koordinering undervejs.

## 12. Bitedesign-positionering

**Valg.** JawTrack er tænkt som supplement til Bitedesign: et planlægningsværktøj, hvor teknikeren stadig overfører og finjusterer i CAD/CAM.

**Hvorfor.** Det holder kravene til præcision og regulatorisk klassificering nede i første omgang. MDR-spørgsmålet er dog ikke afklaret for bevægelsesmåling (se `KENDTE_PROBLEMER.md` punkt 7).
