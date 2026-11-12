# JawTrack v2 — optagelse med hudmarkører

Intet sidder på tænderne. Du skal bruge: markørarket, en saks, en lille læbe-/kindholder, iPhone på stativ og din intraorale scanner.

## 1. Print og klip
1. Print `hudmarkoerer_A4.pdf` i 100 % på mat papir (eller mat etiketark). Mål skalabjælken: 100 mm.
2. Klip ét sæt ud langs den stiplede linje. Behold den hvide kant.

## 2. Sæt markørerne på
| Markør | Sted |
|---|---|
| 20 | Næseryg (på knoglen, ikke næsetippen) |
| 21, 22 | Pande, højre og venstre, ca. 3 cm over øjenbrynene |
| 30 | Hagespids |
| 31, 32 | Ca. 4 cm fra hagen langs kæberanden, højre og venstre |

- 20–22 skal sidde på kraniet, 30–32 på underkæben. Præcis placering og rotation er ligegyldig.
- Alle seks skal kunne ses forfra af kameraet. Sæt 31 og 32 så langt fremme, at de ikke vender mere end ca. 45° væk.
- Markørerne skal ligge fladt. Prøv vaseline først. Hvis de glider, så brug dobbeltklæbende tape. Notér, hvad du brugte.
- Skæg: sæt dem, hvor de ligger mest fladt og stabilt, og notér det.

## 3. iPhone
- Bagkamera 1×, 4K, 60 fps. HDR fra, forbedret stabilisering fra, lås objektivet (ingen makroskift), AE/AF-lås på ansigtet.
- Stativ 30–40 cm foran ansigtet, i højde med munden eller lidt under. Hele området fra pande til under hagen skal være i billedet.
- Diffust lys forfra. Ingen blanke reflekser på markørerne.

## 4. Optag fire klip
Læbeholderen sidder i under alle fire klip, så tænderne kan ses. Start hvert klip med 2 sekunder i MIP og slut i MIP. Hold hovedet nogenlunde stille.

| Fil | Bevægelse |
|---|---|
| `gab.mov` | To maksimale gabebevægelser, roligt tempo |
| `hoejre.mov` | Fra MIP helt til højre og tilbage |
| `venstre.mov` | Fra MIP helt til venstre og tilbage |
| `frem.mov` | Fra MIP så langt frem som muligt, tilbage til MIP og så langt tilbage som muligt |

## 5. Kamerakalibrering (én gang pr. telefon, valgfrit)
Optag 20–30 sekunder af kalibreringsarket (`charuco_calibration_A4.pdf` fra det første kit) med samme indstillinger, mens du bevæger og vipper arket langsomt på 30–40 cm afstand. Uden dette klip bruger pipelinen standardværdier for iPhone-kameraet, hvilket er mindre præcist.

## 6. Scan og send
1. Scan over- og underkæbe, eksportér som STL i okklusion (mm).
2. Send de fire klip, de to STL'er og evt. kalibreringsklippet.
3. Du får `reg_frame.png` tilbage, klikker 8–10 tandpunkter i vieweren under "Registrering" og sender filen med punkterne retur.

## Tjekliste
- [ ] Skalabjælke målt til 100 mm
- [ ] Seks markører flade og synlige forfra
- [ ] 4K / 60 fps, HDR fra, stabilisering fra, AE/AF låst
- [ ] Læbeholder i, tænder synlige i MIP
- [ ] Hvert klip starter med 2 s i MIP
- [ ] To STL'er i okklusion
