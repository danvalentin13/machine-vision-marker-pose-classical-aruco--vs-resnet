# Task 3 - camera invariance (invariance_1)

1499 of 1499 frames with both markers (F not held). The markers stood still; the laptop was moved.

| quantity | camera motion (^C T_F) | relative pose (^F T_M) |
|---|---|---|
| position change | span 126 mm (F seen at 38-50 cm) | RMS jitter 2.7 mm, max deviation 11.4 mm |
| rotation change | up to 11.0 deg | RMS 0.94 deg, max 3.8 deg |

- Relative position of M in F (mean): x -276.8, y 0.1, z 7.3 mm; distance between the centres 276.9 mm (std 0.4).
- Expected distance between the centres (measured by hand): 280.0 mm -> difference -3.1 mm (-1.1 %).
- **Camera invariance: shown: the camera moved 126 mm / 11 deg while ^F T_M stayed within 11.4 mm / 3.8 deg (ratio 11:1).**
- Hand measurements: centres 28 cm apart (ruler, read to the cm: +-0.5 cm), black edges 20 cm apart; 20 cm + one side L (77 mm) = 277 mm.
  Measured by the pipeline: 276.9 mm -> -0.1 mm against 277 (edges + L), -3.1 mm against 280 (centres, cm reading); both within the hand-measurement precision.
