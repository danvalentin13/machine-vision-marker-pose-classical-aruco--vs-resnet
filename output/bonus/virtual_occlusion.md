# Bonus - predictor on uncovered passes hidden in software

Virtual obstacle: M hidden while its left corner is at 78-235 mm on the ruler. Truth = the classical pose measured in the hidden frames.
Predictor = constant-velocity Kalman (position) + alpha-beta on SO(3) (rotation); baseline = hold the last pose. Medians over the passes.

| recording | passes | speed while hidden [mm/s] | hidden [s] | e_t predictor / hold at 0.25 s | at 0.5 s | at 1 s | at the end | e_R at the end | lost |
|---|---|---|---|---|---|---|---|---|---|
| E7_nocover_slow_2 | 4 | 71.0 | 2.2 | 7.6 / 22.4 mm | 14.7 / 42.2 | 24.5 / 81.8 | - / 158.8 | - / 8.3 deg | 4 of 4 |
| E7_nocover_fast | 7 | 172.0 | 0.9 | 23.1 / 51.9 mm | 56.3 / 91.2 | 94.3 / 165.2 | 81.7 / 165.2 | 10.6 / 9.7 deg | 2 of 7 |

Horizon = hidden time at which the position error first passes the threshold (median over the passes that pass it; n = how many do).

| recording | 20 mm: predictor | 20 mm: hold | 40 mm: predictor | 40 mm: hold |
|---|---|---|---|---|
| E7_nocover_slow_2 | 0.8 s (n 4 of 4) | 0.3 s (n 4 of 4) | not reached (0 of 4) | 0.5 s (n 4 of 4) |
| E7_nocover_fast | 0.2 s (n 7 of 7) | 0.1 s (n 7 of 7) | 0.5 s (n 7 of 7) | 0.2 s (n 7 of 7) |
