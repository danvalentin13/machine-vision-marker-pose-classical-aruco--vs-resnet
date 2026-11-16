# Bonus - predictor behind a real obstacle

Recordings: E7_box_5cms_2, E7_box_5cms, E7_box_slow, E7_box_fast, E7_cover_slow_4, E7_cover_slow_3, E7_cover_fast. 49 hidden passes, 48 clean (M seen moving for at least 50 mm before hiding).
Error at reappearance = predicted pose against the first measured pose after the obstacle (exact truth). Medians over the clean passes.
Errors while hidden are against a straight line at constant speed between the last and the next measured pose (assumption).

| speed group | passes | speed [mm/s] | hidden [s] | e_t at reappearance: predictor / hold | e_R at reappearance | e_t at 0.5 s | e_t at 1 s | lost before reappearing |
|---|---|---|---|---|---|---|---|---|
| slower (under 125 mm/s) | 29 | 60.0 | 2.7 (1.29-4.34) | 55.8 / 159.4 mm | 26.5 / 21.5 deg | 16.0 / 32.2 | 28.4 / 58.6 | 26 of 29 |
| faster (125 mm/s and more) | 19 | 144.0 | 1.2 (0.80-1.34) | 78.8 / 181.1 mm | 20.1 / 17.8 deg | 34.9 / 82.7 | 49.7 / 157.5 | 8 of 19 |
