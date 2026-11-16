# Task 2 - detection limits (DICT_6X6_250, marker M id 23, side 77.0 mm, 1280x720)

Cell = frames with a detection out of 10; PASS = at least 9. Distance from the camera lens, tilt about the marker's vertical axis.

| distance [cm] | px per cell at 0 deg | 0 deg | 30 deg | 45 deg | 60 deg | 70 deg | 75 deg | 80 deg | 85 deg | largest tilt PASS |
|---|---|---|---|---|---|---|---|---|---|---|
| 30 | 29.7 | 10/10 PASS | 10/10 PASS | 10/10 PASS | 10/10 PASS | 10/10 PASS | 10/10 PASS | 10/10 PASS | 0/10 FAIL | 80 deg |
| 50 | 17.8 | 10/10 PASS | 10/10 PASS | 10/10 PASS | 10/10 PASS | 10/10 PASS | 10/10 PASS | 10/10 PASS | 0/10 FAIL | 80 deg |
| 75 | 11.0 | 10/10 PASS | 10/10 PASS | 10/10 PASS | 10/10 PASS | 10/10 PASS | 10/10 PASS | 10/10 PASS | 0/10 FAIL | 80 deg |
| 100 | 8.4 | 10/10 PASS | 10/10 PASS | 10/10 PASS | 10/10 PASS | 10/10 PASS | 10/10 PASS | 5/10 FAIL | not tested | 75 deg |

- **Distance limit (tilt 0):** detected at all tested distances up to 100 cm (the largest of the set-up): the limit was NOT reached.
  At 100 cm the marker has 8.4 px per cell. With the lecture rule (failure near 2 px per cell) and px per cell ~ 1/distance, the expected limit is about 4.2 m.
- **Tilt limit at 30 cm:** 80 deg (largest tilt with PASS).
- **Tilt limit at 50 cm:** 80 deg (largest tilt with PASS).
- **Tilt limit at 75 cm:** 80 deg (largest tilt with PASS).
- **Tilt limit at 100 cm:** 75 deg (largest tilt with PASS).
- Frame rate during the tests: 29 fps (median).
