# Data

`comma2k19_rav4_steps.npz`: per-step arrays (0.05 s grid) derived from the
[comma2k19](https://github.com/commaai/comma2k19) dataset (Schafer, Santana, Haden,
Biasini, "A Commute in Data: The comma2k19 Dataset", 2018; MIT licence), chunks 1
(`learn_*`) and 2 (`test_*`), both a Toyota RAV4 on California highway 280:

| array | meaning |
|---|---|
| `*_v` | speed from CAN [m/s] |
| `*_sigma` | steering-wheel angle from CAN [deg] |
| `*_r` | yaw rate from the gyroscope [rad/s], 5-sample running median |
| `*_drive` | index of the one-minute segment the step comes from |

Made by `experiments/11_real_lane.py` from the raw arrays that
`experiments/10_comma_download.py` fetches (not in git; about 0.3 MB per segment).
