# 統一実験スイート結果要約

## 完了状態

- e2: `completed`
- e3b: `completed`
- e3c: `completed`
- e4: `completed`
- e5: `pilot_completed`

## E2A 頑健性（抜粋）

| 条件 | 値 | model | dim | RMSE |
|---|---:|---|---:|---:|
| gaussian_noise_sigma | 0 | oracle_tm | 32 | 0.0128952 |
| gaussian_noise_sigma | 0 | oracle_fourier | 32 | 0.00380648 |
| gaussian_noise_sigma | 0 | direct_tm | 32 | 0.023 |
| gaussian_noise_sigma | 0 | direct_fourier | 32 | 0.0522656 |
| gaussian_noise_sigma | 0.1 | oracle_tm | 32 | 0.024932 |
| gaussian_noise_sigma | 0.1 | oracle_fourier | 32 | 0.0038372 |
| gaussian_noise_sigma | 0.1 | direct_tm | 32 | 0.0226795 |
| gaussian_noise_sigma | 0.1 | direct_fourier | 32 | 0.0522688 |
| gaussian_noise_sigma | 0.3 | oracle_tm | 32 | 0.124999 |
| gaussian_noise_sigma | 0.3 | oracle_fourier | 32 | 0.00472296 |
| gaussian_noise_sigma | 0.3 | direct_tm | 32 | 0.0422111 |
| gaussian_noise_sigma | 0.3 | direct_fourier | 32 | 0.0522145 |
| missing_rate | 0.2 | oracle_tm | 32 | 0.0274388 |
| missing_rate | 0.2 | oracle_fourier | 32 | 0.379501 |
| missing_rate | 0.2 | direct_tm | 32 | 0.026371 |
| missing_rate | 0.2 | direct_fourier | 32 | 0.178533 |
| missing_rate | 0.4 | oracle_tm | 32 | 0.0562425 |
| missing_rate | 0.4 | oracle_fourier | 32 | 0.687349 |
| missing_rate | 0.4 | direct_tm | 32 | 0.0314475 |
| missing_rate | 0.4 | direct_fourier | 32 | 0.287866 |
| observation_length | 128 | oracle_tm | 32 | 0.0418352 |
| observation_length | 128 | oracle_fourier | 32 | 0.0468962 |
| observation_length | 128 | direct_tm | 32 | 0.0541467 |
| observation_length | 128 | direct_fourier | 32 | 0.0655718 |
| observation_length | 256 | oracle_tm | 32 | 0.0284405 |
| observation_length | 256 | oracle_fourier | 32 | 0.0329141 |
| observation_length | 256 | direct_tm | 32 | 0.0388715 |
| observation_length | 256 | direct_fourier | 32 | 0.0546398 |

## E3B 局所安定性と大域basin

| beta | form | local stable grid | range | best basin |
|---:|---|---:|---|---:|
| 1.05 | output_cross | 11 | 0.25–0.5 | 1 |
| 1.05 | state_diffusive | 13 | 0.35–0.95 | 0.648438 |
| 1.1 | output_cross | 9 | 0.3–0.5 | 1 |
| 1.1 | state_diffusive | 10 | 0.45–0.9 | 0.285156 |
| 1.2 | output_cross | 7 | 0.35–0.5 | 1 |
| 1.2 | state_diffusive | 5 | 0.6–0.8 | 0.078125 |
| 1.3 | output_cross | 6 | 0.375–0.5 | 1 |
| 1.3 | state_diffusive | 3 | 0.65–0.75 | 0.0390625 |

## E3C 同期と入力保持（TM）

| K | balanced acc | median sync RMS | effective rank |
|---:|---:|---:|---:|
| 0 | 0.46789 | 10.406 | 19.127 |
| 0.3 | 0.53211 | 5.97052 | 7.2083 |
| 0.45 | 0.527523 | 3.28278 | 8.12403 |
| 0.5 | 0.59633 | 0.11313 | 7.54528 |
| 0.52 | 0.591743 | 0.00621033 | 7.84255 |
| 0.6 | 0.577982 | 4.52593e-08 | 7.9328 |
| 0.8 | 0.449541 | 0 | 7.87644 |
| 0.95 | 0.536697 | 0 | 7.61791 |

## E4A d=16

| representation | test NMSE | test RMSE |
|---|---:|---:|
| input_oracle | 0.0066058 | 0.0532583 |
| reservoir_flat | 1.01472 | 0.66008 |
| tm_spectrum | 1.01703 | 0.660831 |
| raw_mean_dct | 1.02044 | 0.661938 |

## E5A d=16

| representation | NMSE | PSNR | SSIM | class balanced acc |
|---|---:|---:|---:|---:|
| input_oracle | 0.153263 | 19.8805 | 0.953204 | 0.961244 |
| tm_spectrum | 1.01296 | 11.2247 | 0.571758 | 0.102053 |
| raw_mean_dct | 1.01746 | 11.2352 | 0.572689 | 0.0765144 |
| reservoir_flat | 1.02992 | 11.2308 | 0.572328 | 0.0969605 |