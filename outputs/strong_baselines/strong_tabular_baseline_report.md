# Stronger Non-LLM Baseline Report

## Scope

This report strengthens the non-LLM comparison set for 2022 household trip-count prediction. All baselines train on historical NHTS years and evaluate on 2022. The covariate-shift baseline may use 2022 household covariates but does not use 2022 trip-count labels.

## Results

| Method | Device | Weighted MAE | Weighted bias | Weighted R2 |
|---|---|---:|---:|---:|
| catboost_gpu | gpu | 4.2196 | +3.4873 | -0.5712 |
| xgboost_stronger_cuda | cuda | 4.4173 | +3.7364 | -0.6990 |
| xgboost_covariate_shift_reweighted_cuda | cuda | 4.4304 | +3.7523 | -0.7141 |
| lightgbm_cpu | cpu | 4.4430 | +3.7646 | -0.7269 |

## Reference Against Primary Event Adapter

| Comparator | Weighted MAE | Weighted bias | Weighted R2 |
|---|---:|---:|---:|
| best stronger non-LLM: catboost_gpu | 4.2196 | +3.4873 | -0.5712 |
| primary gated event adapter | 2.5023 | -0.0230 | 0.2480 |

The primary gated event adapter is `1.7173` weighted-MAE lower than the best stronger non-LLM baseline, a relative reduction of `40.70%`.

## Interpretation

These baselines are designed to test whether stronger tabular learning or label-free covariate-shift correction can explain away the LLM event-adaptation result. They still overpredict 2022 and remain above the gated LLM adapter, supporting the claim that event mechanisms add information beyond routine household covariates.

LightGBM is reported with CPU execution because the local LightGBM GPU backend failed while creating the Boost.Compute cache. XGBoost and CatBoost baselines were trained through CUDA/GPU paths on the RTX 4090.

## GPU Environment

```text
Wed Jun 10 09:34:57 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 560.94                 Driver Version: 560.94         CUDA Version: 12.6     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 4090      WDDM  |   00000000:01:00.0  On |                  Off |
| 30%   43C    P0             56W /  450W |     431MiB /  24564MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
                                                                                         
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI        PID   Type   Process name                              GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A      2696    C+G   C:\Windows\System32\dwm.exe                 N/A      |
|    0   N/A  N/A      3100    C+G   ...7\extracted\runtime\WeChatAppEx.exe      N/A      |
|    0   N/A  N/A      6328    C+G   ...oogle\Chrome\Application\chrome.exe      N/A      |
|    0   N/A  N/A      6620    C+G   ...1.0_x64__8wekyb3d8bbwe\Video.UI.exe      N/A      |
|    0   N/A  N/A      7396    C+G   ....Search_cw5n1h2txyewy\SearchApp.exe      N/A      |
|    0   N/A  N/A      9788    C+G   ...5n1h2txyewy\ShellExperienceHost.exe      N/A      |
|    0   N/A  N/A     11464    C+G   C:\Windows\explorer.exe                     N/A      |
|    0   N/A  N/A     12960    C+G   ...oogle\Chrome\Application\chrome.exe      N/A      |
|    0   N/A  N/A     13600    C+G   ...ejd91yc\AdobeNotificationClient.exe      N/A      |
|    0   N/A  N/A     14192    C+G   ...on\148.0.3967.96\msedgewebview2.exe      N/A      |
|    0   N/A  N/A     14408    C+G   ...ware\Seafile\bin\seafile-applet.exe      N/A      |
|    0   N/A  N/A     15908    C+G   D:\Software\ToDesk\ToDesk.exe               N/A      |
|    0   N/A  N/A     18088    C+G   ...voice\logioptionsplus_logivoice.exe      N/A      |
|    0   N/A  N/A     18224    C+G   ...x64__qmba6cd70vzyy\ArmouryCrate.exe      N/A      |
|    0   N/A  N/A     18540    C+G   ...t.LockApp_cw5n1h2txyewy\LockApp.exe      N/A      |
|    0   N/A  N/A     19500    C+G   ...ekyb3d8bbwe\PhoneExperienceHost.exe      N/A      |
|    0   N/A  N/A     20300    C+G   ...CBS_cw5n1h2txyewy\TextInputHost.exe      N/A      |
|    0   N/A  N/A     24556    C+G   D:\Software\Microsoft VS Code\Code.exe      N/A      |
|    0   N/A  N/A     24896    C+G   ...aam7r\AcrobatNotificationClient.exe      N/A      |
+-----------------------------------------------------------------------------------------+
```
