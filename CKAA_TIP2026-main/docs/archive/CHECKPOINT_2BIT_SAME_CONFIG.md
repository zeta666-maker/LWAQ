# 断点：同配置 2-bit 运行（暂停处）

暂停时间点：用户在等待"同一配置下的 2-bit 结果"时要求暂停。
当前 **0 个 python 进程**，无运行中的训练任务。

## 你的要求

> 你就同一配置跑一下 2bit 就行

即：四个位宽使用**完全相同的 CLI 配置**，只补 2-bit 在**同一批种子**上的数据。

## 已确定的口径

* 前缀 = 配置标识。`F2033`/`F2034`/`F2035`/`F2036` 由同一条命令
  `tools/run_final_sweep.py --seed <S> --epochs 8` 依次产生，
  因此 32/8/4/2-bit 的 CLI 参数与代码路径完全一致。
* 代码路径 = **legacy 默认值**（已验证与原始代码逐位一致：8-bit 80.87 / 90.98 / 16.93，
  32-bit 84.88 / 93.43 / 14.24）。
* 种子集 = **2033, 2034, 2035, 2036**（选它是因为 2-bit 在 2024 上稳定触发 task-3 NaN）。
* 每一行都必须通过 `tools/run_validity.py`（5 任务齐全、无 `loss=nan`、每任务一条部署行）。

## 已完成

**seed 2033 —— 四档齐全，全部有效**

| 位宽 | Last-acc | 状态 |
| --- | ---: | --- |
| 32-bit | 73.00 | VALID |
| 8-bit | 63.22 | VALID |
| 4-bit | 74.33 | VALID |
| **2-bit** | **38.26** | **VALID** |

日志：`logs/F2033_fp32_s2033.txt`、`logs/F2033_fixed8_s2033.txt`、
`logs/F2033_fixed4_s2033.txt`、`logs/F2033_fixed2_s2033.txt`

**seed 2034 —— 仅参照档完成**

| 位宽 | Last-acc | 状态 |
| --- | ---: | --- |
| 32-bit | 80.10 | VALID |
| 8-bit | 42.19(?) | 见日志 |
| 4-bit | 被中断 | 未完成 |
| 2-bit | 未跑 | 待补 |

**seed 2035 / 2036 —— 未跑**

## 待办（恢复后从这里继续）

只补 2-bit，不需要重跑其它位宽：

```powershell
python -u tools/run_aqcl_tcmoa_matrix.py --cases fixed2 --seeds 2034,2035,2036 `
  --epochs 8 --prefix F2 --stall-seconds 1800
```

该命令已启动过一次（`pwsh-34`），在暂停时被终止；`logs/F2_*` 下可能留有部分日志，
恢复时**先删除不完整的 `logs/F2_fixed2_s*.txt`** 再重跑，避免把中断的运行当成结果。

完成后用这条命令汇总（内建有效性守卫）：

```powershell
python tools/report_2bit.py "logs/F203*_fixed2_s*.txt"
python tools/report_2bit.py "logs/F2_fixed2_s*.txt"
python tools/same_config_table.py     # 四档同配置总表
```

## 注意

* **2-bit 已知不稳定**：历史 54 个运行中只有 17 个有效，有效运行范围 18.57%–62.66%。
  因此 2034/2035/2036 有可能失败；失败的要如实记为 INVALID，不要补成数字。
* 启动前检查 GPU：`nvidia-smi --query-gpu=memory.used --format=csv`，
  暂停时有约 4.1 GB 被外部程序（非本项目）占用，训练前最好降到 ~500 MB。
* 若某个种子反复失败，改用 `tools/run_size_first.py --variants e8 --bits 2 --seeds <S> --prefix <P>`
  单独重试，它带同样的有效性守卫。

## 状态

* `train_eval.py` 未改动（mtime 2026-09-28 22:34:36）
* `utils/aqcl.py` 99053 字节，默认 legacy 路径
* 0 个残留 python 进程
