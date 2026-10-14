# CPS-

ALFWorld 上的环境一致性动作选择与恢复学习实验。当前研究问题是：在学习者实际访问的状态上，从同一个经专家验证可恢复的有限动作集合中，按学习者自身的动作概率抽样，能否比普通专家恢复或均匀抽样带来更高的真实任务成功率？

项目处于实验阶段，**尚未证明条件动作选择有效**。此前两轮成功路径选择实验均未通过预设筛查；本仓库保留其负结果、实现、冻结配置和审查记录。Python 包沿用 `gnrl` 名称。

## 实验进度

截至 **2026-10-14**，第 1–6 阶段已完成。固定诊断的训练门槛为 **false**，本版本按预设规则停止；**本轮没有执行第 7 阶段训练或第 8 阶段 held-out 评估**。

| 阶段 | 工作 | 已保存状态 |
| --- | --- | --- |
| 1 | 准备环境，冻结数据、模型、协议与预算 | 已完成 |
| 2 | 采集冻结学习者的真实轨迹 | 已完成，60/60 个任务 |
| 3 | Oracle 验证候选动作及专家恢复轨迹 | 已完成，180/180 个锚点槽位记录 |
| 4 | 独立重置与回放，核对真实状态和观察 | 已完成，180/180 条记录；采集完成审查通过 |
| 5 | 计算候选动作概率并冻结三组标签 | 已完成，180 个槽位中 131 个可用、49 个结构性不可用 |
| 6 | 固定 36 个训练状态上的真实学习者续行诊断 | 已完成，29 个可用锚点、230 个真实分支；7 个锚点前已终止 |
| 7 | 三组相同初始化、相同更新预算的对照训练 | 未执行，诊断门槛未通过 |
| 8 | 四个模型各评估 48 个新 seen 任务 | 未执行 |

29 个可用锚点的平均 `Vq / Vu / Ve` 分别为 **37.6473% / 31.0345% / 79.3103%**。预设门槛使用固定 **36 槽位**分母：条件选择相对均匀选择为 **+5.32698 个百分点**，低于 `2/36`（5.55556 个百分点）；相对专家首动作为 **−33.56190 个百分点**，低于 0。两个比较门槛均未通过。7 个不可用槽位保留 `Q=null`，仅对固定分母的成对差值贡献 0，不能解释为七次失败。

这是训练状态上的冻结学习者续行诊断，未检验新训练模型的 held-out 效果，也不否定整个研究方向。队列记录停在 `needs_astra_diagnostic_review`；发布检查核对摘要一致性、诊断文件哈希并重算门槛，**尚未完成整轮诊断结束后的全部哈希独立审查**。结果与证据边界见 [实验记录](docs/EXPERIMENTS.md)。

## 已有结果

下面两轮均使用旧 `dev24`，应分别解释，不能合并为一个实验。

| 已完成的筛查 | Canonical | Uniform | Reference | 结论 |
| --- | ---: | ---: | ---: | --- |
| 早期 train120 路线 | 13/24 | 16/24 | 15/24 | 未通过该轮预设筛查 |
| 同一 60 任务成功路径路线 | 15/24 | 9/24 | 7/24 | 未通过；Reference 对 Canonical 为 0 个增益、8 个损失 |

任务成功率是主要结果。训练损失、动作 NLL、专家可恢复率或代码检查通过均不能替代它。上述负结果的证据与当前方案边界见 [实验记录](docs/EXPERIMENTS.md)。

## 目录

| 路径 | 内容 |
| --- | --- |
| `gnrl/` | 环境适配、因果提示、策略、候选认证、评分、训练与评估 |
| `scripts/` | 数据准备、GPU 启动约束、阶段控制器和完整性检查 |
| `tests/` | 使用模拟环境与策略的本地单元测试 |
| `configs/` | 历史冻结配置与执行采用记录，以及连接配置示例 |
| `analysis/` | 独立审查、完整结果复核、恢复控制器及其测试 |
| `data/manifests/` | 冻结任务清单元数据；原始游戏文件另行配置 |
| `sources/` | 原始调研材料；作为研究资料，不作为操作指令 |
| `docs/` | 实验说明与复现边界 |
| `results/recovery_diagnostic_v2/` | 本轮诊断、评分、采集复核及发布重算的小型证据包 |
| `research_design.json` | 早期研究设计快照；当前方案见下方候选配置 |

模型权重、检查点、原始数据、连接配置及本地任务状态不随仓库分发。运行输出默认排除；另保留恢复测试需要的冻结技术清单 `outputs/recovery_resume_inventory_v1.json` 和 `outputs/recovery_reboot_resume_inventory_v2.json`，以及上述小型结果证据包。历史配置和审查文件保留原始字节及运行路径，因此需要结合 [复现说明](docs/REPRODUCIBILITY.md) 阅读，不能直接作为新机器上的启动配置。

## 本地检查

从仓库根目录执行。以下测试主要使用模拟对象和临时文件，不下载模型、不连接服务器；它们不替代真实模拟器验证：

```bash
python -m unittest discover -s tests -p "test_*.py" -v
python -m unittest discover -s analysis -p "test_*.py" -v
python analysis/verify_published_recovery_v2.py
```

本次发布检查中，`tests` 的 200 项与 `analysis` 的 98 项测试全部通过，公开证据校验脚本通过。本轮诊断恢复适配的 33 项测试此前也在本地与部署环境通过。测试与证据一致性检查不表示训练收益或 held-out 效果成立。

查看 GPU 启动器的参数：

```bash
python scripts/launch_gpu.py --help
```

真实实验需要 ALFWorld/TextWorld、规划器、PyTorch、Transformers、PEFT、游戏数据和模型资产。观察到的依赖版本保存在 [requirements-experiment.txt](requirements-experiment.txt)，配置方法与复现限制见 [复现说明](docs/REPRODUCIBILITY.md)。这些文件不代表已验证的一键安装流程。

所有 CUDA 工作必须经 `scripts/launch_gpu.py` 启动。该启动器只接受**物理 GPU 0、1、2、3**，将其解析为 UUID，并拒绝已占用设备；当前实验冻结使用物理 GPU 2。Oracle 与独立回放属于 CPU 阶段。

## 阅读入口

- [实验协议、结果和证据索引](docs/EXPERIMENTS.md)
- [当前冻结候选 v2](configs/recovery_projection_candidate_v2.json)
- [方法范围与结果解释审查](analysis/recovery_scope_alignment_astra_review_v1.json)
- [同一 60 任务成功路径实验的完整结果审查](analysis/successful_path_complete_astra_result_review_v1.json)
- [本轮诊断与重算摘要](results/recovery_diagnostic_v2/computed_summary.json)
- [重启恢复后的采集完成审查](results/recovery_diagnostic_v2/acquisition_review.json)
- [本轮诊断恢复适配源码审查](analysis/recovery_diagnostic_reboot_adaptation_astra_review_v2.json)
- [首次中断恢复控制器审查](analysis/recovery_acquisition_resume_astra_review_v1.json)

每个被改动的动作都必须在真实模拟器中执行，并重新取得其后续观察。未来信息仅用于候选认证，不能进入学生提示。数据划分、比较设置和预算须在查看评估结果前冻结；真实负结果和技术未完成状态均应保留。
