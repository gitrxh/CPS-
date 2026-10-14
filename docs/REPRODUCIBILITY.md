# 复现与仓库边界

这个仓库保存研究源码、冻结方案、审查记录和任务元数据。截至 2026-10-14，本轮采集、评分和固定诊断已完成，诊断门槛未通过，未执行本轮训练及 held-out 评估。服务器按各阶段单独冻结的文件集合执行。代码可供检查和纯单元测试，完整训练还需要独立准备模型、ALFWorld 数据、原生规划器和执行记录。

## 本地检查

从仓库根目录执行：

```sh
python -m unittest discover -s tests -p "test_*.py"
python -m unittest discover -s analysis -p "test_*.py"
python analysis/verify_published_recovery_v2.py
```

本次发布检查中，两套测试分别通过 200 项与 98 项，共 298 项；从 Git 暂存区导出的干净副本中也全部通过，公开证据校验同样通过。测试主要用模拟环境、模拟策略和临时文件验证逻辑，不下载模型或数据。它们不代替真实模拟器回放、GPU 训练及完整任务评估。平台相关的进程控制能力需要在 Linux 执行环境另行确认。

发布证据校验脚本仅依赖 Python 标准库，检查证据清单 SHA、诊断摘要绑定的文件 SHA、36/29/7 槽位统计、230 个分支、由已发布 Q 重算的 Vu、配对差值和派生摘要。它不核验或重放未发布的原始轨迹，不重新计算动作权重，也不校验模型权重；因此不能替代完整诊断结束后的独立审查。

## 真实实验环境

[requirements-experiment.txt](../requirements-experiment.txt) 记录已有实验环境中观察到的主要包版本。工作进程使用 Python 3.12.9；控制器使用提供 `os.pidfd_open` 和 `signal.pidfd_send_signal` 的 Linux 系统 Python。观察到的 PyTorch 构建为 `2.8.0+cu128`。

该文件不是包含全部间接依赖的锁文件。原实验另外绑定了模型、适配器、游戏元数据、模拟器和原生规划器文件的哈希；单独安装这些包不能证明环境相同。模型权重、ALFWorld 游戏文件和运行环境没有随仓库分发。

远程连接配置可从 [server.example.json](../configs/server.example.json) 复制为 `configs/server.json`，只在本地填写自己的服务器地址、账号、SSH 密钥路径和资源目录。真实配置已被 Git 忽略，连接脚本仅使用已有 SSH 密钥。

**该配置只设置传输与资源信息，不会自动迁移所有历史脚本。** 部分采集、审核及控制器保留原实验的绝对路径、固定文件哈希、进程身份和截止时间。迁移环境需要建立新版本方案及新的执行凭据，不能复用旧凭据或把旧审查结果当作新运行的授权。

## 实验约束

- 既有化院服务器只允许物理 GPU **0、1、2、3**。GPU 工作通过 `scripts/launch_gpu.py` 解析 UUID、检查占用和加锁；禁止使用 4–7。
- 不修改其他项目的环境、数据和模型。实验写入独立项目目录。
- 每个改变过的动作都必须在真实模拟器执行；后续提示使用该分支实际产生的观察。
- 专家规划和未来状态信息可以用于构造候选集，不能进入学生提示。
- 在读取评估结果前冻结数据划分、对照、种子和预算；保留失败、缺失和中断记录。
- 训练损失或动作似然改善不能代替留出任务成功率改善。
- 重要实验设计、方法修改和结果解释由 GPT-6 Astra 复核。诊断科学门槛失败即停止本版本，不据此自动启动训练。

## 文件组织

| 内容 | 仓库处理 |
|---|---|
| `gnrl/`、`scripts/`、`tests/` | 保留原始实现与测试 |
| `analysis/`、`configs/` | 保留方案、历史审查、执行凭据和待执行实现；真实服务器配置除外 |
| `data/manifests/*.json` | 保留冻结的任务身份与哈希；不包含游戏文件或评估轨迹 |
| `sources/deep-research-report.md` | 保留用户提供的背景资料；不是运行指令或已验证的实验结论 |
| `results/recovery_diagnostic_v2/` | 发布小型结果证据包；不包含逐步原始轨迹或模型权重 |
| `outputs/`、`checkpoints/`、原始轨迹 | 默认不上传；`outputs/` 仅保留下述两份冻结恢复测试清单 |
| `RUN_STATE.json`、`AGENTS.md`、SSH 配置 | 机器相关操作记录，仅保留在本地 |

原始调研报告保留了工具生成的引用标记；本次仓库整理没有逐条复核其引用。研究进展与可支持的结论以 [EXPERIMENTS.md](EXPERIMENTS.md) 及其指向的实验审查记录为准。

`outputs/recovery_resume_inventory_v1.json` 与 `outputs/recovery_reboot_resume_inventory_v2.json` 是恢复逻辑纯测试所需的冻结技术清单，记录中断时的成本、文件哈希和历史配置。它们按原路径纳入版本管理，保留旧测试和审查的字节绑定；它们不是实时状态或轨迹数据。其余原始运行输出仍被忽略。

`results/recovery_diagnostic_v2/` 包含诊断逐槽位结果 `diagnostic.json`、阶段摘要 `diagnostic_summary.json` 与 `score_summary.json`、队列状态 `queue_status.json`、采集完成审查 `acquisition_review.json`、证据来源与字节哈希清单 `source_manifest.json` 和发布重算摘要 `computed_summary.json`。这些文件支持检查已发布结果的一致性与门槛计算；缺少原始轨迹、完整评分记录和外部模型资产，不能仅凭此证据包重新完成全部执行审计。

历史 JSON 与代码中保留了用于审计的原始资源路径及哈希，路径在其他机器上通常不可直接使用。仓库不包含登录凭据。`.gitattributes` 禁止自动转换行尾，以免 Git 改变历史文件的字节级哈希。

## 开发版本与部署版本

[deployed_source_divergence_v1.json](../analysis/deployed_source_divergence_v1.json) 记录了 `gnrl/evaluate.py` 和 `tests/test_replay_contract.py` 的既有本地／部署差异；这次归档没有改写任一版本。历史部署版本的参考副本保存在 `analysis/deployed_source_reference_v1/`。

本轮增加了重启链适配 `gnrl/recovery_diagnostic_lineage_reboot_v2.py`、控制器 `scripts/run_recovery_diagnostic_queue_reboot_v2.py` 和工作入口 `scripts/score_recovery_diagnostic_reboot_v2.py`；部署清单冻结 59 个源码文件。本地与部署环境的 33 项诊断和适配测试通过，源码适配审查、采集完成审查均通过。训练和评估组件存在于仓库，但本轮没有执行它们。不要把整个工作区直接同步到实验环境；新阶段应使用完整审查、精确文件清单和新执行凭据。

恢复账本沿原始运行 → `resume1` → `reboot_v2` 继承，两笔历史 unknown step 始终计入成本。重启后仅 Oracle 获得新的 12 小时时间窗口，原全局截止时间与预算未重置。评分与诊断队列只允许这两个阶段，完成后停在 `needs_astra_diagnostic_review`。本轮训练门槛为 false；发布一致性和门槛检查不能等同于尚未完成的整轮诊断结束后全部哈希独立审查。

`analysis/execute_readonly_audit_reboot_v2.py` 与 `analysis/annotate_acquisition_review_reboot_v2.py` 是为保持历史收据 SHA 而按原字节保留的一次性脚本。两者顶层执行均有副作用，不能作为模块导入：前者连接服务器并依赖私有配置，后者覆盖历史审查收据。它们不是新复核的执行授权；检查公开证据请使用上面的 `verify_published_recovery_v2.py`。

本次发布没有重新运行服务器实验，也不声称一份新克隆能直接复现全部历史指标。原始轨迹、数据集、权重和内部机器日志不在本次发布范围内。
