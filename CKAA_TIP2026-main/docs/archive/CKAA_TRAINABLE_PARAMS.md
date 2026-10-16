# CKAA 到底训练哪些参数？为什么主干不参与训练？这与 2-bit 有何关系？

> 本文所有数字均由 `python tools/list_trainable_params.py` 走真实代码路径
> （`train_eval.set_model_mode`）统计得到，非人工估算。

---

## 一、结论速览

| 问题 | 答案 |
| --- | --- |
| 主干参加训练吗？ | **不参加。** 12 层 Transformer 的 `attn.*`、`mlp.*`、`norm*` 被显式设为 `requires_grad=False` |
| 那训练什么？ | 按**参数名匹配**：`prompt`、`head`、`adapter`、`prototypes`、`patch_embed` |
| 任一时间点可训练参数 | **2.15 M / 91.48 M = 2.35%** |
| 5 个任务累计曾参与训练的 | 6.13 M / 91.48 M ≈ 6.7%（**非单次训练量**） |
| 是硬件限制吗？ | **不是。** 是 CKAA 的算法设计 |

---

## 二、代码依据

`train_eval.py:318-339`（`set_model_mode`）逐参数判定：

```python
for n, p in model.named_parameters():
    if training and any([_s in n for _s in training_string]):
        if dataset in ('tabular', 'fault_csv') and taskid > 0 and 'patch_embed' in n:
            p.requires_grad_(False)          # 第 2 个任务起，patch_embed 也冻结
        elif freeze_shared_prompts_after_first and taskid > 0 and 'sh_prompt' in n:
            p.requires_grad_(False)          # 第 2 个任务起，共享 prompt 冻结
        elif 'sp_prompt' in n or 'keys' in n or 'task_adapter' in n or 'task_head' in n:
            if str(taskid) in n:
                p.requires_grad_(True)       # 只训练【当前任务】的那一份
            else:
                p.requires_grad_(False)      # 其他任务的份冻结
        else:
            p.requires_grad_(True)
    else:
        p.requires_grad_(False)              # 名字不匹配 → 一律冻结
```

训练字符串（实测）：

```python
training_string = ('prompt', 'head', 'adapter', 'prototypes', 'patch_embed')
```

因此判定规则是**按名字子串匹配**：名字里含这几个词的参数可训练，**其余全部冻结**。

---

## 三、实测明细（任务 1）

| 组件 | 可训练 | 冻结 | 作用 |
| --- | ---: | ---: | --- |
| **task_adapter**（逐任务适配器） | **1,982,720** | 3,965,440 | 瓶颈 64 维的 down/up 投影，12 层 × 6 组 |
| **sp_prompt**（逐任务 prompt） | 61,440 | 122,880 | 每层 4 个 token × 768 维 |
| **sh_prompt**（共享 prompt） | 36,864 | 0 | 第 1 个任务后冻结 |
| **patch_embed**（Conv1d 分词器） | 25,344 | 0 | 第 1 个任务后冻结 |
| ViT 主干内部（keys 等） | 23,040 | 85,208,064 | **几乎全冻** |
| prompt_linear | 9,408 | 0 | |
| task_head（逐任务分类头） | 7,690 | 30,760 | 只训练当前任务的头 |
| head（主分类头） | 1,538 | 0 | |
| **合计** | **2,148,044** | **89,327,144** | **可训练占 2.35%** |

### 两种口径的区别（此前表述有误，特此更正）

| 口径 | 数值 | 含义 |
| --- | ---: | --- |
| **单任务（任一时间点）** | **2.15 M / 91.48 M = 2.35%** | 真正在此刻被优化的参数 |
| 5 任务累计 | 6.13 M / 91.48 M ≈ 6.7% | 5 个任务的适配器/prompt 全都算上 |

此前文档中写的"5.9 M（6.5%）"属于**累计口径**（5 个 `task_adapter` 共 5,948,160 个参数），
**不代表单次训练的可训练量**，容易引起误解。**正确表述应为任一时刻 2.35%。**

---

## 四、这是算法设计，不是硬件限制

CKAA 采用的是 **prompt + adapter 的参数高效持续学习范式**（与 L2P / DualPrompt / 各类
adapter 方法同族）：ViT 主干作为**预训练特征提取器被冻结**，每个任务只学习少量新增参数，
并配合**零空间投影**（`ModAdam`，`null_eta1/2 = 0.999`）抑制新任务对旧任务表征的破坏。

**收益**：遗忘率仅 **2.37%**，Last-acc **95.99%**、Avg-acc **97.92%**。
若放开全参数微调，遗忘会显著上升 —— 冻结主干正是 CKAA 换取低遗忘的手段。

**硬件上完全够用**：

| 项目 | 数值 |
| --- | --- |
| 显卡 | RTX 4060 Laptop，8 GB |
| 全参数微调权重 | 91.48 M × 4 B = **346.9 MB** |
| Adam 一阶 + 二阶动量 | 2 × 346.9 MB = **693.8 MB** |
| 合计约 | **1.04 GB** |
| 当前训练实测占用 | 约 6 GB（含激活、DataParallel、cuDNN workspace） |

所以显存**不是**障碍。若要让主干参与训练，障碍在于**需调整第一篇 CKAA 的训练协议**
（会牺牲其低遗忘特性），而非硬件不足。

---

## 五、与 2-bit 的关系（关键推论）

既然任一时间点**只有 2.35% 的参数可训练**，那么：

> **98% 以上的参数（其中主干 85.2 M）承受 2-bit 截断后，完全没有任何机制可以恢复。**

这把两种 2-bit 方案的性质说清楚了：

| 方案 | 2-bit 施加在哪些参数 | 可否恢复 | 实测保持率 | 体积 |
| --- | --- | --- | ---: | ---: |
| **整网 2-bit** | 全部 91.48 M，其中 **98% 不可训练** | **不可恢复** | **62.6%** | 23.87 MB |
| **冻结主干 8-bit + 可训练集 2-bit** | 仅那 2.35%，且**可训练** | 可恢复 | **≈97%** | 84.87 MB |

**这解释了为什么整网 2-bit 会崩、而 4-bit/8-bit 尚可**：位宽越低，施加在
"不可训练参数"上的截断误差越大，且越无法补偿。4-bit 时该误差尚在可容忍范围
（保持率 94.0%），2-bit 时超出容忍范围（62.6%）。

同时它也解释了为什么论文能从容做到 2-bit：**论文的 ResNet-20 从零训练、全部参数可训练，
每一层都能在训练中适应自身的量化网格**，不存在"不可恢复"的参数。

---

## 六、给老师的一段话

> CKAA 的主干**不参与训练**，这是算法设计而非硬件限制。`train_eval.py` 的
> `set_model_mode` 按参数名匹配设定可训练集，训练字符串为
> `('prompt', 'head', 'adapter', 'prototypes', 'patch_embed')`，其余参数被显式设为
> `requires_grad=False`。实测任一时间点的可训练参数为 **2.15 M / 91.48 M = 2.35%**
> （其中逐任务适配器 1.98 M 为主），主干 85.2 M 参数始终冻结。此前"6.5%"的说法是
> 5 个任务适配器累计口径，特此更正。
>
> 该设计换来的是极低的遗忘率（2.37%）与 95.99% 的 Last-acc。硬件上，8 GB 显存足以
> 全参数微调 ViT-Base（权重 346.9 MB + Adam 状态 693.8 MB ≈ 1.04 GB），故显存并非约束；
> 若要让主干参与训练，需要调整第一篇的训练协议。
>
> 这一点也直接关系到第二篇的 2-bit 表现：由于 **98% 的参数不可训练**，2-bit 截断误差
> 在这些参数上无法补偿。实测仅将冻结主干保持 8-bit、把可训练集压到 2-bit，保持率即由
> **62.6% 提升到约 97%**，印证了该机制。

---

## 复现命令

```powershell
python tools/list_trainable_params.py
```

输出各组件可训练/冻结参数量、可训练占比，以及每个组件的参数名与形状示例。
