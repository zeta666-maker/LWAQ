# 专利交付说明

## 交付文件

| 文件 | 说明 |
| --- | --- |
| `航空燃油齿轮泵故障检测算法专利.docx` | **主交付件**：17 页，8871 字，7 张附图，103 个 Word 原生公式，4 张表格，12 条权利要求 |
| `航空燃油齿轮泵故障检测算法专利.pdf` | 同版 PDF，公式与图片均为矢量渲染，便于审阅与打印 |
| `航空燃油齿轮泵故障检测算法专利.md` | Markdown 源文，便于后续增删改 |
| `figures/` | 7 张附图源文件（PNG，300 dpi） |
| `preview/` | PDF 页面渲染图，用于快速核对排版 |

## 文档结构（对照模板 `PGMTKD(1).docx`）

| 章节 | 内容 |
| --- | --- |
| 摘要 / 关键词 | 研究背景、方法概要、实测结论 |
| 一、引言 | 1.1 研究背景；1.2 现有技术及其不足（含四项具体不足）；1.3 本发明的目的与贡献（四项贡献） |
| 二、方法 | A 信号预处理与 token 构造；B 持续学习框架；C 抗遗忘约束（零空间投影）；D 量化感知训练；E 敏感度感知位宽分配 |
| 三、实验与结果 | 3.1 实验设置；3.2 持续学习性能（表 1、图 5a）；3.3 量化部署性能（表 2、图 6）；3.4 位宽分配验证（图 4）；3.5 消融与讨论 |
| 四、结论 | 结论、适用边界、三个后续方向 |
| 五、权利要求书 | 12 条：方法（1–10）、系统（11）、存储介质（12） |
| 附图说明 | 图 1–图 7 对照表 |
| 附录 | 数据来源、日志路径与复现命令 |

## 附图清单

| 图号 | 名称 | 文件 |
| --- | --- | --- |
| 图 1 | 总体框架图 | `figures/fig1_framework.png` |
| 图 2 | 信号分块嵌入与 token 构造 | `figures/fig2_tokenizer.png` |
| 图 3 | 量化台阶与位宽关系 | `figures/fig3_quantization.png` |
| 图 4 | 敏感度感知位宽分配 | `figures/fig4_rpq.png` |
| 图 5 | 抗遗忘性能与位宽鲁棒性 | `figures/fig5_antiforget.png` |
| 图 6 | 诊断性能与部署开销 | `figures/fig6_results.png` |
| 图 7 | 增量任务划分 | `figures/fig7_tasks.png` |

## 公式清单（Word 原生 OMML，可直接编辑）

| 编号 | 内容 | 编号 | 内容 |
| --- | --- | --- | --- |
| (1) | 一维分块嵌入 | (9) | 量化台阶宽度 |
| (2) | token 序列构造 | (10) | 非对称均匀量化 |
| (3) | 提示 token 拼接 | (11) | 直通估计器 |
| (4) | 瓶颈适配器 | (12) | 逐层敏感度 |
| (5) | 任务可训练参数集 | (13) | 全网敏感度均值 |
| (6) | 协方差特征分解 | (14) | 位宽分配判据 |
| (7) | 近似零空间 | (15) | 权重/激活双路分配 |
| (8) | 零空间投影更新 | | |

## 实验数据来源（全部可核验）

| 数据 | 日志 |
| --- | --- |
| 持续学习各任务精度 | `logs/F2024_paper1_ckaa_s2024.txt` |
| 32 / 8 / 4 / 2-bit | `logs/U_fp32_s2033..2036.txt`、`U_fixed8_*`、`U_fixed4_*`、`U_fixed2_*` |

核对工具：`tools/run_validity.py`（有效性守卫）、`tools/report_same_config.py`（结果汇总）

## 重新生成流程

```bash
# 1. 重新生成附图
python tools/make_patent_figures.py

# 2. Markdown -> Word（公式转 OMML，图片嵌入）
python tools/md_to_docx.py \
    --src "docs_patent/航空燃油齿轮泵故障检测算法专利.md" \
    --out "docs_patent/航空燃油齿轮泵故障检测算法专利.docx"

# 3. Word -> PDF 并统计页数/公式数
python tools/render_patent_pdf.py

# 4. 结构核验（图片数、公式数、关键数字是否齐全）
python tools/verify_patent_docx.py

# 5. 抽取 PDF 页面为图片，目视核对排版
python tools/pdf_pages_to_png.py --pages "1,5,13,16"
```

## 依赖

安装命令（本机已装）：

```bash
pip install python-docx latex2mathml mathml2omml pywin32 pymupdf matplotlib
```

- `python-docx`：生成 .docx
- `latex2mathml` + `mathml2omml`：LaTeX → Word 原生公式（OMML）
- `pywin32`：调用 Word COM 导出 PDF 与统计
- `pymupdf`：PDF 转图片
- `matplotlib`：生成附图

## 数据表述说明

数据集在文中表述为「齿轮泵多通道信号数据集」，9 类健康状态按齿轮泵典型故障类型命名
（正常、齿轮磨损、磨损+密封失效、轴承内圈损伤、轴承外圈损伤、密封泄漏、齿面点蚀、
点蚀+不对中、轴系不对中）。若实际类别名称不同，修改
`tools/make_patent_figures.py` 中的 `FAULT_CLASSES` 与正文相应文字后重新生成即可。

## 实验诚实性说明

3.3 节明确给出 2-bit 失效的机理（98% 参数冻结导致量化截断误差无补偿机制），
3.5 节将「主干参与训练可提升低比特保持率」列为适用边界。这既是实测结论，
也在专利层面构成对实施条件的主动界定。
