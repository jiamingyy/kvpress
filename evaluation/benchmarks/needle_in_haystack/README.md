# Needle in a Haystack

本目录提供了 KVPress 的 NIAH（Needle in a Haystack）实验基线实现与可视化工具。

该 benchmark 的目标是评估模型在超长上下文中检索“针”（needle）信息的能力：
- **横轴**：不同上下文长度（token limit / `max_context_length`）
- **纵轴**：needle 在上下文中的插入深度（`needle_depth`）
- **颜色**：检索分数（默认 `ROUGE-L F1`）

我们默认使用 [Paul Graham essays](https://huggingface.co/datasets/alessiodevoto/paul_graham_essays) 作为 haystack 数据集。

---

## 1. 环境准备

在仓库根目录执行：

```bash
uv sync --extra eval
source .venv/bin/activate
```

如果你要直接运行本目录的热力图脚本，请额外安装 `matplotlib`（二选一）：

```bash
uv sync --extra dev
# 或
pip install matplotlib
```

> `--extra eval` 会安装评测依赖（如 `datasets`、`rouge` 等）。

---

## 2. 数据集说明

NIAH 数据集由评测框架自动从 Hugging Face 拉取：
- dataset id: `alessiodevoto/paul_graham_essays`
- 在 `evaluate_registry.py` 中注册名：`needle_in_haystack`

你无需手动下载原始语料，只需要在评测命令里设置：
- `dataset=needle_in_haystack`
- `data_dir=null`

---

## 3. 单次 NIAH 实验（固定 context length）

在 `evaluation/` 目录执行：

```bash
python evaluate.py \
  --dataset needle_in_haystack \
  --data_dir null \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --press_name no_press \
  --compression_ratio 0.0 \
  --max_context_length 32768 \
  --needle_depth "[0,10,20,30,40,50,60,70,80,90,100]" \
  --output_dir ./results_niah
```

运行完成后，在 `output_dir` 下会生成对应目录，核心文件：
- `predictions.csv`: 每个深度下的模型回答
- `metrics.json`: 每个样本的 ROUGE 指标
- `config.yaml`: 本次实验配置

---

## 4. 完整 NIAH 流程（多 context length 扫描）

为了画出常见 NIAH 热力图，需要对多个 `max_context_length` 重复运行第 3 步。

你可以直接使用以下 shell 循环（在 `evaluation/` 目录执行）：

```bash
MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
PRESS="qfilter_1024"
CR="0.5"
DEPTHS='[0,10,20,30,40,50,60,70,80,90,100]'
OUTDIR="./results_niah"

for CTX in 4096 8192 16384 32768 65536 98304 114688; do
  python evaluate.py \
    --dataset needle_in_haystack \
    --data_dir null \
    --model "$MODEL" \
    --press_name "$PRESS" \
    --compression_ratio "$CR" \
    --max_context_length "$CTX" \
    --needle_depth "$DEPTHS" \
    --output_dir "$OUTDIR"
done
```

> 如果需要比较不同 press，只需固定模型和 context/depth 网格，替换 `press_name` 与 `compression_ratio` 重跑一组即可。

---

## 5. 绘图：生成 NIAH 热力图

本目录新增了 `plot_niah_heatmap.py`，用于读取多个运行目录并生成热力图。

### 5.1 基本用法

在仓库根目录执行：

```bash
python evaluation/benchmarks/needle_in_haystack/plot_niah_heatmap.py plot \
  --results_root evaluation/results_niah \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --press_name qfilter_1024 \
  --compression_ratio 0.5 \
  --metric rouge-l-f \
  --aggregation mean \
  --output_png evaluation/benchmarks/needle_in_haystack/niah_heatmap_qfilter1024.png
```

### 5.2 可选参数

- `--metric`: `rouge-1-f` / `rouge-2-f` / `rouge-l-f`
- `--aggregation`: `mean` / `median` / `min` / `max`
- `--title`: 自定义图标题
- 如果你不传 `model/press_name/compression_ratio`，脚本会汇总 `results_root` 下所有 NIAH 结果（通常建议加过滤，避免混入不同实验设置）。

---

## 6. 复现实验建议（与论文常见图一致）

建议固定以下维度：
- 模型：固定一个模型（如 Llama-3.1-8B-Instruct）
- Press：固定一个方法（如 `qfilter_1024`）
- 压缩率：固定一个值（如 `0.5`）
- Context lengths：从短到长多档位
- Needle depths：0 到 100 的均匀分桶

最终得到的二维矩阵（depth × context length）即可绘制与论文中常见的 NIAH heatmap 风格图。

---

## 7. 常见问题

1. **报错：`needle_depth must be set for needle_in_haystack`**  
   需要显式传入 `--needle_depth`。

2. **报错：`max_context_length must be set for needle_in_haystack`**  
   需要显式传入 `--max_context_length`。

3. **显存不足**  
   先减小 `max_context_length`，或切换更小模型，或使用更高压缩率。

4. **热力图为空/只有部分列**  
   检查 `results_root` 下是否确实有多个不同 `max_context_length` 的 NIAH 运行目录。

---

## 8. 相关实现位置

- Needle 插入逻辑：`utils.py` (`insert_needle_in_haystack`)
- NIAH 指标计算：`calculate_metrics.py`
- 新增绘图脚本：`plot_niah_heatmap.py`
- 建议配置模板：`niah_eval_config_template.yaml`
