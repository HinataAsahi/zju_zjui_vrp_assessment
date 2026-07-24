# VRP 技术学习笔记设计

日期：2026-07-24

## 目标

创建一份中文技术学习笔记，帮助学生理解本次 CVRP 考核项目，达到可以开始动手实现的程度。这份笔记应该偏入门但保持实践导向：它需要解释问题本身、数据集、可行性规则、评价指标、baseline 思路，以及可选的 AI 方法；不应依赖大量复杂数学推导。

这份笔记不是最终报告、展示 PPT，也不是代码实现。它是后续编写 `solve.py`、设计实验、撰写报告和制作 slides 的基础文档。

## 项目上下文

项目目录包含：

- `VRP_project/vrp_project_outline.docx`：官方项目说明。
- `VRP_project/VRPData/train_data.pkl`：7000 个带标签的 CVRP-50 实例。
- `VRP_project/VRPData/validation_data.pkl`：1000 个带标签的 CVRP-50 实例。
- `VRP_project/VRPData/check_data_to_students.pkl`：公开测试集，包含 1000 个 CVRP-50 实例和 500 个 CVRP-100 实例，没有标签。

带标签的数据集中的每个实例存储为：

```text
(depot, loc, demand, capacity, routes, cost)
```

公开测试集中的每个实例存储为：

```text
(depot, loc, demand, capacity)
```

官方推理接口要求提交一个顶层脚本 `solve.py`。该脚本需要读取输入 `.pkl` 文件，并写出一个 UTF-8 编码的 JSON 文件。JSON 中必须包含 `format_version: "cvrp_v1"`，并且要为每个输入实例生成一组路径方案。

## 读者和深度

这份笔记面向具备基础 Python 和机器学习知识、但刚开始接触 CVRP 和神经组合优化的学生。

深度目标：从入门理解到可以开始动手。

这份笔记应该：

- 用中文解释核心概念。
- 使用本项目中的具体例子。
- 避免过多公式。
- 保留与后续实现决策之间的连接。
- 明确指出常见错误和扣分风险。

## 建议结构

### 1. 用一句话理解项目

解释本任务是一个容量约束车辆路径问题：车辆从同一个 depot 出发，访问所有客户，遵守车辆容量限制，并尽量最小化总行驶距离。

### 2. 如何读取数据

解释每个 `.pkl` 文件的用途、tuple 字段含义、带标签数据和无标签数据的区别、客户编号规则、车辆容量，以及 CVRP-50 / CVRP-100 的划分。

### 3. 什么样的 CVRP 解是合法的

定义可行性规则：

- 每个客户必须恰好访问一次。
- 不能漏掉客户。
- 不能重复访问客户。
- 每条路径上的总 demand 不能超过 vehicle capacity。
- depot 是隐含的，不能写进 route 列表。
- JSON 输出中的客户编号必须从 1 开始。

### 4. 如何理解评价指标

解释：

- Average Total Route Cost，平均总路径代价。
- Feasibility Rate，可行解比例。
- Average Gap to a reference baseline，相对参考 baseline 的平均差距。
- Average Inference Time，平均单实例推理时间。

需要强调：可行性是第一优先级，然后才是路径 cost 和推理时间。一个 cost 看起来很低但不可行的解，在这个项目里没有实际意义。

### 5. 为什么先做 baseline

解释 baseline 在本次考核中的作用：

- 验证数据读取和输出格式是否正确。
- 在神经模型之前先获得稳定、合法、可提交的解。
- 为最终报告提供比较对象。

介绍实用 baseline 思路：

- 容量感知的最近邻算法。
- 贪心路径构造。
- 按容量拆分 route。
- 2-opt 局部优化。

### 6. 如何理解 AI-based 方法

用入门级方式介绍主要模型家族：

- Pointer Network 风格的序列构造方法。
- Attention / Transformer-based routing policy。
- POMO 风格的强化学习方法。
- Learning-based improvement 方法。

每类方法都需要解释它在“学什么”，以及为什么可能帮助改进路径质量。

### 7. 推荐项目路线

建议分阶段推进：

1. 理解数据，并实现评价工具。
2. 构建可行 baseline 和官方接口 `solve.py`。
3. 尝试简单的 supervised imitation 或轻量 neural heuristic。
4. 分析性能和局限，用于报告和展示。

### 8. 常见坑检查清单

列出常见错误：

- 把测试集实例当成长度为 6 的 tuple 处理。
- 在输出中使用 0-based 客户编号。
- 把 depot 写进 route。
- 漏客户或重复客户。
- route 超过车辆容量。
- JSON 格式不符合官方要求。
- 不能处理 CVRP-100。
- 只报告低 cost，却没有检查 feasibility。

## 输出文件

计划中的正式中文学习笔记应写入：

```text
docs/vrp_project_technical_learning_note_zh.md
```

## 不在本阶段范围内的内容

这份笔记不会：

- 实现 `solve.py`。
- 训练神经模型。
- 生成公开测试集预测结果。
- 编写最终报告或展示 slides。
- 安装新的依赖。

这些任务会在这份笔记被 review 之后，于后续实现阶段处理。

## 验收标准

这份笔记完成后应满足：

- 初学者可以用自己的话解释项目目标、数据格式、可行性规则和评价指标。
- 笔记能够说明为什么应该先做可行 baseline，再做复杂 AI 建模。
- 笔记中的每个概念都能连接到本项目的文件和提交接口。
- 笔记给出足够方向，使后续 baseline 实现可以开始设计。
- 笔记不会偏离到无关的 VRP 理论，也不会堆砌过多数学细节。
