# VRP Assessment 答辩提纲

## Slide 1: 项目目标

- 任务：求解 CVRP，输出满足容量约束的车辆路径。
- 目标：保证可行性的前提下尽量降低总路径长度。
- 最终入口：`solve.py`。

## Slide 2: 数据与约束

- 每个实例包含 depot、customer 坐标、demand 和车辆容量。
- 每个 customer 必须访问一次且只能访问一次。
- 每条 route 的 demand 总和不能超过车辆容量。
- 输出 route 使用 1-based customer 编号，不包含 depot。

## Slide 3: 评价指标

- Feasible count：是否全部实例可行。
- Average cost：平均总路径长度。
- Average gap：与参考解相比的平均差距。
- Runtime：public check 需要控制在时间要求内。

## Slide 4: 方法演进

| 阶段 | 方法 | 作用 |
| --- | --- | --- |
| 1 | 最近邻 | 快速构造可行解 |
| 2 | 2-opt | 优化单条 route 内顺序 |
| 3 | Relocate | 在不同 route 间移动客户 |
| 4 | Swap | 在不同 route 间交换客户 |

## Slide 5: 最终默认方法

```text
nearest_2opt_relocate_limited_swap
```

- 容量感知最近邻构造初始解。
- Route 内 2-opt 消除局部绕路。
- 有限轮 relocate 改善 route 间分配。
- 有限轮 swap 进一步改善不同 route 的客户组合。

## Slide 6: Validation 结果

| 方法 | Feasible | Average cost | Average gap |
| --- | ---: | ---: | ---: |
| Nearest | 1000/1000 | 13.934951704742822 | 0.31894172859263836 |
| Nearest + 2-opt | 1000/1000 | 13.410121525149266 | 0.2688820821667571 |
| Limited relocate | 1000/1000 | 11.675451253965944 | 0.10360446980377136 |
| Limited relocate + limited swap | 1000/1000 | 11.513862926474488 | 0.08817281823708832 |

## Slide 7: 时间与默认选择

- 最终方法 public check 1500/1500 可行。
- 生成 `outputs/heuristic/predictions.json` 耗时约 148 秒。
- 在质量和时间之间选择 `nearest_2opt_relocate_limited_swap` 作为默认提交方法。
- 可插入图片：`docs/assets/cvrp_sample_solution.png`。

## Slide 8: 学习方法探索

- 尝试 supervised imitation：学习客户优先级。
- 尝试 pairwise ranking loss：增强排序学习。
- 尝试 priority RL finetuning：直接用路径 cost 作为强化学习信号。
- 结论：学习路线可行，但当前效果弱于 heuristic 默认方案。

## Slide 9: RL 结果解释

- Priority RL full validation gap 为 0.158813。
- 默认 heuristic full validation gap 为 0.088173。
- 当前 RL 不是完整 POMO，只是在优先级模型上做轻量微调。
- 因此 RL 作为实验路线记录，不替换默认提交。
- 可插入图片：`docs/assets/method_comparison_gap.png`。

## Slide 10: 最终结论

- 默认提交方法：`nearest_2opt_relocate_limited_swap`。
- 优点：全实例可行、CPU 可运行、无需训练依赖、public check 时间可控。
- 学习方法作为 future work：可进一步研究完整 POMO 或更强的神经组合优化框架。
