# VRP 技术学习笔记实施计划

> **给 agentic workers：** 必需子技能：使用 `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans`，按任务逐步执行本计划。步骤使用 checkbox（`- [ ]`）语法进行跟踪。

**目标：** 为 ZJU 的 ZJUI 学院 CVRP 考核项目撰写正式中文技术学习笔记。

**架构：** 这是一个纯文档任务。已批准的中文设计 spec 规定了笔记结构，最终产物包括一份 Markdown 源文件和一份由该 Markdown 导出的 PDF 文件。两个文件应包含相同内容，用聚焦的章节把项目从问题理解讲到可以开始做 baseline 的程度。

**技术栈：** Markdown、Pandoc 与 `--pdf-engine=weasyprint`、本地 `.pkl` 数据集观察结果、从 `VRP_project/vrp_project_outline.docx` 解析得到的官方项目说明。

## 全局约束

- 正式 Markdown 笔记路径：`docs/vrp_project_technical_learning_note_zh.md`。
- 正式 PDF 笔记路径：`docs/vrp_project_technical_learning_note_zh.pdf`。
- 来源 spec：`docs/superpowers/specs/2026-07-24-vrp-technical-learning-note-design.zh.md`。
- 本任务不实现 `solve.py`。
- 本任务不训练神经模型。
- 本任务不生成公开测试集预测结果。
- 本任务不编写最终报告或展示 slides。
- 如果生成 Markdown 或 PDF 交付物需要新的依赖，可以按需安装。安装时需要说明原因，并记录可复现信息。
- 除非用户明确要求，否则不提交原始项目数据目录。

---

### 任务 1：撰写中文技术学习笔记

**文件：**
- 创建：`docs/vrp_project_technical_learning_note_zh.md`
- 创建：`docs/vrp_project_technical_learning_note_zh.pdf`
- 阅读：`docs/superpowers/specs/2026-07-24-vrp-technical-learning-note-design.zh.md`
- 阅读：`.firecrawl/vrp_project_outline.md`

**接口：**
- 输入：已批准的 spec 章节，以及来自官方说明和数据集检查的项目观察。
- 输出：一份 Markdown 格式和一份 PDF 格式的中文学习笔记，后续 baseline、`solve.py`、评估、报告和展示材料都可以引用它。

- [ ] **步骤 1：阅读已批准的中文 spec**

运行：

```bash
sed -n '1,260p' docs/superpowers/specs/2026-07-24-vrp-technical-learning-note-design.zh.md
```

预期结果：命令输出已批准的学习笔记设计，包括目标读者、结构、输出路径、不在范围内的内容和验收标准。

- [ ] **步骤 2：阅读官方项目说明的解析结果**

运行：

```bash
sed -n '1,220p' .firecrawl/vrp_project_outline.md
```

预期结果：命令输出官方项目目标、数据集格式、评价指标、提交接口和交付物要求。

- [ ] **步骤 3：创建正式学习笔记**

创建 `docs/vrp_project_technical_learning_note_zh.md`，并使用以下精确的顶层章节：

```markdown
# VRP 考核项目技术学习笔记

## 1. 用一句话理解这个项目

## 2. CVRP 到底是什么

## 3. 本项目的数据应该怎么读

## 4. 什么样的 routes 才是合法解

## 5. 评价指标怎么理解

## 6. 为什么第一步应该先做 baseline

## 7. Baseline 可以怎么设计

## 8. AI-based 方法怎么入门理解

## 9. 推荐的项目推进路线

## 10. 常见坑检查清单

## 11. 学完这份笔记后应该能做什么
```

正文必须用中文解释概念，使用本项目中的真实文件名和 tuple 格式，并把深度控制在“从入门到可以开始动手”的水平。

- [ ] **步骤 4：导出 PDF 版本**

运行：

```bash
pandoc docs/vrp_project_technical_learning_note_zh.md -o docs/vrp_project_technical_learning_note_zh.pdf --pdf-engine=weasyprint
```

预期结果：从 Markdown 源文件生成 `docs/vrp_project_technical_learning_note_zh.pdf`。

- [ ] **步骤 5：验证笔记覆盖已批准章节**

运行：

```bash
rg -n "用一句话理解|数据应该怎么读|合法解|评价指标|baseline|AI-based|项目推进路线|常见坑" docs/vrp_project_technical_learning_note_zh.md
```

预期结果：命令能找到每个已批准章节对应的标题或正文内容。

- [ ] **步骤 6：验证 PDF 文件存在**

运行：

```bash
test -s docs/vrp_project_technical_learning_note_zh.pdf
```

预期结果：命令以状态码 0 退出，证明 PDF 文件存在且不是空文件。

- [ ] **步骤 7：扫描未完成标记**

运行：

```bash
rg -n "T[B]D|TO[D]O|place[ -]?holder|[?][?][?]|待[定]|未[定]|以后再[说]" docs/vrp_project_technical_learning_note_zh.md
```

预期结果：命令以状态码 1 退出，并且不输出任何匹配项。

- [ ] **步骤 8：检查 git diff 范围**

运行：

```bash
git status --short
git diff -- docs/vrp_project_technical_learning_note_zh.md
```

预期结果：能看到目标 Markdown 学习笔记和 PDF 导出文件的新内容。原始项目数据仍保持未跟踪状态，且没有被暂存。

- [ ] **步骤 9：提交并推送学习笔记**

运行：

```bash
git add docs/vrp_project_technical_learning_note_zh.md docs/vrp_project_technical_learning_note_zh.pdf
git commit -m "docs: add VRP technical learning note"
git push
```

预期结果：提交只包含正式中文学习笔记的 Markdown 和 PDF 文件，并被推送到 `origin/main`。
