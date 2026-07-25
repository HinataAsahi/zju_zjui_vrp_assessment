# VRP Technical Learning Note Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Write the formal Chinese technical learning note for the ZJU/ZJUI CVRP assessment project.

**Architecture:** This is a documentation-only task. The approved Chinese design spec defines the structure, and the final note will be a single Markdown file with focused sections that explain the project from problem understanding to practical baseline readiness.

**Tech Stack:** Markdown, local project `.pkl` dataset observations, official project outline parsed from `VRP_project/vrp_project_outline.docx`.

## Global Constraints

- Final note path: `docs/vrp_project_technical_learning_note_zh.md`.
- Source spec: `docs/superpowers/specs/2026-07-24-vrp-technical-learning-note-design.zh.md`.
- Do not implement `solve.py` in this task.
- Do not train a neural model in this task.
- Do not generate public-set predictions in this task.
- Do not write the final report or presentation slides in this task.
- Do not install new dependencies in this task.
- Do not commit the raw project data directory unless the user explicitly asks for it.

---

### Task 1: Write The Chinese Technical Learning Note

**Files:**
- Create: `docs/vrp_project_technical_learning_note_zh.md`
- Read: `docs/superpowers/specs/2026-07-24-vrp-technical-learning-note-design.zh.md`
- Read: `.firecrawl/vrp_project_outline.md`

**Interfaces:**
- Consumes: Approved spec sections and project observations from the official outline and dataset inspection.
- Produces: A Chinese learning note that later baseline, `solve.py`, evaluation, report, and presentation work can reference.

- [ ] **Step 1: Read the approved Chinese spec**

Run:

```bash
sed -n '1,260p' docs/superpowers/specs/2026-07-24-vrp-technical-learning-note-design.zh.md
```

Expected: The command prints the approved note design, including target audience, structure, output path, out-of-scope items, and acceptance criteria.

- [ ] **Step 2: Read the official outline parse**

Run:

```bash
sed -n '1,220p' .firecrawl/vrp_project_outline.md
```

Expected: The command prints the official project objective, dataset format, evaluation metrics, submission interface, and deliverables.

- [ ] **Step 3: Create the formal learning note**

Create `docs/vrp_project_technical_learning_note_zh.md` with these exact top-level sections:

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

The body must explain concepts in Chinese, use this project's actual file names and tuple formats, and keep the depth at "from beginner to able to start work".

- [ ] **Step 4: Verify the note covers the approved sections**

Run:

```bash
rg -n "用一句话理解|数据应该怎么读|合法解|评价指标|baseline|AI-based|项目推进路线|常见坑" docs/vrp_project_technical_learning_note_zh.md
```

Expected: The command finds headings or body text for every approved section.

- [ ] **Step 5: Scan for unfinished markers**

Run:

```bash
rg -n "T[B]D|TO[D]O|place[ -]?holder|[?][?][?]|待[定]|未[定]|以后再[说]" docs/vrp_project_technical_learning_note_zh.md
```

Expected: The command exits with status 1 and prints no matches.

- [ ] **Step 6: Check git diff scope**

Run:

```bash
git status --short
git diff -- docs/vrp_project_technical_learning_note_zh.md
```

Expected: The intended new note appears. Raw project data remains untracked and is not staged.

- [ ] **Step 7: Commit and push the note**

Run:

```bash
git add docs/vrp_project_technical_learning_note_zh.md
git commit -m "docs: add VRP technical learning note"
git push
```

Expected: The commit contains only the formal Chinese learning note and is pushed to `origin/main`.
