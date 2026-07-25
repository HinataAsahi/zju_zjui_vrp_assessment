# VRP Technical Learning Note Design

Date: 2026-07-24

## Purpose

Create a Chinese technical learning note that helps the student understand the
CVRP assessment project well enough to start implementation. The note should be
introductory but practical: it should explain the problem, dataset, feasibility
rules, evaluation metrics, baseline ideas, and AI method options without relying
on heavy mathematical derivations.

The note is not the final report, presentation, or code implementation. It is a
foundation document for later work on `solve.py`, experiments, report writing,
and slides.

## Source Context

The project directory contains:

- `VRP_project/vrp_project_outline.docx`: official project description.
- `VRP_project/VRPData/train_data.pkl`: 7000 labeled CVRP-50 instances.
- `VRP_project/VRPData/validation_data.pkl`: 1000 labeled CVRP-50 instances.
- `VRP_project/VRPData/check_data_to_students.pkl`: 1000 CVRP-50 and 500
  CVRP-100 public test instances without labels.

The labeled datasets store instances as:

```text
(depot, loc, demand, capacity, routes, cost)
```

The public test dataset stores instances as:

```text
(depot, loc, demand, capacity)
```

The official inference interface requires a top-level `solve.py` that reads an
input `.pkl` file and writes a UTF-8 JSON file with `format_version:
"cvrp_v1"` and one route solution per instance.

## Audience And Depth

The note should target a student who has basic Python and machine learning
knowledge but is new to CVRP and neural combinatorial optimization.

Depth target: "from beginner to able to start work".

The note should:

- Explain core concepts in Chinese.
- Use concrete examples from this project.
- Avoid excessive formulas.
- Keep links to later implementation decisions visible.
- Make common failure modes explicit.

## Proposed Structure

### 1. Project In One Sentence

Explain that the task is a Capacitated Vehicle Routing Problem: vehicles start
from one depot, visit all customers, obey vehicle capacity limits, and minimize
total travel distance.

### 2. How To Read The Data

Explain the role of each `.pkl` file, the tuple fields, the difference between
labeled and unlabeled data, customer indexing, capacity values, and CVRP-50 /
CVRP-100 split.

### 3. What Makes A CVRP Solution Feasible

Define feasibility rules:

- Every customer is visited exactly once.
- No customer is missing.
- No customer is duplicated.
- Every route demand is no greater than vehicle capacity.
- The depot is implicit and must not appear in route lists.
- Customer IDs in JSON output are 1-based.

### 4. How To Understand The Evaluation Metrics

Explain:

- Average Total Route Cost.
- Feasibility Rate.
- Average Gap to a reference baseline.
- Average Inference Time.

Clarify why feasibility is the first priority, then route cost and inference
time.

### 5. Why Build A Baseline First

Explain the role of a baseline in this assessment:

- It proves data parsing and output format.
- It gives a safe valid solution before neural modeling.
- It creates a comparison point for the report.

Introduce practical baseline ideas:

- Capacity-aware nearest neighbor.
- Greedy route construction.
- Route splitting.
- 2-opt local improvement.

### 6. How To Understand AI-Based Methods

Introduce the main model families at an introductory level:

- Pointer Network style sequential construction.
- Attention / Transformer-based routing policies.
- POMO-style reinforcement learning.
- Learning-based improvement methods.

For each method, explain what the model learns and why it might help.

### 7. Recommended Project Roadmap

Propose a staged path:

1. Understand data and implement evaluation tools.
2. Build a feasible baseline and official `solve.py`.
3. Try a simple supervised imitation or lightweight neural heuristic.
4. Analyze performance and limitations for report and presentation.

### 8. Common Pitfalls Checklist

List common errors:

- Treating test instances as length-6 tuples.
- Using 0-based customer IDs in output.
- Including depot in routes.
- Missing or duplicated customers.
- Exceeding route capacity.
- Producing invalid JSON format.
- Failing on CVRP-100.
- Reporting low cost without checking feasibility.

## Output Files

The planned learning note should be written as both Markdown and PDF:

```text
docs/vrp_project_technical_learning_note_zh.md
docs/vrp_project_technical_learning_note_zh.pdf
```

## Out Of Scope

This note will not:

- Implement `solve.py`.
- Train a neural model.
- Generate public-set predictions.
- Write the final report or presentation slides.
- Install new dependencies.

Those tasks should be handled in later implementation phases after the note is
reviewed.

## Acceptance Criteria

The note is complete when:

- A beginner can explain the project goal, dataset format, feasibility rules,
  and metrics in their own words.
- The note makes clear why a feasible baseline should come before complex AI
  modeling.
- The note connects each concept to this project's files and submission
  interface.
- The Markdown note and PDF export contain the same approved content.
- The note gives enough direction to start designing the baseline implementation.
- The note does not drift into unrelated VRP theory or excessive mathematical
  detail.
