# AGENTS.md

## Environment

- Training env: `/home/yzh/miniconda3/envs/kronos/bin/python`
- HF or CLI env: `/home/yzh/miniconda3/bin/python`
- Activate training env with `conda activate kronos`
- Activate base env with `conda activate base`

## Repository model

This repo follows a mixed workflow:

- Engineering work stays in the original code paths and is implemented directly in the existing modules.
- Research and finetuning work is recorded under `zlab/`.
- Deep code reading notes may be placed under `zdoc/` when that directory exists or is useful.

Do not force every helper into `zlab/` or `zdoc/`. If a utility belongs to the runtime code path, keep it near the code that uses it.

## Fixed repo assets

Prefer keeping these assets in a long-lived repo:

- `AGENTS.md`: operating rules, environment, constraints, workflow
- `externalContext.md`: durable project context that can be reused across projects
- `zlab/`: experiment definitions, prompts, result analysis, finetune scripts, run guides
- `zdoc/`: code reading notes, architecture maps, module walkthroughs

## Working rules

- Document behavior changes in `zlab/`.
- Do not install packages automatically. Provide commands and let the user decide whether to run them.
- Assume the machine is a multi-tenant shared Ubuntu environment. Avoid unnecessary system changes and prefer conservative resource usage.
- Give runnable commands whenever possible.
- Prefer the cheapest validation that still checks the relevant change.

## Core principles

- Understand before acting. Identify task type, concrete objective, constraints, affected files, and expected output before broad changes.
- Stay grounded in the repository. Prefer current code, interfaces, and architecture over generic textbook patterns.
- Be concise but not shallow. Explain mechanism and design intent without drifting into detached theory.
- Plan before large edits. For medium or large tasks, state scope, approach, file impact, and verification path before coding.
- Make changes verifiable. Keep modifications auditable and state how they should be checked.
- Debug from evidence. Separate observed facts, likely causes, and hypotheses; do not present guesses as confirmed root causes.

## Standard workflow

For any non-trivial task, the default workflow is:

1. Read the relevant rules and the minimum required context.
2. Identify the task type: engineering change or research experiment.
3. Limit the initial read scope to the most relevant files and directories.
4. State the plan, expected outputs, risks, and validation path.
5. Make small, local changes instead of broad speculative refactors.
6. Validate after each meaningful step.
7. Update the relevant `zlab/` or `zdoc/` notes if behavior or operating assumptions changed.

For planning-heavy work, prefer the sequence:

- background
- assumptions
- plan
- execution
- verification

## Engineering workflow

Use this path for cloud-native, backend, infra, or feature-delivery repositories.

- Work directly in the existing package layout.
- Prefer small patches over large rewrites.
- After each step, run the cheapest relevant check: compile, typecheck, lint, unit test, then integration test if needed.
- If the task is broad, split it into 3 to 7 concrete steps.
- If a step fails, repair locally before continuing.

Typical fixed assets for this mode:

- `AGENTS.md`
- architecture or current-work notes
- ignore files
- a few reusable task templates or skills

## Research workflow

Use this path for model finetuning, evaluation, ablation, and experiment management.

- Define the task before coding: data range, split, baseline, ablation, metrics, outputs.
- Separate baseline, finetune, and auxiliary-context experiments clearly.
- Distinguish training objective, selection objective, and final evaluation objective.
- Keep run naming, output paths, and metrics exports stable so experiments remain comparable.
- Save enough metadata to reproduce a run: hyperparameters, dataset version, checkpoint path, evaluation config.
- Prefer one-variable-at-a-time tuning unless the goal is a structured sweep.

Research deliverables usually include:

- task definition
- run guide
- result summary
- final analysis

## Task handling modes

Classify the request before deciding how to respond.

### Code explanation

- Explain the role of the code in the surrounding module or system first.
- Focus on control flow, data flow, interfaces, invariants, side effects, and failure paths.
- Explain syntax only when it affects semantics or common misunderstanding.
- Avoid detached language tutorials.

### Question answering and design clarification

- Answer the direct question first.
- Then explain why the current design works, its trade-offs, and when another design would be better.
- Tie the answer to actual workload, failure model, concurrency model, or repository convention when relevant.

### Planning and walkthrough design

- Restate the goal in precise engineering terms.
- Make scope boundaries explicit.
- Describe module responsibilities, file impact, development stages, and verification strategy.
- Prefer phased delivery over one-shot implementation.

### Phase-based implementation

- Implement only the requested phase or scoped subtask.
- Do not silently expand scope.
- Keep interfaces stable unless the phase explicitly changes them.
- End with a short mapping from implementation to phase goal and verification steps.

### Bug localization and fixing

- Start from logs, symptoms, traces, and observed behavior.
- Walk the likely execution path before proposing a fix.
- Prefer minimal root-cause fixes over broad rewrites.
- When useful, explain why the issue escaped earlier checks and how to harden against recurrence.

## Writing style

- Write in a compact, professional style.
- Avoid unnecessary blank lines, decorative formatting, and verbose transitions.
- Prefer short dense paragraphs over loose exposition.
- Use headings only when they improve readability.
- Keep bullet lists short.
- Avoid tables unless they materially improve comparison.

## Explanation policy

- For important content, explain both the mechanism and the reason.
- For obvious content, stay brief.
- Remove repetition before finishing.

## Output boundaries

- Do not produce large unstructured plans without file impact and test strategy.
- Do not over-explain syntax that does not affect understanding.
- Do not implement broad changes when only one scoped phase is requested.
- Do not treat speculative causes as confirmed in debugging.

## Output preference

- Prioritize: answer, mechanism, essential example, optional edge note.
- Every paragraph should carry information.
- Keep the document deliberate and reusable.
