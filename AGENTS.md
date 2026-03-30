# AGENTS.md

## Environment

- train env Python is `/home/yzh/miniconda3/envs/kronos/bin/python`, use `conda activate kronos` to activate.
- huggingface or hf clie env Python is `/home/yzh/miniconda3/bin/python`, user `conda activate base` to activate.

## Repository expectations

- Document changes in `zlab/` when you change behavior.
- Do not install packages automactically, please let user to install
- you should known the env is in Multi-tenant shared Ubuntu environment，Provide operation commands whenever possible, allowing users to decide whether to execute them.

## Writing style
- When generating documents, write in a compact, professional style.
- Avoid unnecessary blank lines, excessive spacing, decorative formatting, and verbose transitions.
- Keep the layout tight and efficient. Do not spread short ideas across many lines.
- Use headings only when they improve readability. Do not over-segment the document.
- Prefer short paragraphs over long, loose exposition, but do not fragment the text excessively.

### Content density
- Explain key ideas, design decisions, trade-offs, and mechanisms in sufficient depth.
- For secondary or obvious points, give only the necessary description.
- Do not pad the output with generic background, repeated summaries, or restatements of the prompt.
- Do not include filler or motivational text unless explicitly requested.

### Structure
- Prioritize information in this order:
  1. Core conclusion or answer
  2. Important reasoning or mechanism
  3. Essential examples
  4. Optional edge notes only if useful
- Keep bullet lists concise. Do not create long lists for simple topics.
- Do not use tables unless they clearly improve comparison or density.

### Formatting
- Use clean Markdown.
- Do not use uppercase letters in markdown document names.
- Avoid excessive indentation.
- Avoid isolated single-sentence paragraphs unless emphasis is necessary.
- Avoid redundant section headers such as “Summary”, “Conclusion”, and “Final Thoughts” unless they add real value.
- Do not add extra spacing simply for visual looseness.

### Math and formula formatting
- Use inline formulas only for short and simple expressions.
- Do not squeeze complex formulas, derivations, or multi-term expressions into a sentence.
- For complex equations, use display math on separate lines.
- For multi-step derivations, use multi-line aligned math formatting so the structure is readable.
- Prefer readability over compactness for mathematical content.
- When an equation is important, introduce it in text first, then present it as a separate block.

### Explanation policy
- For important content, explain thoroughly enough that the reader can understand the “why” and “how,” not only the “what.”
- For straightforward content, be brief.
- When in doubt, compress obvious parts and spend space on the non-obvious parts.

### Output preference
- The document should feel dense, clear, and deliberate.
- Every paragraph should carry information.
- Remove repetition before finishing.