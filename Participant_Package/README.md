The supplied Participant Package, retained as the brief requires (§8):
`Challenge_Brief.md`, `Setup_Instructions.md`, the `handout/` guides,
`public_questions.jsonl`, and the `answer_template.json`,
`questions_template.json`, `submission_template.json` and `validate.json`
contract samples.

The 15 public questions are the calibration set, run by
`training/run_calibration.py` with **only** the `prompt` field passed through.
Their published reference values are also what the determinism tests in
`training/` assert against — keyed by tool arguments, never by question id
(CON-9, AC-13).
