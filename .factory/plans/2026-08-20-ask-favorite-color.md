# Plan: Ask the human their favorite color

## Task
Create `answer.txt` at the repo root with the human's favorite color.

## Steps

1. Use `dk_ask_human` to ask the human what their favorite color is.
2. Write the human's answer to `answer.txt`.
3. Run `git status` and `git diff` to verify the change.
4. Return `{"passed": true}` or `{"passed": false, "findings": "..."}`.

## Notes
- The `dk_ask_human` MCP tool sends the question and blocks until the human replies.
- No tests or lint/typecheck needed for this task.
- The answer must be exactly the human's response (trimmed of whitespace).
