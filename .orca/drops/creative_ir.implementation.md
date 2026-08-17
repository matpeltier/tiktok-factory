# Gemini CreativeIR implementation note

- Video: `7106594312292453675`
- Model: `gemini-3.6-flash`
- Prompt: `gemini-single-pass-creative-ir-v0.1`
- Parsed output: `creative_ir.parsed.json`
- Raw model response: `creative_ir.raw.json`
- Usage and cost record: `creative_ir.usage.json`
- Validation: repository `schemas/creative_ir_v0_1.json` with Draft 2020-12 plus ordered temporal/reference checks.

## Evidence

The executed source run reports 24.00 seconds, five ordered shots, and hard cuts near 5.00, 7.00, 19.50, 22.00 seconds. The broad structure, challenge card OCR, reveal count and visual continuity match manual frame inspection.
The single pass also produced unsupported shot-1 OCR; exact watermark/OCR timing and audio identity remain high-risk details.

## Recommendation

multi-pass-needed

Use this single-pass result as a baseline; add deterministic frame/time sampling, scene-boundary detection, and focused OCR/audio passes before relying on the output operationally.
