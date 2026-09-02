# Multi-step CreativeIR implementation note

- Video: `7106594312292453675`
- Model: `google/gemini-3.8-flash`
- Pipeline: `issue-4-multistep-perception-v0.1`
- Shot analysis prompt: `openrouter-gemini-shot-analysis-v0.1`
- Global synthesis prompt: `openrouter-gemini-global-synthesis-v0.1`
- Parsed output: `creative_ir.parsed.json`
- Raw shot analysis: `creative_ir.shot_analysis.raw.json`
- Raw global synthesis: `creative_ir.global_synth.raw.json`
- Usage record: `creative_ir.usage.json`
- Deterministic perception: `perception.json`
- Validation: repository `schemas/creative_ir_v0_1.json` with Draft 2020-12 plus ordered temporal/reference checks.

## Deterministic facts (from ffprobe)

- Duration: 24.402s (authoritative)
- Resolution: 576x1024 (vertical_9_16)
- FPS: 29.97
- Video codec: h264
- Audio codec: aac
- File size: 3308379 bytes

## Detected scenes (PySceneDetect)

6 scenes with boundaries: [4.972, 6.974, 7.708, 19.753, 21.722] seconds.

## Recommendation

validated-for-pilot

Multi-step pipeline with deterministic preprocessing produces materially better shot boundaries and media facts than the single-pass baseline.
