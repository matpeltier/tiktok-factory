# CreativeIR v0.1 notes

The repository now contains two projections of the same creative analysis:

- `schemas/creative_ir_v0_1.json`: Full CreativeIR, rich and auditable.
- `schemas/canonical_ir_v0_1.json`: compact CanonicalIR for ranking and modeling.

`schemas/performance_v0_1.json` is deliberately separate from both. CreativeIR describes what the video is; performance describes how it performed.

## Full IR versus CanonicalIR

The Full IR keeps the complete analysis in a compact hierarchy:

```text
source + decompilation
observed
├── context / hook / narrative / marketing / commercial
└── shots[]
    ├── visual / camera / text / dialogue / audio / editing
    ├── inferred semantic role and mechanisms
    └── generation prompt and continuity constraints
inferred
└── concept / audience / hook / narrative / marketing / commercial
generation
└── global reconstruction brief
```

Shot-level details are not re-aggregated into duplicate top-level `visual_description`, `camera`, `audio` or `editing` blocks. The narrative and hook refer to shot/text IDs instead.

CanonicalIR keeps only compact `observed` and `inferred` fields: controlled labels, timing, subjects, text roles, camera motion, audio presence, narrative structure and commercial status. It intentionally has no `generation` block. The initial ranker should consume CanonicalIR or a projection of `observed + inferred`, not Gemini's reconstruction prose.

## Separation rule

- `observed`: what is visible/audible or present in source metadata, with `evidence` in the Full IR.
- `inferred`: a creative, audience, attention or commercial interpretation, with `confidence` and `rationale` where interpretation is rich.
- `generation`: instructions for reconstructing the creative result, only in the Full IR.

The `decompilation` block records `model`, `prompt_version`, `schema_version`, `created_at`, `pipeline_version` and whether the annotation was automated, manual or hybrid. This makes annotations comparable when the decompiler or prompt changes.

## Commercial block

The Full IR includes a reusable commercial layer for affiliate and product videos without forcing non-commercial examples to invent ad content:

- observed: product presence, first appearance, exact product/claim/offer/CTA text when present;
- inferred: problem, desire, promise, proof type, trust signals, objections and CTA type;
- `not_applicable` is valid for a non-ad such as the Minecraft reference example.

This lets later modeling distinguish attention mechanisms from persuasion mechanisms such as problem, proof, product and CTA.

## Ambiguous fields and enums

- `source.observed.creator_handle` identifies the publishing account. It does not claim that the publishing account is the original content creator.
- `time_range` uses seconds from the beginning of the media. Boundaries are manual annotations in this v0.1 example; `end_seconds` is exclusive by convention.
- `evidence.kind` identifies the evidence channel, not its reliability. Interpretation reliability belongs in `confidence`.
- `spoken_dialogue` and sound effects allow `uncertain`; absence should only be used when the review supports absence.
- `camera.motion` describes deliberate camera movement, not subject or world animation.
- Narrative, semantic-role, hook and persuasion enums describe editorial function or mechanism, not genre. Free-text summaries remain available where the controlled vocabulary is insufficient.
- `high`, `medium` and `low` confidence apply to interpretation, not to the existence of a source file.

## Examples and validation

The hand-written Full IR example is [the Minecraft TikTok example](../examples/creative_ir_v0_1_tiktok_7106594312292453675.json). The compact projection is [the CanonicalIR example](../examples/canonical_ir_v0_1_tiktok_7106594312292453675.json). The referenced media assets are not committed to the repository.

When `jsonschema` is installed, validate both examples with:

```bash
python3 - <<'PY'
import json
from pathlib import Path
from jsonschema import Draft202012Validator

checks = [
    ("schemas/creative_ir_v0_1.json", "examples/creative_ir_v0_1_tiktok_7106594312292453675.json"),
    ("schemas/canonical_ir_v0_1.json", "examples/canonical_ir_v0_1_tiktok_7106594312292453675.json"),
]
for schema_path, example_path in checks:
    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    example = json.loads(Path(example_path).read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(example)
    print(f"valid: {example_path}")
PY
```
