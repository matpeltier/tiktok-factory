export const meta = {
  name: "dark-kitchen-issue",
  description: "Implement, review, and verify one GitHub issue.",
  whenToUse: "For a Dark Kitchen AI-managed GitHub issue.",
  phases: [
    { title: "Architecture" },
    { title: "Implementation" },
    { title: "Independent review" },
    { title: "Fix and reverify" },
  ],
};

import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
const configPath = process.env.FACTORY_CONFIG_PATH ?? path.join(process.cwd(), ".factory", "config.json");
const factoryConfig = JSON.parse(await readFile(configPath, "utf8")) as { agents: Record<string, string> };

// codex-dynamic-workflows injects agent(), phase(), and args into this script.
const issue = args as { number: number; title: string; body: string; labels: string[]; resultPath?: string };

const IMPLEMENTATION_SCHEMA = {
  type: "object", additionalProperties: false,
  properties: {
    status: { type: "string", enum: ["success", "needs_human"] },
    summary: { type: "string" }, question: { type: "string" },
    category: { type: "string" }, recommendation: { type: "string" },
    tests: { type: "array", items: { type: "string" } }
  },
    required: ["status", "summary", "tests"]
};

const REVIEW_SCHEMA = {
  type: "object", additionalProperties: false,
  properties: {
    hasBlockingFindings: { type: "boolean" }, summary: { type: "string" },
    findings: { type: "array", items: { type: "string" } }
  },
  required: ["hasBlockingFindings", "summary", "findings"]
};

const FIX_SCHEMA = {
  type: "object", additionalProperties: false,
  properties: {
    status: { type: "string", enum: ["fixed", "needs_human"] },
    summary: { type: "string" }, question: { type: "string" },
    category: { type: "string" }, recommendation: { type: "string" }
  },
    required: ["status", "summary"]
};

function provider(role: keyof typeof factoryConfig.agents): string {
  return factoryConfig.agents[role];
}

function warrantsArchitecture(): boolean {
  return /architecture|design|schema|migration|refactor|integration|api|database/i.test(`${issue.title}\n${issue.body}`)
    || issue.body.length > 900;
}

async function save(result: unknown): Promise<unknown> {
  const resultPath = issue.resultPath ?? path.join(process.cwd(), ".factory", "runtime", String(issue.number), "result.json");
  await mkdir(path.dirname(resultPath), { recursive: true });
  await writeFile(resultPath, JSON.stringify(result, null, 2) + "\n", "utf8");
  return result;
}

let finalResult: any;
try {
  let architecture = "No separate architecture phase was needed for this issue.";
  if (warrantsArchitecture()) {
    phase("Architecture");
    const plan = await agent(
      `Read AGENTS.md and issue #${issue.number}: ${issue.title}\n\n${issue.body}\n\nPlan the smallest implementation that satisfies the stated acceptance criteria. Do not invent product requirements. If materially ambiguous or impossible, return a human blocker with a concrete question.`,
      { label: "architect", phase: "Architecture", provider: provider("architect"), schema: IMPLEMENTATION_SCHEMA }
    );
    if (!plan) {
      finalResult = { status: "failed", summary: "The architect returned no structured result.", attempts: ["Architecture agent returned null after retries."] };
    } else if (plan.status === "needs_human") {
      finalResult = { status: "needs_human", category: plan.category, summary: plan.summary, question: plan.question, recommendation: plan.recommendation, evidence: [] };
    } else {
      architecture = plan.summary;
    }
  }

  if (!finalResult) {
    phase("Implementation");
    const implementation = await agent(
      `You are the implementation owner for GitHub issue #${issue.number}: ${issue.title}.\n\nIssue body and acceptance criteria:\n${issue.body}\n\nArchitecture context:\n${architecture}\n\nRead AGENTS.md and inspect the repository. Implement only this issue. Run relevant tests, lint, and typecheck where configured. Do not change product requirements, launch other issues, or ask about routine coding/debugging choices. Before reporting success, inspect the final diff and commit meaningful changes on the current branch. If a genuinely ambiguous/impossible requirement, missing credential, destructive approval, or repeated failure blocks the work, return needs_human with a precise question and evidence.`,
      { label: "implementer", phase: "Implementation", provider: provider("implementer"), schema: IMPLEMENTATION_SCHEMA }
    );
    if (!implementation) {
      finalResult = { status: "failed", summary: "The implementer returned no structured result.", attempts: ["Implementation agent returned null after retries."] };
    } else if (implementation.status === "needs_human") {
      finalResult = { status: "needs_human", category: implementation.category, summary: implementation.summary, question: implementation.question, recommendation: implementation.recommendation, evidence: implementation.tests };
    } else {
      let reviewSummary = "No blocking review findings.";
      let tests = implementation.tests;
      let unresolved: string[] = [];
      for (let loop = 0; loop < 2; loop += 1) {
        phase("Independent review");
        const review = await agent(
          `Independently review issue #${issue.number} and the current worktree. Read the issue, AGENTS.md, git diff, committed changes, and test results. Check every acceptance criterion, correctness, regressions, and missing tests. Do not rewrite code in this review session. Return only actionable blocking findings.`,
          { label: `reviewer-${loop + 1}`, phase: "Independent review", provider: provider("reviewer"), schema: REVIEW_SCHEMA }
        );
        if (!review) {
          finalResult = { status: "failed", summary: "The reviewer returned no structured result.", attempts: ["Review agent returned null after retries."] };
          break;
        }
        reviewSummary = review.summary;
        unresolved = review.findings;
        if (!review.hasBlockingFindings) break;
        phase("Fix and reverify");
        const fixed = await agent(
          `Fix the blocking review findings for issue #${issue.number}. Findings:\n- ${review.findings.join("\n- ")}\n\nMake the smallest correct changes, rerun relevant tests, inspect the diff, and commit the fix. Do not ask about routine debugging. If a finding exposes a real product blocker, return needs_human.`,
          { label: `fixer-${loop + 1}`, phase: "Fix and reverify", provider: provider("fixer"), schema: FIX_SCHEMA }
        );
        if (!fixed) {
          finalResult = { status: "failed", summary: "The fixer returned no structured result.", attempts: ["Fix agent returned null after retries."] };
          break;
        }
        if (fixed.status === "needs_human") {
          finalResult = { status: "needs_human", category: fixed.category, summary: fixed.summary, question: fixed.question, recommendation: fixed.recommendation, evidence: review.findings };
          break;
        }
        tests = [...tests, fixed.summary];
        if (loop === 1) {
          finalResult = { status: "needs_human", category: "repeated_failure", summary: "The independent review still has blocking findings after two fix loops.", question: "How should the remaining review findings be resolved?", recommendation: "Review the preserved worktree and choose the intended behavior.", evidence: unresolved };
        }
      }
      if (!finalResult) finalResult = { status: "success", summary: implementation.summary, tests, reviewSummary };
    }
  }
} catch (error) {
  finalResult = { status: "failed", summary: error instanceof Error ? error.message : String(error), attempts: ["Workflow orchestration or an agent call failed."] };
}

await save(finalResult);
finalResult;
