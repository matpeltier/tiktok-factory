import { readFileSync } from "node:fs";
import path from "node:path";

// Provider names and role assignments live in .factory/config.json. Read it from
// disk because codex-workflow evaluates provider configs from a data URL, where a
// relative JSON module import cannot be resolved. Credentials are still read by
// codex-dynamic-workflows from the environment and never stored here.
const configPath = process.env.FACTORY_CONFIG_PATH
  ?? (process.cwd().endsWith(path.sep + ".factory")
    ? path.join(process.cwd(), "config.json")
    : path.join(process.cwd(), ".factory", "config.json"));
const factoryConfig = JSON.parse(readFileSync(configPath, "utf8"));

export default {
  providers: Object.fromEntries(
    Object.entries(factoryConfig.providers).map(([name, provider]) => [
      name,
      provider.backend === "codex"
        ? { ...provider, sandbox: provider.sandbox ?? "danger-full-access" }
        : provider,
    ]),
  ),
  default: factoryConfig.agents.implementer,
};
