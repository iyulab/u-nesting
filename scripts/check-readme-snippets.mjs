#!/usr/bin/env node
// Runs the JavaScript examples a package's README teaches, against the package
// as it is installed.
//
//   node check-readme-snippets.mjs <package-name> [--readme <path>]
//
// Run it from a directory where <package-name> is installed (the publish
// workflow's smoke directory). The README defaults to the one the package
// ships -- node_modules/<package-name>/README.md -- because that is the text
// a consumer reads on the registry page.
//
// A fenced ```js / ```javascript / ```mjs / ```cjs block that imports or
// requires the package is a self-contained example and must run without
// throwing. Blocks that do not name the package continue an earlier example
// (they use its variables) and are not run on their own. A README with no
// runnable example at all fails: every published package owes its reader one.
//
// This catches what an import-only smoke test cannot: an example written for a
// different build target (`import init ... await init()` against a bundler/node
// build that has no default export) passes `import('<pkg>')` and fails on the
// first line a reader copies.

import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";

const args = process.argv.slice(2);
const pkg = args[0];
if (!pkg || pkg.startsWith("--")) {
  console.error("usage: check-readme-snippets.mjs <package-name> [--readme <path>]");
  process.exit(2);
}
const readmeFlag = args.indexOf("--readme");
const readmePath =
  readmeFlag >= 0 ? args[readmeFlag + 1] : join("node_modules", pkg, "README.md");

const text = readFileSync(readmePath, "utf8");
const fence = /^```(js|javascript|mjs|cjs)[ \t]*\r?\n([\s\S]*?)^```/gm;
const escaped = pkg.replace(/[.*+?^${}()|[\]\\/]/g, "\\$&");
const names = new RegExp(`(from\\s*|require\\(\\s*|import\\(\\s*)["']${escaped}["']`);

const blocks = [];
for (const m of text.matchAll(fence)) {
  if (!names.test(m[2])) continue;
  blocks.push({ line: text.slice(0, m.index).split("\n").length, body: m[2] });
}

if (blocks.length === 0) {
  console.error(
    `${readmePath}: no runnable example -- no js block imports or requires '${pkg}'. ` +
      "A reader of the published README has nothing to start from.",
  );
  process.exit(1);
}

const dir = mkdtempSync(join(process.cwd(), ".readme-snippets-"));
let failed = 0;
try {
  blocks.forEach((block, i) => {
    // ES module syntax or top-level await needs a module; otherwise CommonJS.
    const esm = /^\s*(import|export)\s/m.test(block.body) || /^\s*await\s/m.test(block.body);
    const file = join(dir, `block-${i}${esm ? ".mjs" : ".cjs"}`);
    writeFileSync(file, block.body);
    try {
      execFileSync(process.execPath, [file], { stdio: ["ignore", "pipe", "pipe"], timeout: 60_000 });
      console.log(`ok    ${readmePath}:${block.line}`);
    } catch (e) {
      failed++;
      // Node prints the source line, a caret and the error first; the stack
      // below it points into the loader, not the example.
      const lines = `${e.stderr ?? ""}`.trim().split("\n");
      const at = lines.findIndex((l) => /^\s+at\s/.test(l));
      const out = lines.slice(0, at > 0 ? at : 6).join("\n      ");
      console.error(`FAIL  ${readmePath}:${block.line}\n      ${out || e.message}`);
    }
  });
} finally {
  rmSync(dir, { recursive: true, force: true });
}

if (failed) {
  console.error(`${failed} of ${blocks.length} README example(s) failed against the installed '${pkg}'.`);
  process.exit(1);
}
console.log(`README examples: ${blocks.length} ran against the installed '${pkg}'.`);
