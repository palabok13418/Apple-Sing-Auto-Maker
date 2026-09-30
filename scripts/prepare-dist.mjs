import { copyFileSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { resolve } from "node:path";
import ts from "typescript";

const root = resolve(process.cwd());
const dist = resolve(root, "dist");
mkdirSync(dist, { recursive: true });

copyFileSync(resolve(root, "index.html"), resolve(dist, "index.html"));
copyFileSync(resolve(root, "src", "style.css"), resolve(dist, "style.css"));

const sourcePath = resolve(root, "src", "main.ts");
const mainSource = readFileSync(sourcePath, "utf8");
const transpiled = ts.transpileModule(mainSource, {
  compilerOptions: {
    target: ts.ScriptTarget.ES2022,
    module: ts.ModuleKind.ESNext,
    moduleResolution: ts.ModuleResolutionKind.Bundler,
    sourceMap: false,
  },
  fileName: "main.ts",
});
writeFileSync(resolve(dist, "main.js"), transpiled.outputText, "utf8");

const indexPath = resolve(dist, "index.html");
let html = readFileSync(indexPath, "utf8");
html = html
  .replace('href="./src/style.css"', 'href="./style.css"')
  .replace('src="./dist/main.js"', 'src="./main.js"');
writeFileSync(indexPath, html);

execFileSync(process.execPath, ["--check", resolve(dist, "main.js")], { stdio: "inherit" });
console.log("dist/main.js syntax check passed");
