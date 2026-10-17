import { copyFileSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { resolve } from "node:path";

const root = resolve(process.cwd());
const dist = resolve(root, "dist");
mkdirSync(dist, { recursive: true });

copyFileSync(resolve(root, "index.html"), resolve(dist, "index.html"));
copyFileSync(resolve(root, "src", "style.css"), resolve(dist, "style.css"));
copyFileSync(resolve(root, "src", "main.js"), resolve(dist, "main.js"));
copyFileSync(resolve(root, "src", "whisper-worker.js"), resolve(dist, "whisper-worker.js"));

const indexPath = resolve(dist, "index.html");
let html = readFileSync(indexPath, "utf8");
html = html
  .replace('href="./src/style.css"', 'href="./style.css"')
  .replace('src="./dist/main.js"', 'src="./main.js"');
writeFileSync(indexPath, html);

execFileSync(process.execPath, ["--check", resolve(dist, "main.js")], { stdio: "inherit" });
console.log("dist/main.js syntax check passed");
