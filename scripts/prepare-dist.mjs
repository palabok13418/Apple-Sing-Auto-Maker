import { copyFileSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";

const root = resolve(process.cwd());
const dist = resolve(root, "dist");
mkdirSync(dist, { recursive: true });

copyFileSync(resolve(root, "index.html"), resolve(dist, "index.html"));
copyFileSync(resolve(root, "src", "style.css"), resolve(dist, "style.css"));

const indexPath = resolve(dist, "index.html");
let html = readFileSync(indexPath, "utf8");
html = html
  .replace('href="./src/style.css"', 'href="./style.css"')
  .replace('src="./dist/main.js"', 'src="./main.js"');
writeFileSync(indexPath, html);
