import assert from "node:assert/strict";
import fs from "node:fs";

const index = fs.readFileSync(new URL("../index.html", import.meta.url), "utf8");
const bridge = fs.readFileSync(new URL("../embed-bridge.js", import.meta.url), "utf8");
const css = fs.readFileSync(new URL("../src/style.css", import.meta.url), "utf8");
const vercel = JSON.parse(fs.readFileSync(new URL("../vercel.json", import.meta.url), "utf8"));
const build = fs.readFileSync(new URL("./prepare-dist.mjs", import.meta.url), "utf8");

assert.match(index, /src="\.\/embed-bridge\.js"/);
assert.match(bridge, /MUS_API_APP_PING/);
assert.match(bridge, /APPLE_SING_TTML_READY/);
assert.match(bridge, /https:\/\/mus-api\.vercel\.app/);
assert.match(css, /html\.mus-api-embedded \.topbar/);
assert.ok(vercel.headers.some(({ headers = [] }) => headers.some(({ key, value }) => key === "Content-Security-Policy" && value.includes("frame-ancestors") && value.includes("https://mus-api.vercel.app"))));
assert.match(build, /embed-bridge\.js/);
console.log("Apple Sing Mus-API embedding checks passed.");
