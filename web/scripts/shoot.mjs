// Screenshot script: spin up the dev server, drive a real Chromium with
// WebGL, capture a few frames of the rendered attractor.
import { chromium } from "playwright";
import { spawn } from "node:child_process";
import { setTimeout as wait } from "node:timers/promises";
import { mkdirSync } from "node:fs";

const OUT = new URL("../shots/", import.meta.url).pathname;
mkdirSync(OUT, { recursive: true });

const vite = spawn("npx", ["vite", "--port", "5173", "--strictPort"], {
  stdio: ["ignore", "pipe", "pipe"],
  env: process.env,
});
vite.stdout.on("data", (d) => process.stdout.write(`[vite] ${d}`));
vite.stderr.on("data", (d) => process.stderr.write(`[vite!] ${d}`));

// Wait for the server to be reachable.
async function waitForServer(url, timeoutMs = 15000) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    try {
      const res = await fetch(url);
      if (res.ok) return;
    } catch (_) {}
    await wait(200);
  }
  throw new Error(`Server at ${url} never came up`);
}
await waitForServer("http://localhost:5173/");

const browser = await chromium.launch({
  args: [
    "--use-gl=swiftshader",
    "--enable-webgl",
    "--ignore-gpu-blocklist",
    "--no-sandbox",
  ],
});
const context = await browser.newContext({
  viewport: { width: 1600, height: 1000 },
  deviceScaleFactor: 1,
});
const page = await context.newPage();
page.on("console", (msg) => console.log(`[page:${msg.type()}] ${msg.text()}`));
page.on("pageerror", (err) => console.error(`[page error] ${err.message}`));

// Capture several attractors so the user gets a sense of how the
// sphere-impostor shader reads across different cloud shapes.
const shots = [
  { idx: 0, name: "01_idx0_top_lambda.png", warm: 2500 },
  { idx: 1, name: "02_idx1.png", warm: 3500 },
  { idx: 2, name: "03_idx2.png", warm: 3500 },
  { idx: 6, name: "04_idx6.png", warm: 3500 },
];
for (const s of shots) {
  await page.goto(`http://localhost:5173/?index=${s.idx}`, { waitUntil: "networkidle" });
  await wait(s.warm);
  await page.screenshot({ path: `${OUT}${s.name}`, omitBackground: false });
  console.log(`captured ${s.name}`);
}

await browser.close();
vite.kill("SIGTERM");
await wait(200);
console.log("done");
process.exit(0);
