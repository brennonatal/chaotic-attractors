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

// Capture a few attractors with vivid colours, plus exercise the
// keyboard navigation: load idx=1, then press ArrowRight twice to verify
// the cycling actually swaps attractors live.
await page.goto("http://localhost:5173/?index=1", { waitUntil: "networkidle" });
await wait(3500);
await page.screenshot({ path: `${OUT}01_initial_idx1.png` });
console.log("captured 01_initial_idx1.png");

await page.keyboard.press("ArrowRight");
await wait(3500);
await page.screenshot({ path: `${OUT}02_after_right.png` });
console.log("captured 02_after_right.png");

await page.keyboard.press("ArrowRight");
await wait(3500);
await page.screenshot({ path: `${OUT}03_after_right_again.png` });
console.log("captured 03_after_right_again.png");

await page.keyboard.press("p");
await wait(3500);
await page.screenshot({ path: `${OUT}04_after_p_reroll.png` });
console.log("captured 04_after_p_reroll.png");

for (const idx of [0, 5, 10]) {
  await page.goto(`http://localhost:5173/?index=${idx}`, { waitUntil: "networkidle" });
  await wait(3500);
  await page.screenshot({ path: `${OUT}10_idx${idx}.png` });
  console.log(`captured 10_idx${idx}.png`);
}

await browser.close();
vite.kill("SIGTERM");
await wait(200);
console.log("done");
process.exit(0);
