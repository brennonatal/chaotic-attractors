import * as THREE from "three";
import { Attractor, loadEntries } from "./attractor.js";
import { ParticleField } from "./particles.js";
import { PALETTES } from "./palettes.js";
import { QUALITY_PRESETS, DEFAULT_QUALITY } from "./config.js";

const QUALITY = DEFAULT_QUALITY;

async function main() {
  const entries = await loadEntries("attractors.json");
  if (entries.length === 0) {
    throw new Error("attractors.json is empty. Re-run tools/dump_coeffs.py.");
  }

  const preset = QUALITY_PRESETS[QUALITY];
  const palette = PALETTES[0];

  const canvas = document.getElementById("app");
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(palette.background[0], palette.background[1], palette.background[2]);

  const camera = new THREE.PerspectiveCamera(40, window.innerWidth / window.innerHeight, 0.01, 1e6);
  camera.up.set(0, 0, 1);

  const entry = entries[0];
  const attractor = new Attractor(entry, preset);
  const particles = new ParticleField(attractor, palette);
  scene.add(particles.mesh);

  const hud = document.getElementById("hud");
  hud.textContent =
    `CHAOTIC ATTRACTORS  ·  01/${String(entries.length).padStart(2, "0")}  ` +
    `seed ${entry.seed}  ·  λ ${entry.lyapunov.toFixed(3)}  ·  ${QUALITY}  ·  ${palette.name}`;

  resize();
  window.addEventListener("resize", resize);
  function resize() {
    const w = window.innerWidth;
    const h = window.innerHeight;
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }

  let frame = 0;
  function loop() {
    for (let i = 0; i < preset.stepsPerFrame; i++) attractor.step();
    particles.update();

    const t = frame / 60;
    attractor.orbitCamera(camera, t);
    particles.updatePerFrameUniforms(camera);

    renderer.render(scene, camera);
    frame++;
    requestAnimationFrame(loop);
  }

  requestAnimationFrame(loop);
}

main().catch((err) => {
  console.error(err);
  const hud = document.getElementById("hud");
  if (hud) hud.textContent = `error: ${err.message}`;
});
