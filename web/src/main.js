import * as THREE from "three";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";

import { Attractor, loadEntries } from "./attractor.js";
import { ParticleField } from "./particles.js";
import { TrailField } from "./trails.js";
import { StarField } from "./starfield.js";
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
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.1;

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(palette.background[0], palette.background[1], palette.background[2]);

  const camera = new THREE.PerspectiveCamera(40, window.innerWidth / window.innerHeight, 0.01, 1e6);
  camera.up.set(0, 0, 1);

  const params = new URLSearchParams(window.location.search);
  const startIndex = Math.max(0, Math.min(entries.length - 1, parseInt(params.get("index") ?? "0", 10) || 0));
  const entry = entries[startIndex];

  const attractor = new Attractor(entry, preset);
  const particles = new ParticleField(attractor, palette);
  const trails = new TrailField(attractor, palette);
  const stars = new StarField(attractor, palette, preset.starCount);
  scene.add(stars.mesh);
  scene.add(trails.mesh);
  scene.add(particles.mesh);

  // Post-processing: bloom for the cinematic glow, OutputPass for tone mapping
  // + sRGB encode at the very end of the pipeline.
  const composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, camera));
  const bloomPass = new UnrealBloomPass(
    new THREE.Vector2(window.innerWidth, window.innerHeight),
    0.55,   // strength — subtle, not neon
    0.35,   // radius   — tight kernel keeps single-pixel halos round
    0.0,    // threshold — bloom everything
  );
  composer.addPass(bloomPass);
  composer.addPass(new OutputPass());

  const hud = document.getElementById("hud");
  hud.textContent =
    `CHAOTIC ATTRACTORS  ·  ${String(startIndex + 1).padStart(2, "0")}/${String(entries.length).padStart(2, "0")}  ` +
    `seed ${entry.seed}  ·  λ ${entry.lyapunov.toFixed(3)}  ·  ${QUALITY}  ·  ${palette.name}`;

  resize();
  window.addEventListener("resize", resize);
  function resize() {
    const w = window.innerWidth;
    const h = window.innerHeight;
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(w, h, false);
    composer.setSize(w, h);
    bloomPass.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }

  let frame = 0;
  function loop() {
    for (let i = 0; i < preset.stepsPerFrame; i++) attractor.step();
    particles.update();
    trails.update();

    const t = frame / 60;
    attractor.orbitCamera(camera, t);
    particles.updatePerFrameUniforms(camera);

    composer.render();
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
