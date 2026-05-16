import * as THREE from "three";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";

import { Attractor, loadEntries } from "./attractor.js";
import { ParticleField } from "./particles.js";
import { TrailField } from "./trails.js";
import { StarField } from "./starfield.js";
import { makeColorScheme } from "./palettes.js";
import { QUALITY_PRESETS, DEFAULT_QUALITY } from "./config.js";

const QUALITY = DEFAULT_QUALITY;

async function main() {
  const entries = await loadEntries("attractors.json");
  if (entries.length === 0) {
    throw new Error("attractors.json is empty. Re-run tools/dump_coeffs.py.");
  }

  const preset = QUALITY_PRESETS[QUALITY];

  const canvas = document.getElementById("app");
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.1;

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(40, window.innerWidth / window.innerHeight, 0.01, 1e6);
  camera.up.set(0, 0, 1);

  const composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, camera));
  const bloomPass = new UnrealBloomPass(
    new THREE.Vector2(window.innerWidth, window.innerHeight),
    0.55,
    0.35,
    0.0,
  );
  composer.addPass(bloomPass);
  composer.addPass(new OutputPass());

  const hud = document.getElementById("hud");
  const params = new URLSearchParams(window.location.search);
  let idx = Math.max(0, Math.min(entries.length - 1, parseInt(params.get("index") ?? "0", 10) || 0));
  let paletteVariant = 0;
  let paused = false;
  let showHud = true;
  let frame = 0;
  let active = null;

  function disposeMesh(mesh) {
    if (!mesh) return;
    scene.remove(mesh);
    mesh.geometry?.dispose();
    const m = mesh.material;
    if (Array.isArray(m)) m.forEach((mat) => mat.dispose());
    else m?.dispose();
  }

  function load(i) {
    const entry = entries[i];
    if (active) {
      disposeMesh(active.particles.mesh);
      disposeMesh(active.trails.mesh);
      disposeMesh(active.stars.mesh);
    }
    const scheme = makeColorScheme(entry.seed ^ (paletteVariant * 0x9E3779B9), preset.points);
    scene.background = new THREE.Color(scheme.background[0], scheme.background[1], scheme.background[2]);

    const attractor = new Attractor(entry, preset);
    const particles = new ParticleField(attractor, scheme);
    const trails = new TrailField(attractor, scheme);
    const stars = new StarField(attractor, scheme, preset.starCount);
    scene.add(stars.mesh);
    scene.add(trails.mesh);
    scene.add(particles.mesh);

    active = { entry, scheme, attractor, particles, trails, stars };
    frame = 0;
    updateHud();
  }

  function updateHud() {
    if (!showHud || !active) {
      hud.textContent = "";
      return;
    }
    const { entry, scheme } = active;
    hud.textContent =
      `CHAOTIC ATTRACTORS  ·  ${String(idx + 1).padStart(2, "0")}/${String(entries.length).padStart(2, "0")}  ` +
      `seed ${entry.seed}  ·  λ ${entry.lyapunov.toFixed(3)}  ·  ${QUALITY}  ·  ${scheme.name}` +
      `${paused ? "  ·  paused" : ""}`;
  }

  load(idx);

  window.addEventListener("keydown", (e) => {
    switch (e.key) {
      case "ArrowRight":
      case "ArrowDown":
        idx = (idx + 1) % entries.length;
        load(idx);
        break;
      case "ArrowLeft":
      case "ArrowUp":
        idx = (idx - 1 + entries.length) % entries.length;
        load(idx);
        break;
      case " ":
        paused = !paused;
        updateHud();
        e.preventDefault();
        break;
      case "r":
      case "R":
        load(idx);
        break;
      case "p":
      case "P":
        paletteVariant = (paletteVariant + 1) | 0;
        load(idx);
        break;
      case "h":
      case "H":
        showHud = !showHud;
        updateHud();
        break;
      case "f":
      case "F":
        if (document.fullscreenElement) document.exitFullscreen();
        else document.documentElement.requestFullscreen?.();
        break;
    }
  });

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

  function loop() {
    if (!paused) {
      for (let i = 0; i < preset.stepsPerFrame; i++) active.attractor.step();
      active.particles.update();
      active.trails.update();
    }
    const t = frame / 60;
    active.attractor.orbitCamera(camera, t);
    active.particles.updatePerFrameUniforms(camera);

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
