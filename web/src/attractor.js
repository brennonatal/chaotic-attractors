// 3D quadratic-polynomial attractor with the same dynamics as
// animations/live_display.py: derivatives + softened pairwise gravity,
// per-particle dt modulation, proximity speed-up, divergence reset.
// All buffers are reused across frames; no per-step allocations in step().

import { MIN_BBOX_RANGE } from "./config.js";

export async function loadEntries(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to load ${url}: ${res.status}`);
  return res.json();
}

function mulberry32(seed) {
  let a = (seed >>> 0) || 1;
  return () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gauss(rng) {
  // Box–Muller. Reproducibility doesn't need to match numpy bit-for-bit —
  // attractor seeds reproduce dynamics, the cloud jitter is just aesthetic.
  let u = 0;
  let v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(Math.PI * 2 * v);
}

function bboxCenterAndScale(bbox) {
  const cx = (bbox[0][0] + bbox[0][1]) / 2;
  const cy = (bbox[1][0] + bbox[1][1]) / 2;
  const cz = (bbox[2][0] + bbox[2][1]) / 2;
  const rx = Math.max(bbox[0][1] - bbox[0][0], MIN_BBOX_RANGE);
  const ry = Math.max(bbox[1][1] - bbox[1][0], MIN_BBOX_RANGE);
  const rz = Math.max(bbox[2][1] - bbox[2][0], MIN_BBOX_RANGE);
  return { center: [cx, cy, cz], ranges: [rx, ry, rz], scale: Math.max(rx, ry, rz) };
}

export class Attractor {
  constructor(entry, preset) {
    this.entry = entry;
    this.preset = preset;
    this.coeffs = entry.coeffs;             // (3, 10)
    this.initialState = entry.initialState; // [x0, y0, z0]
    this.points = preset.points;
    this.dt = preset.dt;

    this.velocityGain = 1.28;
    this.gravityStrength = 0.12;
    this.proximityGain = 0.85;

    const fr = bboxCenterAndScale(entry.bbox);
    this.center = fr.center;
    this.scale = fr.scale;
    this.displayScale = Math.max(fr.scale, 1.0);
    this.bounds = this.displayScale * 2.6 + 2.0;

    const lyapunov = entry.lyapunov ?? 0.5;
    this.gravityAdapt = Math.max(0, Math.min(1, Math.pow(lyapunov / 1.5, 2)));
    this._updateGravityCoupling();
    this.gravitySofteningSq = Math.pow(Math.max(this.displayScale * 0.18, 0.18), 2);
    this.proximityRadius = Math.max(this.displayScale * 0.30, 0.30);

    this.rng = mulberry32(entry.seed + this.points);
    this.jitter = Math.max(this.displayScale * 0.0012, 0.008);

    const n = this.points;
    this.states = new Float32Array(n * 3);
    this.statesNext = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      this.states[i * 3 + 0] = this.initialState[0] + gauss(this.rng) * this.jitter;
      this.states[i * 3 + 1] = this.initialState[1] + gauss(this.rng) * this.jitter;
      this.states[i * 3 + 2] = this.initialState[2] + gauss(this.rng) * this.jitter;
    }

    this.prevVelocity = new Float32Array(n * 3);
    this.speedEma = new Float32Array(n);
    this.currentSpeedNorm = new Float32Array(n);
    this.currentCurvatureNorm = new Float32Array(n);
    this.speedPhase = new Float32Array(n);
    this.baseSpeed = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      this.speedPhase[i] = this.rng() * Math.PI * 2;
      this.baseSpeed[i] = 0.84 + this.rng() * (1.52 - 0.84);
    }

    this._k1 = new Float32Array(n * 3);
    this._k2 = new Float32Array(n * 3);
    this._k3 = new Float32Array(n * 3);
    this._k4 = new Float32Array(n * 3);
    this._tmp = new Float32Array(n * 3);
    this._deriv = new Float32Array(n * 3);
    this._grav = new Float32Array(n * 3);
    this._h = new Float32Array(n);
    this._prox = new Float32Array(n);
    this._sortBuf = new Float32Array(n);

    this.simTick = 0;
    this._fitFraming();
    // _warmupAndFitFraming() will be re-enabled once trails are in: without
    // trails the visited bbox leaves the instantaneous cluster looking tiny.
  }

  _updateGravityCoupling() {
    this.gravityCoupling =
      this.gravityStrength * this.gravityAdapt * this.displayScale * this.displayScale * 0.18;
  }

  _derivatives(states, out) {
    const c = this.coeffs;
    const c0 = c[0];
    const c1 = c[1];
    const c2 = c[2];
    const n = this.points;
    for (let i = 0; i < n; i++) {
      const x = states[i * 3 + 0];
      const y = states[i * 3 + 1];
      const z = states[i * 3 + 2];
      const xx = x * x, yy = y * y, zz = z * z;
      const xy = x * y, xz = x * z, yz = y * z;
      out[i * 3 + 0] = c0[0] + c0[1]*x + c0[2]*y + c0[3]*z + c0[4]*xx + c0[5]*yy + c0[6]*zz + c0[7]*xy + c0[8]*xz + c0[9]*yz;
      out[i * 3 + 1] = c1[0] + c1[1]*x + c1[2]*y + c1[3]*z + c1[4]*xx + c1[5]*yy + c1[6]*zz + c1[7]*xy + c1[8]*xz + c1[9]*yz;
      out[i * 3 + 2] = c2[0] + c2[1]*x + c2[2]*y + c2[3]*z + c2[4]*xx + c2[5]*yy + c2[6]*zz + c2[7]*xy + c2[8]*xz + c2[9]*yz;
    }
  }

  _gravityField(states, out) {
    const G = this.gravityCoupling;
    if (G === 0) {
      out.fill(0);
      return;
    }
    const softSq = this.gravitySofteningSq;
    const n = this.points;
    for (let i = 0; i < n; i++) {
      let ax = 0, ay = 0, az = 0;
      const xi = states[i * 3 + 0];
      const yi = states[i * 3 + 1];
      const zi = states[i * 3 + 2];
      for (let j = 0; j < n; j++) {
        if (j === i) continue;
        const dx = states[j * 3 + 0] - xi;
        const dy = states[j * 3 + 1] - yi;
        const dz = states[j * 3 + 2] - zi;
        const d2 = dx * dx + dy * dy + dz * dz + softSq;
        const invR3 = 1.0 / (d2 * Math.sqrt(d2));
        ax += dx * invR3;
        ay += dy * invR3;
        az += dz * invR3;
      }
      out[i * 3 + 0] = G * ax;
      out[i * 3 + 1] = G * ay;
      out[i * 3 + 2] = G * az;
    }
  }

  _field(states, out) {
    this._derivatives(states, this._deriv);
    this._gravityField(states, this._grav);
    const n = this.points * 3;
    for (let i = 0; i < n; i++) out[i] = this._deriv[i] + this._grav[i];
  }

  _computeTimestep(tick) {
    const n = this.points;
    const dt = this.dt;
    const vg = this.velocityGain;
    for (let i = 0; i < n; i++) {
      const ph = this.speedPhase[i];
      const breath = 1 + 0.30  * Math.sin(tick * 0.007 + ph);
      const pulse  = 1 + 0.11  * Math.sin(tick * 0.019 + ph * 0.43);
      const micro  = 1 + 0.035 * Math.sin(tick * 0.043 + ph * 1.71);
      const h = dt * vg * this.baseSpeed[i] * breath * pulse * micro;
      this._h[i] = Math.max(dt * 0.45, Math.min(dt * 2.25, h));
    }
  }

  _computeProximity(states) {
    const n = this.points;
    if (this.proximityGain === 0) {
      for (let i = 0; i < n; i++) this._prox[i] = 1.0;
      return;
    }
    const r = this.proximityRadius;
    for (let i = 0; i < n; i++) {
      const xi = states[i * 3 + 0];
      const yi = states[i * 3 + 1];
      const zi = states[i * 3 + 2];
      let nearestSq = Infinity;
      for (let j = 0; j < n; j++) {
        if (j === i) continue;
        const dx = states[j * 3 + 0] - xi;
        const dy = states[j * 3 + 1] - yi;
        const dz = states[j * 3 + 2] - zi;
        const d2 = dx * dx + dy * dy + dz * dz + 1e-9;
        if (d2 < nearestSq) nearestSq = d2;
      }
      const d = Math.sqrt(nearestSq);
      const closeness = Math.max(0, Math.min(1, 1 - d / r));
      this._prox[i] = 1 + this.proximityGain * closeness * closeness;
    }
  }

  step() {
    const n = this.points;
    const old = this.states;
    const next = this.statesNext;

    this._computeTimestep(this.simTick);
    this._computeProximity(old);
    for (let i = 0; i < n; i++) this._h[i] *= this._prox[i];

    this._field(old, this._k1);
    for (let i = 0; i < n; i++) {
      const h2 = this._h[i] * 0.5;
      this._tmp[i * 3 + 0] = old[i * 3 + 0] + h2 * this._k1[i * 3 + 0];
      this._tmp[i * 3 + 1] = old[i * 3 + 1] + h2 * this._k1[i * 3 + 1];
      this._tmp[i * 3 + 2] = old[i * 3 + 2] + h2 * this._k1[i * 3 + 2];
    }
    this._field(this._tmp, this._k2);
    for (let i = 0; i < n; i++) {
      const h2 = this._h[i] * 0.5;
      this._tmp[i * 3 + 0] = old[i * 3 + 0] + h2 * this._k2[i * 3 + 0];
      this._tmp[i * 3 + 1] = old[i * 3 + 1] + h2 * this._k2[i * 3 + 1];
      this._tmp[i * 3 + 2] = old[i * 3 + 2] + h2 * this._k2[i * 3 + 2];
    }
    this._field(this._tmp, this._k3);
    for (let i = 0; i < n; i++) {
      const h = this._h[i];
      this._tmp[i * 3 + 0] = old[i * 3 + 0] + h * this._k3[i * 3 + 0];
      this._tmp[i * 3 + 1] = old[i * 3 + 1] + h * this._k3[i * 3 + 1];
      this._tmp[i * 3 + 2] = old[i * 3 + 2] + h * this._k3[i * 3 + 2];
    }
    this._field(this._tmp, this._k4);
    for (let i = 0; i < n; i++) {
      const h = this._h[i] / 6;
      next[i * 3 + 0] = old[i * 3 + 0] + h * (this._k1[i * 3 + 0] + 2 * this._k2[i * 3 + 0] + 2 * this._k3[i * 3 + 0] + this._k4[i * 3 + 0]);
      next[i * 3 + 1] = old[i * 3 + 1] + h * (this._k1[i * 3 + 1] + 2 * this._k2[i * 3 + 1] + 2 * this._k3[i * 3 + 1] + this._k4[i * 3 + 1]);
      next[i * 3 + 2] = old[i * 3 + 2] + h * (this._k1[i * 3 + 2] + 2 * this._k2[i * 3 + 2] + 2 * this._k3[i * 3 + 2] + this._k4[i * 3 + 2]);
    }

    // Track speed (EMA, then 90th-percentile normalize) + curvature.
    const ds = Math.max(this.displayScale, 1e-6);
    for (let i = 0; i < n; i++) {
      const vx = next[i * 3 + 0] - old[i * 3 + 0];
      const vy = next[i * 3 + 1] - old[i * 3 + 1];
      const vz = next[i * 3 + 2] - old[i * 3 + 2];
      const speed = Math.sqrt(vx * vx + vy * vy + vz * vz);
      const raw = Math.log1p(speed / ds);
      this.speedEma[i] = this.speedEma[i] * 0.84 + raw * 0.16;
      this._sortBuf[i] = this.speedEma[i];

      const pvx = this.prevVelocity[i * 3 + 0];
      const pvy = this.prevVelocity[i * 3 + 1];
      const pvz = this.prevVelocity[i * 3 + 2];
      const cx = pvy * vz - pvz * vy;
      const cy = pvz * vx - pvx * vz;
      const cz = pvx * vy - pvy * vx;
      const cross = Math.sqrt(cx * cx + cy * cy + cz * cz);
      const lp = Math.sqrt(pvx * pvx + pvy * pvy + pvz * pvz);
      this.currentCurvatureNorm[i] = Math.max(0, Math.min(1, cross / (lp * speed + 1e-9)));
      this.prevVelocity[i * 3 + 0] = vx;
      this.prevVelocity[i * 3 + 1] = vy;
      this.prevVelocity[i * 3 + 2] = vz;
    }

    const sorted = Array.from(this._sortBuf).sort((a, b) => a - b);
    const p90 = Math.max(sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * 0.9))], 1e-6);
    for (let i = 0; i < n; i++) {
      this.currentSpeedNorm[i] = Math.max(0, Math.min(1, this.speedEma[i] / p90));
    }

    // Divergence reset — any particle that goes non-finite or escapes the
    // attractor's bounding sphere is replanted near the initial state.
    const cx0 = this.center[0];
    const cy0 = this.center[1];
    const cz0 = this.center[2];
    const bndSq = this.bounds * this.bounds;
    for (let i = 0; i < n; i++) {
      const x = next[i * 3 + 0];
      const y = next[i * 3 + 1];
      const z = next[i * 3 + 2];
      const finite = Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z);
      const dx = x - cx0, dy = y - cy0, dz = z - cz0;
      const inBounds = finite && (dx * dx + dy * dy + dz * dz) < bndSq;
      if (!inBounds) {
        next[i * 3 + 0] = this.initialState[0] + gauss(this.rng) * this.jitter;
        next[i * 3 + 1] = this.initialState[1] + gauss(this.rng) * this.jitter;
        next[i * 3 + 2] = this.initialState[2] + gauss(this.rng) * this.jitter;
        this.prevVelocity[i * 3 + 0] = 0;
        this.prevVelocity[i * 3 + 1] = 0;
        this.prevVelocity[i * 3 + 2] = 0;
        this.speedEma[i] = 0;
        this.currentSpeedNorm[i] = 0;
        this.currentCurvatureNorm[i] = 0;
      }
    }

    this.states = next;
    this.statesNext = old;
    this.simTick++;
  }

  _fitFraming() {
    // Initial conservative fit from the attractor's full bbox. Replaced by
    // _warmupAndFitFraming below once we know where the cloud actually lives.
    const fr = bboxCenterAndScale(this.entry.bbox);
    this.framingCenter = fr.center.slice();
    const diag = Math.sqrt(fr.ranges[0] ** 2 + fr.ranges[1] ** 2 + fr.ranges[2] ** 2) * 0.5;
    this.framingScale = Math.max(fr.scale * 1.45 + 1.0, diag * 1.15 + 1.0);
  }

  _warmupAndFitFraming(steps = 2400) {
    // Mirror of live_display.precompute_camera_framing: run the integrator
    // forward, measure the bounds the cloud actually visits, then fit to
    // those bounds rather than the (often much larger) attractor bbox. This
    // is what stops gravity-bound clouds from rendering as tiny dots.
    const n = this.points;
    const initialStates = this.states.slice();

    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    for (let i = 0; i < n; i++) {
      const x = this.states[i * 3 + 0];
      const y = this.states[i * 3 + 1];
      const z = this.states[i * 3 + 2];
      if (x < minX) minX = x; if (x > maxX) maxX = x;
      if (y < minY) minY = y; if (y > maxY) maxY = y;
      if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
    }
    for (let s = 0; s < steps; s++) {
      this.step();
      const st = this.states;
      for (let i = 0; i < n; i++) {
        const x = st[i * 3 + 0], y = st[i * 3 + 1], z = st[i * 3 + 2];
        if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;
        if (x < minX) minX = x; if (x > maxX) maxX = x;
        if (y < minY) minY = y; if (y > maxY) maxY = y;
        if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
      }
    }

    // Restore the cloud to its initial seeded state and clear the velocity /
    // speed-EMA trackers so the rendered run starts fresh.
    this.states.set(initialStates);
    this.simTick = 0;
    this.prevVelocity.fill(0);
    this.speedEma.fill(0);
    this.currentSpeedNorm.fill(0);
    this.currentCurvatureNorm.fill(0);

    const rx = Math.max(maxX - minX, MIN_BBOX_RANGE);
    const ry = Math.max(maxY - minY, MIN_BBOX_RANGE);
    const rz = Math.max(maxZ - minZ, MIN_BBOX_RANGE);
    this.framingCenter = [(minX + maxX) / 2, (minY + maxY) / 2, (minZ + maxZ) / 2];
    const maxAxis = Math.max(rx, ry, rz);
    const diag = Math.sqrt(rx * rx + ry * ry + rz * rz) * 0.5;
    this.framingScale = Math.max(maxAxis * 1.45 + 1.0, diag * 1.15 + 1.0);
  }

  orbitCamera(camera, t) {
    // Same azimuth/elevation/roll motion as live_display.update_camera_motion.
    const azimuthDeg = 35 + t * 5.0;
    const elevationDeg = 24 + Math.sin(t * 0.15) * 7.0;
    const rollDeg = Math.sin(t * 0.09) * 1.8;
    const fovDeg = 40;

    const az = (azimuthDeg * Math.PI) / 180;
    const el = (elevationDeg * Math.PI) / 180;
    const fov = (fovDeg * Math.PI) / 180;
    // vispy's scale_factor on TurntableCamera is the visible extent at the
    // focus point. Convert to a perspective-camera distance.
    const dist = this.framingScale / (2 * Math.tan(fov / 2));

    const cx = this.framingCenter[0];
    const cy = this.framingCenter[1];
    const cz = this.framingCenter[2];

    const cosEl = Math.cos(el);
    const px = cx + dist * cosEl * Math.cos(az);
    const py = cy + dist * cosEl * Math.sin(az);
    const pz = cz + dist * Math.sin(el);

    camera.fov = fovDeg;
    camera.up.set(0, 0, 1);
    camera.position.set(px, py, pz);
    camera.lookAt(cx, cy, cz);
    if (rollDeg !== 0) {
      camera.rotateZ((rollDeg * Math.PI) / 180);
    }
    camera.near = Math.max(0.01, dist * 0.005);
    camera.far = dist * 6.0;
    camera.updateProjectionMatrix();
  }
}
