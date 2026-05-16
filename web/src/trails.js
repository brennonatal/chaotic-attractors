// Per-particle fading ribbons. Stored as one big LineSegments BufferGeometry —
// for n particles and trailLength T, we draw n * (T - 1) segments.
//
// Each frame we walk the attractor's ring buffer backwards from trailHead and
// rewrite the position + RGBA color attributes. Alpha falls off exponentially
// with segment age, so old segments disappear instead of lingering as smudges.
// With UnrealBloomPass on top, the bright head reads as a velocity streak.

import * as THREE from "three";

export class TrailField {
  constructor(attractor, palette) {
    this.attractor = attractor;
    this.palette = palette;
    this.n = attractor.points;
    this.trailLength = attractor.trailLength;

    const segs = this.n * (this.trailLength - 1);
    const verts = segs * 2;

    this._positions = new Float32Array(verts * 3);
    // RGB only — additive blending means brightness IS the trail strength,
    // so we bake the exponential age-fade and head boost into the colour
    // values instead of fighting three.js's vertex-alpha plumbing.
    this._colors = new Float32Array(verts * 3);

    const geom = new THREE.BufferGeometry();
    geom.setAttribute(
      "position",
      new THREE.BufferAttribute(this._positions, 3).setUsage(THREE.DynamicDrawUsage),
    );
    geom.setAttribute(
      "color",
      new THREE.BufferAttribute(this._colors, 3).setUsage(THREE.DynamicDrawUsage),
    );

    const mat = new THREE.LineBasicMaterial({
      vertexColors: true,
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });

    this.mesh = new THREE.LineSegments(geom, mat);
    this.mesh.frustumCulled = false;

    // Per-particle base colour (cool→hot ramp across instances).
    this._particleColors = new Float32Array(this.n * 3);
    const cool = palette.cool;
    const hot = palette.hot;
    for (let i = 0; i < this.n; i++) {
      const t = this.n > 1 ? i / (this.n - 1) : 0;
      this._particleColors[i * 3 + 0] = cool[0] * (1 - t) + hot[0] * t;
      this._particleColors[i * 3 + 1] = cool[1] * (1 - t) + hot[1] * t;
      this._particleColors[i * 3 + 2] = cool[2] * (1 - t) + hot[2] * t;
    }
  }

  update() {
    const a = this.attractor;
    const trails = a.trails;
    const head = a.trailHead;
    const fill = a.trailFill;
    const tl = this.trailLength;
    const n = this.n;
    const pos = this._positions;
    const col = this._colors;
    const speedNorm = a.currentSpeedNorm;
    const denom = tl > 1 ? tl - 1 : 1;

    let v = 0;
    let c = 0;
    for (let p = 0; p < n; p++) {
      const cr = this._particleColors[p * 3 + 0];
      const cg = this._particleColors[p * 3 + 1];
      const cb = this._particleColors[p * 3 + 2];
      const sn = speedNorm[p];
      const fadeRate = 2.4 - 0.85 * sn;
      const brightBase = 1.4 + 0.6 * sn;

      for (let s = 0; s < tl - 1; s++) {
        // Segment s connects trail samples (head-1-s) and (head-2-s) modulo tl.
        const t0 = (head - 1 - s + 2 * tl) % tl; // newer endpoint
        const t1 = (head - 2 - s + 2 * tl) % tl; // older endpoint

        const off0 = (t0 * n + p) * 3;
        const off1 = (t1 * n + p) * 3;

        pos[v + 0] = trails[off0 + 0];
        pos[v + 1] = trails[off0 + 1];
        pos[v + 2] = trails[off0 + 2];
        pos[v + 3] = trails[off1 + 0];
        pos[v + 4] = trails[off1 + 1];
        pos[v + 5] = trails[off1 + 2];
        v += 6;

        const valid = s < fill - 1;
        const age = s / denom;
        const fade = valid ? Math.exp(-age * fadeRate) : 0;
        // Bake fade + head boost into RGB; with additive blending this IS
        // the trail's perceived brightness on screen.
        const headBoost = 1.0 + 0.8 * (1.0 - age);
        const k = fade * brightBase * headBoost;
        const r = cr * k;
        const g = cg * k;
        const b = cb * k;

        col[c + 0] = r; col[c + 1] = g; col[c + 2] = b;
        col[c + 3] = r; col[c + 4] = g; col[c + 5] = b;
        c += 6;
      }
    }

    this.mesh.geometry.attributes.position.needsUpdate = true;
    this.mesh.geometry.attributes.color.needsUpdate = true;
  }
}
