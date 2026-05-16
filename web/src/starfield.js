// Faint depth backdrop — tiny additive points distributed around the
// attractor center. Without these the empty space behind the cloud feels
// dead. With bloom enabled they bloom into soft dust motes.

import * as THREE from "three";

export class StarField {
  constructor(attractor, scheme, count) {
    const positions = new Float32Array(count * 3);
    const colors = new Float32Array(count * 3);
    const cx = attractor.center[0];
    const cy = attractor.center[1];
    const cz = attractor.center[2];
    const r0 = Math.max(attractor.displayScale * 1.65, 1.0);
    const star = scheme.starColor;
    for (let i = 0; i < count; i++) {
      positions[i * 3 + 0] = cx + (Math.random() * 2 - 1) * r0;
      positions[i * 3 + 1] = cy + (Math.random() * 2 - 1) * r0;
      positions[i * 3 + 2] = cz + (Math.random() * 2 - 1) * r0;
      colors[i * 3 + 0] = star[0];
      colors[i * 3 + 1] = star[1];
      colors[i * 3 + 2] = star[2];
    }
    const geom = new THREE.BufferGeometry();
    geom.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geom.setAttribute("color", new THREE.BufferAttribute(colors, 3));

    const mat = new THREE.PointsMaterial({
      size: 2.2,
      vertexColors: true,
      transparent: true,
      opacity: star[3] ?? 0.32,
      sizeAttenuation: false,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    this.mesh = new THREE.Points(geom, mat);
    this.mesh.frustumCulled = false;
  }
}
