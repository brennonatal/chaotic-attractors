// Particles as fragment-shader sphere impostors.
//
// Each particle renders as one view-aligned billboard. The fragment shader
// reconstructs the sphere's surface normal from billboard-local UVs, discards
// pixels outside the unit disc, and shades with Lambert + Fresnel rim +
// Blinn-Phong specular. The result reads as a lit material body, not a flat
// disc — and one quad per particle is enough geometry for tens of thousands.

import * as THREE from "three";

const VERT = /* glsl */ `
in vec3 iPosition;
in vec3 iColor;
in float iRadius;

out vec3 vColor;
out vec3 vCenterView;
out vec2 vLocal;
out float vRadius;

void main() {
  vColor = iColor;
  vRadius = iRadius;
  vLocal = position.xy * 2.0;

  vec4 centerView = modelViewMatrix * vec4(iPosition, 1.0);
  vCenterView = centerView.xyz;

  // Quad corners are in [-0.5, 0.5]; scale by 2*radius to get a [-r, r] billboard
  // in view space — i.e. screen-facing because view-space XY is screen-aligned.
  vec3 offset = vec3(position.xy * 2.0 * iRadius, 0.0);
  gl_Position = projectionMatrix * vec4(centerView.xyz + offset, 1.0);
}
`;

const FRAG = /* glsl */ `
precision highp float;

in vec3 vColor;
in vec3 vCenterView;
in vec2 vLocal;
in float vRadius;

out vec4 fragColor;

uniform mat4 uProjectionMatrix;
uniform vec3 uLightDir;     // view space, normalized
uniform vec3 uRimColor;
uniform vec3 uSpecColor;
uniform float uAmbient;
uniform float uRimStrength;
uniform float uSpecStrength;
uniform float uSpecPower;

void main() {
  float r2 = dot(vLocal, vLocal);
  if (r2 > 1.0) discard;
  float z = sqrt(1.0 - r2);

  // The sphere's view-space normal is exactly the billboard-local position on
  // the unit hemisphere because the billboard is camera-facing.
  vec3 normal = vec3(vLocal, z);

  // Lambert diffuse with a flat ambient floor.
  float ndotl = max(dot(normal, uLightDir), 0.0);
  vec3 col = vColor * (uAmbient + (1.0 - uAmbient) * ndotl);

  // Fresnel rim — brighter near the silhouette.
  float fresnel = pow(1.0 - z, 3.0);
  col += uRimColor * fresnel * uRimStrength;

  // Blinn-Phong specular highlight.
  vec3 viewDir = vec3(0.0, 0.0, 1.0);
  vec3 halfV = normalize(uLightDir + viewDir);
  float spec = pow(max(dot(normal, halfV), 0.0), uSpecPower);
  col += uSpecColor * spec * uSpecStrength;

  // Push the surface depth out to where the sphere actually is, so spheres
  // intersect correctly instead of all sitting at the billboard's flat depth.
  vec3 spherePosView = vCenterView + vec3(vLocal, z) * vRadius;
  vec4 clip = uProjectionMatrix * vec4(spherePosView, 1.0);
  gl_FragDepth = (clip.z / clip.w) * 0.5 + 0.5;

  fragColor = vec4(col, 1.0);
}
`;

const QUAD_POSITIONS = new Float32Array([
  -0.5, -0.5, 0,
   0.5, -0.5, 0,
  -0.5,  0.5, 0,
   0.5,  0.5, 0,
]);
const QUAD_INDICES = new Uint16Array([0, 1, 2, 1, 3, 2]);

const WORLD_LIGHT = new THREE.Vector3(0.55, 1.0, 0.6).normalize();
const _tmpLight = new THREE.Vector3();

export class ParticleField {
  constructor(attractor, scheme) {
    this.attractor = attractor;
    this.scheme = scheme;
    this.count = attractor.points;

    const geom = new THREE.InstancedBufferGeometry();
    geom.setAttribute("position", new THREE.BufferAttribute(QUAD_POSITIONS, 3));
    geom.setIndex(new THREE.BufferAttribute(QUAD_INDICES, 1));
    geom.instanceCount = this.count;

    this.iPosition = new THREE.InstancedBufferAttribute(new Float32Array(this.count * 3), 3);
    this.iColor = new THREE.InstancedBufferAttribute(new Float32Array(this.count * 3), 3);
    this.iRadius = new THREE.InstancedBufferAttribute(new Float32Array(this.count), 1);
    geom.setAttribute("iPosition", this.iPosition);
    geom.setAttribute("iColor", this.iColor);
    geom.setAttribute("iRadius", this.iRadius);

    // Particle radius is a fraction of the attractor's display scale, so
    // spheres read at the same screen size regardless of how big the
    // underlying coordinate range happens to be.
    this.baseRadius = Math.max(attractor.displayScale * 0.018, 0.06);

    this.material = new THREE.ShaderMaterial({
      glslVersion: THREE.GLSL3,
      uniforms: {
        uProjectionMatrix: { value: new THREE.Matrix4() },
        uLightDir: { value: new THREE.Vector3(0, 0, 1) },
        uRimColor: { value: new THREE.Color(scheme.rimColor[0], scheme.rimColor[1], scheme.rimColor[2]) },
        uSpecColor: { value: new THREE.Color(scheme.specColor[0], scheme.specColor[1], scheme.specColor[2]) },
        uAmbient: { value: 0.18 },
        uRimStrength: { value: 0.42 },
        uSpecStrength: { value: 0.55 },
        uSpecPower: { value: 48.0 },
      },
      vertexShader: VERT,
      fragmentShader: FRAG,
    });

    this.mesh = new THREE.Mesh(geom, this.material);
    this.mesh.frustumCulled = false;

    this._writeStaticAttributes();
  }

  _writeStaticAttributes() {
    // Per-particle colour comes straight from the scheme (vivid hues at the
    // golden angle around the wheel). Radii are uniform across the cloud.
    this.iColor.array.set(this.scheme.particleColors);
    const radii = this.iRadius.array;
    for (let i = 0; i < this.count; i++) radii[i] = this.baseRadius;
    this.iColor.needsUpdate = true;
    this.iRadius.needsUpdate = true;
  }

  // Pushes the latest particle positions to the GPU. Cheap — just a copy +
  // needsUpdate flag.
  update() {
    this.iPosition.array.set(this.attractor.states);
    this.iPosition.needsUpdate = true;
  }

  updatePerFrameUniforms(camera) {
    // Transform the world-space light into view space so highlights stay
    // anchored to a global direction instead of swinging with the camera.
    _tmpLight.copy(WORLD_LIGHT).transformDirection(camera.matrixWorldInverse);
    this.material.uniforms.uLightDir.value.copy(_tmpLight).normalize();
    this.material.uniforms.uProjectionMatrix.value.copy(camera.projectionMatrix);
  }
}
