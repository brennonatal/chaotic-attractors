// Per-attractor vivid colour scheme. Each attractor's seed picks a base
// hue rotation; particles are then laid out at the golden angle around
// the hue wheel so neighbouring indices are always visually distinct,
// and lightness wobbles a touch so the bloom doesn't read as monotone.
//
// Everything returned here is in *linear* RGB space — three.js's
// WebGLRenderer gamma-encodes to sRGB at output, so feeding it raw 0..1
// would double-encode and wash everything out.

function srgbToLinear(c) {
  return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}

function hslToLinearRgb(h, s, l) {
  const hp = ((((h % 360) + 360) % 360)) / 60;
  const c = (1 - Math.abs(2 * l - 1)) * s;
  const x = c * (1 - Math.abs((hp % 2) - 1));
  let r, g, b;
  if (hp < 1)      { r = c; g = x; b = 0; }
  else if (hp < 2) { r = x; g = c; b = 0; }
  else if (hp < 3) { r = 0; g = c; b = x; }
  else if (hp < 4) { r = 0; g = x; b = c; }
  else if (hp < 5) { r = x; g = 0; b = c; }
  else             { r = c; g = 0; b = x; }
  const m = l - c / 2;
  return [srgbToLinear(r + m), srgbToLinear(g + m), srgbToLinear(b + m)];
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

// Golden angle in turns — multiplying by 360 gives the equivalent in degrees.
const GOLDEN_ANGLE_TURNS = 0.6180339887;

export function makeColorScheme(seed, particleCount) {
  const rng = mulberry32(((seed | 0) ^ 0xC0FFEE) >>> 0);
  const baseHue = rng() * 360;
  const accentHue = (baseHue + 60 + rng() * 240) % 360;

  // Per-particle vivid colours. Golden-angle hue spacing keeps adjacent
  // indices distinct; saturation pinned high (synthwave), lightness gently
  // jittered around 0.55 so bloom doesn't fuse everything into one blob.
  const particleColors = new Float32Array(particleCount * 3);
  for (let i = 0; i < particleCount; i++) {
    const h = baseHue + i * 360 * GOLDEN_ANGLE_TURNS;
    const l = 0.52 + 0.10 * (rng() - 0.5);
    const [r, g, b] = hslToLinearRgb(h, 0.95, l);
    particleColors[i * 3 + 0] = r;
    particleColors[i * 3 + 1] = g;
    particleColors[i * 3 + 2] = b;
  }

  // Deep near-black background tinted faintly toward the base hue, so the
  // void isn't pure RGB-zero (which can read as a dead screen).
  const [bgR, bgG, bgB] = hslToLinearRgb(baseHue, 0.5, 0.025);
  // Stars: lightly toward the accent hue so they tie the scene together.
  const [stR, stG, stB] = hslToLinearRgb(accentHue, 0.3, 0.78);

  return {
    name: `vivid ${Math.round(baseHue).toString().padStart(3, "0")}°`,
    background: [bgR, bgG, bgB, 1.0],
    particleColors,
    rimColor:  [0.95, 0.97, 1.0, 1.0],
    specColor: [1.00, 1.00, 1.0, 1.0],
    starColor: [stR, stG, stB, 0.34],
  };
}
