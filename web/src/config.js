// Mirror of QUALITY_PRESETS in animations/live_display.py.
// (points, trailLength, dt, stepsPerFrame, starCount, glowWidth)
export const QUALITY_PRESETS = {
  balanced: { points: 21, trailLength: 320, dt: 0.0030, stepsPerFrame: 5, starCount: 45,  glowWidth: 4.4 },
  cinema:   { points: 33, trailLength: 420, dt: 0.0026, stepsPerFrame: 6, starCount: 80,  glowWidth: 6.0 },
  ultra:    { points: 55, trailLength: 520, dt: 0.0022, stepsPerFrame: 7, starCount: 120, glowWidth: 7.2 },
};

export const DEFAULT_QUALITY = "cinema";
export const AUTO_ADVANCE_SECONDS = 45;
export const MIN_BBOX_RANGE = 1e-6;
