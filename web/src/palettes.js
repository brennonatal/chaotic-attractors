// Direct port of the Pantone-inspired palette set from animations/live_display.py.
// Each colour is RGBA in [0, 1] so the JS side can interpolate without conversion.

function hex(value, alpha = 1.0) {
  const v = value.replace(/^#/, "");
  return [
    parseInt(v.slice(0, 2), 16) / 255,
    parseInt(v.slice(2, 4), 16) / 255,
    parseInt(v.slice(4, 6), 16) / 255,
    alpha,
  ];
}

export const PALETTES = [
  {
    name: "mocha periwinkle",
    background: hex("0B0A10"),
    fog:        hex("2A1E22", 0.14),
    head:       hex("F7E1D2", 0.98),
    hot:        hex("A47864", 0.76),
    cool:       hex("6667AB", 0.66),
    ghost:      hex("0B0A10", 0.00),
    star:       hex("F2D8C2", 0.22),
  },
  {
    name: "peach ink",
    background: hex("080A12"),
    fog:        hex("281B25", 0.15),
    head:       hex("FFE4D6", 0.98),
    hot:        hex("FFBE98", 0.78),
    cool:       hex("5B7C99", 0.64),
    ghost:      hex("080A12", 0.00),
    star:       hex("FFD6BF", 0.20),
  },
  {
    name: "viva cyan",
    background: hex("0A0710"),
    fog:        hex("2B0E22", 0.16),
    head:       hex("FDE7F0", 0.98),
    hot:        hex("BB2649", 0.78),
    cool:       hex("00A6A6", 0.64),
    ghost:      hex("0A0710", 0.00),
    star:       hex("F4A3B7", 0.20),
  },
  {
    name: "serenity coral",
    background: hex("071018"),
    fog:        hex("10253A", 0.15),
    head:       hex("F4FBFF", 0.98),
    hot:        hex("F7786B", 0.76),
    cool:       hex("92A8D1", 0.66),
    ghost:      hex("071018", 0.00),
    star:       hex("C9D8F2", 0.19),
  },
  {
    name: "greenery ultraviolet",
    background: hex("070C09"),
    fog:        hex("102618", 0.14),
    head:       hex("F4FFE8", 0.98),
    hot:        hex("88B04B", 0.76),
    cool:       hex("5F4B8B", 0.66),
    ghost:      hex("070C09", 0.00),
    star:       hex("D9F2B4", 0.18),
  },
];
