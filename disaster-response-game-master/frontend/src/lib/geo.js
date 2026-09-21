// Anchors the simulation's abstract 40x26 grid onto a real lat/lng bounding box, so the
// tactical map can render on real satellite/street tiles instead of a bare grid.
//
// Default anchor: Chennai, India's coastline (flood/cyclone-prone, so it fits the
// scenario set). Swap ANCHOR below for any other real place — nothing else in the app
// needs to change, since every consumer goes through gridToLatLng/metersPerCell.

export const ANCHOR = {
  north: 13.095,
  south: 13.03,
  west: 80.22,
  east: 80.295,
};

/** Convert a [x, y] grid cell (0..grid.width, 0..grid.height) to [lat, lng]. */
export function gridToLatLng([x, y], grid) {
  const { north, south, east, west } = ANCHOR;
  const lng = west + (x / grid.width) * (east - west);
  // grid y grows downward (screen-style); lat grows upward, so invert.
  const lat = north - (y / grid.height) * (north - south);
  return [lat, lng];
}

/** Bounds usable directly by react-leaflet's <MapContainer bounds=...>. */
export function anchorBounds() {
  const { north, south, east, west } = ANCHOR;
  return [
    [south, west],
    [north, east],
  ];
}

/** Approx meters per grid cell (x-direction), for sizing radii that should scale with real distance. */
export function metersPerCell(grid) {
  const { north, south, east, west } = ANCHOR;
  const midLatRad = ((north + south) / 2) * (Math.PI / 180);
  const metersPerDegreeLng = 111_320 * Math.cos(midLatRad);
  return ((east - west) / grid.width) * metersPerDegreeLng;
}
