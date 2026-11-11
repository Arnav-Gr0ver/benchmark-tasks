"""Build files for the real board, from the same tables as the simulation (maze.py). Pure Python (math only).

    python make_cad.py cad

Writes:
- maze_board.stl   base plate and walls as one print (mm); the walls are boxes along each arc, as simulated
- maze_plan.svg    1:1 top view (print at 100 %): walls, gaps, spokes, the ball's start mark and the centre cell
- maze_plan.dxf    the same outlines for a laser cutter or CNC router (mm)
- maze_lid.dxf     the clear lid: a 280 mm disc
"""

from __future__ import annotations

import math
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import maze  # noqa: E402

MM = 1000.0


def _box_corners(b) -> list[tuple[float, float]]:
    c, s = math.cos(b.yaw), math.sin(b.yaw)
    return [(MM * (b.x + c * sx * b.hx - s * sy * b.hy), MM * (b.y + s * sx * b.hx + c * sy * b.hy))
            for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1))]


def _prism(outline, z0, z1):
    """Triangles of a closed prism over a convex outline (counterclockwise seen from above)."""
    tris = []
    n = len(outline)
    cx = sum(p[0] for p in outline) / n
    cy = sum(p[1] for p in outline) / n
    for i in range(n):
        a, b = outline[i], outline[(i + 1) % n]
        tris.append(((cx, cy, z1), (a[0], a[1], z1), (b[0], b[1], z1)))
        tris.append(((cx, cy, z0), (b[0], b[1], z0), (a[0], a[1], z0)))
        tris.append(((a[0], a[1], z0), (b[0], b[1], z0), (b[0], b[1], z1)))
        tris.append(((a[0], a[1], z0), (b[0], b[1], z1), (a[0], a[1], z1)))
    return tris


def _normal(t):
    (ax, ay, az), (bx, by, bz), (cx, cy, cz) = t
    ux, uy, uz, vx, vy, vz = bx - ax, by - ay, bz - az, cx - ax, cy - ay, cz - az
    nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
    n = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
    return nx / n, ny / n, nz / n


def write_stl(path: Path) -> int:
    disc = [(MM * maze.BOARD_R * math.cos(2 * math.pi * i / 180), MM * maze.BOARD_R * math.sin(2 * math.pi * i / 180))
            for i in range(180)]
    tris = _prism(disc, -MM * maze.BASE_T, 0.0)
    for b in maze.wall_boxes():
        tris += _prism(_box_corners(b), -0.2, MM * maze.WALL_H)  # 0.2 mm into the base, so the slicer merges them
    with path.open("wb") as f:
        f.write(b"tilt maze: base and walls, mm".ljust(80, b" "))
        f.write(struct.pack("<I", len(tris)))
        for t in tris:
            f.write(struct.pack("<3f", *_normal(t)))
            for v in t:
                f.write(struct.pack("<3f", *v))
            f.write(b"\0\0")
    return len(tris)


def _dxf(entities: list[str]) -> str:
    head = ["0", "SECTION", "2", "HEADER", "9", "$INSUNITS", "70", "4", "0", "ENDSEC", "0", "SECTION", "2", "ENTITIES"]
    return "\n".join(head + entities + ["0", "ENDSEC", "0", "EOF"]) + "\n"


def _dxf_circle(x, y, r, layer):
    return ["0", "CIRCLE", "8", layer, "10", f"{x:.4f}", "20", f"{y:.4f}", "30", "0", "40", f"{r:.4f}"]


def _dxf_poly(points, layer):
    out = ["0", "LWPOLYLINE", "8", layer, "90", str(len(points)), "70", "1"]
    for x, y in points:
        out += ["10", f"{x:.4f}", "20", f"{y:.4f}"]
    return out


def write_dxf(plan_path: Path, lid_path: Path) -> None:
    ents = _dxf_circle(0, 0, MM * maze.BOARD_R, "BASE")
    for b in maze.wall_boxes():
        ents += _dxf_poly(_box_corners(b), "WALLS")
    ents += _dxf_circle(MM * maze.START_XY[0], MM * maze.START_XY[1], MM * maze.BALL_R, "START_MARK")
    ents += _dxf_circle(0, 0, MM * (maze.WALLS[0] - maze.WALL_T / 2), "CENTRE_CELL")
    plan_path.write_text(_dxf(ents))
    lid_path.write_text(_dxf(_dxf_circle(0, 0, MM * maze.BOARD_R, "LID")))


def write_svg(path: Path) -> None:
    r = MM * maze.BOARD_R
    size = 2 * r + 20
    # SVG y points down; the drawing is the board seen from above with the robot at the bottom: board +x (far side)
    # up the page and board +y (robot's left) to the left, so (x, y) -> (-y, -x).
    def pt(x, y):
        return f"{-y:.3f},{-x:.3f}"

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}mm" height="{size + 14}mm" '
             f'viewBox="{-size / 2} {-size / 2} {size} {size + 14}">',
             f'<circle cx="0" cy="0" r="{r}" fill="#f2f2ee" stroke="#555" stroke-width="0.4"/>',
             f'<circle cx="0" cy="0" r="{MM * (maze.WALLS[0] - maze.WALL_T / 2)}" fill="#3aa65a"/>']
    for b in maze.wall_boxes():
        pts = " ".join(pt(x, y) for x, y in _box_corners(b))
        parts.append(f'<polygon points="{pts}" fill="#1c1c20"/>')
    sx, sy = MM * maze.START_XY[0], MM * maze.START_XY[1]
    parts.append(f'<circle cx="{-sy}" cy="{-sx}" r="{MM * maze.BALL_R}" fill="none" stroke="#d01010" '
                 f'stroke-width="0.6"/>')
    for deg in range(0, 360, 30):
        a = math.radians(deg)
        x, y = (r + 5) * math.cos(a), (r + 5) * math.sin(a)
        parts.append(f'<text x="{-y:.2f}" y="{-x + 1.2:.2f}" font-size="3.2" text-anchor="middle" '
                     f'font-family="sans-serif">{deg}</text>')
    parts.append(f'<text x="0" y="{size / 2 + 6}" font-size="3.5" text-anchor="middle" font-family="sans-serif">'
                 f'Robot side. Board {2 * r:.0f} mm; print at 100 %. Red circle: ball start mark.</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n")


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "cad")
    out.mkdir(parents=True, exist_ok=True)
    n = write_stl(out / "maze_board.stl")
    write_dxf(out / "maze_plan.dxf", out / "maze_lid.dxf")
    write_svg(out / "maze_plan.svg")
    print(f"wrote {out}/maze_board.stl ({n} triangles), maze_plan.svg, maze_plan.dxf, maze_lid.dxf")
