"""Builds the jigsaw's geometry and its print-and-cut template. Not needed to run the task (pieces.json is checked in).

    pip install shapely mapbox-earcut
    python make_pieces.py          -> pieces.json (read by task.py), print_and_cut.svg (the real build)

Board frame: millimetres, origin at the board's centre, X away from the robot, Y to the robot's left. The picture is
drawn in image coordinates (u to the robot's right, v away from the robot, origin at the board's near-left corner),
so it reads upright from the robot's side.

Pieces: a 4 x 4 grid of 44 mm cells. Each inner edge has one round-headed tab (a neck and a round head, as on a
cardboard jigsaw) owned by one of its two cells, with a socket in the other. A tab's position along its edge and its
head's size vary a little from edge to edge (a fixed draw), so a piece fits its hole only one way round. Every piece
is its cell's share of the board shrunk by half the clearance, so neighbours have CLEARANCE between them everywhere.
"""

from __future__ import annotations

import json
import math
import pathlib
import random

import mapbox_earcut as earcut
import numpy as np
from shapely import affinity
from shapely.geometry import MultiPolygon, Point, Polygon, box
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

HERE = pathlib.Path(__file__).resolve().parent

N = 4
CELL = 44.0
HALF_BOARD = N * CELL / 2
CLEARANCE = 1.6            # between neighbouring pieces, everywhere
NECK_W = 7.0               # a tab's neck, across
HEAD_R = 6.0               # its round head
HEAD_C = 8.0               # head centre, out from the edge (a tab reaches 14 mm)
JITTER_OFFSET = 4.0        # a tab sits up to this far off its edge's middle
JITTER_R = 0.5             # and its head is up to this much bigger or smaller
SEED = 11

HOLE = (1, 1)              # (i, j): i along X from the robot's side, j along Y from the robot's right
DIRS = {"+X": (1, 0), "+Y": (0, 1), "-X": (-1, 0), "-Y": (0, -1)}
# The missing piece's edges: sockets toward and away from the robot (its neighbours there own the tabs), tabs to the
# robot's left and right.
HOLE_EDGES = {"+X": "S", "-X": "S", "+Y": "T", "-Y": "T"}


def cell_centre(c):
    return ((c[0] - 1.5) * CELL, (c[1] - 1.5) * CELL)


def cells():
    return [(i, j) for i in range(N) for j in range(N)]


def inner_edges():
    out = []
    for i, j in cells():
        for d in ("+X", "+Y"):
            ux, uy = DIRS[d]
            if i + ux < N and j + uy < N:
                out.append(((i, j), (i + ux, j + uy)))
    return out


def draw_edges():
    """Each inner edge: (owner of the tab, offset along the edge, head radius)."""
    rng = random.Random(SEED)
    edges = {}
    for a, b in inner_edges():
        owner = rng.choice((a, b))
        offset = rng.uniform(-JITTER_OFFSET, JITTER_OFFSET)
        r = HEAD_R + rng.uniform(-JITTER_R, JITTER_R)
        edges[frozenset((a, b))] = [owner, offset, r]
    for d, kind in HOLE_EDGES.items():
        other = (HOLE[0] + DIRS[d][0], HOLE[1] + DIRS[d][1])
        edges[frozenset((HOLE, other))][0] = HOLE if kind == "T" else other
    return edges


EDGES = draw_edges()


def tab(owner, other, offset, r):
    """The tab from ``owner`` into ``other``: a neck and a round head."""
    (ax, ay), (bx, by) = cell_centre(owner), cell_centre(other)
    nx, ny = (bx - ax) / CELL, (by - ay) / CELL          # outward
    tx, ty = -ny, nx                                    # along the edge
    mx, my = (ax + bx) / 2 + tx * offset, (ay + by) / 2 + ty * offset
    neck = Polygon([(mx + tx * s * NECK_W / 2 + nx * t, my + ty * s * NECK_W / 2 + ny * t)
                    for s, t in ((-1, -0.5), (1, -0.5), (1, HEAD_C), (-1, HEAD_C))])
    head = Point(mx + nx * HEAD_C, my + ny * HEAD_C).buffer(r, quad_segs=5)
    return unary_union([neck, head])


def raw_piece(c, kinds=None):
    """Cell ``c``'s share of the board (no clearance). ``kinds`` overrides which edges are tabs ("T") or sockets
    ("S"), for the decoys, using the same edges' tab shapes."""
    x, y = cell_centre(c)
    shape = box(x - CELL / 2, y - CELL / 2, x + CELL / 2, y + CELL / 2)
    add, cut = [], []
    for d, (ux, uy) in DIRS.items():
        other = (c[0] + ux, c[1] + uy)
        if not (0 <= other[0] < N and 0 <= other[1] < N):
            continue
        owner, offset, r = EDGES[frozenset((c, other))]
        mine = (kinds[d] == "T") if kinds else owner == c
        # the tab's shape belongs to the edge: drawn from whichever side owns it on the board
        board_owner = owner
        board_other = other if owner == c else c
        t = tab(board_owner, board_other, offset, r)
        if mine:
            if owner != c:  # a decoy's tab where the board has a socket: the same tab, mirrored to point outward
                t = affinity.scale(t, xfact=-1 if ux else 1, yfact=-1 if uy else 1,
                                   origin=((x + (x + ux * CELL)) / 2, (y + (y + uy * CELL)) / 2))
            add.append(t)
        else:
            if owner == c:  # a decoy's socket where the board has a tab: the tab's shape, mirrored inward
                t = affinity.scale(t, xfact=-1 if ux else 1, yfact=-1 if uy else 1,
                                   origin=((x + (x + ux * CELL)) / 2, (y + (y + uy * CELL)) / 2))
            cut.append(t)
    shape = unary_union([shape, *add])
    if cut:
        shape = shape.difference(unary_union(cut))
    return shape


def piece_outline(raw):
    p = raw.buffer(-CLEARANCE / 2, join_style="mitre", quad_segs=3)
    p = p.simplify(0.05)
    return orient(p, 1.0)


def edge_kinds(c):
    out = {}
    for d, (ux, uy) in DIRS.items():
        other = (c[0] + ux, c[1] + uy)
        if 0 <= other[0] < N and 0 <= other[1] < N:
            out[d] = "T" if EDGES[frozenset((c, other))][0] == c else "S"
        else:
            out[d] = "F"
    return out


# -- the picture ----------------------------------------------------------------------------------------------------
SKY, SUN, HILL, MEADOW = (0.55, 0.80, 0.97), (1.00, 0.80, 0.10), (0.42, 0.45, 0.62), (0.36, 0.72, 0.28)
RIVER, WALL, ROOF, TRUNK, CROWN = (0.10, 0.32, 0.85), (0.88, 0.18, 0.14), (0.40, 0.17, 0.08), (0.45, 0.28, 0.12), \
    (0.06, 0.40, 0.14)


def picture_uv():
    """(colour, shape) in image mm, painted in order."""
    w = 2 * HALF_BOARD
    us = np.linspace(-120, w + 120, 160)
    horizon = [(u, 80 + 9 * math.sin(u / w * 2 * math.pi * 1.3 + 0.4)) for u in us]
    centre = [(u, 26 + 34 * u / w + 7 * math.sin(u / 34)) for u in us]
    river_top = [(u, v + 8) for u, v in centre]
    river_bot = [(u, v - 8) for u, v in centre]
    return [
        (SKY, box(-120, -120, w + 120, w + 120)),
        (SUN, Point(132, 142).buffer(22, quad_segs=12)),
        (HILL, Polygon([(-120, 100), (-2, 100), (40, 146), (80, 108), (112, 128), (w + 2, 94), (w + 120, 94),
                        (w + 120, -120), (-120, -120)])),
        (MEADOW, Polygon(horizon + [(w + 120, -120), (-120, -120)])),
        (RIVER, Polygon(river_bot + river_top[::-1])),
        (WALL, box(20, 84, 58, 108)),
        (ROOF, Polygon([(14, 108), (39, 128), (64, 108)])),
        (TRUNK, box(138, 70, 146, 88)),
        (CROWN, Point(142, 98).buffer(14, quad_segs=12)),
    ]


def uv_to_board(g):
    return affinity.affine_transform(g, [0, 1, -1, 0, -HALF_BOARD, HALF_BOARD])


def picture(transform=None):
    """The picture in the board frame, as non-overlapping (colour, region) pairs. ``transform`` (in image mm) gives a
    decoy's wrong picture."""
    regions = []
    covered = None
    for colour, g in reversed(picture_uv()):
        if transform:
            g = transform(g)
        g = uv_to_board(g)
        part = g if covered is None else g.difference(covered)
        covered = g if covered is None else unary_union([covered, g])
        if not part.is_empty:
            regions.append((colour, part))
    return regions[::-1]


def colour_at(regions, x, y):
    p = Point(x, y)
    for colour, g in regions:
        if g.covers(p):
            return colour
    return None


# -- the four loose pieces --------------------------------------------------------------------------------------------
# Each is cut at the hole's place; ``kinds`` its tabs and sockets, ``image`` which picture is printed on it.


def _from_the_right(g):
    """The picture as it is one cell to the robot's right: a real part of the scene, in the wrong place."""
    return affinity.translate(g, -CELL, 0.0)


def _upside_down(g):
    return affinity.scale(g, yfact=-1, origin=(HALF_BOARD, HALF_BOARD))


CANDIDATES = {
    # name: (tabs and sockets, picture (None: the right one), what it is)
    "piece_a": ({"+X": "T", "+Y": "T", "-X": "S", "-Y": "S"}, None,
                "the right picture, the wrong tabs (tabs away from the robot and to its left, sockets toward it and "
                "to its right)"),
    "piece_b": (dict(HOLE_EDGES), _from_the_right,
                "the right shape, the wrong picture (the picture of the piece to the hole's right)"),
    "piece_c": (dict(HOLE_EDGES), None, "the missing piece"),
    "piece_d": ({"+X": "T", "+Y": "T", "-X": "T", "-Y": "S"}, _upside_down,
                "neither: three tabs, and the hole's picture upside down"),
}
CORRECT = "piece_c"


# -- meshes ---------------------------------------------------------------------------------------------------------
def polygons(g):
    if isinstance(g, Polygon):
        return [g]
    if isinstance(g, MultiPolygon):
        return list(g.geoms)
    return [p for p in getattr(g, "geoms", []) if isinstance(p, Polygon)]


def triangulate(poly: Polygon):
    poly = orient(poly, 1.0)
    rings = [np.asarray(poly.exterior.coords)[:-1]] + [np.asarray(r.coords)[:-1] for r in poly.interiors]
    verts = np.concatenate(rings)
    ends = np.cumsum([len(r) for r in rings]).astype(np.uint32)
    idx = earcut.triangulate_float64(verts, ends).reshape(-1, 3)
    return [tuple(map(tuple, verts[t])) for t in idx]


def _cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _clean(poly):
    """Drop repeated and collinear corners (they would hide a dent from the convexity test)."""
    out = []
    for v in poly:
        if not out or math.dist(v, out[-1]) > 1e-6:
            out.append(v)
    while len(out) > 1 and math.dist(out[0], out[-1]) <= 1e-6:
        out.pop()
    changed = True
    while changed and len(out) > 3:
        changed = False
        for k in range(len(out)):
            a, b, c = out[k - 1], out[k], out[(k + 1) % len(out)]
            if abs(_cross(a, b, c)) <= 1e-9:
                del out[k]
                changed = True
                break
    return out


def _convex(poly):
    poly = _clean(poly)
    n = len(poly)
    return n >= 3 and all(_cross(poly[k], poly[(k + 1) % n], poly[(k + 2) % n]) > 1e-9 for k in range(n))


def convex_parts(outline: Polygon):
    """Hertel-Mehlhorn: ear-clip, then merge neighbouring pieces while the result stays strictly convex. Every part
    is checked against its own convex hull, since the physics engine collides with the hull."""
    parts = [_clean(list(t)) for t in triangulate(outline)]
    parts = [p if _cross(*p[:3]) > 0 else p[::-1] for p in parts if len(p) == 3 and abs(_cross(*p)) > 1e-9]
    merged = True
    while merged:
        merged = False
        for a in range(len(parts)):
            for b in range(a + 1, len(parts)):
                pa, pb = parts[a], parts[b]
                shared = [(i, k) for i in range(len(pa)) for k in range(len(pb))
                          if pa[i] == pb[(k + 1) % len(pb)] and pa[(i + 1) % len(pa)] == pb[k]]
                if not shared:
                    continue
                i, k = shared[0]
                rest_a = pa[i + 1:] + pa[:i + 1]          # starts at pa[i+1], ends at pa[i]
                rest_b = pb[k + 2:] + pb[:k]              # pb after pa[i], up to before pa[i+1]
                cand = rest_a + rest_b
                if _convex(cand):
                    parts[a] = _clean(cand)
                    del parts[b]
                    merged = True
                    break
            if merged:
                break
    for p in parts:
        poly = Polygon(p)
        assert _convex(p) and abs(poly.convex_hull.area - poly.area) < 1e-6, "a part is not convex"
    return parts


def local(g, centre):
    return affinity.translate(g, -centre[0], -centre[1])


def r4(v):
    return round(float(v) / 1000.0, 5)  # mm -> m


THICK = 6.0         # mm plywood
EDGE = 2.0          # mm: every piece's top edge is chamfered this much in and down (a routed edge), and the
#                     picture is printed on the flat top inside it. Around the hole the glued pieces' chamfers and
#                     the loose piece's bevel below make a funnel.
CHAMFER = 2.0       # mm in from the edge, and
CHAMFER_H = 2.5     # mm up: a loose piece's bottom edge is bevelled, so a piece let go a few millimetres off its hole
#                     lands on the bevel and slides in instead of resting on a neighbour's edge


def bottom_of(part, outline):
    """A convex part's underside: the part less the outline's outer CHAMFER, or None where nothing is left (the part
    then starts CHAMFER above the floor)."""
    p = Polygon(part).intersection(outline.buffer(-CHAMFER, join_style="mitre"))
    if p.is_empty or p.area < 0.3:
        return None
    return list(orient(p.convex_hull, 1.0).exterior.coords)[:-1]


def solid_mesh(outline: Polygon):
    """A glued piece as one closed triangle mesh (points in mm, triangles facing out): flat bottom, straight sides up
    to THICK - EDGE, a chamfer in to the flat top, inset EDGE."""
    inner = orient(outline.buffer(-EDGE, join_style="mitre"), 1.0)
    outer_xy = list(outline.exterior.coords)[:-1]
    inner_xy = list(inner.exterior.coords)[:-1]
    n = len(outer_xy)
    zs = THICK - EDGE
    pts = [(x, y, 0.0) for x, y in outer_xy] + [(x, y, zs) for x, y in outer_xy] + [(x, y, THICK) for x, y in inner_xy]
    index = {(round(x, 9), round(y, 9), z): k for k, (x, y, z) in enumerate(pts)}
    tris = []

    def up(t):  # counterclockwise seen from above
        a, b, c = (pts[k] for k in t)
        return t if _cross(a, b, c) > 0 else (t[0], t[2], t[1])

    for t in triangulate(outline):                                   # bottom, facing down
        a, b, c = (index[(round(x, 9), round(y, 9), 0.0)] for x, y in t)
        tris.append(tuple(reversed(up((a, b, c)))))
    for k in range(n):                                               # sides
        j = (k + 1) % n
        tris += [(k, j, n + j), (k, n + j, n + k)]
    ring = Polygon(outer_xy, [inner_xy])                             # the chamfer
    for t in triangulate(ring):
        ids = []
        for x, y in t:
            key = (round(x, 9), round(y, 9))
            ids.append(index.get((*key, zs), index.get((*key, THICK))))
        tris.append(up(tuple(ids)))
    for t in triangulate(inner):                                     # top
        tris.append(up(tuple(index[(round(x, 9), round(y, 9), THICK)] for x, y in t)))
    return pts, tris


def piece_record(outline: Polygon, regions, centre, chamfer=False, solid=False):
    """What task.py needs: the outline, its convex parts and the printed picture's triangles, in metres in the
    piece's own frame (origin at its cell's centre)."""
    loc = local(outline, centre)
    decals = []
    printed = outline.buffer(-EDGE, join_style="mitre")
    for colour, g in regions:
        part = g.intersection(printed)
        tris = []
        for p in polygons(part):
            if p.area < 0.2:
                continue
            for t in triangulate(local(p, centre)):
                tris.append([[r4(a), r4(b)] for a, b in t])
        if tris:
            decals.append({"color": list(colour), "triangles": tris})
    parts = [p for p in convex_parts(loc) if Polygon(p).area > 0.3]
    rec = {
        "outline": [[r4(a), r4(b)] for a, b in list(loc.exterior.coords)[:-1]],
        "parts": [[[r4(a), r4(b)] for a, b in p] for p in parts],
        "decals": decals,
    }
    if solid:  # a glued piece is one exact triangle-mesh collider
        pts, tris = solid_mesh(loc)
        rec["mesh"] = {"points": [[r4(x), r4(y), r4(z)] for x, y, z in pts], "triangles": [list(t) for t in tris]}
    if chamfer:
        rec["bottoms"] = [None if b is None else [[r4(x), r4(y)] for x, y in b]
                          for b in (bottom_of(p, loc) for p in parts)]
    return rec


def build():
    board = picture()
    out = {"cell": CELL / 1000, "clearance": CLEARANCE / 1000, "bevel": [CHAMFER / 1000, CHAMFER_H / 1000],
           "thick": THICK / 1000, "edge": EDGE / 1000,
           "hole": list(HOLE), "placed": {}, "candidates": {}, "correct": CORRECT}
    for c in cells():
        if c == HOLE:
            continue
        out["placed"][f"piece_{c[0]}{c[1]}"] = {"cell": list(c), "edges": edge_kinds(c),
                                                **piece_record(piece_outline(raw_piece(c)), board, cell_centre(c),
                                                             solid=True)}
    for name, (kinds, transform, what) in CANDIDATES.items():
        outline = piece_outline(raw_piece(HOLE, kinds))
        out["candidates"][name] = {"edges": kinds, "what": what,
                                   **piece_record(outline, picture(transform) if transform else board,
                                                  cell_centre(HOLE), chamfer=True)}
    return out


# -- print-and-cut template -------------------------------------------------------------------------------------------
def svg(data_board, data_candidates):
    """One sheet, 1 unit = 1 mm: the board's picture with all 16 pieces' cut lines, and the three decoys beside it,
    each with its own picture. Print it on sticker paper, stick it on 6 mm plywood and laser-cut the red lines."""
    def rgb(c):
        return "#%02x%02x%02x" % tuple(int(round(255 * v)) for v in c)

    def path(p: Polygon, ox, oy):
        def ring(coords):
            pts = [(u + ox, v + oy) for u, v in coords]
            return "M" + " L".join(f"{a:.2f},{b:.2f}" for a, b in pts) + " Z"
        return ring(p.exterior.coords) + "".join(ring(r.coords) for r in p.interiors)

    def to_sheet(g):  # board frame -> sheet (x right, y down), the picture upright
        return affinity.affine_transform(g, [0, -1, -1, 0, HALF_BOARD, HALF_BOARD])

    margin, gap = 10.0, 30.0
    width = margin * 2 + 2 * HALF_BOARD + gap + 80
    height = margin * 2 + 3 * 80 + 10
    body = []
    for colour, g in picture():
        for p in polygons(to_sheet(g.intersection(box(-HALF_BOARD, -HALF_BOARD, HALF_BOARD, HALF_BOARD)))):
            body.append(f'<path d="{path(p, margin, margin)}" fill="{rgb(colour)}" stroke="none"/>')
    for c in cells():
        body.append(f'<path d="{path(to_sheet(piece_outline(raw_piece(c))), margin, margin)}" fill="none" '
                    f'stroke="#ff0000" stroke-width="0.1"/>')
    notes = ("Board: cut all 16 outlines (each piece is cut on its own, 1.6 mm clear of its neighbours).",
             "Glue 15 to a 200 x 200 mm dark base; the one in row 2 from the bottom, column 2 from",
             "the right is piece_c, the missing piece. Glue a 12 mm square, 14 mm peg on each loose piece.")
    for k, line in enumerate(notes):
        body.append(f'<text x="{margin}" y="{margin + 2 * HALF_BOARD + 7 + 6 * k}" font-size="4">{line}</text>')
    k = 0
    for name, (kinds, transform, _) in CANDIDATES.items():
        if name == CORRECT:
            continue
        ox, oy = margin + 2 * HALF_BOARD + gap, margin + k * 80
        hc = cell_centre(HOLE)
        outline = piece_outline(raw_piece(HOLE, kinds))
        corner = to_sheet(Point(hc))

        def shift(g, corner=corner):
            return affinity.translate(to_sheet(g), -corner.x + 40, -corner.y + 40)
        for colour, g in (picture(transform) if transform else picture()):
            for p in polygons(shift(g.intersection(outline.buffer(3)))):
                body.append(f'<path d="{path(p, ox, oy)}" fill="{rgb(colour)}" stroke="none"/>')
        body.append(f'<path d="{path(shift(outline), ox, oy)}" fill="none" stroke="#ff0000" stroke-width="0.1"/>')
        body.append(f'<text x="{ox}" y="{oy + 78}" font-size="5">{name}</text>')
        k += 1
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}mm" height="{height}mm" '
            f'viewBox="0 0 {width} {height}">\n' + "\n".join(body) + "\n</svg>\n")


if __name__ == "__main__":
    data = build()
    (HERE / "pieces.json").write_text(json.dumps(data, separators=(",", ":")))
    (HERE / "print_and_cut.svg").write_text(svg(None, None))
    print("pieces.json:", sum(len(p["parts"]) for p in data["placed"].values()), "convex parts on the board;",
          {k: len(v["parts"]) for k, v in data["candidates"].items()})
