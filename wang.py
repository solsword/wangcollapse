"""
wang.py

Wang tiling code (corner-based).
"""

import random

import numpy as np

from quiche import dep

import pattern
import wfc

# Size of each supergrid tile
SGSIZE = 16

# Radius of corner seeds
SEED_RADIUS = 2

# Size of seeding template from which corner templates are created
TEMPLATE_SIZE = 64

# Number of corner templates to use
N_COLORS = 4
# TODO: Increase
#N_COLORS = 16

VERTICAL = 0
HORIZONTAL = 1

def hash(x, y):
  """
  Simple hash function for lattice indices.
  """
  return (x * 31 + y) * 17 * 1000 + 36542135464 + (x*y) + ((x*3)^y)

COLOR_COORDS = [
  (x, y)
    for x in range(TEMPLATE_SIZE - 2*SEED_RADIUS)
    for y in range(TEMPLATE_SIZE - 2*SEED_RADIUS)
]

random.shuffle(COLOR_COORDS)

def color_pos(color):
  """
  Mapping from "color" integer to template location.
  """
  return COLOR_COORDS[color % len(COLOR_COORDS)]

def supertile(pos):
  """
  Converts a normal tile position to a supertile position.
  """
  return (pos[0]//SGSIZE, pos[1]//SGSIZE)

def wang_params(spos):
  """
  Returns a set of wang tiling params for the given supertile location. Returns
  corner colors in clockwise order from the top left.
  """
  sgx = spos[0]
  sgy = spos[1]

  # colors
  c_ul = hash(sgx, sgy) % N_COLORS 
  c_ur = hash(sgx+1, sgy) % N_COLORS
  c_br = hash(sgx+1, sgy+1) % N_COLORS
  c_bl = hash(sgx, sgy+1) % N_COLORS

  return (c_ul, c_ur, c_br, c_bl)

def wang_edges(wang_params):
  """
  Converts Wang tile corner color parameters into edge color pairs. Uses north,
  east, south, west ordering and clockwise color ordering within edges.
  """
  ul, ur, br, bl = wang_params
  return ((ul, ur), (ur, br), (br, bl), (bl, ul))

def all_possible_edges(n_colors):
  """
  Lists all possible edge configurations for the given number of colors.
  """
  result = set()
  for orientation in (VERTICAL, HORIZONTAL):
    for c1 in range(n_colors):
      for c2 in range(n_colors):
        result.add((c1, c2, orientation))

  return list(result)

@dep.template_task(
  (
    "{batch}-wang-template",
    "{batch}-wfc-overlapping-compatibilities",
  ),
  "{batch}-{c1}:{c2}:{ori}-edge"
)
def edge_task(name_match, template, compat_tables):
  """
  Task template for create_edge. Ori should be either 'H' for horizontal or 'V'
  for vertical.
  """
  gd = name_match.groupdict()
  c1 = int(gd["c1"])
  c2 = int(gd["c2"])
  orientation = int(gd["ori"] == "H")
  return create_edge(template, compat_tables, c1, c2, orientation)

def create_edge(template, compat_tables, c1, c2, orientation):
  """
  Creates an edge from the given template using the given colors at the start
  and end and the given orientation (0 = vertical; 1 = horizontal). Returns a
  2*SGSIZE × SGSIZE or SGSIZE × 2*SGSIZE possibility field, censored outside of
  the diamond containing the given edge. E.g., for orientation=0 and colors 1
  and 2, if SGSIZE were 8, the 'F's in the diagram below would be filled in,
  the 'C's would be fully collapsed, and the '#'s, '/'s, and '\\'s would be
  completely unknown (all possible patterns allowed):

                   1
    \ # # # # # # C C # # # # # # /
    # \ # # # # / C C \ # # # # / #
    # # \ # # / F C C F \ # # / # #
    # # # \ / F F C C F F \ / # # #
    # # # / \ F F C C F F / \ # # #
    # # / # # \ F C C F / # # \ # #
    # / # # # # \ C C / # # # # \ #
    / # # # # # # C C # # # # # # \\
                   2

  """
  n_pat = compat_tables[0].shape[0]
  if orientation == 0: # vertical edge
    result = np.ones([2*SGSIZE, SGSIZE, n_pat], dtype=np.bool_)
  else:
    result = np.ones([SGSIZE, 2*SGSIZE, n_pat], dtype=np.bool_)

  c1p = color_pos(c1)
  c2p = color_pos(c2)
  seed1 = template[
    c1p[0]-SEED_RADIUS : c1p[0]+SEED_RADIUS,
    c1p[1]-SEED_RADIUS : c1p[1]+SEED_RADIUS,
    :
  ]
  seed2 = template[
    c2p[0]-SEED_RADIUS : c2p[0]+SEED_RADIUS,
    c2p[1]-SEED_RADIUS : c2p[1]+SEED_RADIUS,
    :
  ]

  # Insert the seeds:
  if orientation == 0:
    result[
      SGSIZE-SEED_RADIUS : SGSIZE+SEED_RADIUS,
      : SEED_RADIUS,
      :
    ] = seed1[:, SEED_RADIUS:, :]
    result[
      SGSIZE-SEED_RADIUS : SGSIZE+SEED_RADIUS,
      -SEED_RADIUS :,
      :
    ] = seed2[:, :SEED_RADIUS, :]
  else:
    result[
      : SEED_RADIUS,
      SGSIZE-SEED_RADIUS : SGSIZE+SEED_RADIUS,
      :
    ] = seed1[SEED_RADIUS:, :, :]
    result[
      -SEED_RADIUS :,
      SGSIZE-SEED_RADIUS : SGSIZE+SEED_RADIUS,
      :
    ] = seed2[:SEED_RADIUS, :, :]

  # Collapse the edge:
  if orientation == 0:
    edge = [
      (x, y)
        for x in range(SGSIZE-1,SGSIZE+1)
        for y in range(SGSIZE)
    ]
  else:
    edge = [
      (x, y)
        for x in range(SGSIZE)
        for y in range(SGSIZE-1,SGSIZE+1)
    ]
    
  wfc.collapse(
    result,
    compat_tables,
    edge
  )

  # Censor the results:
  if orientation == 0:
    for x in range(2*SGSIZE):
      for y in range(SGSIZE):
        dx = max(x - SGSIZE, SGSIZE - 1 - x)
        dy = min(y, SGSIZE - 1 - y)
        if y == 0 or y == SGSIZE - 1:
          if dx > dy:
            result[x, y, :] = 1
        elif dx >= dy:
          result[x, y, :] = 1
  else:
    for x in range(SGSIZE):
      for y in range(2*SGSIZE):
        dx = min(x, SGSIZE - 1 - x)
        dy = max(y - SGSIZE, SGSIZE - 1 - y)
        if x == 0 or x == SGSIZE - 1:
          if dy > dx:
            result[x, y, :] = 1
        elif dy >= dx:
          result[x, y, :] = 1

  return result

@dep.template_task(
  (
    "{batch}-wang-template",
    "{batch}-wfc-overlapping-compatibilities",
  ),
  "{batch}-edges"
)
def edge_cache(_, template, compat_tables):
  """
  Creates a cache of all edge templates from the given base template.
  """
  cache = {}
  for (c1, c2, ori) in all_possible_edges(N_COLORS):
    cache[(c1, c2, ori)] = create_edge(template, compat_tables, c1, c2, ori)

  return cache

@dep.template_task(
  (
    "{batch}-edges",
    "{batch}-wfc-patterns",
  ),
  "{batch}-edge-images"
)
def viz_edges(_, edges, patterns):
  results = []
  print("{} edges...".format(len(edges)))
  for k in edges:
    results.append(pattern.create_overlapping_image(edges[k], patterns))

  return results

@dep.template_task(
  (
    "{batch}-edges",
    "{batch}-wfc-overlapping-compatibilities",
  ),
  "{batch}-supertile-{sx}:{sy}"
)
def build_tile_task(name_match, edge_cache, compat_tables):
  gd = name_match.groupdict()
  sx = int(gd["sx"])
  sy = int(gd["sy"])
  return build_tile((sx, sy), edge_cache, compat_tables)

def build_tile(spos, edge_cache, compat_tables):
  """
  Builds the tile at the given supergrid position.
  """
  return wfc.collapse(init_tile(spos, edge_cache), compat_tables)

def init_tile(spos, edge_cache):
  """
  Returns the initial possibility assignments for the given supertile.
  """
  wp = wang_params(spos)
  edges = wang_edges(wp)
  north, east, south, west = edges

  nt = edge_cache[(north[0], north[1], HORIZONTAL)]
  et = edge_cache[(east[0], east[1], VERTICAL)]
  st = edge_cache[(south[1], south[0], HORIZONTAL)]
  wt = edge_cache[(west[1], west[0], VERTICAL)]

  return (
    nt[      :      , SGSIZE:      , : ]
  & et[      :SGSIZE,       :      , : ]
  & st[      :      ,       :SGSIZE, : ]
  & wt[SGSIZE:      ,       :      , : ]
  )

@dep.template_task(
  (
    "{batch}-edges",
    "{batch}-wfc-patterns",
  ),
  "{batch}-init-test"
)
def test_init(_, edge_cache, patterns):
  t1 = init_tile((5, 5), edge_cache)
  t2 = init_tile((6, 5), edge_cache)
  t3 = init_tile((5, 6), edge_cache)
  t4 = init_tile((6, 6), edge_cache)
  region = assemble_region([t1, t2, t3, t4], 2, 2)
  return pattern.create_overlapping_image(region, patterns)


# A simple template alias:
@dep.template_task(
  (
    "{{batch}}-wfc-overlapping-probabilities-{size}".format(
      size="{n}×{n}".format(n=TEMPLATE_SIZE)
    ),
  ),
  "{batch}-wang-template"
)
def create_wang_template(_, value):
  return value

def tile_at(params, pos):
  """
  Function for fetching the tile at the given position.
  """
  batch = wfc.batchname(
    params["filename"],
    params["pattern_radius"],
    params["add_rotations"]
  )
  spos = supertile(pos)
  ts, result = dep.create_brave(
    "{batch}-supertile-{sx}:{sy}".format(
      batch=batch,
      sx=spos[0],
      sy=spos[1]
    )
  )
  ipos = (pos[0] % SGSIZE, pos[1] % SGSIZE)

  possibilities = result[ipos[0], ipos[1], :]
  pidx = np.argmax(possibilities)

  ts, patterns = dep.create_brave("{batch}-wfc-patterns".format(batch=batch))

  return pattern.center_of(patterns[pidx])

def supertile_at(params, spos):
  """
  Fetches a whole supertile at once, putting less strain on the caching system.
  """
  batch = wfc.batchname(
    params["filename"],
    params["pattern_radius"],
    params["add_rotations"]
  )
  ts, result = dep.create_brave(
    "{batch}-supertile-{sx}:{sy}".format(
      batch=batch,
      sx=spos[0],
      sy=spos[1]
    )
  )

  ts, patterns = dep.create_brave("{batch}-wfc-patterns".format(batch=batch))

  return pattern.create_overlapping_image(result, patterns)

def assemble_region(pieces, width, height):
  """
  Assembles N-dimensional pieces by concatenating along their 1st and 2nd
  dimensions.
  """
  rows = []
  for r in range(height):
    rows.append(np.concatenate(pieces[r*width:(r+1)*width], axis=0))
  return np.concatenate(rows, axis=1)

@dep.template_task(
  (
    "{batch}-wfc-patterns",
    "{batch}-supertile-175:342",
    "{batch}-supertile-176:342",
    "{batch}-supertile-177:342",
    "{batch}-supertile-175:343",
    "{batch}-supertile-176:343",
    "{batch}-supertile-177:343",
    "{batch}-supertile-175:344",
    "{batch}-supertile-176:344",
    "{batch}-supertile-177:344",
  ),
  "{batch}-wang-test"
)
def wang_test(_, patterns, *supertiles):
  images = [
    pattern.create_overlapping_image(st, patterns)
    for st in supertiles
  ]
  return assemble_region(images, 3, 3)

@dep.template_task(
  (
    "{batch}-wfc-patterns",
  ) + tuple(
    "{{batch}}-supertile-{x}:{y}".format(
      x=x,
      y=y
    )
      for y in range(19820, 19828)
      for x in range(182, 190)
  ),
  "{batch}-wang-bigtest"
)
def wang_bigtest(_, patterns, *supertiles):
  images = [
    pattern.create_overlapping_image(st, patterns)
    for st in supertiles
  ]
  return assemble_region(images, 8, 8)

def target_name(input_filename, big=False):
  return "{batch}-wang-{big}test".format(
    batch=wfc.batchname(input_filename, 1, True),
    big="big" if big else ""
  )

def edgeviz_target(input_filename):
  return "{batch}-edge-images".format(
    batch=wfc.batchname(input_filename, 1, True),
  )
