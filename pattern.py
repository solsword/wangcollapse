"""
pattern.py

Common functions for managing/extracting patterns from images.
"""

from quiche import dep

import numpy as np

N_CHANNELS = 3

NORTH = 0
EAST = 1
SOUTH = 2
WEST = 3
N_SIDES = 4

def opposite_side(side):
  """
  Computes the opposite of the given side.
  """
  return (side + N_SIDES//2) % N_SIDES

def pattern_at(image, xy, pattern_radius=1):
  """
  Extracts the pattern at the given position in the given image. Treats the
  image as a torus, wrapping at both edges.
  """
  width, height = image.shape[:2]

  x = xy[0]
  y = xy[1]

  while x < 0:
    x += width
  while x >= width:
    x -= width

  while y < 0:
    y += height
  while y >= height:
    y -= height

  pattern = np.zeros([pattern_radius*2+1, pattern_radius*2+1, N_CHANNELS])

  for dx in range(-pattern_radius, pattern_radius+1):
    for dy in range(-pattern_radius, pattern_radius+1):
      ox = x + dx
      oy = y + dy

      if ox < 0:
        ox += width
      if ox >= width:
        ox -= width

      if oy < 0:
        oy += height
      if oy >= height:
        oy -= height

      px = tuple(image[ox,oy])

      pattern[dx + pattern_radius, dy + pattern_radius, :] = px

  return tuple(pattern.flatten())

def edge_of(pattern, side, pattern_radius=1):
  """
  Extracts the edge identity of the given side of the given pattern.
  """
  pw = pattern_radius*2 + 1
  nc = N_CHANNELS
  prow = nc*pw
  if side == NORTH:
    return pattern[0:prow]
  elif side == EAST:
    edge = ()
    x = pw-1
    for y in range(pw):
      edge += pattern[y*prow + nc*x:y*prow + nc*(x+1)]
    return edge
  elif side == SOUTH:
    return pattern[(pw-1)*prow:]
  elif side == WEST:
    edge = ()
    x = 0
    for y in range(pw):
      edge += pattern[y*prow:y*prow + nc]
    return edge

def pattern_match_list(patterns, pattern, side):
  """
  Computes the set of matching patterns from the given set on the given side of
  the given pattern.
  """
  opp = opposite_side(side)
  result = []
  edge = edge_of(pattern, side)
  for i, p in enumerate(patterns):
    if edge_of(p, opp) == edge:
      result.append(i)

  return result

def pattern_tensor(pattern, pattern_radius=1):
  """
  Transforms a pattern (back) into a tensor.
  """
  pw = pattern_radius * 2 + 1
  return np.array(pattern).reshape([pw, pw, N_CHANNELS])

def pattern_crop(ptensor, side):
  """
  Returns the part of the given pattern tensor that excludes one row opposite
  the given side.
  """
  if side == NORTH:
    return ptensor[:,:-1,:]
  elif side == EAST:
    return ptensor[1:,:,:]
  elif side == SOUTH:
    return ptensor[:,1:,:]
  elif side == WEST:
    return ptensor[:-1,:,:]

def overlap_matches(patternA, patternB, side, pattern_radius=1):
  """
  Returns True or False depending on whether the given patterns can overlap
  with the center of pattern B placed adjacent to the center of pattern A in
  the direction indicated by side.
  """
  ta = pattern_tensor(patternA, pattern_radius)
  tb = pattern_tensor(patternB, pattern_radius)
  opp = opposite_side(side)
  return np.array_equal(pattern_crop(ta, side), pattern_crop(tb, opp))

def adjacent_match_list(patterns, pattern, side, pattern_radius=1):
  """
  Instead of asking whether the given patterns can be placed edge-adjacent,
  this asks whether they can be overlapped so that their centers are adjacent.
  """
  result = []
  edge = edge_of(pattern, side)
  for i, p in enumerate(patterns):
    if overlap_matches(pattern, p, side, pattern_radius):
      result.append(i)

  return result

def overlapping_matchbook(patterns, pattern, pattern_radius=1):
  """
  The adjacent matches lists for the given pattern on each side.
  """
  return [
    adjacent_match_list(patterns, pattern, s, pattern_radius)
      for s in range(N_SIDES)
  ]

def adjacent_matchbook(patterns, pattern, pattern_radius=1):
  """
  The matches lists for the given pattern on each side.
  """
  return [
    pattern_match_list(patterns, pattern, s, pattern_radius)
      for s in range(N_SIDES)
  ]

def neighbors(xy):
  """
  Returns side, xy neighbor pairs.
  """
  return [
    (NORTH, (xy[0], xy[1]-1)),
    (EAST, (xy[0]+1, xy[1])),
    (SOUTH, (xy[0], xy[1]+1)),
    (WEST, (xy[0]-1, xy[1])),
  ]

def bfs_order(size, start):
  """
  Returns a list of x/y pairs indicating a breadth-first-search iteration order
  over a space of the given size starting at the given coordinates.
  """
  start = tuple(start)
  visited = { start }
  unexplored = size[0] * size[1] - 1
  queue = [start]
  qi = 0
  while unexplored > 0:
    here = queue[qi]
    qi += 1
    for side, nb in neighbors(here):
      if (
        nb[0] >= 0
    and nb[0] < size[0]
    and nb[1] >= 0
    and nb[1] < size[1]
    and nb not in visited
      ):
        unexplored -= 1
        visited.add(nb)
        queue.append(nb)

  return queue

@dep.template_task(
  ("{batch}-image", "{batch}-params"),
  "{batch}-patterns"
)
def all_patterns(_, image, params):
  """
  Finds all patterns in the given image and returns them as a list.
  """
  pattern_radius = params["pattern_radius"]
  result = set()
  width, height = image.shape[:2]

  for x in range(width):
    for y in range(height):
      result.add(pattern_at(image, (x, y), pattern_radius))

  return list(result)

@dep.template_task(("{batch}-patterns",), "{batch}-pmap")
def pmap(_, patterns):
  return {
    pat: i
      for (i, pat) in enumerate(patterns)
  }

@dep.template_task(
  ("{batch}-patterns", "{batch}-pmap", "{batch}-params"),
  "{batch}-adjacent-matchbooks"
)
def adjacent_matchbooks(_, patterns, pmap, params):
  """
  Computes a full set of matchbooks for the given pattern set (along with a
  mapping from patterns to integers used for keys in the matchbooks).
  """
  books = {}
  for p in patterns:
    books[pmap[p]] = adjacent_matchbook(patterns, p, params["pattern_radius"])

  return books

@dep.template_task(
  ("{batch}-patterns", "{batch}-pmap", "{batch}-params"),
  "{batch}-overlapping-matchbooks"
)
def overlapping_matchbooks(_, patterns, pmap, params):
  """
  Like matchbooks but for adjacency matching.
  """
  books = {}
  for p in patterns:
    books[pmap[p]] = overlapping_matchbook(
      patterns,
      p,
      params["pattern_radius"]
    )

  return books
