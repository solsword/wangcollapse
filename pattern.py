"""
pattern.py

Common functions for managing/extracting patterns from images.
"""

import random

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

def center_of(pattern):
  """
  Returns the central vector of the given pattern as a tuple.
  """
  pw = int((len(pattern) / N_CHANNELS)**0.5)
  pr = pw//2
  pci = (pw*pr + pr) * N_CHANNELS # pattern center index

  return pattern[pci:pci+N_CHANNELS]

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

def iterpairs(size):
  """
  Creates a list of all (x, y) pairs over the given (2D) matrix size, and
  returns a shuffled version.
  """
  pairs = [ (x, y) for x in range(size[0]) for y in range(size[1]) ]
  random.shuffle(pairs)
  return pairs

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
  (),
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
    print("extracting patterns: {}/{}".format(x, width), end="\r")
    for y in range(height):
      pat = pattern_at(image, (x, y), pattern_radius)
      result.add(pat)
      if "add_rotations" in params and params["add_rotations"]:
        pat = rotate_pattern(pat)
        result.add(pat)
        pat = rotate_pattern(pat)
        result.add(pat)
        pat = rotate_pattern(pat)
        result.add(pat)

  print("--done extracting patterns {w}/{w}--".format(w=width))

  return list(result)

def rotate_pattern(pattern):
  """
  Rotates the given pattern 90 degrees clockwise.
  """
  pw = int((len(pattern)//N_CHANNELS)**0.5)
  pr = pw // 2
  result = []
  for x in range(pw):
    for y in range(pw):
      for ch in range(N_CHANNELS):
        result.append(pattern[((pw - y - 1)*pw + x) * N_CHANNELS + ch])

  return tuple(result)

@dep.template_task(("{batch}-patterns",), (), "{batch}-pmap")
def pmap(_, patterns):
  return {
    pat: i
      for (i, pat) in enumerate(patterns)
  }

@dep.template_task(
  ("{batch}-patterns", "{batch}-pmap", "{batch}-params"),
  (),
  "{batch}-adjacent-matchbooks"
)
def adjacent_matchbooks(_, patterns, pmap, params):
  """
  Computes a full set of matchbooks for the given pattern set (along with a
  mapping from patterns to integers used for keys in the matchbooks).
  """
  books = {}
  for i, p in enumerate(patterns):
    print("building matchbooks: {}/{}".format(i, len(patterns)), end="\r")
    books[pmap[p]] = adjacent_matchbook(patterns, p, params["pattern_radius"])

  print("--done building matchbooks: {n}/{n}--".format(n=len(patterns)))
  return books

@dep.template_task(
  ("{batch}-patterns", "{batch}-pmap", "{batch}-params"),
  (),
  "{batch}-overlapping-matchbooks"
)
def overlapping_matchbooks(_, patterns, pmap, params):
  """
  Like matchbooks but for adjacency matching.
  """
  books = {}
  for i, p in enumerate(patterns):
    print("building matchbooks: {}/{}".format(i, len(patterns)), end="\r")
    books[pmap[p]] = overlapping_matchbook(
      patterns,
      p,
      params["pattern_radius"]
    )

  print("--done building matchbooks: {n}/{n}--".format(n=len(patterns)))
  return books

@dep.template_task(("{batch}-matchbooks",), (), "{batch}-compatibilities")
def compatibility_matrices(_, matchbooks):
  """
  Converts a matchbooks dictionary into a pair of vertical and horizontal
  compatibility matrices.
  """
  npat = len(matchbooks)
  vert = np.zeros([npat, npat], dtype=np.bool_)
  horiz = np.zeros([npat, npat], dtype=np.bool_)
  for pi in range(npat):
    mb = matchbooks[pi]
    for ni in range(npat):
      if ni in mb[NORTH]:
        vert[ni,pi] = 1 # north first
      if ni in mb[EAST]:
        horiz[ni,pi] = 1 # east first
      # mb[SOUTH] and mb[WEST] are symmetric

  return (vert, horiz)

def blank_probabilities(n_patterns, size):
  """
  Returns a blank probabilities array of the given (2D) size.
  """
  return np.ones(list(size) + [n_patterns]) / n_patterns

def blank_possibiliies(n_patterns, size):
  """
  Returns a blank possibilities array of the given (2D) size.
  """
  return np.ones(list(size) + [n_patterns], dtype=np.bool_)

def vector_entropy(v):
  """
  Entropy of a 1-dimensional vector of probabilities.
  """
  return -sum(((e * math.log2(e)) if e > 0 else 0) for e in v)

def entropy_at(probabilities, xy):
  """
  Returns entropy at the given position.
  """
  return vector_entropy(probabilities[xy[0],xy[1],:])

def pick_probability(probabilities, xy):
  """
  Randomly picks a pattern according to the probabilities at the given
  position.
  """
  v = random.uniform(0, 1)
  for i, e in enumerate(probabilities[xy[0],xy[1],:]):
    v -= e
    if v <= 0:
      return i
  return probabilities.shape[2]-1

def pick_possibility(possibilities, xy):
  """
  Randomly picks a pattern according to the possibilities at the given
  position, picking completely at random if no possibilities remain.
  """
  npat = possibilities.shape[2]
  start = random.randint(0, npat-1)
  x = xy[0]
  y = xy[1]
  allowed = possibilities[x, y, :]
  pcount = sum(allowed)
  if pcount == 0:
    return random.randint(0, npat-1)
  chosen = random.randint(0, pcount-1)
  for i in range(npat):
    if allowed[i]:
      if chosen == 0:
        return i
      chosen -= 1

  # shouldn't be possible
  raise RuntimeError("Failed to pick a possibility from:\n{}".format(allowed))

@dep.template_task(
  ("{batch}-{mode}-probabilities-{size}", "{batch}-patterns"),
  (),
  "{batch}-{mode}-synth-{size}"
)
def create_image(name_match, probabilities, patterns):
  if name_match.groups(2) == "adjacent":
    return create_adjacent_image(probabilities, patterns)
  else:
    return create_overlapping_image(probabilities, patterns)

@dep.template_task(
  ("{batch}-{mode}-probability-sequence-{size}", "{batch}-patterns"),
  (),
  "{batch}-{mode}-seqsynth-{size}"
)
def create_image_sequence(name_match, pseq, patterns):
  result = []
  for i, frame in enumerate(pseq):
    print("frame {}/{}".format(i, len(pseq)))
    if name_match.groups(2) == "adjacent":
      result.append(create_adjacent_image(frame, patterns))
    else:
      result.append(create_overlapping_image(frame, patterns))

  print("--done creating frames {l}/{l}--".format(l=len(pseq)))

  return result

def create_adjacent_image(probabilities, patterns):
  """
  Creates an image from a pattern probability matrix, using patterns from the
  given ruleset.

  This version puts tiles adjacent to each other.
  """
  pw = int((len(patterns[0]) / N_CHANNELS)**0.5)
  pr = pw//2
  shape = [
    probabilities.shape[0] * pw,
    probabilities.shape[1] * pw
  ] + [N_CHANNELS]
  result = np.zeros(shape, dtype=int)
  for x in range(probabilities.shape[0]):
    for y in range(probabilities.shape[1]):
      colors = np.zeros([pw, pw, N_CHANNELS])
      tw = 0
      for i, pat in enumerate(patterns):
        w = probabilities[x, y, i]
        if w > 0:
          colors += pattern_tensor(pat, pr) * w
          tw += w

      colors /= tw
      result[x*pw:(x+1)*pw, y*pw:(y+1)*pw, :] = colors

  return result

def create_overlapping_image(probabilities, patterns):
  """
  Another version of create_adjacent_image.

  This version overlaps tiles, effectively using only the central pixel of each
  tile.
  """
  nc = N_CHANNELS

  pw = int((len(patterns[0]) / nc)**0.5)
  pr = pw//2
  pci = (pw*pr + pr) * nc # pattern center index

  shape = [probabilities.shape[0], probabilities.shape[1]] + [nc]
  result = np.zeros(shape, dtype=int)
  for x in range(probabilities.shape[0]):
    print("assembling image: {}/{}".format(x, probabilities.shape[0]), end="\r")
    for y in range(probabilities.shape[1]):
      color = np.zeros([N_CHANNELS])
      tw = 0
      for i, pat in enumerate(patterns):
        w = probabilities[x, y, i]
        if w > 0:
          color += np.array(pat[pci:pci+nc]) * w
          tw += w

      color /= tw

      result[x, y, :] = color

  print("--done assembling image {w}/{w}--".format(w=probabilities.shape[0]))

  return result
