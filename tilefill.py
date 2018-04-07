#!/usr/bin/env python

"""
wfc.py

Wang Function Collapse:

  Wave Function Collapse meets Wang tiling to provide locally-accessible values
  that mimic a small sample.
"""

import sys
import os
import math
import random

from quiche import dep

import numpy as np

from skimage.io import imread
from skimage.io import imsave
from skimage.io import imshow

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

def pattern_matches(patterns, pattern, side):
  """
  Computes the set of matching patterns from the given set on the given side of
  the given pattern.
  """
  opp = opposite_side(side)
  result = []
  edge = edge_of(pattern, side)
  for p in patterns:
    if edge_of(p, opp) == edge:
      result.append(p)

  return result

def matchbook(patterns, pattern):
  """
  The matches lists for the given pattern on each side.
  """
  return [pattern_matches(patterns, pattern, s) for s in range(N_SIDES)]

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

@dep.template_task(("{batch}-patterns", "{batch}-pmap"), "{batch}-matchbooks")
def matchbooks(_, patterns, pmap):
  """
  Computes a full set of matchbooks for the given pattern set (along with a
  mapping from patterns to integers used for keys in the matchbooks).
  """
  books = {}
  for p in patterns:
    books[pmap[p]] = matchbook(patterns, p)

  return books

def influence_of(patterns, pmap, matchbooks, pattern, radius=3):
  """
  The influence regime of the given pattern is a (2r+1)×(2r+1)×n_patterns
  tensor that indicates the probability of each possible pattern within the
  given radius, given edge matching rules.

  TODO: As is, this is permissive for non-adjacent tiles. Do proper constraint
  propagation or not?
  """
  w = radius*2+1
  # base distribution is even:
  result = np.ones([w, w, len(patterns)]) / len(patterns)
  # middle tile is collapsed:
  result[w//2, w//2, :] = 0
  result[w//2, w//2, pmap[pattern]] = 1

  fixed = set()
  for xy in bfs_order((w, w), (w//2, w//2)):
    fixed.add(xy)
    entry = result[xy[0], xy[1],:]
    for pi in range(len(patterns)):
      prb = entry[pi]
      if prb > 0: # this pattern is possible here
        mb = matchbooks[pi]
        for side, nb in neighbors(xy): # each neighbor
          if ( # that's in the map & not already fixed
            nb[0] >= 0
        and nb[0] < w
        and nb[1] >= 0
        and nb[1] < w
        and nb not in fixed
          ):
            nbpats = mb[side] # possible neighboring patterns
            for pt in nbpats:
              # TODO: Probability math is hard T_T
              result[nb[0], nb[1], pmap[pt]] += prb / len(nbpats)

  # normalize probabilities in each bin
  for x in range(result.shape[0]):
    for y in range(result.shape[1]):
      result[x,y,:] /= sum(result[x,y,:])

  return result

def batchname(image, pattern_radius=1, influence_radius=3):
  """
  Converts batch parameters to a string.
  """
  return "×".join(str(x) for x in [image, pattern_radius, influence_radius])

def batch_params(batchname):
  """
  Inverse of batchname.
  """
  fields = batchname.split("×")
  fn = fields[0]
  pr = int(fields[1])
  ir = int(fields[2])
  return (fn, pr, ir)

@dep.template_task((), "{batch}-params")
def params(match):
  """
  Task that unpacks params according to batch name.
  """
  return batch_params(match.group(1))

@dep.template_task(("{batch}-params",), "{batch}-image")
def load_image(_, params):
  fn = params[0]
  im = imread(fn)
  im = im[:,:,:N_CHANNELS]
  return im

@dep.template_task(("{batch}-image", "{batch}-params"), "{batch}-patterns")
def all_patterns(_, image, params):
  """
  Finds all patterns in the given image and returns them as a list.
  """
  pattern_radius = params[1]
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
  (
    "{batch}-image",
    "{batch}-patterns",
    "{batch}-pmap",
    "{batch}-matchbooks",
    "{batch}-params"
  ),
  "{batch}-rules"
)
def extract_rules(_, image, patterns, pmap, matchbooks, params):
  """
  Extracts an returns a ruleset for the given image, dictating the possible
  patterns and how each pattern affects surrounding probabilities.
  """
  pattern_radius = params[1]
  influence_radius = params[2]
  rules = []
  for i in range(len(patterns)):
    if i % 20 == 0:
      print("building ruleset: {}/{}".format(i, len(patterns)), end="\r")
    rules.append(
      influence_of(
        patterns,
        pmap,
        matchbooks,
        patterns[i],
        radius=influence_radius
      )
    )

  return {
    "patterns": patterns,
    "pmap": pmap,
    "books": matchbooks,
    "rules": rules,
    "pattern_radius": pattern_radius,
    "influence_radius": influence_radius,
  }

def blank_probabilities(patterns, size):
  """
  Returns a blank probabilities array of the given (2D) size.
  """
  return np.ones(list(size) + [len(patterns)]) / len(patterns)

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

def pick_pattern(probabilities, xy):
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

def apply_influence(probabilities, influence, xy):
  """
  Applies the given influence matrix to the given probabilities centered at the
  given position.
  """
  ps = probabilities.shape
  ir = influence.shape[0]//2
  for dx in range(-ir, ir+1):
    for dy in range(-ir, ir+1):
      ox = xy[0] + dx
      oy = xy[1] + dy

      if ox < 0 or ox >= ps[0] or oy < 0 or oy >= ps[1]:
        # we're off the map; ignore this position
        continue

      # lookup and combine influence vectors
      ipr = influence[ir + dx, ir + dy, :]
      npr = probabilities[ox, oy, :]

      # update probability
      result = np.multiply(ipr, npr)
      if sum(result) == 0:
        # inconsistency: reset to completely even probability
        # TODO: raise + backtrack instead?
        probabilities[ox, oy, :] = 1/len(npr)
      else:
        result /= sum(result)
        probabilities[ox, oy, :] = result

def iterpairs(size):
  """
  Creates a list of all (x, y) pairs over the given (2D) matrix size, and
  returns a shuffled version.
  """
  pairs = [ (x, y) for x in range(size[0]) for y in range(size[1]) ]
  random.shuffle(pairs)
  return pairs

@dep.template_task((), "{size}-size")
def determine_size(match):
  return tuple(int(x) for x in match.group(1).split("×"))

@dep.template_task(("{batch}-rules", "{size}-size"), "{batch}-collapsed-{size}")
def synthsize(_, ruleset, size):
  """
  Synthesizes a probability table of the given size according to the given
  ruleset.
  """
  return collapse(blank_probabilities(ruleset["patterns"], size), ruleset)

def collapse(probabilities, ruleset):
  """
  Collapses remaining undecided probability distributions in the given
  probabilities matrix according to the given ruleset, targeting min-entropy
  distributions first and breaking ties randomly.
  """
  width, height = probabilities.shape[:2]

  # pick a random order for index traversal
  myorder = iterpairs(probabilities.shape[:2])

  # remove already-fixed entries from our iteration list of collapse targets
  fixed = []
  entropies = np.zeros(probabilities.shape[:2])
  for xy in myorder:
    ent = entropy_at(probabilities, xy)
    entropies[xy[0], xy[1]] = ent
    if ent == 0:
      fixed.append(xy)

  for xy in fixed:
    myorder.remove(xy)

  # collapse all remaining entries
  to_collapse = len(myorder)
  for i in range(to_collapse):
    if i % 20 == 0:
      print("collapse: {}/{}".format(i, to_collapse), end="\r")

    # Find the minimum-entropy cell:
    min_index = None
    min_entropy = None
    for xy in myorder:
      ent = entropies[xy[0], xy[1]]
      if min_entropy == None or ent < min_entropy:
        min_entropy = ent
        min_index = xy

    # Chose and apply a pattern:
    chosen = pick_pattern(probabilities, min_index)
    apply_influence(
      probabilities,
      ruleset["rules"][chosen],
      min_index
    )

    # Update affected entropies:
    er = ruleset["influence_radius"]
    for dx in range(-er, er+1):
      for dy in range(-er, er+1):
        ox = min_index[0] + dx
        oy = min_index[1] + dy
        if ox >= 0 and ox < width and oy >= 0 and oy < height:
          entropies[ox, oy] = entropy_at(probabilities, (ox, oy))

    # remove collapsed index from consideration
    myorder.remove(min_index)

  return probabilities

@dep.template_task(
  ("{batch}-collapsed-{size}","{batch}-rules"),
  "{batch}-synth-{size}"
)
def create_image(_, probabilities, ruleset):
  """
  Creates an image from a pattern probability matrix, using patterns from the
  given ruleset.
  """
  patterns = ruleset["patterns"]
  pr = ruleset["pattern_radius"]
  pw = pr*2+1 # pattern width
  chosen = np.argmax(probabilities, axis=2)
  shape = [chosen.shape[0] * pw, chosen.shape[1] * pw] + [N_CHANNELS]
  result = np.zeros(shape, dtype=int)
  for x in range(chosen.shape[0]):
    for y in range(chosen.shape[1]):
      pat = patterns[chosen[x,y]]
      for dx in range(pw):
        for dy in range(pw):
          pidx = (dx + dy*pw) * N_CHANNELS
          result[x*pw + dx, y*pw + dy, :] = pat[pidx:pidx+N_CHANNELS]

  return result

def target_name(input_filename, pattern_radius, influence_radius, size):
  """
  Returns the target name for the given job.
  """
  return "{fn}×{pr}×{ir}-synth-{size[0]}×{size[1]}".format(
    fn=input_filename,
    pr=pattern_radius,
    ir=influence_radius,
    size=size
  )

if __name__ == "__main__":
  for fn in sys.argv[1:]:
    size = (64, 64)
    tn = target_name(fn, 1, 3, (64, 64))
    #print(dep.recursive_target_report(tn))
    #print(dep.find_target_report("samples/caves.png×1×3-rules"))
    ts, result = dep.create(tn)
    nfn = os.path.splitext(fn)[0] + ".synth.png"
    imsave(nfn, result)
