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

import numpy as np

from skimage.io import imread
from skimage.io import imsave
from skimage.io import imshow

N_CHANNELS = 3

def pattern_at(image, xy, pattern_radius=1):
  """
  Extracts the pattern at the given position in the given image. Returns None
  for positions outside the image.
  """
  width, height = image.shape[:2]

  if xy[0] < 0 or xy[0] >= width or xy[1] < 0 or xy[1] >= height:
    return None

  pattern = np.zeros([pattern_radius*2+1, pattern_radius*2+1, N_CHANNELS])

  for dx in range(-pattern_radius, pattern_radius+1):
    for dy in range(-pattern_radius, pattern_radius+1):
      ox = xy[0] + dx
      oy = xy[1] + dy

      if ox < 0 or ox >= width or oy < 0 or oy >= height:
        px = (-1,)*N_CHANNELS
      else:
        px = tuple(image[ox,oy])

      pattern[dx + pattern_radius, dy + pattern_radius, :] = px

  return tuple(pattern.flatten())

def influence_at(image, pattern_map, xy, pattern_radius, influence_radius):
  """
  Computes an influence tensor for the given point in the given image, using
  the given pattern map to map patterns to pattern indices.
  """
  pc = len(pattern_map)
  influence = np.zeros(
    [
      influence_radius*2+1,
      influence_radius*2+1,
      pc
    ]
  )
  for dx in range(-influence_radius, influence_radius+1):
    for dy in range(-influence_radius, influence_radius+1):
      ox = xy[0] + dx
      oy = xy[1] + dy

      pat = pattern_at(image, (ox, oy), pattern_radius)
      if pat == None:
        # set all probabilities to 1: multiplication will have no effect
        influence[influence_radius + dx, influence_radius + dy, :] = 1/pc
      else:
        pidx = pattern_map[pat]
        influence[influence_radius + dx, influence_radius + dy, :] = 0
        influence[influence_radius + dx, influence_radius + dy, pidx] = 1

  return influence

def disjoin_influence(i1, i2, w1=1, w2=2):
  """
  Combines influence values using disjunction and the given weights.
  """
  return (i1 * w1 + i2 * w2) / (w1 + w2)

def conjunct_influence(i1, i2):
  """
  Combines influence values using conjunction.
  """
  result = np.multiply(i1, i2)
  divisors = np.amax(result, [0, 1])
  # TODO: HERE?

def all_patterns(image, pattern_radius=1):
  """
  Finds all patterns in the given image and returns them as a list.
  """
  result = set()
  width, height = image.shape[:2]
  for x in range(width):
    for y in range(height):
      result.add(pattern_at(image, (x, y), pattern_radius))

  return list(result)

def extract_rules(image, pattern_radius=1, effect_radius=3):
  """
  Extracts an returns a ruleset for the given image, dictating the possible
  patterns and how each pattern affects surrounding probabilities.
  """
  patterns = all_patterns(image, pattern_radius)
  pmap = {
    pat: i
      for (i, pat) in enumerate(patterns)
  }
  rules = {
    i:{}
      for i in range(len(patterns))
  }
  width, height = image.shape[:2]
  ic = 0
  for x in range(width):
    print("extract: {}/{}".format(x * height, width * height), end="\r")
    for y in range(height):
      pat = pattern_at(image, (x, y), pattern_radius)

      entry = rules[pmap[pat]]

      if "influence" in entry:
        entry["influence"] = disjoin_influence(
          entry["influence"],
          influence_at(image, pmap, (x, y), pattern_radius, effect_radius),
          entry["count"],
          1
        )
        entry["count"] += 1
      else:
        entry["influence"] = influence_at(
          image,
          pmap,
          (x, y),
          pattern_radius,
          effect_radius
        )
        entry["count"] = 1

  return {
    "patterns": patterns,
    "pmap": pmap,
    "rules": rules,
    "pattern_radius": pattern_radius,
    "effect_radius": effect_radius,
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

      rad = (dx*dx + dy*dy)**0.5
      inf = 1 - rad/ir

      if inf <= 0: # outside radius-of-influence
        continue

      inf = inf**0.25 # strongly quadratic influence

      # lookup and combine influence vectors
      ipr = influence[ir + dx, ir + dy, :]
      npr = probabilities[ox, oy, :]

      # update probability
      if rad == 0: # fully overwrite central spot
        probabilities[ox, oy, :] = inf
      else: # blend probabilities
        # TODO: something more clever here?
        probabilities[ox, oy, :] = ipr * inf + npr * (1 - inf)

def iterpairs(size):
  """
  Creates a list of all (x, y) pairs over the given (2D) matrix size, and
  returns a shuffled version.
  """
  pairs = [ (x, y) for x in range(size[0]) for y in range(size[1]) ]
  random.shuffle(pairs)
  return pairs

def synthsize(ruleset, size):
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
    # TODO: This could be sped up by only recomputing influenced entropies
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
      ruleset["rules"][chosen]["influence"],
      min_index
    )

    # Update affected entropies:
    er = ruleset["effect_radius"]
    for dx in range(-er, er+1):
      for dy in range(-er, er+1):
        ox = min_index[0] + dx
        oy = min_index[1] + dy
        if ox >= 0 and ox < width and oy >= 0 and oy < height:
          entropies[ox, oy] = entropy_at(probabilities, (ox, oy))

    # remove collapsed index from consideration
    myorder.remove(min_index)

  return probabilities

def create_image(probabilities, ruleset):
  """
  Creates an image from a pattern probability matrix, using patterns from the
  given ruleset.
  """
  patterns = ruleset["patterns"]
  pc = len(patterns[0]) // 2
  pcs = pc - N_CHANNELS//2
  pce = pc + N_CHANNELS//2 + 1
  chosen = np.argmax(probabilities, axis=2)
  result = np.zeros(list(chosen.shape[:2]) + [N_CHANNELS], dtype=int)
  for x in range(chosen.shape[0]):
    for y in range(chosen.shape[1]):
      result[x,y,:] = patterns[chosen[x,y]][pcs:pce]

  return result

if __name__ == "__main__":
  for fn in sys.argv[1:]:
    im = imread(fn)
    im = im[:,:,:N_CHANNELS]
    rs = extract_rules(im)
    syn = synthsize(rs, (64, 64))
    ex = create_image(syn, rs)
    nfn = os.path.splitext(fn)[0] + ".synth.png"
    imsave(nfn, ex)
