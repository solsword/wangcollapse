"""
tilefill.py

Tile fill algorithm using limited information propagation and
tile-border-matching.
"""

import math
import random

from quiche import dep

import numpy as np

import pattern

def influence_of(patterns, pmap, matchbooks, pat, radius=3):
  """
  The influence regime of the given pattern is a (2r+1)×(2r+1)×n_patterns
  tensor that indicates the probability of each possible pattern within the
  given radius, given matching rules.

  TODO: As is, this is permissive for non-adjacent tiles. Do proper constraint
  propagation or not?
  """
  w = radius*2+1
  # base distribution is even:
  result = np.ones([w, w, len(patterns)]) / len(patterns)
  # middle tile is filled:
  result[w//2, w//2, :] = 0
  result[w//2, w//2, pmap[pat]] = 1

  fixed = set()
  for xy in pattern.bfs_order((w, w), (w//2, w//2)):
    fixed.add(xy)
    entry = result[xy[0], xy[1],:]
    for pi in range(len(patterns)):
      prb = entry[pi]
      if prb > 0: # this pattern is possible here
        mb = matchbooks[pi]
        for side, nb in pattern.neighbors(xy): # each neighbor
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
              result[nb[0], nb[1], pt] += prb / len(nbpats)

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
  return { "filename": fn, "pattern_radius": pr, "influence_radius": ir }

@dep.template_task((), "{batch}-tile-params")
def params(match):
  """
  Task that unpacks params according to batch name.
  """
  return batch_params(match.group(1))

@dep.template_task(
  (
    "{batch}-tile-patterns",
    "{batch}-tile-pmap",
    "{batch}-tile-{mode}-matchbooks",
    "{batch}-tile-params"
  ),
  "{batch}-tile-{mode}-rules"
)
def extract_rules(_, patterns, pmap, matchbooks, params):
  """
  Extracts an returns a ruleset for the given pattern set, dictating the
  possible patterns and how each pattern affects surrounding probabilities.
  """
  pattern_radius = params["pattern_radius"]
  influence_radius = params["influence_radius"]
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
  print("--ruleset done {}/{}--      ".format(len(patterns), len(patterns)))

  return {
    "patterns": patterns,
    "pmap": pmap,
    "books": matchbooks,
    "rules": rules,
    "pattern_radius": pattern_radius,
    "influence_radius": influence_radius,
  }

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

@dep.template_task(
  ("{batch}-tile-{mode}-rules", "{size}-size"),
  "{batch}-tile-{mode}-probabilities-{size}"
)
def synthesize(_, ruleset, size):
  """
  Synthesizes a probability table of the given size according to the given
  ruleset.
  """
  return fill(
    pattern.blank_probabilities(len(ruleset["patterns"]), size),
    ruleset
  )

def fill(probabilities, ruleset, prefill=8):
  """
  Collapses remaining undecided probability distributions in the given
  probabilities matrix according to the given ruleset, targeting min-entropy
  distributions first and breaking ties randomly.

  prefill sites are collapsed before relying on entropy to pick sites in order
  to give the result more interesting structure.
  """
  width, height = probabilities.shape[:2]

  # pick a random order for index traversal
  myorder = pattern.iterpairs(probabilities.shape[:2])

  # remove already-fixed entries from our iteration list of fill targets
  fixed = []
  entropies = np.zeros(probabilities.shape[:2])
  for xy in myorder:
    ent = pattern.entropy_at(probabilities, xy)
    entropies[xy[0], xy[1]] = ent
    if ent == 0:
      fixed.append(xy)

  for xy in fixed:
    myorder.remove(xy)

  # fill all remaining entries
  to_fill = len(myorder)
  for i in range(to_fill):
    if i % 20 == 0:
      print("fill: {}/{}".format(i, to_fill), end="\r")

    # Just use
    if i < prefill:
      min_index = myorder[i]
    else:
      # Find the minimum-entropy cell:
      min_index = None
      min_entropy = None
      for xy in myorder:
        ent = entropies[xy[0], xy[1]]
        if min_entropy == None or ent < min_entropy:
          min_entropy = ent
          min_index = xy

    # Chose and apply a pattern:
    chosen = pattern.pick_probability(probabilities, min_index)
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
          entropies[ox, oy] = pattern.entropy_at(probabilities, (ox, oy))

    # remove filled index from consideration
    myorder.remove(min_index)

  print("--fill finished {}/{}--".format(to_fill, to_fill))

  return probabilities

def target_name(input_filename, pattern_radius, influence_radius, mode, size):
  """
  Returns the target name for the given job. Mode should be one of
  "overlapping" or "adjacent."
  """
  return "{fn}×{pr}×{ir}-tile-{mode}-synth-{size[0]}×{size[1]}".format(
    fn=input_filename,
    pr=pattern_radius,
    ir=influence_radius,
    mode=mode,
    size=size
  )
