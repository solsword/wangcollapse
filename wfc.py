"""
wfc.py

Wave function collapse: propagates adjacency constraint info to fill in tiles.
"""

import random

from quiche import dep

import numpy as np

import pattern

def batchname(image, pattern_radius=1):
  """
  Converts batch parameters to a string.
  """
  return image + "×" + str(pattern_radius)

def batch_params(batchname):
  """
  Inverse of batchname.
  """
  add_rotations = False
  if batchname.endswith(":R"):
    add_rotations = True
    batchname = batchname[:-2]
  fields = batchname.split("×")
  fn = fields[0]
  pr = int(fields[1])
  return {
    "filename": fn,
    "pattern_radius": pr,
    "add_rotations": add_rotations,
  }

@dep.template_task((), "{batch}-wfc-params")
def params(match):
  """
  Task that unpacks params according to batch name.
  """
  return batch_params(match.group(1))

@dep.template_task(
  (
    "{batch}-wfc-{mode}-compatibilities",
    "{size}-size"
  ),
  "{batch}-wfc-{mode}-probabilities-{size}"
)
def synthesize(_, compatibility_tables, size):
  """
  Synthesizes a collapsed possibility table of the given size according to the
  given ruleset.
  """
  return collapse(
    pattern.blank_possibiliies(compatibility_tables[0].shape[0], size),
    compatibility_tables
  )

@dep.template_task(
  (
    "{batch}-wfc-{mode}-compatibilities",
    "{size}-size"
  ),
  "{batch}-wfc-{mode}-probability-sequence-{size}"
)
def synth_sequence(_, compatibility_tables, size):
  """
  Synthesizes a collapsed possibility table of the given size according to the
  given ruleset.
  """
  return collapse_sequence(
    pattern.blank_possibiliies(compatibility_tables[0].shape[0], size),
    compatibility_tables
  )

def collapse(possibilities, compatibility_tables, targets=None):
  """
  Uses constraint propagation and lowest-entropy-first choices to collapse the
  given possibilities field. If targets are given, at least those points are
  collapsed (others my be via constraint propagation) and once they're
  collapsed, the rest of the field is left alone.
  """
  width, height = possibilities.shape[:2]

  if targets == None:
    # pick a random order for index traversal
    targets = pattern.iterpairs(possibilities.shape[:2])

  # compute possibility count of each entry
  psums = np.sum(possibilities, axis=2)

  # fill all remaining entries
  i = 0
  to_collapse = len(targets)
  while targets:
    print("collapse: {}/{}".format(i, to_collapse), end="\r")
    i += 1

    # Find the minimum-entropy cell:
    min_index = None
    min_posibilities = None
    for xy in targets:
      npos = psums[xy[0], xy[1]]
      if min_posibilities == None or npos < min_posibilities:
        min_posibilities = npos
        min_index = xy

    # Chose and apply a pattern:
    chosen = pattern.pick_possibility(possibilities, min_index)
    collapse_point(
      possibilities,
      psums,
      targets,
      compatibility_tables,
      min_index,
      chosen
    )

  print("--collapse finished {}/{}--".format(to_collapse, to_collapse))

  return possibilities

def collapse_sequence(possibilities, compatibility_tables, targets=None):
  """
  Like collapse, but stores and returns the full sequence of intermediate
  possibility fields during the collapse process.
  """
  width, height = possibilities.shape[:2]

  if targets == None:
    # pick a random order for index traversal
    targets = pattern.iterpairs(possibilities.shape[:2])

  # compute possibility count of each entry
  psums = np.sum(possibilities, axis=2)

  # fill all remaining entries
  i = 0
  to_collapse = len(targets)
  sequence = [np.copy(possibilities)]
  while targets:
    print("collapse: {}/{}".format(i, to_collapse), end="\r")
    i += 1

    # Find the minimum-entropy cell:
    min_index = None
    min_posibilities = None
    for xy in targets:
      npos = psums[xy[0], xy[1]]
      if min_posibilities == None or npos < min_posibilities:
        min_posibilities = npos
        min_index = xy

    # Chose and apply a pattern:
    chosen = pattern.pick_possibility(possibilities, min_index)
    collapse_point(
      possibilities,
      psums,
      targets,
      compatibility_tables,
      min_index,
      chosen
    )
    sequence.append(np.copy(possibilities))

  print("--collapse finished {}/{}--".format(to_collapse, to_collapse))

  return sequence

def collapse_point(
  possibilities,
  psums,
  indices,
  compatibility_tables,
  xy,
  chosen
):
  """
  Takes the entry at the given xy position and collapses it, putting the chosen
  pattern there and propagating adjacency constraints throughout the
  possibility field, updating entropies and removing chain collapsed points
  from the indices list as it goes.
  """
  try:
    indices.remove(xy)
  except:
    pass
  npat = possibilities.shape[2]
  x = xy[0]
  y = xy[1]
  possibilities[x, y, :] = 0
  possibilities[x, y, chosen] = 1
  psums[x, y] = 1
  update_all_possibilities(
    possibilities,
    psums,
    indices,
    compatibility_tables,
    xy
  )

def allowed_values(possibilities, compatibility_tables, xy, heading):
  """
  Computes the vector of allowed patterns in the given direction from the given
  position based on the possibilities at the given position.
  """
  here = possibilities[xy[0], xy[1], :]
  if heading == pattern.NORTH:
    return compatibility_tables[0] @ here
  elif heading == pattern.EAST:
    return compatibility_tables[1] @ here
  elif heading == pattern.SOUTH:
    return compatibility_tables[0].T @ here
  elif heading == pattern.WEST:
    return compatibility_tables[1].T @ here

def update_all_possibilities(
  possibilities,
  psums,
  indices,
  compatibility_tables,
  start_at
):
  """
  Iteratively updates all possibilities in the given space, updating psums as
  well, and removing collapsed positions from the given indices list. The given
  position should have just been changed.
  """
  width = possibilities.shape[0]
  height = possibilities.shape[1]
  changed = [ start_at ]
  while changed:
    xy = changed.pop()
    if psums[xy] == 0:
      # don't propagate
      continue
    for heading, nb in pattern.neighbors(xy):
      nbx = nb[0]
      nby = nb[1]

      # respect map boundaries
      if nbx < 0 or nbx >= width or nby < 0 or nby >= height:
        continue

      # ignore already-collapsed and inconsistent spaces
      if psums[nbx, nby] <= 1:
        continue

      # TODO: Cache these directly?
      allowed = allowed_values(possibilities, compatibility_tables, xy, heading)

      old = possibilities[nbx, nby, :]

      new = old & allowed

      if not np.array_equal(old, new):
        # update and recurse
        possibilities[nbx, nby, :] = new
        nsum = np.sum(new)
        psums[nbx, nby] = nsum
        if nsum <= 1:
          try:
            indices.remove(nb)
          except:
            pass
        if nsum >= 1:
          changed.append(nb)
        # don't propagate from inconsistent spaces
      # else do nothing

def target_name(input_filename, pattern_radius, mode, size):
  """
  Returns the target name for the given job. Mode should be one of
  "overlapping" or "adjacent."
  """
  return "{fn}×{pr}:R-wfc-{mode}-synth-{size[0]}×{size[1]}".format(
    fn=input_filename,
    pr=pattern_radius,
    mode=mode,
    size=size
  )

def seq_target_name(input_filename, pattern_radius, mode, size):
  """
  As above, but for the sequence of possibilities during collapse.
  """
  return "{fn}×{pr}:R-wfc-{mode}-seqsynth-{size[0]}×{size[1]}".format(
    fn=input_filename,
    pr=pattern_radius,
    mode=mode,
    size=size
  )
