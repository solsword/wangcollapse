#!/usr/bin/env python
"""
run.py

Executable script that calls functions from the various modules based on
command-line arguments.
"""

import os
import sys

from skimage.io import imread
from skimage.io import imsave
from skimage.io import imshow

from quiche import dep

import pattern
import tilefill

# Common targets

@dep.template_task((), "{size}-size")
def determine_size(match):
  return tuple(int(x) for x in match.group(1).split("×"))

@dep.template_task(("{batch}-params",), "{batch}-image")
def load_image(_, params):
  fn = params["filename"]
  im = imread(fn)
  im = im[:,:,:pattern.N_CHANNELS]
  return im

def fill_tiles(fn):
  size = (24, 24)
  tn = tilefill.target_name(fn, 1, 3, "overlapping", (64, 64))
  #print(dep.recursive_target_report(tn))
  #print(dep.find_target_report("samples/caves.png×1×3-rules"))
  ts, result = dep.create(tn)
  nfn = os.path.splitext(fn)[0] + ".synth.png"
  imsave(nfn, result)

def collapse_wave_functions(fn):
  size = (48, 48)
  pass

if __name__ == "__main__":
  for fn in sys.argv[1:]:
    fill_tiles(fn)
