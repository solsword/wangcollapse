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

from moviepy.editor import VideoClip

from quiche import dep

import pattern
import tilefill
import wfc

FRAMERATE = 24

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
  tn = tilefill.target_name(fn, 1, 2, "overlapping", size)
  #print(dep.recursive_target_report(tn))
  #print(dep.find_target_report("samples/caves.png×1×3-rules"))
  ts, result = dep.create(tn)
  nfn = os.path.splitext(fn)[0] + ".filled.png"
  imsave(nfn, result)

def collapse_wave_functions(fn):
  size = (48, 48)
  tn = wfc.target_name(fn, 1, "overlapping", size)
  ts, result = dep.create(tn)
  nfn = os.path.splitext(fn)[0] + ".collapsed.png"
  imsave(nfn, result)

def animate_wave_function_collapse(fn, seconds_per_state=0.5):
  size = (32, 32)
  tn = wfc.seq_target_name(fn, 1, "overlapping", size)
  ts, result = dep.create(tn)
  full_duration = len(result) * seconds_per_state
  def makeframe(t):
    state_index = int(t / seconds_per_state)
    return result[min(state_index, len(result)-1)]

  anim = VideoClip(makeframe, duration=full_duration)
  nfn = os.path.splitext(fn)[0] + ".collapse.mp4"
  anim.write_videofile(nfn, fps=FRAMERATE)
  nfn = os.path.splitext(fn)[0] + ".collapsed.png"
  imsave(nfn, result[-1])

if __name__ == "__main__":
  for fn in sys.argv[1:]:
    #fill_tiles(fn)
    collapse_wave_functions(fn)
    #animate_wave_function_collapse(fn)
