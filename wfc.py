"""
wfc.py

Wave function collapse: propagates adjacency constraint info to fill in tiles.
"""

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
