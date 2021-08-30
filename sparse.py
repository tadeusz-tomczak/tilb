# vim: set expandtab:

import taichi as ti
import numpy  as np
import math
import random

import tilb
import shapes
import img


def setBBifFluid (i, j):
  nodeType = shapes.getNodeType (i,j)
  if (tilb.FLUID == nodeType):
    shapes.setNodeType (i, j, tilb.BOUNDARY_BB)

def setWalls():

  nx = tilb.nx
  ny = tilb.ny
  
  for i in range (nx):
    setBBifFluid (i, 0)
    setBBifFluid (i, ny - 1)

  for j in range (ny):
    i = 0
    nodeType = shapes.getNodeType (i,j)
    if (tilb.FLUID == nodeType):
      shapes.setNodeType (i, j, tilb.BOUNDARY_P_WEST)
      tilb.bcVel [i, j] = [tilb.NaN, 0.0]
      tilb.bcRho [i, j] = 1.016

  for j in range (ny):
    i = nx - 1
    nodeType = shapes.getNodeType (i,j)
    if (tilb.FLUID == nodeType):
      shapes.setNodeType (i, j, tilb.BOUNDARY_P_EAST)
      tilb.bcVel [i, j] = [tilb.NaN, 0.0]
      tilb.bcRho [i, j] = 1.0

    

def buildRegular (width, height, shape, radius, distance):

  if ("circle" == shape):
    placeFunc = shapes.circle
    
  print (f"sparse geometry regular {width} x {height}, "
         f"shape = {shape}, radius = {radius}, distance = {distance}")

  # WARNING - Allocates all memory.
  shapes.setAllNodes (tilb.FLUID)

  def column (x):
    for y in np.arange (0, height + 2*radius, distance):
      placeFunc (x, y, radius)

  for x in np.arange (0, width + 2*radius, distance):
    column (x)

  setWalls()

  tilb.initializeGeometry()
  tilb.setNu (0.1)  
  tilb.initializeAtEquilibrium (rho0 = 1.0, vx0 = 0.0, vy0 = 0.0)


def buildRandom (width, height, shape, porosity):

  if ("circle" == shape):
    placeFunc = shapes.circle

  print (f"sparse geometry random {width} x {height}, "
         f"shape = {shape}, porosity = {porosity}")

  # WARNING - Allocates all memory.
  shapes.setAllNodes (tilb.FLUID)

  nNodes = width * height
  nSolidNodes = nNodes * (1 - porosity)

  random.seed (1)

  minR = 8
  maxR = 256


  nEstimatedSolid = 0 
  
  while True:
    radii = []
    while True:
    
      r = random.uniform (minR, maxR)
      radii.append (r) 
    
      nEstimatedSolid += math.pi * r * r
      if nEstimatedSolid >= nSolidNodes:
        break
    
    for r in radii:
      x = random.uniform (0, width)
      y = random.uniform (0, height)
      placeFunc (x, y, r)
    
    tilb.computeGeometryStatistics()
    p = tilb.nComputationalNodes / tilb.nNodes
    
    if (p <= porosity):
      break
    nEstimatedSolid = tilb.nNodes - tilb.nComputationalNodes


  setWalls()
  
  tilb.initializeGeometry()
  tilb.setNu (0.1)  
  tilb.initializeAtEquilibrium (rho0 = 1.0, vx0 = 0.0, vy0 = 0.0)



def circleRadius (porosity, distance):

  r = (distance * distance * (1 - porosity) / math.pi)**0.5
  return r


def simulate (N, porosity):

  numberOfTimeSteps = 1000000
  saveInterval      = 100000

  shape    = "circle"
  distance = N / 8.0
  r = circleRadius (porosity, distance)

  buildRegular (width    = N,
                height   = N,
                shape    = "circle",
                radius   = r,
                distance = distance
                )

  #buildRandom (width    = N,
  #             height   = N,
  #             shape    = shape,
  #             porosity = porosity
  #             )

  #text = f"PERFORMANCE layout \"{tilb.layout}\" sparse {shape} random porosity {porosity} geometry {N} x {N}: "
  #tilb.run (100)  # warm-up
  #tilb.measurePerformance (nSteps = 1000, text = text)

  tilb.simulate (numberOfTimeSteps = numberOfTimeSteps, 
                 saveInterval = saveInterval, 
                 shouldSave = True
                 )

  tilb.computeRhoV (tilb.f_old)
  tilb.saveVtk ("out")
  tilb.saveState (fileName = "state.npz", f = tilb.f_old)



if ("__main__" == __name__):

  ti.init (arch = ti.gpu                         , 
                  kernel_profiler        = True  ,
                  use_unified_memory     = False ,
                  device_memory_fraction = 0.90  ,
                  #async_mode             = True  ,
                  excepthook             = True
                  )

  dataType = "f32"
  layout   = "dense"
  N        = 4096

  tilb.setDataType (dataType)
  tilb.setLayout   (layout)
  tilb.setResolution (x = N, y = N)
  tilb.allocate()
  shapes.initialize()
  img.initialize()

  porosities = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
  for porosity in porosities:
    print (f"porosity = {porosity}")
    simulate (N, porosity)
  
  #img.initialize()
  #img.saveNodeTypes()

  ti.kernel_profiler_print()

