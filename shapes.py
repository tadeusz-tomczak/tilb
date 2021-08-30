# vim: set expandtab:

import taichi as ti
import math

import tilb


def initialize():
  
  nx = tilb.nx
  ny = tilb.ny

  global circleKernel
  global squareKernel

  DT = ti.f64

  @ti.kernel
  def circleKernel (x0 : DT, y0 : DT, R : DT):

    for i,j in ti.ndrange (nx, ny):
      
      if (R > 0):
        x = ti.cast (i, DT)
        y = ti.cast (j, DT)

        r2 = (x - x0)**2 + (y - y0)**2

        if (r2 <= R*R):
          tilb.setNodeType (i, j, tilb.SOLID)

  @ti.kernel
  def squareKernel (x0 : DT, y0 : DT, halfA : DT):

    for i,j in ti.ndrange (nx, ny):
      
      if (halfA > 0):
        x = ti.cast (i, DT)
        y = ti.cast (j, DT)

        if ( abs (x - x0) <= halfA  and  abs (y - y0) <= halfA):
          tilb.setNodeType (i, j, tilb.SOLID)


def setAllNodes (nodeType):

  nx = tilb.nx
  ny = tilb.ny

  DT = tilb.DataType

  @ti.kernel
  def setAllNodesKernel (nodeType : ti.u16):

    for i,j in ti.ndrange (nx, ny):

        tilb.setNodeType (i, j, nodeType)

  setAllNodesKernel (nodeType)
  

def setNodeType (i, j, nodeType):
  nx = tilb.nx
  ny = tilb.ny

  if ( i < nx  and  i >= 0  and  j < ny  and  j >= 0):
    tilb.nodeType [i, j] = nodeType


def getNodeType (i, j):
  nx = tilb.nx
  ny = tilb.ny

  if ( i < nx  and  i >= 0  and  j < ny  and  j >= 0):
    return tilb.nodeType [i, j] & 0x0F
  else:
    return -1


def circle (centerX, centerY, r):
  
  circleKernel (centerX, centerY, r)

def square (centerX, centerY, halfA):
  
  squareKernel (centerX, centerY, halfA)


