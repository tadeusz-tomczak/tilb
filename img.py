# vim: set expandtab:

import taichi as ti
import numpy  as np
import png

import tilb


img = ti.Vector.field (3, dtype = ti.u8)


def initialize():
  ti.root.dense (ti.ij, (tilb.nx, tilb.ny)).place (img)

@ti.kernel
def buildImageNodeTypeKernel():
  
  scale = 255
   
  for i,j in img:
    img [i,j] = [1.0 * scale, 0.6 * scale, 0.8 * scale]

  for i,j in tilb.nodeType:
    bt = ti.cast (tilb.getNodeType (i,j), ti.u32)
    if (tilb.SOLID == bt):
      img [i,j] = [1 * scale, 1 * scale, 1 * scale]
    elif (tilb.FLUID == bt):
      img [i,j] = [0.5 * scale, 0.5 * scale, 0.5 * scale]
    elif (tilb.BOUNDARY_BB == bt):
      img [i,j] = [1 * scale,1 * scale, 0 * scale]
    elif (tilb.BOUNDARY_VEL_NEQN_NORTH == bt):
      img [i,j] = [0 * scale, 0 * scale, 1 * scale]
    elif (tilb.BOUNDARY_VEL_NEQN_SOUTH == bt):
      img [i,j] = [1 * scale, 0 * scale, 1 * scale]
    elif (tilb.BOUNDARY_VEL_NEQN_EAST == bt):
      img [i,j] = [0 * scale, 1 * scale, 1 * scale]
    elif (tilb.BOUNDARY_VEL_NEQN_WEST == bt):
      img [i,j] = [1 * scale, 0 * scale, 0 * scale]
    else:
      img [i,j] = [0,0,0]


def saveNodeTypes():
  bc_np = np.rot90 (tilb.nodeType.to_numpy(), k = 1, axes = (0,1))
  np.savetxt ("nodeTypes.txt", bc_np, fmt="%4x")
  buildImageNodeTypeKernel()
  ti.imwrite (img, "nodeTypes.png")
  
