# vim: set expandtab:

import taichi as ti
import time

################################################################################
#
#   Data types
#

DataType = ti.f32
DataSize = 4


def setDataType (dataType):

  global DataType
  global DataSize
  
  if ("f32" == dataType):
    DataType = ti.f32
    DataSize = 4
  elif ("f64" == dataType):
    DataType = ti.f64
    DataSize = 8
  else:
    raise Exception (f"Unsupported dataType {dataType}")

  print (f"DataType = \"{dataType}\", DataSize = {DataSize}")


################################################################################
#
#   Data layouts.
#

def setResolution (x, y):

  global nx
  global ny

  nx = x
  ny = y


def defineHelperFields():

  global nodeType
  global vel
  global rho
  global bcVel
  global bcRho

  nodeType = ti.       field (   dtype = ti.u16)
  vel      = ti.Vector.field (2, dtype = DataType)
  rho      = ti.       field (   dtype = DataType)
  bcVel    = ti.Vector.field (2, dtype = DataType)
  bcRho    = ti.       field (   dtype = DataType)

def allocateHelperFields():

  global nodeType
  global vel
  global rho
  global bcVel
  global bcRho

  defineHelperFields()

  ti.root.dense (ti.ij, (nx, ny)).place (nodeType)
  ti.root.dense (ti.ij, (nx, ny)).place (vel)
  ti.root.dense (ti.ij, (nx, ny)).place (rho)
  ti.root.dense (ti.ij, (nx, ny)).place (bcVel)
  ti.root.dense (ti.ij, (nx, ny)).place (bcRho)


layout = "dense"

def setLayout (newLayout):

  if (   "dense"         == newLayout
      or "bitmask_node"  == newLayout
      or "tile"          == newLayout
      or "pointer_tile"  == newLayout
      ):
    global layout
    layout = newLayout
  else:
    raise Exception (f"Unsupported layout \"{newLayout}\"")



# Use these for Vector fields.
#
@ti.func
def getFVec (F, i,j,k):
  val = ti.cast (0.0, DataType)
  if   (0 == k):
    val = F [i,j][0]
  elif (1 == k):
    val = F [i,j][1]
  elif (2 == k):
    val = F [i,j][2]
  elif (3 == k):
    val = F [i,j][3]
  elif (4 == k):
    val = F [i,j][4]
  elif (5 == k):
    val = F [i,j][5]
  elif (6 == k):
    val = F [i,j][6]
  elif (7 == k):
    val = F [i,j][7]
  elif (8 == k):
    val = F [i,j][8]
  return val

@ti.func
def setFVec (F, i,j,k, val):
  if   (0 == k):
    F [i,j][0] = val
  elif (1 == k):
    F [i,j][1] = val
  elif (2 == k):
    F [i,j][2] = val
  elif (3 == k):
    F [i,j][3] = val
  elif (4 == k):
    F [i,j][4] = val
  elif (5 == k):
    F [i,j][5] = val
  elif (6 == k):
    F [i,j][6] = val
  elif (7 == k):
    F [i,j][7] = val
  elif (8 == k):
    F [i,j][8] = val


# Use these for dense fields.
#
@ti.func
def getFScalar (F, i,j,k):
  return F [i,j,k]

@ti.func
def setFScalar (F, i,j,k, val):
  F [i,j,k] = val


def defineVectorFields():

  global f_old
  global f_new
  global getF
  global setF

  f_old = ti.Vector.field (9, dtype = DataType)
  f_new = ti.Vector.field (9, dtype = DataType)

  getF = getFVec
  setF = setFVec



def allocateFDense():

  global f_new
  global f_old

  ti.root.dense (ti.ij, (nx, ny)).place (f_new(0))
  ti.root.dense (ti.ij, (nx, ny)).place (f_new(1))
  ti.root.dense (ti.ij, (nx, ny)).place (f_new(2))
  ti.root.dense (ti.ij, (nx, ny)).place (f_new(3))
  ti.root.dense (ti.ij, (nx, ny)).place (f_new(4))
  ti.root.dense (ti.ij, (nx, ny)).place (f_new(5))
  ti.root.dense (ti.ij, (nx, ny)).place (f_new(6))
  ti.root.dense (ti.ij, (nx, ny)).place (f_new(7))
  ti.root.dense (ti.ij, (nx, ny)).place (f_new(8))
  
  ti.root.dense (ti.ij, (nx, ny)).place (f_old(0))
  ti.root.dense (ti.ij, (nx, ny)).place (f_old(1))
  ti.root.dense (ti.ij, (nx, ny)).place (f_old(2))
  ti.root.dense (ti.ij, (nx, ny)).place (f_old(3))
  ti.root.dense (ti.ij, (nx, ny)).place (f_old(4))
  ti.root.dense (ti.ij, (nx, ny)).place (f_old(5))
  ti.root.dense (ti.ij, (nx, ny)).place (f_old(6))
  ti.root.dense (ti.ij, (nx, ny)).place (f_old(7))
  ti.root.dense (ti.ij, (nx, ny)).place (f_old(8))


def allocateMemoryVectorSeparate():

  defineVectorFields()
  allocateFDense()
  allocateHelperFields()


def allocateMemoryBitmaskedSingleNodeOnlyType():
  
  defineVectorFields()

  global nodeType
  global vel
  global rho
  global bcVel
  global bcRho

  defineHelperFields()

  ti.root.bitmasked (ti.ij, (nx,ny)).place (nodeType)

  ti.root.dense (ti.ij, (nx, ny)).place (vel)
  ti.root.dense (ti.ij, (nx, ny)).place (rho)
  ti.root.dense (ti.ij, (nx, ny)).place (bcVel)
  ti.root.dense (ti.ij, (nx, ny)).place (bcRho)

  allocateFDense()


def allocateMemoryTiled():

  defineVectorFields()
  defineHelperFields()
  
  global nodeType
  global vel
  global rho
  global bcVel
  global bcRho

  ti.root.dense (ti.ij, (nx, ny)).place (vel)
  ti.root.dense (ti.ij, (nx, ny)).place (rho)
  ti.root.dense (ti.ij, (nx, ny)).place (bcVel)
  ti.root.dense (ti.ij, (nx, ny)).place (bcRho)

  tileEdge = 16
  tnx = nx // tileEdge
  tny = ny // tileEdge
  
  if (0 != (nx % tileEdge)):
     tnx += 1
  if (0 != (ny % tileEdge)):
     tny += 1

  tileLayout = ti.root.dense (ti.ij, (tnx, tny))

  tileLayout.dense (ti.ij, (tileEdge,tileEdge)).place (nodeType)

  # Reduced uncoalesced memory transactions, but it seems that 
  # taichi 0.7.26 does not change order of indices.
  #
  tileLayout.dense (ti.ij, (tileEdge, tileEdge)).place (f_new (0))
  tileLayout.dense (ti.ji, (tileEdge, tileEdge)).place (f_new (1))
  tileLayout.dense (ti.ij, (tileEdge, tileEdge)).place (f_new (2))
  tileLayout.dense (ti.ji, (tileEdge, tileEdge)).place (f_new (3))
  tileLayout.dense (ti.ij, (tileEdge, tileEdge)).place (f_new (4))
  tileLayout.dense (ti.ij, (tileEdge, tileEdge)).place (f_new (5))
  tileLayout.dense (ti.ij, (tileEdge, tileEdge)).place (f_new (6))
  tileLayout.dense (ti.ij, (tileEdge, tileEdge)).place (f_new (7))
  tileLayout.dense (ti.ij, (tileEdge, tileEdge)).place (f_new (8))

  tileLayout.dense (ti.ij, (tileEdge, tileEdge)).place (f_old (0))
  tileLayout.dense (ti.ji, (tileEdge, tileEdge)).place (f_old (1))
  tileLayout.dense (ti.ij, (tileEdge, tileEdge)).place (f_old (2))
  tileLayout.dense (ti.ji, (tileEdge, tileEdge)).place (f_old (3))
  tileLayout.dense (ti.ij, (tileEdge, tileEdge)).place (f_old (4))
  tileLayout.dense (ti.ij, (tileEdge, tileEdge)).place (f_old (5))
  tileLayout.dense (ti.ij, (tileEdge, tileEdge)).place (f_old (6))
  tileLayout.dense (ti.ij, (tileEdge, tileEdge)).place (f_old (7))
  tileLayout.dense (ti.ij, (tileEdge, tileEdge)).place (f_old (8)) 


def allocateMemoryPointerTiled():

  defineVectorFields()
  defineHelperFields()
  
  global nodeType
  global vel
  global rho
  global bcVel
  global bcRho

  ti.root.dense (ti.ij, (nx, ny)).place (vel)
  ti.root.dense (ti.ij, (nx, ny)).place (rho)
  ti.root.dense (ti.ij, (nx, ny)).place (bcVel)
  ti.root.dense (ti.ij, (nx, ny)).place (bcRho)

  tileEdge = 16
  tnx = nx // tileEdge
  tny = ny // tileEdge
  
  if (0 != (nx % tileEdge)):
     tnx += 1
  if (0 != (ny % tileEdge)):
     tny += 1

  # One level of pointers.
  tileLayout = ti.root.pointer (ti.ij, (tnx, tny))

  # Two levels of pointers.
  #t2 = 8
  #tnx2 = tnx // t2
  #tny2 = tny // t2
  #if (0 != (tnx % t2)):
  #   tnx2 += 1
  #if (0 != (tny % t2)):
  #   tny2 += 1
  #
  #tileLayout = ti.root.pointer (ti.ij, (tnx2, tny2)).pointer (ti.ij, (t2, t2))

  tileLayout.dense (ti.ij, (tileEdge, tileEdge)).place (nodeType)

  tileLayout.dense (ti.ij, (tileEdge, tileEdge)).place (f_new (0))
  tileLayout.dense (ti.ji, (tileEdge, tileEdge)).place (f_new (1))
  tileLayout.dense (ti.ji, (tileEdge, tileEdge)).place (f_new (2))
  tileLayout.dense (ti.ji, (tileEdge, tileEdge)).place (f_new (3))
  tileLayout.dense (ti.ji, (tileEdge, tileEdge)).place (f_new (4))
  tileLayout.dense (ti.ij, (tileEdge, tileEdge)).place (f_new (5))
  tileLayout.dense (ti.ij, (tileEdge, tileEdge)).place (f_new (6))
  tileLayout.dense (ti.ij, (tileEdge, tileEdge)).place (f_new (7))
  tileLayout.dense (ti.ij, (tileEdge, tileEdge)).place (f_new (8))

  tileLayout.dense (ti.ij, (tileEdge, tileEdge)).place (f_old (0))
  tileLayout.dense (ti.ji, (tileEdge, tileEdge)).place (f_old (1))
  tileLayout.dense (ti.ji, (tileEdge, tileEdge)).place (f_old (2))
  tileLayout.dense (ti.ji, (tileEdge, tileEdge)).place (f_old (3))
  tileLayout.dense (ti.ji, (tileEdge, tileEdge)).place (f_old (4))
  tileLayout.dense (ti.ij, (tileEdge, tileEdge)).place (f_old (5))
  tileLayout.dense (ti.ij, (tileEdge, tileEdge)).place (f_old (6))
  tileLayout.dense (ti.ij, (tileEdge, tileEdge)).place (f_old (7))
  tileLayout.dense (ti.ij, (tileEdge, tileEdge)).place (f_old (8)) 
       


def allocate():

  print (f"layout = \"{layout}\"")
  
  if ("dense" == layout):
    allocateMemoryVectorSeparate()

  elif ("bitmask_node" == layout):
    allocateMemoryBitmaskedSingleNodeOnlyType() 

  elif ("tile" == layout):
    allocateMemoryTiled()

  elif ("pointer_tile" == layout):
    allocateMemoryPointerTiled()

  else:
    raise Exception (f"Unsupported layout \"{layout}\"")


  ti.root.place (nFluidNodesField, nComputationalNodesField, nBounceBackNodesField)


################################################################################
#
#   LBM computations.
#
def setNu (newNu):

  global nu
  global tau
  global omega

  nu = newNu
  tau = 3.0 * nu + 0.5
  omega = 1.0 / tau
  print (f"nu = {nu}, tau = {tau}, omega = {omega}")



SOLID                   =  0
FLUID                   =  1
BOUNDARY_BB             =  2
BOUNDARY_VEL_NEQN_NORTH =  3      
BOUNDARY_VEL_NEQN_SOUTH =  4
BOUNDARY_VEL_NEQN_EAST  =  5
BOUNDARY_VEL_NEQN_WEST  =  6
BOUNDARY_P_WEST         =  7
BOUNDARY_P_EAST         =  8

@ti.pyfunc
def isBoundaryVel (boundaryType):
  result = False
  if ( BOUNDARY_VEL_NEQN_NORTH == boundaryType  or
       BOUNDARY_VEL_NEQN_SOUTH == boundaryType  or
       BOUNDARY_VEL_NEQN_EAST  == boundaryType  or
       BOUNDARY_VEL_NEQN_WEST  == boundaryType ):
    result = True

  return result

@ti.pyfunc
def isBoundaryP (boundaryType):
  result = False
  if ( BOUNDARY_P_EAST  == boundaryType  or
       BOUNDARY_P_WEST  == boundaryType ):
    result = True

  return result



#   
#     6    2   5
#       \  |  /
#        \ | /
#   3 ---- 0 ---- 1
#        / | \
#       /  |  \
#      7   4   8
#
inv_k = [0, 3,4,1,2,7,8,5,6]
ec = [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]] 
wc = [ 4.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 36.0,
       1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0]


NaN = float('nan')


@ti.func
def getNodeType (i, j):
  nType = ti.cast (nodeType [i,j], ti.u32)
  return nType & 0x0F

@ti.func
def setNodeType (i, j, nType):
  tmp = ti.cast (nodeType [i,j], ti.u32)
  nodeType [i,j] = ti.cast ((tmp & 0xF0) | (nType & 0x0F), ti.u16)

@ti.func
def getPackedNeighborMap (i, j):
  nType = ti.cast (nodeType [i,j], ti.u32)
  return nType >> 8
  
@ti.func
def setPackedNeighborMap (i, j, neighborMap):
  tmp = ti.cast (nodeType [i,j], ti.u32)
  nodeType [i,j] = (tmp & 0x0F) | (neighborMap << 8)
  
@ti.func
def hasNeighborInMask (neighborMask, k):
  nghbrBit = 1 << (k - 1)
  bitMask = neighborMask & nghbrBit
  return (0 != bitMask)
  
@ti.func
def hasNeighbor (i, j, k):
  return hasNeighborInMask (ti.cast (getPackedNeighborMap (i,j), ti.u32), k)

@ti.func
def f_eq (vel, rho, e, w):
  eu =  e[0] * vel [0] + e[1] * vel [1]
  uv = vel [0]**2.0 +  vel [1]**2.0
  return w * rho * (1.0 + 3.0 * eu + 4.5 * eu**2 - 1.5 * uv)


def initializeAtEquilibrium (rho0, vx0, vy0):

  @ti.kernel
  def initializeAtEquilibriumKernel (rho0 : DataType, 
                                     vx0  : DataType, 
                                     vy0  : DataType):     
    for i, j in nodeType:

      node = getNodeType (i, j)

      if (SOLID != node):

        vel [i, j][0] =  vx0
        vel [i, j][1] =  vy0

        if (isBoundaryVel (node)):
          vel [i,j] = bcVel [i,j]
         
        rho [i, j] = rho0

        if (isBoundaryP (node)):
          rho [i,j] = bcRho [i,j]
        
        if (SOLID == node):
          for k in ti.static (range (9)):
            setF (f_new, i,j, ti.static (k), NaN)
            setF (f_old, i,j, ti.static (k), NaN)
        else:
          for k in ti.static (range (9)):
            lfeq = f_eq (vel [i,j], rho [i,j], ec[k], wc[k])
            setF (f_old, i,j, ti.static (k), lfeq)
            if (BOUNDARY_BB == node):
              setF (f_new, i,j, ti.static (k), lfeq) 
            else:
              setF (f_new, i,j, ti.static (k), NaN) 


  initializeAtEquilibriumKernel (rho0, vx0, vy0)


@ti.func
def rhoV (f ,i, j):
  lrho  = ti.cast (0.0, DataType)
  lvel0 = ti.cast (0.0, DataType)
  lvel1 = ti.cast (0.0, DataType)
  for k in ti.static(range(9)):
    lf = getF (f, i,j,k)
    lrho += lf
    lvel0 += (ti.cast (ec [k][0], DataType) * lf)
    lvel1 += (ti.cast (ec [k][1], DataType) * lf)
  lvel0 /= lrho 
  lvel1 /= lrho 
  return lrho, lvel0, lvel1

def computeRhoV (f):
  
  @ti.kernel
  def computeRhoVKernel():
    
    for i, j in nodeType:
      nType = getNodeType (i,j)
      if (isBoundaryVel (nType)):
        rho [i,j] = NaN
        vel [i,j][0] = bcVel [i,j][0]
        vel [i,j][1] = bcVel [i,j][1]
      else:
        rho [i,j], vel[i,j][0], vel[i,j][1] = rhoV (f, i, j)

  computeRhoVKernel()


@ti.func
def gatherFull (f, i,j,k, ei,ej, enable, nodeType):
  
  # When no neighbor, load current value of PDF - used for bounce-back nodes
  ni = i
  nj = j

  if (True == enable):

    ni = i - ei
    nj = j - ej

  elif (BOUNDARY_BB == nodeType):
  
    # Gather missing PDFs for bounce-back nodes. Needed, because after processing
    # of boundary nodes ALL PDF are stored as fA. Without the below code, fA
    # values for bounce-back nodes were uncorrectly set to 0.

    ni = i
    nj = j

  return getF (f, ni, nj, k)


@ti.func
def streamCollideFunc (fA, fB, i, j, omega):

  # Overlaps with load of bType - all indices for f0 do not change.
  # However, may generate unnecessary loads for solid nodes.
  f0 = ti.cast (0.0, DataType)
  f0 = getF (fB, i,j,0)

  localF = ti.cast (0.0, DataType)
  nType        = ti.cast (getNodeType          (i,j), ti.u32)
  neighborMask = ti.cast (getPackedNeighborMap (i,j), ti.u32)

  lrho  = ti.cast (0.0, DataType)
  lvel0 = ti.cast (0.0, DataType)
  lvel1 = ti.cast (0.0, DataType)
  lvel  = [lvel0, lvel1]

  lrho = f0

  f1 = ti.cast (0.0, DataType)
  f2 = ti.cast (0.0, DataType)
  f3 = ti.cast (0.0, DataType)
  f4 = ti.cast (0.0, DataType)
  f5 = ti.cast (0.0, DataType)
  f6 = ti.cast (0.0, DataType)
  f7 = ti.cast (0.0, DataType)
  f8 = ti.cast (0.0, DataType)

  if (   FLUID       == nType
      or BOUNDARY_BB == nType
      or isBoundaryVel (nType)
      or isBoundaryP   (nType)
      ):
   
    # Compute velocity in-fly. For bounce-back and velocity boundary this is not needed,
    # but does not hurt.
    k = ti.static (1)
    f = gatherFull (fB, i,j,k, ec [k][0], ec [k][1], hasNeighborInMask (neighborMask, k), nType)
    lvel0 += (ti.cast (ec [k][0], DataType) * f)
    lvel1 += (ti.cast (ec [k][1], DataType) * f)
    lrho  += f
    f1 = f

    k = ti.static (2)
    f = gatherFull (fB, i,j,k, ec [k][0], ec [k][1], hasNeighborInMask (neighborMask, k), nType)
    lvel0 += (ti.cast (ec [k][0], DataType) * f)
    lvel1 += (ti.cast (ec [k][1], DataType) * f)
    lrho  += f
    f2 = f
    
    k = ti.static (3)
    f = gatherFull (fB, i,j,k, ec [k][0], ec [k][1], hasNeighborInMask (neighborMask, k), nType)
    lvel0 += (ti.cast (ec [k][0], DataType) * f)
    lvel1 += (ti.cast (ec [k][1], DataType) * f)
    lrho  += f
    f3 = f
    
    k = ti.static (4)
    f = gatherFull (fB, i,j,k, ec [k][0], ec [k][1], hasNeighborInMask (neighborMask, k), nType)
    lvel0 += (ti.cast (ec [k][0], DataType) * f)
    lvel1 += (ti.cast (ec [k][1], DataType) * f)
    lrho  += f
    f4 = f
    
    k = ti.static (5)
    f = gatherFull (fB, i,j,k, ec [k][0], ec [k][1], hasNeighborInMask (neighborMask, k), nType)
    lvel0 += (ti.cast (ec [k][0], DataType) * f)
    lvel1 += (ti.cast (ec [k][1], DataType) * f)
    lrho  += f
    f5 = f
    
    k = ti.static (6)
    f = gatherFull (fB, i,j,k, ec [k][0], ec [k][1], hasNeighborInMask (neighborMask, k), nType)
    lvel0 += (ti.cast (ec [k][0], DataType) * f)
    lvel1 += (ti.cast (ec [k][1], DataType) * f)
    lrho  += f
    f6 = f
    
    k = ti.static (7)
    f = gatherFull (fB, i,j,k, ec [k][0], ec [k][1], hasNeighborInMask (neighborMask, k), nType)
    lvel0 += (ti.cast (ec [k][0], DataType) * f)
    lvel1 += (ti.cast (ec [k][1], DataType) * f)
    lrho  += f
    f7 = f
    
    k = ti.static (8)
    f = gatherFull (fB, i,j,k, ec [k][0], ec [k][1], hasNeighborInMask (neighborMask, k), nType)
    lvel0 += (ti.cast (ec [k][0], DataType) * f)
    lvel1 += (ti.cast (ec [k][1], DataType) * f)
    lrho  += f
    f8 = f


  if (FLUID == nType):

    lvel [0] = lvel0 / lrho
    lvel [1] = lvel1 / lrho

  elif (isBoundaryVel (nType)):

    lvel [0] = bcVel [i,j][0]
    lvel [1] = bcVel [i,j][1]
    ux = lvel [0]
    uy = lvel [1]

    if (BOUNDARY_VEL_NEQN_NORTH == nType):

      # Load missing values.
      # TODO: Use better way of handling corner nodes.
      k = ti.static (1)
      if (not hasNeighborInMask (neighborMask, k)):
        f1 = getF (fB, i - ec[k][0], j - ec[k][1], k)
      k = ti.static (2)
      if (not hasNeighborInMask (neighborMask, k)):
        f2 = getF (fB, i - ec[k][0], j - ec[k][1], k)
      k = ti.static (3)
      if (not hasNeighborInMask (neighborMask, k)):
        f3 = getF (fB, i - ec[k][0], j - ec[k][1], k)
      k = ti.static (5)
      if (not hasNeighborInMask (neighborMask, k)):
        f5 = getF (fB, i - ec[k][0], j - ec[k][1], k)
      k = ti.static (6)
      if (not hasNeighborInMask (neighborMask, k)):
        f6 = getF (fB, i - ec[k][0], j - ec[k][1], k)

      lrho = (f0 + f1 + f3 + 2.0 * (f2 + f5 + f6)) / (1 + uy)

      rho_y = lrho * uy * (1.0 / 6.0)
      rho_x = lrho * ux * 0.5
      f3_f1 = (f3 - f1) * 0.5
      term  = rho_x + f3_f1
      f4 = f2 - 4.0 * rho_y
      f7 = f5 - term - rho_y
      f8 = f6 + term - rho_y

    elif (BOUNDARY_VEL_NEQN_SOUTH == nType):
    
      # Load missing values.
      # TODO: Use better way of handling corner nodes.
      k = ti.static (1)
      if (not hasNeighborInMask (neighborMask, k)):
        f1 = getF (fB, i - ec[k][0], j - ec[k][1], k)
      k = ti.static (3)
      if (not hasNeighborInMask (neighborMask, k)):
        f3 = getF (fB, i - ec[k][0], j - ec[k][1], k)
      k = ti.static (4)
      if (not hasNeighborInMask (neighborMask, k)):
        f4 = getF (fB, i - ec[k][0], j - ec[k][1], k)
      k = ti.static (7)
      if (not hasNeighborInMask (neighborMask, k)):
        f7 = getF (fB, i - ec[k][0], j - ec[k][1], k)
      k = ti.static (8)
      if (not hasNeighborInMask (neighborMask, k)):
        f8 = getF (fB, i - ec[k][0], j - ec[k][1], k)
  
      lrho = (f0 + f1 + f3 + 2.0 * (f4 + f7 + f8)) / (1 - uy)
      
      f2 = f4 + lrho * uy * 2.0 / 3.0
      f5 = f7 - 0.5 * (f1 - f3) + 0.5 * lrho * ux + lrho * uy / 6.0
      f6 = f8 + 0.5 * (f1 - f3) - 0.5 * lrho * ux + lrho * uy / 6.0

    elif (BOUNDARY_VEL_NEQN_WEST == nType):
        
      k = ti.static (1)
      if (not hasNeighborInMask (neighborMask, k)):
        f1 = getF (fB, i - ec[k][0], j - ec[k][1], k)
      k = ti.static (2)
      if (not hasNeighborInMask (neighborMask, k)):
        f2 = getF (fB, i - ec[k][0], j - ec[k][1], k)
      k = ti.static (4)
      if (not hasNeighborInMask (neighborMask, k)):
        f4 = getF (fB, i - ec[k][0], j - ec[k][1], k)
      k = ti.static (5)
      if (not hasNeighborInMask (neighborMask, k)):
        f5 = getF (fB, i - ec[k][0], j - ec[k][1], k)
      k = ti.static (8)
      if (not hasNeighborInMask (neighborMask, k)):
        f8 = getF (fB, i - ec[k][0], j - ec[k][1], k)
  
      lrho = (f0 + f2 + f4 + 2.0 * (f3 + f6 + f7)) / (1 - ux)

      f1 = f3 + lrho * ux * (2.0 / 3.0)
      f5 = f7 - 0.5 * (f2 - f4) + 0.5 * lrho * uy + lrho * ux * (1.9 / 6.0)
      f8 = f6 + 0.5 * (f2 - f4) - 0.5 * lrho * uy + lrho * ux * (1.9 / 6.0)

  elif (isBoundaryP (nType)):

    lrho = bcRho [i,j]

    if (BOUNDARY_P_WEST == nType):

      if (not hasNeighborInMask (neighborMask, 4)): 
        
        # Bottom corner

        ux = 0
        uy = 0
        lvel [0] = ux
        lvel [1] = uy

        f1 = f3
        f2 = f4
        f5 = f7
        
        f6 = 0.5 * (lrho - (f0 + f1 + f2 + f3 + f4 + f5 + f7))
        f8 = f6

      elif (not hasNeighborInMask (neighborMask, 2)): 
        
        # Top corner

        ux = 0
        uy = 0
        lvel [0] = ux
        lvel [1] = uy

        f1 = f3
        f4 = f2
        f8 = f6
        
        f5 = 0.5 * (lrho - (f0 + 2.0 * f2 + 2.0 * f3 + 2.0 * f6))
        f7 = f5

      else:

        lvel [1] = 0
        uy = lvel [1]

        ux = 1 - (f0 + f2 + f4 + 2.0 * (f3 + f6 + f7)) / lrho
        lvel [0] = ux

        f1 = f3 + lrho * ux * (2.0 / 3.0)
        f5 = f7 - 0.5 * (f2 - f4) + lrho * ux * (1.0 / 6.0)
        f8 = f6 + 0.5 * (f2 - f4) + lrho * ux * (1.0 / 6.0)

    elif (BOUNDARY_P_EAST == nType):

      if (not hasNeighborInMask (neighborMask, 4)): 
        
        # Bottom corner

        ux = 0
        uy = 0
        lvel [0] = ux
        lvel [1] = uy
        
        f3 = f1
        f2 = f4
        f6 = f8

        f5 = 0.5 * (lrho - f0) - f1 - f4 - f8
        f7 = f5

      elif (not hasNeighborInMask (neighborMask, 2)): 
        
        # Top corner

        ux = 0
        uy = 0
        lvel [0] = ux
        lvel [1] = uy
        
        f3 = f1
        f4 = f2
        f7 = f5

        f6 = 0.5 * (lrho - f0) - f1 - f2 - f5
        f8 = f6

      else:

        lvel [1] = 0
        uy = lvel [1]

        ux = (f0 + f2 + f4 + 2.0 * (f1 + f5 + f8)) / lrho - 1.0
        lvel [0] = ux
        
        f3 = f1 - lrho * ux * (2.0 / 3.0)
        f6 = f8 - 0.5 * (f2 - f4) - lrho * ux * (1.0 / 6.0)
        f7 = f5 + 0.5 * (f2 - f4) - lrho * ux * (1.0 / 6.0)

  #
  # Collide.
  #

  f0_new = ti.cast (0.0, DataType)
  f1_new = ti.cast (0.0, DataType)
  f2_new = ti.cast (0.0, DataType)
  f3_new = ti.cast (0.0, DataType)
  f4_new = ti.cast (0.0, DataType)
  f5_new = ti.cast (0.0, DataType)
  f6_new = ti.cast (0.0, DataType)
  f7_new = ti.cast (0.0, DataType)
  f8_new = ti.cast (0.0, DataType)

  if (   FLUID == nType
      or isBoundaryVel (nType)
      or isBoundaryP   (nType)
      ):

    omega_1 = 1.0 - omega
    rho_9   = lrho * (1.0 / 9.0)
    uv      = lvel [0]**2 + lvel [1]**2
    uv_3_2  = 1.5 * uv
    uv_term = 1 - uv_3_2

    local_feq = 4.0 * rho_9 * uv_term
    new_f = omega_1 * f0 + omega * local_feq
    f0_new = new_f

    
    eu = lvel [0]
    eu_term = 3.0 * eu
    eu2_term = 4.5 * eu * eu
    term_1 = uv_term + eu2_term

    local_feq = rho_9 * (term_1 + eu_term)
    new_f = omega_1 * f1 + omega * local_feq
    f1_new = new_f

    local_feq = rho_9 * (term_1 - eu_term)
    new_f = omega_1 * f3 + omega * local_feq
    f3_new = new_f


    eu = lvel [1]
    eu_term = 3.0 * eu 
    eu2_term = 4.5 * eu * eu
    term_1 = uv_term + eu2_term

    local_feq = rho_9 * (term_1 + eu_term)
    new_f = omega_1 * f2 + omega * local_feq
    f2_new = new_f

    local_feq = rho_9 * (term_1 - eu_term)
    new_f = omega_1 * f4 + omega * local_feq
    f4_new = new_f


    rho_36   = lrho * (1.0 / 36.0)

    eu = lvel [0] + lvel [1]
    eu_term = 3.0 * eu
    eu2_term = 4.5 * eu * eu
    term_1 = uv_term + eu2_term
    
    local_feq = rho_36 * (term_1 + eu_term)
    new_f = omega_1 * f5 + omega * local_feq
    f5_new = new_f

    local_feq = rho_36 * (term_1 - eu_term)
    new_f = omega_1 * f7 + omega * local_feq
    f7_new = new_f


    eu = lvel [1] - lvel [0]
    eu_term = 3.0 * eu
    eu2_term = 4.5 * eu * eu
    term_1 = uv_term + eu2_term

    local_feq = rho_36 * (term_1 + eu_term)
    new_f = omega_1 * f6 + omega * local_feq
    f6_new = new_f

    local_feq = rho_36 * (term_1 - eu_term)
    new_f = omega_1 * f8 + omega * local_feq
    f8_new = new_f

  # Do not collide for bounce-back nodes.
  elif (BOUNDARY_BB == nType):
    
    f0_new = f0
    f1_new = f3
    f2_new = f4
    f3_new = f1
    f4_new = f2
    f5_new = f7
    f6_new = f8
    f7_new = f5
    f8_new = f6


  if (   FLUID       == nType
      or BOUNDARY_BB == nType
      or isBoundaryVel (nType)
      or isBoundaryP   (nType)
      ):
  
    setF (fA, i,j, 0, f0_new)
    setF (fA, i,j, 1, f1_new)
    setF (fA, i,j, 2, f2_new)
    setF (fA, i,j, 3, f3_new)
    setF (fA, i,j, 4, f4_new)
    setF (fA, i,j, 5, f5_new)
    setF (fA, i,j, 6, f6_new)
    setF (fA, i,j, 7, f7_new)
    setF (fA, i,j, 8, f8_new)

 
@ti.kernel
def streamCollideA (omega: DataType): 

  fA, fB = ti.static (f_new, f_old)

  ti.block_dim (512) 
  for i, j in nodeType:
    streamCollideFunc (fA, fB, i, j, omega)


@ti.kernel
def streamCollideB (omega: DataType):

  fA, fB = ti.static (f_old, f_new)

  ti.block_dim (512)
  for i, j in nodeType:
    streamCollideFunc (fA, fB, i, j, omega)


def run (numberOfTimeSteps):

  nIterations = numberOfTimeSteps // 2

  for i in range (nIterations):
    streamCollideA (omega)
    streamCollideB (omega)

  return 2 * nIterations


################################################################################
#
#   Geometry.
#

@ti.kernel
def initNeighborMap():

  for i, j in nodeType:
    hasAllNeighbors = True
    neighborBitMask = 0
    
    for k in ti.static (range (1,9)):
      ni = i - ec [k][0]
      nj = j - ec [k][1]
      if (ni < 0  or  nj < 0):
        hasAllNeighbors = False
      elif (ni >= nx  or  nj >= ny):
        hasAllNeighbors = False
      elif (SOLID == getNodeType (ni,nj)):
        hasAllNeighbors = False
      else:
        neighborBitMask |= (1 << (k - 1))
    
    setPackedNeighborMap (i,j, neighborBitMask)

    if (False == hasAllNeighbors  and FLUID == getNodeType (i,j)):
      setNodeType (i,j, BOUNDARY_BB)


nFluidNodesField         = ti.field (dtype = ti.u64)
nComputationalNodesField = ti.field (dtype = ti.u64)
nBounceBackNodesField    = ti.field (dtype = ti.u64)


@ti.kernel
def computeGeometryStatisticsKernel():

  for i,j in nodeType:
    nType = getNodeType (i,j)

    if (SOLID != nType):
      one = ti.cast (1, ti.u64)

      nComputationalNodesField [None] += one

      if (FLUID       == nType): nFluidNodesField      [None] += one
      if (BOUNDARY_BB == nType): nBounceBackNodesField [None] += one


def computeGeometryStatistics():

  global nNodes
  global nFluidNodes
  global nComputationalNodes
  global nBounceBackNodes

  nFluidNodesField         [None] = 0
  nComputationalNodesField [None] = 0
  nBounceBackNodesField    [None] = 0

  computeGeometryStatisticsKernel()

  nNodes = nx * ny
  nFluidNodes         = nFluidNodesField         [None]
  nComputationalNodes = nComputationalNodesField [None]
  nBounceBackNodes    = nBounceBackNodesField    [None]


def printGeometryStatistics():
  
  global nNodes
  global nFluidNodes
  global nComputationalNodes
  global nBounceBackNodes

  nBoundaryNodes = nComputationalNodes - nFluidNodes
  porosity       = 1.0 * nComputationalNodes / nNodes

  print (f"nNodes              = {nNodes}")
  print (f"nComputationalNodes = {nComputationalNodes}" 
         f" {nComputationalNodes / nNodes} of all")
  print (f"nBoundaryNodes      = {nBoundaryNodes}" 
         f" {nBoundaryNodes / nComputationalNodes} of computational")
  print (f"nBounceBackNodes    = {nBounceBackNodes}" 
         f" {nBounceBackNodes / nBoundaryNodes} of boundary" 
         f" {nBounceBackNodes / nComputationalNodes} of computational")
  print (f"porosity = {porosity}  solidity = {1 - porosity}")



def initializeGeometry():

  initNeighborMap()
  computeGeometryStatistics()
  printGeometryStatistics()

################################################################################
#
#   File I/O.
#
import numpy as np
import evtk.hl


def saveRhoV (fileName):

  velnp = vel.to_numpy()
  rhonp = rho.to_numpy()

  np.savez_compressed (fileName, vel = velnp, rho = rhonp)


def saveState (fileName, f):

  velnp = vel.to_numpy()
  rhonp = rho.to_numpy()
  fnp = f.to_numpy()

  np.savez_compressed (fileName, vel = velnp, rho = rhonp, f = fnp)


def loadState (fileName):
  
  state = np.load (fileName)

  f_old.from_numpy (state ["f"  ])
  vel  .from_numpy (state ["vel"])
  rho  .from_numpy (state ["rho"])  


# Based on 
# https://github.com/paulo-herrera/PyEVTK/blob/master/evtk/examples/structured.py
def saveVtk (fileName):

  velnp = vel.to_numpy()
  vel_mag =  (velnp[:, :, 0]**2.0 + velnp[:, :, 1]**2.0)**0.5
  rhonp = rho.to_numpy()

  # Coordinates
  X = np.arange (0, nx, 1, dtype='float32')
  Y = np.arange (0, ny, 1, dtype='float32')

  x = np.zeros ((nx, ny, 1))
  y = np.zeros ((nx, ny, 1))
  z = np.zeros ((nx, ny, 1))

  for j in range (ny):
      for i in range (nx):
          x [i,j,0] = X [i]
          y [i,j,0] = Y [j]
          z [i,j,0] = 0

  evtk.hl.structuredToVTK (fileName, x, y, z, pointData = {
    "v_mag" : vel_mag.flatten (order = 'F'), 
    "vx"    : velnp [:,:, 0].flatten (order = 'F'), 
    "vy"    : velnp [:,:, 1].flatten (order = 'F'), 
    "rho"   : rhonp.flatten (order = 'F')
    })
  
################################################################################
#
#   Performance.
#
def minBandwidthPerNode():
  return 2 * 9 * DataSize


def computePerformance (nTimeSteps, duration):

  latticeUpdates = nTimeSteps * nComputationalNodes
  MLUPS = (latticeUpdates / 1.0e6) / duration
  bandwidthGBs = latticeUpdates * minBandwidthPerNode() / (duration * 1.0e9)
  return MLUPS, bandwidthGBs


def showPerformance (nTimeSteps, duration, text = ""):

  MLUPS, bandwidthGBs = computePerformance (nTimeSteps, duration)

  print (f"{text}nTimeSteps = {nTimeSteps} duration = {duration} s "  \
         f"MLUPS = {MLUPS} bandwidth = {bandwidthGBs} GB/s")


def measurePerformance (nSteps, text):

  ti.sync()

  begin = time.perf_counter()
  nComputedTimeSteps = run (nSteps)
  ti.sync()
  end = time.perf_counter()

  duration = end - begin
  
  showPerformance (nComputedTimeSteps, duration, text)


def simulate (numberOfTimeSteps, saveInterval, shouldSave = True):

  i = 0
  while i < numberOfTimeSteps:

    ti.sync()

    begin = time.perf_counter()
    nComputedSteps = run (numberOfTimeSteps = saveInterval)
    ti.sync()
    end = time.perf_counter()

    i += nComputedSteps
    text = f"{i} "
    duration = end - begin
    showPerformance (nTimeSteps = nComputedSteps, duration = duration, 
                     text = text)

    if shouldSave:
      computeRhoV (f_old)
      saveRhoV (f"rho_v_{i}.npz")
      #saveVtk (f"out_{i}")
      #saveState (fileName = f"state_{i}.npz", f = f_old)


