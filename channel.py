# vim: set expandtab

import taichi as ti
import numpy  as np

import tilb
import img


def setSize (h):

  ny = h
  nx = 8 * h  

  tilb.setResolution (x = nx, y = ny)
  tilb.allocate()


def buildVelocity (h, vx0, nu):

  setSize (h)

  nx = tilb.nx
  ny = tilb.ny

  @ti.kernel
  def buildVelocityKernel (vx0 : tilb.DataType):

    for i,j in ti.ndrange (nx, ny):
      tilb.setNodeType (i,j, tilb.FLUID)
    
    # Top
    for i in ti.ndrange (nx):
      j = ny - 1
      tilb.setNodeType (i, j, tilb.BOUNDARY_BB)

    # Bottom
    for i in ti.ndrange (nx):
      j = 0
      tilb.setNodeType (i, j, tilb.BOUNDARY_BB)

    # Left
    for j in range (1, ny-1):
      i = 0
      tilb.setNodeType (i, j, tilb.BOUNDARY_VEL_NEQN_WEST)
      tilb.bcVel [i,j] = [vx0, 0.0]

    ## Right
    for j in range (0, ny):
      i = nx-1
      tilb.setNodeType (i, j, tilb.BOUNDARY_P_EAST)
      tilb.bcVel [i, j] = [tilb.NaN, 0.0]
      tilb.bcRho [i, j] = 1.0


  buildVelocityKernel (vx0)

  tilb.initializeGeometry()
  tilb.setNu (nu)  
  tilb.initializeAtEquilibrium (rho0 = 1.0, vx0 = 0.0, vy0 = 0.0)


def buildPressure (h, nu):

  setSize (h)

  nx = tilb.nx
  ny = tilb.ny

  @ti.kernel
  def buildPressureKernel ():

    for i,j in ti.ndrange (nx, ny):
      tilb.setNodeType (i,j, tilb.FLUID)
    
    # Top
    for i in ti.ndrange (nx):
      j = ny - 1
      tilb.setNodeType (i, j, tilb.BOUNDARY_BB)

    # Bottom
    for i in ti.ndrange (nx):
      j = 0
      tilb.setNodeType (i, j, tilb.BOUNDARY_BB)

    # Left
    for j in range (1, ny-1):
      i = 0
      tilb.setNodeType (i, j, tilb.BOUNDARY_P_WEST)
      tilb.bcVel [i, j] = [tilb.NaN, 0.0]
      tilb.bcRho [i, j] = 1.016

    ## Right
    for j in range (0, ny):
      i = nx-1
      tilb.setNodeType (i, j, tilb.BOUNDARY_P_EAST)
      tilb.bcVel [i, j] = [tilb.NaN, 0.0]
      tilb.bcRho [i, j] = 1.0


  buildPressureKernel ()

  tilb.initializeGeometry()
  tilb.setNu (nu)  
  tilb.initializeAtEquilibrium (rho0 = 1.008, vx0 = 0.0, vy0 = 0.0)


def simulate ():

  h                 = 512
  numberOfTimeSteps = 1000000
  saveInterval      = 10000  


  buildPressure (h = h, nu = 0.25)
  #buildVelocity (h = h, vx0 = 0.1, nu = 0.25)


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
                  excepthook             = True
                  )

  dataType = "f32"
  layout   = "dense"

  tilb.setDataType (dataType)
  tilb.setLayout   (layout)

  simulate()

  img.initialize()
  img.saveNodeTypes()

  ti.kernel_profiler_print()  

