# vim: set expandtab:

import taichi as ti
import numpy  as np
import matplotlib
import matplotlib.pyplot as plt

import tilb


def build (resolution, Re):

  N = resolution

  tilb.setResolution (x = N, y = N)
  tilb.allocate()

  @ti.kernel
  def buildKernel (U : tilb.DataType):

    nx = tilb.nx
    ny = tilb.ny

    for i,j in ti.ndrange (nx, ny):
      tilb.setNodeType (i,j, tilb.FLUID)
    
    # Top
    for i in ti.ndrange (nx):
      j = ny - 1
      tilb.setNodeType (i, j, tilb.BOUNDARY_VEL_NEQN_NORTH)
      tilb.bcVel [i,j] = [U, 0.0]

    # Bottom
    for i in ti.ndrange (1,nx-1):
      j = 0
      tilb.setNodeType (i, j, tilb.BOUNDARY_BB)

    # Left
    for j in range (0, ny-1):
      i = 0
      tilb.setNodeType (i, j, tilb.BOUNDARY_BB)

    ## Right
    for j in range (0, ny-1):
      i = nx-1
      tilb.setNodeType (i, j, tilb.BOUNDARY_BB)

  U  = 0.1
  L  = N - 1
  nu = (U * L) / Re

  print (f"case = cavity2D, resolution = {N} x {N}, "
         f"U = {U}, L = {L}, Re = {Re}, nu = {nu}")
  buildKernel (U)
  tilb.initializeGeometry()
  tilb.setNu (nu)  
  tilb.initializeAtEquilibrium (rho0 = 1.0, vx0 = 0.0, vy0 = 0.0)


def indicesGhia (Re):

  if (    100 == Re):
    x = 1
  elif ( 1000 == Re):
    x = 2
  elif ( 3200 == Re):
    x = 3
  elif ( 5000 == Re):
    x = 4
  elif (10000 == Re):
    x = 5
  else:
    raise Exception(f"There are no reference results for Re = {Re}")

  y = x + 6

  return x, y


def verify (velocity2D, Re, imageName):
  v = velocity2D

  shape = v.shape
  nx = shape [0]
  ny = shape [1]

  uIdx, vIdx = indicesGhia (Re)

  y_ref, u_ref = np.loadtxt('ghia1982.dat', unpack=True, 
                             skiprows=2, usecols=(0, uIdx))
  x_ref, v_ref = np.loadtxt('ghia1982.dat', unpack=True, 
                             skiprows=2, usecols=(6, vIdx))

  fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 3), dpi=200)

  axes.plot (np.linspace(0, 1.0, nx), v[:,:,0][nx // 2, :] / 0.1, 'b-', 
            label='LBM vx')
  axes.plot (y_ref, u_ref, 'rs', label='Ghia et al. 1982')

  axes.plot (np.linspace(0, 1.0, nx), v[:,:,1][:, ny // 2] / 0.1, 'b-', 
             label='LBM vy')
  axes.plot (x_ref, v_ref, 'rs', label='Ghia et al. 1982')
  
  axes.legend()
  axes.set_xlabel(r'Y')
  axes.set_ylabel(r'U')

  fig.savefig (imageName)
  plt.close (fig)


def verifyVelocity (f, Re, imageName):
  tilb.computeRhoV (f)
  vel = tilb.vel.to_numpy()
  verify (vel, Re, imageName)


def simulate ( resolution ,
               Re ,
               numberOfTimeSteps ,
               saveInterval ,
               dataType = "f32",
               layout = "dense"
               ):

  #
  #   DataType have to be set before creation of kernels and data fields.
  #
  tilb.setDataType (dataType)
  tilb.setLayout   (layout)
  build (resolution = resolution, Re = Re)

  tilb.simulate (numberOfTimeSteps = numberOfTimeSteps, saveInterval = saveInterval)
  
  tilb.computeRhoV (tilb.f_old)
  verifyVelocity (f = tilb.f_old, Re = Re, imageName = "plot.png")
  tilb.saveVtk ("out")
  tilb.saveState (fileName = "state.npz", f = tilb.f_old)


if ("__main__" == __name__):

  ti.init (arch = ti.gpu                         , 
                  kernel_profiler        = True  ,
                  use_unified_memory     = False ,
                  device_memory_fraction = 0.90  ,
                  excepthook             = True
                  )

  #
  #   Complete simulations.
  #
  simulate (resolution = 128, Re = 100, 
            numberOfTimeSteps = 12000,
            saveInterval = 2000,
            dataType = "f32")
  #simulate (resolution = 256, Re = 1000, 
  #          numberOfTimeSteps = 100000,
  #          saveInterval = 2000,
  #          dataType = "f32")
  #simulate (resolution = 1024, Re = 3200, 
  #          numberOfTimeSteps = 1000000,
  #          saveInterval = 50000,
  #          dataType = "f32")
  #simulate (resolution = 2048, Re = 5000, 
  #          numberOfTimeSteps = 3000000,
  #          saveInterval = 50000,
  #          dataType = "f32")
  #simulate (resolution = 4096, Re = 10000, 
  #          numberOfTimeSteps = 10000000,
  #          saveInterval = 100000,
  #          dataType = "f32")

  ti.kernel_profiler_print()
  #ti.print_profile_info()

