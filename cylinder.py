# vim: set expandtab:

import taichi as ti

import tilb
import channel
import shapes
import img



def simulate ():

  h                 = 512
  nu                = 0.1
  numberOfTimeSteps = 1000000
  saveInterval      = 10000  


  channel.buildVelocity (h = h, vx0 = 0.1, nu = nu)

  shapes.initialize()

  #shapes.circle (2 * h, h / 2.0, h / 4.0)
  shapes.square (2 * h, h / 2.0, h / 4.0)

  tilb.initializeGeometry()
  tilb.setNu (nu)  
  tilb.initializeAtEquilibrium (rho0 = 1.008, vx0 = 0.0, vy0 = 0.0)  


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

