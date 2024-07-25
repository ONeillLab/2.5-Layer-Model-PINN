import numpy as np
from netCDF4 import Dataset
from name_list import *
import access_data as ad


# [:,l] move columns 1 right np.roll(v1, 1, axis=1)
# [:,r] move columns 1 left  np.roll(v1, -1, axis=1)
# [l,:] move rows 1 down     np.roll(u1, 1, axis=0)
# [r,:] move rows 1 up       np.roll(u1, -1, axis=0)

def create_ex_file(data_name):
    """
    Creates a netCDF file on the disk
    """
    rootgroup = Dataset(data_name, "a") # creates the file
    rootgroup.tmax = tmax # creates attributes
    rootgroup.c22h = c22h
    rootgroup.c12h = c12h
    rootgroup.H1H2 = H1H2
    rootgroup.Bt = Bt
    rootgroup.Br2 = Br2
    rootgroup.p1p2 = p1p2
    rootgroup.tstf = tstf
    rootgroup.tstpf = tstpf
    rootgroup.tradf = tradf
    rootgroup.dragf = dragf
    rootgroup.Ar = Ar
    rootgroup.Re = Re
    rootgroup.Wsh = Wsh
    rootgroup.gm = gm 
    rootgroup.aOLd = aOLd 
    rootgroup.L = L 
    rootgroup.num = num 
    rootgroup.deglim = deglim  
    rootgroup.Lst = Lst
    rootgroup.AB = AB  
    rootgroup.layers = layers  
    rootgroup.n = n 
    rootgroup.kappa = kappa
    rootgroup.ord = ord 
    rootgroup.spongedrag1 = spongedrag1
    rootgroup.spongedrag2 = spongedrag2
    rootgroup.dx = dx
    rootgroup.dt = dt
    rootgroup.dtinv = dtinv
    rootgroup.sampfreq = sampfreq
    rootgroup.tpl = tpl
    rootgroup.N = N
    rootgroup.L = L
    rootgroup.EpHat = EpHat

    rootgroup.createDimension("x", 36) 
    rootgroup.createDimension("y", 36)
    rootgroup.createDimension("io", 9)
    rootgroup.createDimension("example", None)


    examples = rootgroup.createVariable("examples", "f8", ("example", "io", "x", "y",),compression='zlib') # variables (list of arrays)

    rootgroup.close()

def store_example(xmat, x, file_name):
    rootgroup = Dataset(file_name, "a")
    rootgroup.variables[xmat][rootgroup.variables[xmat].shape[0],:,:,:] = x.astype("float64") 
    rootgroup.close()










# v1 + v1[:,l]