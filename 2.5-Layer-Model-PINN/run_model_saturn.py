import numpy as np
import math
import helper_functions as hf
import time
from name_list import *
from numba import jit, objmode, threading_layer, config
import psutil
from netCDF4 import Dataset
import access_data as ad
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable


config.THREADING_LAYER = 'omp'
                                    
#@jit(nopython=True, parallel=True)                                     
def run_sim(u1, u2, v1, v2, h1, h2, locs, lasttime):
    if restart_name == None:
        locs = hf.genlocs(num, N, 0) ### use genlocs instead of paircount
        
    mode = 1

    wlayer = hf.pairshapeN2(locs, lasttime)
    Wmat = hf.pairfieldN2(L, h1, wlayer)

    # TIME STEPPING
    if AB == 2:
        u1_p = u1.copy()
        v1_p = v1.copy()
        h1_p = h1.copy()
        u2_p = u2.copy()
        v2_p = v2.copy()
        h2_p = h2.copy()

    ts = []
    zeta2mat = []
    u1mat = []
    psi2 = np.zeros_like(x)
    zeta1 = psi2.copy()
    zeta2 = psi2.copy()
    B2 = psi2.copy()
    B1p = B2.copy()

    ii = 0

    KEmat = []
    APEmat = []

    with objmode(timer='f8'):
        timer = time.perf_counter()

        
    t = lasttime
    tc = round(t/dt)

    #####


    while t <= tmax + lasttime + dt / 2:
        if t == 1:    
            print(np.max(Wmat))
            print(np.min(Wmat))

        if AB == 2:
            tmp = u1.copy()
            u1 = 1.5 * u1 - 0.5 * u1_p
            u1_p = tmp  #
            tmp = u2.copy()
            u2 = 1.5 * u2 - 0.5 * u2_p
            u2_p = tmp  #
            tmp = v1.copy()
            v1 = 1.5 * v1 - 0.5 * v1_p
            v1_p = tmp
            tmp = v2.copy()
            v2 = 1.5 * v2 - 0.5 * v2_p
            v2_p = tmp
            tmp = h1.copy()
            h1 = 1.5 * h1 - 0.5 * h1_p
            h1_p = tmp
            if layers == 2.5:
                tmp = h2.copy()
                h2 = 1.5 * h2 - 0.5 * h2_p
                h2_p = tmp

        # add friction
        du1dt = hf.viscND(u1, Re, n)
        du2dt = hf.viscND(u2, Re, n)
        dv1dt = hf.viscND(v1, Re, n)
        dv2dt = hf.viscND(v2, Re, n)

        """ if spongedrag1 > 0:
            du1dt = du1dt - spdrag1 * (u1)
            du2dt = du2dt - spdrag2 * (u2)
            dv1dt = dv1dt - spdrag1 * (v1)
            dv2dt = dv2dt - spdrag2 * (v2) """

        # absolute vorticity
        zeta1 = 1 - Bt * rdist**2 + (1 / dx) * (v1 - v1[:,l] + u1[l,:] - u1)
        
        zeta2 = 1 - Bt * rdist**2 + (1 / dx) * (v2 - v2[:,l] + u2[l,:] - u2)


        # add vorticity flux, zeta*u
        zv1 = zeta1 * (v1 + v1[:,l])
        zv2 = zeta2 * (v2 + v2[:,l])

        du1dt = du1dt + 0.25 * (zv1 + zv1[r,:])

        du2dt = du2dt + 0.25 * (zv2 + zv2[r,:])

        zu1 = zeta1 * (u1 + u1[l,:])
        zu2 = zeta2 * (u2 + u2[l,:])

        dv1dt = dv1dt - 0.25 * (zu1 + zu1[:,r])
        dv2dt = dv2dt - 0.25 * (zu2 + zu2[:,r])

        ### Cumulus Drag (D) ###
        du1dt = du1dt - (1 / dx) * u1 / dragf
        du2dt = du2dt - (1 / dx) * u2 / dragf
        dv1dt = dv1dt - (1 / dx) * v1 / dragf
        dv2dt = dv2dt - (1 / dx) * v2 / dragf

        B1p, B2p = hf.BernN2(u1, v1, u2, v2, gm, c22h, c12h, h1, h2, ord)

        du1dtsq = du1dt - (1 / dx) * (B1p - B1p[:,l])
        du2dtsq = du2dt - (1 / dx) * (B2p - B2p[:,l])

        dv1dtsq = dv1dt - (1 / dx) * (B1p - B1p[l,:])
        dv2dtsq = dv2dt - (1 / dx) * (B2p - B2p[l,:])

        if AB == 2:
            u1sq = u1_p + dt * du1dtsq
            u2sq = u2_p + dt * du2dtsq

            v1sq = v1_p + dt * dv1dtsq
            v2sq = v2_p + dt * dv2dtsq

        ##### new storm forcing -P #####

        remove_layers = [] # store weather layers that need to be removed here

        if mode == 1:
            for i in range(len(locs)):
                if (t-locs[i][-1]) >= locs[i][3] and t != 0:
                    remove_layers.append(i) # tag layer for removal if a storm's 

            add = len(remove_layers) # number of storms that were removed

            if add != 0:
                newlocs = hf.genlocs(add, N, t)

                for i in range(len(remove_layers)):
                    locs[remove_layers[i]] = newlocs[i]

                wlayer = hf.pairshapeN2(locs, t) ### use pairshapeBEGIN instead of pairshape
                Wmat = hf.pairfieldN2(L, h1, wlayer)

        ##### new storm forcing -P #####

        Fx1 = hf.xflux(h1, u1) - kappa / dx * (h1 - h1[:,l])
        Fy1 = hf.yflux(h1, v1) - kappa / dx * (h1 - h1[l,:])
        dh1dt = -(1 / dx) * (Fx1[:,r] - Fx1 + Fy1[r,:] - Fy1)

        if layers == 2.5:
            Fx2 = hf.xflux(h2, u2) - kappa / dx * (h2 - h2[:,l])
            Fy2 = hf.yflux(h2, v2) - kappa / dx * (h2 - h2[l,:])

            dh2dt = -(1 / dx) * (Fx2[:,r] - Fx2 + Fy2[r,:] - Fy2)

        if tradf > 0:
            dh1dt = dh1dt - 1 / tradf * (h1 - 1)
            dh2dt = dh2dt - 1 / tradf * (h2 - 1)

        if mode == 1:
            dh1dt = dh1dt + Wmat.astype(np.float64)
            if layers == 2.5:
                dh2dt = dh2dt - H1H2 * Wmat.astype(np.float64)

        if AB == 2:
            h1 = h1_p + dt * dh1dt
            if layers == 2.5:
                h2 = h2_p + dt * dh2dt

        u1 = u1sq
        u2 = u2sq
        v1 = v1sq
        v2 = v2sq

        if tc % tpl == 0:
            with objmode(timer='f8'):
                print(f"t={t}, mean h1 is {round(np.mean(np.mean(h1)), 4)}, num storms {locs.shape[0]}. Time elapsed, {round(time.perf_counter()-timer, 3)}s. CPU usage, {psutil.cpu_percent()}")
                timer = time.perf_counter()

            ii += 1
        
            ts.append(t)
            #zeta2mat.append(zeta2)
            u1mat.append(zeta2)

            """ with objmode():
                ad.save_data(u1,u2,v1,v2,h1,h2,locs,t,lasttime,new_name) """

            KEmat.append(hf.calculate_KE(u1,u2,v1,v2,h1,h2))
            APEmat.append(hf.calculate_APE(h1, h2))
                
        if math.isnan(h1[0, 0]):
            break

        tc += 1
        t = tc * dt

    fig = plt.figure()

    frames4 = np.asarray(u1mat)
    ax4 = fig.add_subplot(111)
    cv4 = frames4[0]
    print(np.shape(cv4))
    vminlist = []
    vmaxlist = []
    for j in frames4:
        vminlist.append(np.min(j))
        vmaxlist.append(np.max(j))
    vmin = np.min(vminlist)
    vmax = np.max(vmaxlist)
    im4 = ax4.imshow(cv4, cmap="bwr", vmin=vmin, vmax=vmax)
    cb = fig.colorbar(im4)
    tx4 = ax4.set_title("main")

    """ frames0 = np.asarray(u1mat)[:,:79*2,:79*2]
    ax0 = fig.add_subplot(231)
    frames0len = int(len(frames0))
    cv0 = frames0[0]
    im0 = ax0.imshow(cv0, cmap="bwr", vmin=vmin, vmax=vmax)
    tx0 = ax0.set_title(f"1")

    frames1 = np.asarray(u1mat)[:,:79*2,79*2:]
    ax1 = fig.add_subplot(232)
    cv1 = frames1[0]
    im1 = ax1.imshow(cv1, cmap="bwr", vmin=vmin, vmax=vmax)
    tx1 = ax1.set_title(f"2")

    frames2 = np.asarray(u1mat)[:,79*2:,:79*2]
    ax2 = fig.add_subplot(234)
    cv2 = frames2[0]
    im2 = ax2.imshow(cv2, cmap="bwr", vmin=vmin, vmax=vmax)
    tx2 = ax2.set_title(f"3")

    frames3 = np.asarray(u1mat)[:,79*2:,79*2:]
    ax3 = fig.add_subplot(235)
    cv3 = frames3[0]
    im3 = ax3.imshow(cv3, cmap="bwr", vmin=vmin, vmax=vmax)
    tx3 = ax3.set_title(f"time: {0}") """

    def animate(i):
        arr4 = frames4[i] 
        vmax = np.max(arr4)
        vmin = np.min(arr4)
        tx4.set_text(f"time: {i}")
        im4.set_data(arr4)

    print("animating")
    ani = animation.FuncAnimation(fig, animate, interval=ani_interval, frames=len(frames4))
    plt.show()

""" if restart_name == None:
    ad.create_file(new_name)
    lasttime = 0
    locs = np.zeros((num,5))
else:
    ad.create_file(new_name)
    u1, u2, v1, v2, h1, h2, locs, lasttime = ad.last_timestep(restart_name) """

def compute_u1(u1, u1_p, v1, v1_p, h1, h2):
    tmp = u1.copy()
    u1 = 1.5 * u1 - 0.5 * u1_p
    u1_p = tmp
    tmp = v1.copy()
    v1 = 1.5 * v1 - 0.5 * v1_p
    v1_p = tmp

    du1dt = hf.viscND(u1, Re, n)
    dv1dt = hf.viscND(v1, Re, n)

    du1dt = du1dt - spdrag1 * (u1)
    dv1dt = dv1dt - (1 / dx) * v1 / dragf

    zeta1 = 1 - Bt * rdist**2 + (1 / dx) * (v1 - v1[:,l] + u1[l,:] - u1) #[:,l] move columns 1 right [l,:] move rows 1 down

    zv1 = zeta1 * (v1 + v1[:,l]) 
    du1dt = du1dt + 0.25 * (zv1 + zv1[r,:]) #[r,:] move rows 1 up 
    #du1dt = du1dt + 0.25 * zeta1 * ( (v1 + v1[:,l]) + (v1 + v1[:,l])[r,:] ) 

    zu1 = zeta1 * (u1 + u1[l,:])
    dv1dt = dv1dt - 0.25 * (zu1 + zu1[:,r]) #[:,r] move columns 1 left
    # dv1dt = dv1dt - 0.25 * zeta1 * ((u1 + u1[l,:]) + (u1 + u1[l,:])[:,r])

    du1dt = du1dt - (1 / dx) * u1 / dragf
    dv1dt = dv1dt - (1 / dx) * v1 / dragf

    B1p = c12h * h1 + c22h * h2 + 0.25 * (u1**2 + u1[:,r]**2 + v1**2 + v1[r,:]**2)

    du1dtsq = du1dt - (1 / dx) * (B1p - B1p[:,l])
    # (u1**2 + u1[:,r]**2 + v1**2 + v1[r,:]**2) - (u1**2 + u1[:,r]**2 + v1**2 + v1[r,:]**2)[:,l]

    dv1dtsq = dv1dt - (1 / dx) * (B1p - B1p[l,:])

    u1sq = u1_p + dt * du1dtsq
    v1sq = v1_p + dt * dv1dtsq

    u1 = u1sq
    v1 = v1sq

    return u1, u1_p, v1, v1_p
    
def compute_u1_(u1, u1_p, v1, v1_p, h1, h2, h1up, h2up, h1left, h2left, v1up, v1left, v1bottom, u1up, u1right, u1left):
    tmp = u1.copy()
    u1 = 1.5 * u1 - 0.5 * u1_p
    u1_p = tmp
    tmp = v1.copy()
    v1 = 1.5 * v1 - 0.5 * v1_p
    v1_p = tmp

    du1dt = hf.viscND(u1, Re, n)
    dv1dt = hf.viscND(v1, Re, n)

    du1dt = du1dt - spdrag1 * (u1)
    dv1dt = dv1dt - (1 / dx) * v1 / dragf

    v1_l = v1[:,l]
    v1_l[:,0] = v1left[1:-1]

    u1_r = u1[:,r]
    u1_r[:,-1] = u1right[1:-1]

    u1_l = u1[:,l]
    u1_l[:,0] = u1left[1:-1]
    
    u1l_ = u1[l,:]
    u1l_[0,:] = u1up[1:-1]

    v1l_ = v1[l,:]
    v1l_[0,:] = v1up[1:-1]

    v1r_ = v1[r,:]
    v1r_[-1,:] = v1bottom[1:-1]

    h1_l = h1[:,l]
    h1_l[:,0] = h1left[1:-1]

    h1l_ = h1[l,:]
    h1l_[0,:] = h1up[1:-1]

    h2_l = h2[:,l]
    h2_l[:,0] = h2left[1:-1]

    h2l_ = h2[l,:]
    h2l_[0,:] = h2up[1:-1]

    zeta1 = 1 - Bt * rdist**2 + (1 / dx) * (v1 - v1_l + u1l_ - u1) #[:,l] move columns 1 right [l,:] move rows 1 down

    v1shift = (v1bottom[0:-1] + v1bottom[1:])[1:-1]
    v1plusv1_l = (v1+v1_l)[r,:] 
    v1plusv1_l[-1,:] = v1shift

    du1dt = du1dt + 0.25 * ((zeta1 * (v1 + v1_l)) + (zeta1 * v1plusv1_l))#)(v1 + v1_l))[r,:]) 

    u1shift = (u1right[0:-1] + u1right[1:])[1:-1]
    u1plusu1l_ = (u1 + u1l_)[:,r]
    u1plusu1l_[:,-1] = u1shift

    dv1dt = dv1dt - 0.25 * ((zeta1 * (u1 + u1l_)) + (zeta1 * u1plusu1l_)) #[:,r] move columns 1 left

    du1dt = du1dt - (1 / dx) * u1 / dragf
    dv1dt = dv1dt - (1 / dx) * v1 / dragf

    B1p = c12h * h1 + c22h * h2 + 0.25 * (u1**2 + u1_r**2 + v1**2 + v1r_**2)
    v1[r,:][:,l]
    v1r_l_ = v1r_.copy()
    v1r_l_ = v1r_l_[:,l]
    v1r_l_[:,0] = v1left[2:]
    B1p_l = (c12h * h1_l + c22h * h2_l + 0.25 * (u1_l**2 + u1**2 + v1_l**2 + v1r_l_**2))
    u1_rl_ = u1_r.copy()
    u1_rl_ = u1_rl_[l,:]
    u1_rl_[0,:] = u1up[2:]
    B1pl_ = c12h * h1l_ + c22h * h2l_ + 0.25 * (u1l_**2 + u1_rl_**2 + v1l_**2 + v1**2)

    du1dtsq = du1dt - (1 / dx) * (B1p - B1p_l)
    dv1dtsq = dv1dt - (1 / dx) * (B1p - B1pl_)

    u1sq = u1_p + dt * du1dtsq
    v1sq = v1_p + dt * dv1dtsq

    u1 = u1sq
    v1 = v1sq

    return u1, u1_p, v1, v1_p


lasttime = 0
locs = np.zeros((num,5))  
run_sim(u1,u2,v1,v2,h1,h2,locs,lasttime)

print("Threading layer chosen: %s" % threading_layer())
print("Num Threads: %s" % config.NUMBA_NUM_THREADS)