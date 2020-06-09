import numpy as np
import glob
import os
a = []
b = []
c = []
tsd = []
for d in range( len(glob.glob('min.*.dat'))):
    if d == 0:
        a = np.loadtxt('min.'+str(d)+'.dat')

        cmd = 'more vol.'+str(d)+'.dat > vol.dat'
        os.system(cmd)

        cmd = 'more crv.'+str(d)+'.dat > crv.dat'
        os.system(cmd)

        cmd = 'more tsd.'+str(d)+'.dat > tsd.dat'
        os.system(cmd)
        # tsd = np.loadtxt('tsd.'+str(d)+'.dat')
    else:
        aT = np.loadtxt('min.'+str(d)+'.dat')
        cmd = 'more vol.'+str(d)+'.dat >> vol.dat'
        os.system(cmd)

        cmd = 'more crv.'+str(d)+'.dat >> crv.dat'
        os.system(cmd)

        cmd = 'more tsd.'+str(d)+'.dat >> tsd.dat'
        os.system(cmd)

        a = np.concatenate([a,aT])


a = a[a[:,0].argsort()]

np.savetxt('minsort.dat',a[:,1])
