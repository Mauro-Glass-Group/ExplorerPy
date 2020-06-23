import numpy as np
import glob
import time
import random
import sys
import os

import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Union


try:
    import ctypes as ct
except:
    print('Missing :|: Ctypes Python Library')
    exit()

try:
    import scipy
    import scipy.optimize
except:
    print('Missing :|: Scipy Python Library')
    exit()

try:
    from mep.neb import NEB
    from mep.path import Path
    from mep.path import Image
    from mep.optimize import Optimizer, SGD

except:
    print('Missing :|: Minimum Energy Pathway (MEP) Python Library')
    exit()
### PartialHessian Package from U Illinois
### PyLammps Install - lammps as a shared library


parallel = True
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    sub_comm = comm.Split(comm.Get_rank(),comm.Get_rank())
    print ('Splitting ... ' + str(comm.Get_rank()))

except:
    parallel = False
    print('Could not Find mpi4py: MPI Package Missing.')
    exit()

if comm.Get_size() ==1:
    parallel = False

def reduce(delta,Lx,Ly,Lz):
    s = False
    if delta.shape[0] == len(delta.flatten()):
        d = np.reshape(delta[:-1], (int((len(delta)-1)/3) , 3 ) ).T
        s = True
    else:
        d = delta

    x = d[0,:]
    y = d[1,:]
    z = d[2,:]

    x -= np.around(x/Lx) * Lx
    y -= np.around(y/Ly) * Ly
    z -= np.around(z/Lz) * Lz

    d[0,:] = x
    d[1,:] = y
    d[2,:] = z

    if s:
        delta[:-1] = d.T.flatten()
        return np.copy(delta)
    else:
        return d

class periodicPath(Path):
    lx = 0.
    ly = 0.
    lz = 0.
    @property
    def image_distances(self):
        r = [0]
        for i,j in zip(self.coords[:-1], self.coords[1:]):
            delta = i-j
            if i[-1] > j[-1]:
                L = i[-1]
            else:
                L = j[-1]

            s = L/self.lx
            delta = reduce(delta, self.lx*s , self.ly*s , self.lz*s)

            r.append(np.linalg.norm(delta))
        return r

    @classmethod
    def from_linear_end_points(cls, image_start, image_end, a, b, c, n, k=-5):
        cls.lx = a
        cls.ly = b
        cls.lz = c

        delta = np.array(image_end) - np.array(image_start)

        if image_end[-1] > image_start[-1]:
            L = image_end[-1]
        else:
            L = image_start[-1]

        s = L/a
        delta = reduce(delta,a*s,b*s,c*s)

        dximage = delta / (n-1)
        return cls([np.array(image_start) + dximage * i for i in range(n)], k=k)

    @property
    def spring_forces(self):
        def dist(a,b):
            if a[0][-1] > b[0][-1]:
                L = a[0][-1]
            else:
                L = b[0][-1]
            delta = a[0] - b[0]

            s = L/self.lx

            delta = reduce(delta,s*self.lx,s*self.ly,s*self.lz)
            return np.linalg.norm(delta)

        _spring_forces = []
        for k, i in enumerate(self.inner_images, start=1):
            _spring_forces.append(i.prev.data * dist(self.images[k+1].data, i.data) -
                                  i.next.data * dist(i.data, self.images[k-1].data))
        return _spring_forces

    def get_unit_tangent(self, i: int, energies: List[float]=None):
        """
        As described in Henkelmana et al. Journal of Chemical Physics
        https://aip.scitation.org/doi/pdf/10.1063/1.1323224?class=pdf
        Args:
            i: index of the image
            energies: list of energies for all images
        Returns:
        """
        if i < 1 or i > self.n_images - 2:
            raise ValueError('Only internal images can be calculated')
        vs = energies[(i - 1):(i + 2)]
        coords = [self.images[k].data for k in [i - 1, i, i + 1]]

        L = coords[0][0][-1]
        s = L/self.lx
        # print(L)
        # exit()

        tau_plus = coords[-1] - coords[1]
        tau_plus = reduce(tau_plus.T.reshape((tau_plus.shape[1],)),s*self.lx,s*self.ly,s*self.lz)

        tau_minus = coords[1] - coords[0]
        tau_minus = reduce(tau_minus.T.reshape((tau_minus.shape[1],)),s*self.lx,s*self.ly,s*self.lz)

        dv_plus = vs[2] - vs[1]
        dv_minus = vs[1] - vs[0]

        if (dv_plus > 0) and (dv_minus > 0):
            tau = tau_plus
        elif (dv_plus < 0) and (dv_minus < 0):
            tau = tau_minus
        else:
            dv_max = max(abs(dv_plus), abs(dv_minus))
            dv_min = min(abs(dv_plus), abs(dv_minus))
            if vs[2] - vs[0] > 0:
                tau = tau_plus * dv_max + tau_minus * dv_min
            else:
                tau = tau_plus * dv_min + tau_minus * dv_max
        if np.linalg.norm(tau) == 0:
            return tau
        else:
            return tau / np.linalg.norm(tau)

class kMCNN:
    atoms = []
    n = 1
    i = 0

    B_THRESH = 1e-5 ### Was 1e-5
    F_THRESH = 1. #0.3 ### Was 1.
    P_THRESH = 20.

    freqMD = 0
    freqSV = 0

    tolerance = 5 ### Was 20
    stepSize = 0.12 ### Was 0.15
    rate = 0.0 ### Doesn't work leave as 0.
    c = False

    press = 999.
    lStep = 0.02
    dL = 0.001

    ref=[]

    def eval(self, expr):
        self.center()
        self.atoms.command('run 1')
        v = self.atoms.get_thermo(str(expr))
        return v

    def predict_energy(self, image):
        if isinstance(image, Image):
            c = np.copy( image.data[0] )
        else:
            c = np.copy( image )

        i = self.getCoords()
        self.setCoords(c)
        e = self.eval('pe')
        self.setCoords(i)
        # c[:-1] *= c[-1]
        return np.asarray([e])

    def predict_energy_and_forces(self, image, delta=0.00001):
        # image.data[0][:-1]*=image.data[0][-1]
        i = self.getCoords()
        self.setCoords( np.copy(image.data[0]) )
        f = self.getForces()

        if self.press == 999.:
            f = np.append(f,[0])
        else:
            f = np.append(f, [( self.press - self.eval('press') )*6.3e-7] )

        e = self.eval('pe')
        self.setCoords(i)

        return self.predict_energy(image.data[0]),np.asarray(f)

    def NEB(self,start,end,nimages=7,spring_constant=1.):
        self.setCoords(start)
        U = (self.energy(start))
        U = self.eval('pe')

        path = periodicPath.from_linear_end_points(np.copy(start), np.copy(end),
                        self.atoms.extract_global("boxxhi", 1) - self.atoms.extract_global("boxxlo", 1),
                        self.atoms.extract_global("boxyhi", 1) - self.atoms.extract_global("boxylo", 1),
                        self.atoms.extract_global("boxzhi", 1) - self.atoms.extract_global("boxzlo", 1),
                        nimages, spring_constant)  # set 101 images, and k=1

        neb =NEB(self, path) # initialize NEB
        history = neb.run(verbose=False,n_steps=100, optimizer=SGD(0.003)) # run

        E = []
        # d=[]
        # j =[]

        E.append(U)
        # d.append(0)
        # j.append(self.eval('pe'))

        for i in history.history[-1][1:-1]:
            self.setCoords(i[0])
            E.append(self.energy())
            # d.append(self.distance(start)[1])
            # j.append(self.eval('pe'))

        self.setCoords(end)
        E.append(self.energy())

        # d.append(self.distance(start)[1])
        # j.append(self.eval('pe'))
        # if np.max(E) - self.energy(start) < -1e-4 or np.max(E) == self.energy(start) or np.max(E) == self.energy(end):
        # print(d)
        # print(E)
        # plt.plot(d,E,'.',markersize=9,color='blue')
        # plt.plot(d,E,'-',color='blue')
        # plt.plot(d,j,'.',markersize=9,color='orange')
        # plt.plot(d,j,'-',color='orange')
        #
        # plt.axvline(d[np.argmax(E)])
        # plt.show()
        # exit()
        # exit()

        self.setCoords(end)

        return np.max(E), history.history[-1][np.argmax(E)][0] ###Problem Child!

    def MD(self, prev, mThresh = 0.1, eMax = 20., steps=200,custom=0.):
        s = time.time()
        U = self.energy()

        start = self.getCoords()
        begin = self.getCoords()

        barr = 0.

        while barr > eMax or barr < 1e-4:
            numStep = steps
            self.setCoords(np.copy(begin))

            m = 0.
            while m < mThresh:
                if custom == 0.:
                    cmd = 'velocity all create 1500. ' + str( int( 100*(time.time()-s)*(comm.Get_rank()+1) )%1000000 + 1 ) + ' rot yes dist gaussian'
                    self.atoms.command('fix md all nvt temp 1500. 1500. 0.1')
                    self.atoms.command(cmd)

                    d=0.
                    while d < mThresh:
                        self.atoms.command('run ' + str(numStep) )
                        d, t = self.distance(start)
                else:
                    self.atoms.file(custom)

                mid = self.getCoords()
                self.atoms.command('unfix md')
                self.atoms.command('velocity all set 0. 0. 0.')

                # print('Begin: ' + str(self.energy(begin)) )
                # print('Start: ' + str(self.energy(start)) )
                self.minimize()

                # print('Begin: ' + str(self.energy(start)) )
                # print('Start: ' + str(self.energy(start)) )
                # if self.energy(start) != U:
                #     print('FUCKING FUCK')
                #     exit()
                end = self.getCoords()
                newEnergy = self.energy()
                m, disp = self.distance(start)
                if m < 0.1:
                    self.setCoords(mid)
                    numStep+=2
                else:
                    print ('Max Displacment Found: ' + str(round(m,3)) + ' Total Displacment: ' + str(round(disp,3)) + ' New Energy: ' + str(round(newEnergy,3) ) )
            # print(prev)
            # print('\n\n\n')
            self.load(prev)
            begin = self.getCoords()

            # begin = self.getCoords()
            barr, tPoint = self.NEB( np.copy(begin) ,np.copy(end) )
            # print('Barrier: ' + str(barr) )
            barr -= U

            self.setCoords(end)
            if barr == 0:
                barr = 0
            else:
                print ('Stats| Energy Found: ' + str(round(barr,6)) + ' Time Since Start: ' + str(round(time.time()-s,3))
                        + 's dE: ' + str( round(U - newEnergy,3) ) )
        return True, barr+U, tPoint, -1

    def shove(self,n,eMax=15,dist=0.25,mThresh=0.05):
        a,b = self.hessian()
        initCurve =a[n]

        def distAtom():
            s = self.getCoords()
            s = np.reshape(s[:-1],(self.atoms.get_natoms(),3))
            min=100.

            for an in range(len(s[:,0])):
                Lx = self.atoms.extract_global("boxxhi", 1) - self.atoms.extract_global("boxxlo", 1)
                Ly = self.atoms.extract_global("boxyhi", 1) - self.atoms.extract_global("boxylo", 1)
                Lz = self.atoms.extract_global("boxzhi", 1) - self.atoms.extract_global("boxzlo", 1)

                d = reduce(s - s[an,:],Lx,Ly,Lz)
                d = np.linalg.norm(d,axis=1)

                if np.sort(d)[1] < min:
                    min = np.sort(d)[1]
            return min

        s = time.time()
        st = self.getCoords()
        temp = self.getCoords()
        initial = self.energy()

        m = 0.

        d = np.copy(temp)
        dL = 1.0

        closeparm = 0.3
        while m < mThresh:# or dL < closeparm:
            a,b = self.hessian()

            dat = np.copy(b[:,n])
            dat = np.reshape(dat,(self.atoms.get_natoms(),3))
            lmd = dist / np.max(np.linalg.norm(dat,axis=1))

            d += lmd*np.append(b[:,n],[0])

            self.setCoords(d)
            temp = self.getCoords()

            # print(dL)
            # dL = distAtom()

            # if dL > closeparm:
            # print('')
            # print(self.energy())
            self.minimize()
            m,total = self.distance(st)

            if m < mThresh:
                self.setCoords(temp)

            print(str(comm.Get_rank()) + ' :Minimized Dist: ' + str(round(m,2)) + ' A')

        end = self.getCoords()
        m,total = self.distance(st)
        print('Shoving Distance: ' + str(m) + ' Total Displacement: ' + str(total) + ' New Energy: ' +str(self.energy()))
        print('')

        if np.abs(self.energy() - initial) > eMax*10.:
            return False, 0., self.getCoords(), initCurve

        barr, tPoint = self.NEB(st,self.getCoords())

        print ('Stats| Energy Found: ' + str(round(barr-initial,6)) + ' Time Since Start: ' + str(round(time.time()-s,3))
                + 's dE: ' + str( -round(self.energy(st) - self.energy(),3) ) + '\n')
        self.setCoords(end)

        if barr-initial < eMax and barr > initial and barr > self.energy() and barr-initial > 1e-6:
            return True, barr, tPoint, initCurve
        else:
            return False, barr, tPoint, initCurve

    def __init__(self,sys, p, c, n, f , sf ): ### Initializer
        self.atoms = sys
        self.press = p
        self.c = c
        self.freqMD = f
        self.freqSV = sf
        self.P_THRESH = 50.

        x = self.getCoords()
        self.ref = [x[n],x[n+1],x[n+2]]

        A = glob.glob('./kMC.*.xyz')
        # self.FPT = self.F_THRESH/20.
        self.center()
        self.atoms.command('velocity all set 0. 0. 0.')
        comm.Barrier()
        if len(A) == 0:
            self.n =0
            self.i =0

            print ('\nStaring New Run.\n')
            self.atoms.command('min_style cg')
            self.atoms.command('run 0')

            self.minimize()
            if comm.Get_rank() == 0:
                self.write(1)

                e = round(self.energy() + 20.)

                min = open('min.'+str(comm.Get_rank())+'.dat','w')
                trd = open('tsd.'+str(comm.Get_rank())+'.dat','w')
                vol = open('vol.'+str(comm.Get_rank())+'.dat','w')
                crv = open('crv.'+str(comm.Get_rank())+'.dat','w')

                min.write('## Min State Information\n')
                vol.write('## Volume of each basin\n')

                min.write('1\t' + str( self.energy() ) + '\t' + str( comm.Get_rank() ) +'\n' )
                vol.write('1\t' + str( (self.atoms.extract_global("boxxhi", 1) - self.atoms.extract_global("boxxlo", 1))**3 ) + '\n' )
                trd.write('## Transition State Information\n')
                crv.write('## Vibrational Information\n')

                min.close()
                vol.close()
                trd.close()
                crv.close()

                info = open('dinfo','w')
                info.write('DELTA 0.05\n')
                info.write('FIRST ' + str(e) + '\n')
                info.write('MAXTSENERGY ' + str(e) + '\n')
                info.write('\n')
                info.write('CENTREGMIN\n')
                info.write('CONNECTMIN 0\n')
                info.write('\n')
                info.write('LEVELS 1000\n')
                info.write('MINIMA minsort.dat\n')
                info.write('TS tsd.dat\n')
                info.write('ENERGY_LABEL \"Energy [eV]\"\n')
                info.write('TRVAL 0 vol.dat\n')
                info.write('COLOUR_BAR_LABEL \"Volume [Angstrom * 3]\"')
                info.close()

            else:

                min = open('min.'+str(comm.Get_rank())+'.dat','w')
                trd = open('tsd.'+str(comm.Get_rank())+'.dat','w')
                vol = open('vol.'+str(comm.Get_rank())+'.dat','w')
                crv = open('crv.'+str(comm.Get_rank())+'.dat','w')

                min.write('## Min Energy Data\n')
                trd.write('## Transition State Data\n')
                vol.write('## Volume Data\n')
                crv.write('## Vibrational Data\n')

                min.close()
                vol.close()
                trd.close()
                crv.close()

        else:
            print ('\nContinuing Run.\n')
            self.n = len(A)
            self.load(A[self.n-comm.Get_size()])
            self.i = self.n-comm.Get_size()

    def center(self):
        if self.c:
            L = self.atoms.extract_global("boxxhi", 1) - self.atoms.extract_global("boxxlo", 1)

            data = np.array(self.atoms.gather_atoms("x", 1, 3), dtype=ct.c_double)
            natoms = self.atoms.get_natoms()
            coord = (data.reshape(natoms, 3))

            xs = np.asarray(coord[:,0])
            ys = np.asarray(coord[:,1])
            zs = np.asarray(coord[:,2])

            dx = self.ref[0] - xs[0]
            dy = self.ref[1] - ys[0]
            dz = self.ref[2] - zs[0]

            d = self.getCoords()
            for i in range(natoms):
                d[3*i]+=dx
                d[(3*i)+1]+=dy
                d[(3*i)+2]+=dz

            natoms = self.atoms.get_natoms()
            Lx = self.atoms.extract_global("boxxhi", 1) - self.atoms.extract_global("boxxlo", 1)
            Ly = self.atoms.extract_global("boxyhi", 1) - self.atoms.extract_global("boxylo", 1)
            Lz = self.atoms.extract_global("boxzhi", 1) - self.atoms.extract_global("boxzlo", 1)

            for i in range(natoms):
                d[3*i] = ( d[(3*i)] - self.atoms.extract_global("boxxlo", 1) )%Lx + self.atoms.extract_global("boxxlo", 1)
                d[(3*i) + 1] = (d[(3*i)+1] - self.atoms.extract_global("boxylo", 1) )%Ly + self.atoms.extract_global("boxylo", 1)
                d[(3*i) + 2] = (d[(3*i)+2] - self.atoms.extract_global("boxzlo", 1) )%Lz + self.atoms.extract_global("boxzlo", 1)

            n3 = 3*natoms + 1
            x = (n3*ct.c_double)()

            for i in range( len (d) ):
                x[i] = d[i]

            self.atoms.scatter_atoms('x',1,3,x)

    def getCoords(self):
        _data = np.array(self.atoms.gather_atoms("x", 1, 3), dtype=ct.c_double)
        natoms = self.atoms.get_natoms()
        coord = (_data.reshape(int(natoms), 3))
        coord = coord.flatten()

        L = self.atoms.extract_global("boxxhi", 1) - self.atoms.extract_global("boxxlo", 1)
        C = np.append(np.asarray(coord).flatten(),L)
        return np.copy(C)

    def setCoords(self,data):

        natoms = self.atoms.get_natoms()
        d = np.copy(data) ## Copying Data
        Lx = self.atoms.extract_global("boxxhi", 1) - self.atoms.extract_global("boxxlo", 1)
        Ly = self.atoms.extract_global("boxyhi", 1) - self.atoms.extract_global("boxylo", 1)
        Lz = self.atoms.extract_global("boxzhi", 1) - self.atoms.extract_global("boxzlo", 1)

        for i in range(natoms):
            d[3*i] = ( d[(3*i)] - self.atoms.extract_global("boxxlo", 1) )%Lx + self.atoms.extract_global("boxxlo", 1)
            d[(3*i) + 1] = (d[(3*i)+1] - self.atoms.extract_global("boxylo", 1) )%Ly + self.atoms.extract_global("boxylo", 1)
            d[(3*i) + 2] = (d[(3*i)+2] - self.atoms.extract_global("boxzlo", 1) )%Lz + self.atoms.extract_global("boxzlo", 1)

        n3 = 3*natoms
        x = (n3*ct.c_double)()

        for i in range( len (d[:-1]) ):
            x[i] = d[i]

        self.atoms.scatter_atoms('x',1,3,x)
        self.center()

        if (self.press != 999. and d[-1] != self.atoms.extract_global("boxxhi", 1) - self.atoms.extract_global("boxxlo", 1)):
            scale = np.abs(d[-1]/Lx)
            if scale > 1.:
                cmd = 'change_box all x scale ' + str( scale ) + ' y scale ' + str( scale ) + ' z scale ' + str( scale ) + ' remap'
                self.atoms.command(cmd)

            Lx = self.atoms.extract_global("boxxhi", 1) - self.atoms.extract_global("boxxlo", 1)
            Ly = self.atoms.extract_global("boxyhi", 1) - self.atoms.extract_global("boxylo", 1)
            Lz = self.atoms.extract_global("boxzhi", 1) - self.atoms.extract_global("boxzlo", 1)

            for j in range(natoms):
                d[3*j] -= Lx/2.
                d[3*j] *= scale
                d[3*j] += Lx/2.

                d[(3*j)+1] -= Ly/2.
                d[(3*j)+1] *= scale
                d[(3*j)+1] += Ly/2.

                d[(3*j)+2] -= Lz/2.
                d[(3*j)+2] *= scale
                d[(3*j)+2] += Lz/2.

            if scale < 1.:
                cmd = 'change_box all x scale ' + str( scale ) + ' y scale ' + str( scale ) + ' z scale ' + str( scale ) + ' remap'
                self.atoms.command(cmd)

            Lx = self.atoms.extract_global("boxxhi", 1) - self.atoms.extract_global("boxxlo", 1)
            Ly = self.atoms.extract_global("boxyhi", 1) - self.atoms.extract_global("boxylo", 1)
            Lz = self.atoms.extract_global("boxzhi", 1) - self.atoms.extract_global("boxzlo", 1)

            cmd = 'change_box all x final 0. ' + str( Lx ) + ' y final 0. ' + str( Ly ) + ' z final 0. ' + str( Lz ) + ' remap'
            self.atoms.command(cmd)

        self.center()
        self.atoms.command('run 0')

    def write(self,num):

        data = self.getCoords()

        self.center()

        self.atoms.command('reset_timestep 0')
        cmd = 'dump dumpall all custom 1 kMC.' + str(num) + '.' + str(comm.Get_rank()) + '.xyz id type x y z'
        print('Writing on Proc: ' + str(comm.Get_rank()) + ' #: ' + str(num))
        self.atoms.command(cmd)
        self.atoms.command('run 0')
        self.atoms.command('undump dumpall')

    def writeTS(self,c,num):
        self.setCoords(c)
        self.center()
        cmd = 'dump dumpall all atom 1 TP.' + str(num) + '.' + str(comm.Get_rank()) + '.xyz'
        self.atoms.command(cmd)
        self.atoms.command('run 0')
        self.atoms.command('undump dumpall')

    def load(self,struct):
        try:
            struct= int(struct)
            if struct == 1:
                file = 'kMC.' + str(struct) + '.' + str(0) + '.xyz'
            else:
                file = 'kMC.' + str(struct) + '.' + str(comm.Get_rank()) + '.xyz'
        except:
            file = struct

        # print ('Loading File: ' + file)
        cmd = 'read_dump ' + file + ' 0 x y z box yes'
        self.atoms.command(cmd)

    def energy(self,d=[]):
        self.center()
        if self.press == 999.:
            if len(d) != 0:
                a = self.getCoords()
                self.setCoords(d)
                e = self.eval('pe')
                self.setCoords(a)

            else:
                e = self.eval('pe')
        else:
            if len(d) != 0:
                a = np.copy(self.getCoords())
                self.setCoords(np.copy(d))
                e = self.eval('enthalpy')
                self.setCoords(np.copy(a))

            else:
                e = self.eval('enthalpy')

        return e

    def minimize(self):

        # print('minimization')

        if self.press != 999.:
        #     print(self.eval(press))
        #     print(self.energy())
        #     print ('')

            self.atoms.command('fix 1 all box/relax iso ' + str(self.press))
            def pr(l):
                # i = self.atoms.extract_global("boxxhi", 1) - self.atoms.extract_global("boxxlo", 1)
                dxl = 1
                self.atoms.command('minimize 1e-9 1e-9 100000 1000000')
                a = ( self.eval('press') - self.press )

                while np.abs(a) > self.P_THRESH:
                    # print(self.eval('density'))
                    # print(self.eval('press'))
                    # print(self.eval('vol'))
                    # print(self.energy())
                    # print('')
                    L = self.atoms.extract_global("boxxhi", 1) - self.atoms.extract_global("boxxlo", 1)
                    dL = dxl*a*6.3e-7
                    if dL > 0.01:
                        dL = 0.01
                    elif dL < -0.01:
                        dL = -0.01
                    L+=dL

                    scale = L/(self.atoms.extract_global("boxxhi", 1) - self.atoms.extract_global("boxxlo", 1))

                    cmd = 'change_box all x scale ' + str( scale ) + ' y scale ' + str( scale ) + ' z scale ' + str( scale ) + ' remap'
                    self.atoms.command(cmd)
                    self.atoms.command('minimize 1e-9 1e-9 100000 100000')
                    # print(self.eval('press'))

                    a = (self.eval('press') - self.press)

            # self.atoms.command('minimize 1e-9 1e-9 100000 100000')
            l = self.getCoords()[-1]
            pr(l)
            # print('Minimized:  ' + str(self.eval('density')))

        else:
            self.atoms.command("minimize 1e-9 1e-9 1000000 10000000") ### Pressure Issues

        # print('/Minimization')
    def getForces(self):
        self.atoms.command('run 0')
        _data = np.array(self.atoms.gather_atoms("f", 1, 3), dtype=ct.c_double)
        natoms = self.atoms.get_natoms()
        f = (_data.reshape(natoms, 3))

        return np.asarray(f).flatten()

    def hessian (self):
        self.atoms.command('compute r all partialHessian 1e-2')
        self.atoms.command('fix out all ave/time 1 1 1 c_r[*] mode vector file new_data' + str(comm.Get_rank()) + '.out')

        self.atoms.command('run 0')
        self.atoms.command('uncompute r')
        self.atoms.command('unfix out')


        hess = pd.read_csv('new_data' + str(comm.Get_rank()) + '.out',sep=" ", skiprows=3 , index_col=0).values.reshape((self.atoms.get_natoms()*3,self.atoms.get_natoms()*3))
        hess = 0.5*(hess+(hess.T))

        a,b = np.linalg.eig(hess)

        a = np.real(a)
        b = np.real(b)

        return a,b

    def pressure(self):
        if self.press != 999.:
            self.center()

            def pr(l):
                x = self.getCoords()
                x[-1] = l
                self.setCoords(x)
                return self.eval('press') - self.press

            l = self.getCoords()[-1]
            z = scipy.optimize.newton(pr,l ,tol=1e-4)
            if np.abs(z - l ) > self.lStep:
                if l-z > self.lStep:
                    z = l - self.lStep
                else:
                    z = l + self.lStep
            pr(z)

    def distance(self,data):
        self.center()

        i = np.copy(data)
        Lx = self.atoms.extract_global("boxxhi", 1) - self.atoms.extract_global("boxxlo", 1)
        Ly = self.atoms.extract_global("boxyhi", 1) - self.atoms.extract_global("boxylo", 1)
        Lz = self.atoms.extract_global("boxzhi", 1) - self.atoms.extract_global("boxzlo", 1)

        a = self.getCoords()

        delta = (i-a)
        delta = reduce(delta,Lx,Ly,Lz)
        d = np.reshape(delta[:-1], (self.atoms.get_natoms(),3) ).T

        distances = np.sum(d**2,axis=0)**0.5

        return np.max(distances),np.sum(distances)

    def findTransition(self,n,energyThresh, iEnergy, dx, maxTime=25, mThresh=0.01):
        if comm.Get_rank() == 0:
            print('EVF: FThresh: ' + str(self.F_THRESH) + ' BThresh: ' + str(self.B_THRESH) + ' Stepsize: ' + str(self.stepSize))

        initial = self.getCoords()
        atoms = np.copy(initial)
        prevStep = np.zeros(len(atoms))
        stime = time.time()

        iL = self.atoms.extract_global("boxxhi", 1) - self.atoms.extract_global("boxxlo", 1)

        curve =[]
        newton=[]

        t = 0
        pV = 0.
        curv = 0

        # if round(iEnergy,3) != round(self.energy(),3):
        #     print ('err\n\n')
        #     exit()

        while t < maxTime:
            xL = self.atoms.extract_global("boxxhi", 1) - self.atoms.extract_global("boxxlo", 1)
            if (np.abs(self.energy()-iEnergy) > np.abs(iEnergy)) or np.abs(xL - iL) > iL or self.energy() > 0.:
                print ( "Unstable Step | Returning to Start\n")
                return False, self.energy(), newton, curv

            startTime = time.time()
            # dx = self.stepSize

            a,b = self.hessian()
            if t ==0:
                curv = a[n]
                # if a[n] > 1:
                #     dx /= np.sqrt(a[n])

            grad = -self.getForces()
            curve.append(a[n])
            newton.append(-grad[n])

            if np.sum(a < -self.B_THRESH ) == 1 and t > 1 and np.abs(grad[n]) < self.F_THRESH and np.abs(pV) < self.P_THRESH and self.energy() > iEnergy:
                bar = self.energy()-iEnergy
                tst = self.getCoords()
                print (str(comm.Get_rank()) + ':Step ' + str(t) + ': Time: ' + str(round(time.time()-startTime,3)) + 's  Energy: ' + str(round(self.energy(),7))
                        + ' Force: ' + str(round(-grad[n],3)) + ' Barrier: ' + str(bar))
                tPoint = np.copy(atoms)

                if np.sum(tot**2)**0.5 == 0:
                    print ('Error | Failed.')
                    return False,self.energy(),newton,curv

                normalize = 10.*dx*( tot / (np.sum(tot**2)**0.5) ) ### Collin Reinturpt this!

                d1 = np.sum(((tPoint + normalize)-initial)**2)**0.5
                d2 = np.sum(((tPoint - normalize)-initial)**2)**0.5
                if d1 > d2:
                    atoms = tPoint + normalize
                else:
                    atoms = tPoint - normalize

                self.setCoords(atoms)
                self.minimize()

                U = iEnergy + bar
                maxD, netDisplacment = self.distance(initial)
                if (maxD < mThresh or bar < energyThresh or self.energy() > 0 or np.abs(self.energy()-iEnergy) > np.abs(iEnergy) ): ### Changed line
                    print ('')
                    print ('Failure to Converge   | Time: ' + str(round(time.time()-stime,3)) + 's')
                    print ('Barrier               | '+ str(bar))
                    print ('Energies              | OE: ' + str(round(iEnergy,3)) + ' NE: ' + str(round(self.energy(),3)) + ' TP: ' + str(round(iEnergy+bar,3)) )
                    print ('Statistical Analysis  | Total Displacment: ' + str(round(netDisplacment,3)) + ' Max Atom Displacment: ' + str(round(maxD,3)))
                    # print ('Failures              | C: ' + str( not np.sum(a < -self.B_THRESH ) > 1 ) + ' T: ' + str( not np.abs(U-iEnergy) < energyThresh) + ' E: ' + str(not np.abs(U-iEnergy) > 100.) )
                    # print ('Failures              | E: ' + str( not self.energy() > 0 ) + ' E: ' + str( not self.energy() < -1.e10) + ' E! ' + str( not self.energy() > 0 ))
                    print ('')
                    self.setCoords(initial)
                    return False, self.energy(), newton, curv #-1, tPoint
                else:
                    if self.energy() > U or iEnergy > U:
                        e = self.getCoords()
                        energy, tPoint = self.NEB(initial, self.getCoords())
                        self.setCoords(e)

                    print ('')
                    print ('Transition Point Found| Energy: ' + str(round(U,3)) + ' Barrier: ' + str( round ( U-iEnergy,3 )) + ' Density: ' + str(round( sim.eval('density') , 5 )))
                    print ('New Inherent Structure| Old Energy: ' + str(round(iEnergy,3)) + ' New Energy: ' + str(round(self.energy(),3)) )
                    print ('Statistical Analysis  | Total Displacment: ' + str(round(netDisplacment,3)) + ' Max Atom Displacment: ' + str(round(maxD,3)))
                    print ('Time                  | Time: ' + str(round(time.time()-stime,3)) + 's [Rank Processor]: ' + str( comm.Get_rank() ))
                    print ('')

                    return True, U, tst, curv
            else:
                # for z in range(0,len(a)):
                #     F = np.sum(grad[z]*b[:,z])
                #     if z == n:
                #         tot = tot + (dx*b[:,z])
                #     else:
                #         tot = tot - (dx*b[:,z])*F
                # if self.energy(tot+atoms) < self.energy():
                #     tot = -tot

                tot = np.zeros(self.atoms.get_natoms()*3 + 1)
                for z in range(0,len(a)):
                    lm = 0 ### All non-n values are 0
                    F = np.sum(grad[z]*b[:,z])
                    step = np.zeros(self.atoms.get_natoms()*3)

                    if a[z] < -self.B_THRESH:
                        if z != n: ### Minmize in direction
                            if np.abs(F) > self.F_THRESH:
                                lm = a[z] - np.abs(F / dx )
                                step = b[:,z] * F / (lm - a[z])
                            else:
                                step = dx*b[:,z]
                        else: ### Maximize in direction
                            if np.abs(F) > self.F_THRESH:
                                step = b[:,z] * F/ (lm - a[z])
                            else:
                                step = 0*step

                    elif a[z] > self.B_THRESH:
                        if z != n: ### Minimize in direction
                            if np.abs(F) < self.F_THRESH:
                                step = 0*step
                            else:
                                step = b[:,z] * F/(0-a[z])
                        else: ### Maximizing in this direction
                            if np.abs(F) < self.F_THRESH: ### This is the problem
                                step = b[:,z] * dx
                            else:
                                lm = a[z] + np.abs(F / dx)
                                step = b[:,z] * F / (lm - a[z])
                    else:
                        if z != n: #### Minimize
                            if F < -self.F_THRESH:
                                step = dx*b[:,z]*np.abs(F)#/self.B_THRESH
                            elif F > self.F_THRESH:
                                step = -dx*b[:,z]*np.abs(F)#/self.B_THRESH
                            else:
                                step = 0*step
                        else: ### Maximize
                            if F < -self.F_THRESH:
                                step = -dx*b[:,n]*np.abs(F)#/self.B_THRESH
                            elif F > self.F_THRESH:
                                step = dx*b[:,n]*np.abs(F)#/self.B_THRESH
                            else:
                                n = random.randrange(len(grad))
                                step = 0*step
                    tot = tot + np.append(step,0.)

                atoms = atoms + tot
                atoms[-1] = self.atoms.extract_global("boxxhi", 1) - self.atoms.extract_global("boxxlo", 1)
                prevStep = tot
                self.setCoords(atoms)

                if self.press != 999.:
                    self.pressure()
                    pV = np.abs(self.press - self.eval('press'))

                t+=1
                print (str(comm.Get_rank()) + ':Step ' + str(t) + ': Time: ' + str(round(time.time()-startTime,3)) + 's  Enthalpy: ' + str(round(self.energy(),3)) + ' Box Length: ' + str(round(self.atoms.extract_global("boxxhi", 1) - self.atoms.extract_global("boxxlo", 1),3)) + ' Force: ' + str(round(-grad[n],3)) )

        self.setCoords(initial)
        print('Timed Out of Solution.')
        return False,self.energy(),newton,curv

    def exploreLandscape(self,num,timeLimit,energyThresh,fthresh=0.,bthresh=0.,ss=0.,custom=0.):

        if fthresh != 0:
            self.F_THRESH = fthresh
        if bthresh != 0:
            self.B_THRESH = bthresh
        if ss != 0:
            self.stepSize = ss

        start = time.time()
        # self.minimize()
        suc = self.n
        curr = self.n
        skip = False
        curv = 0.
        prev = self.energy()

        min = open('min.'+str(comm.Get_rank())+'.dat','a')
        trd = open('tsd.'+str(comm.Get_rank())+'.dat','a')
        vol = open('vol.'+str(comm.Get_rank())+'.dat','a')
        crv = open('crv.'+str(comm.Get_rank())+'.dat','a')

        curr = self.i
        prevStruct = self.i+1
        self.n = curr

        while len(glob.glob('./kMC.*.xyz')) < self.i + num + comm.Get_size() and (time.time() - start)/3600. < timeLimit :

            # self.n = len(glob.glob('./kMC.*.xyz'))+comm.Get_rank()

            # pnum = self.n #len(glob.glob('./kMC.*.xyz')) + comm.Get_rank()

            begin = time.time()
            tEnergy = 0

            a,b = self.hessian()

            value = np.sort(a)
            F = False
            at = 1
            n=6

            if self.i == curr:
                n += comm.Get_rank()
                self.n += comm.Get_rank()+1
            else:
                self.n = len(glob.glob('./kMC.*.xyz'))
                # print(glob.glob('./kMC.*.xyz'))
                # print(self.n)
                # exit()
            pnum = self.n+1
            self.write(pnum)

            startLocation = np.copy(self.getCoords())
            iE = self.energy()

            while not F and len(glob.glob('./kMC.*.xyz')) < self.i + num + comm.Get_size() and (time.time() - start)/3600. < timeLimit :
                self.load(pnum)

                if value[n] < self.B_THRESH+1.:
                    n=n+1

                loc = np.argmin( np.abs(value[n] - a) )

                print ( 'Exploring Branch: ' + str(self.n) + ' Attempt: '+ str(at)+ ' Direction: ' + str( loc ) + ' Current Energy: ' + str(self.energy()) +  ' Target: ' + str(self.i + num -1) + ' Rank: ' + str(comm.Get_rank()) )

                if (self.freqMD==0 or (self.n)%self.freqMD != 0) and (self.freqSV == 0 or (self.n)%self.freqSV != 0) :
                    F, tEnergy, transition, curv = self.findTransition(loc,energyThresh,prev,self.stepSize) ### Search transition
                elif ( self.freqMD!=0 and (self.n)%self.freqMD == 0 ):
                    F, tEnergy, transition, curv = self.MD(prevStruct,custom=custom)
                else:
                    # n = random.randrange(3*self.atoms.get_natoms())
                    # if value[n] < self.B_THRESH+1.:
                    #     n=n+1
                    F, tEnergy, transition, curv = self.shove(n)

                at += 1 ### Iterating the attempts
                n+=1

                if at == self.tolerance:
                    while not F:
                        self.setCoords(startLocation)
                        F, tEnergy, transition, curv = self.MD(curr)

                if not F or iE+energyThresh > tEnergy or self.energy()+energyThresh > tEnergy or tEnergy < prev or tEnergy < self.energy():
                    F = False
                    print('|| Transition Point Did Not Meet Criteria || Restarting Search ||')

                newLoc = self.getCoords()

            if F and len(glob.glob('./kMC.*.xyz')) < self.i + num + comm.Get_size() and (time.time() - start)/3600. < timeLimit:
                # pnum = len(glob.glob('./kMC.*.xyz'))-1
                # print('WRITING OUT FINAL '+ str(pnum))
                self.write(pnum)

                ne = self.energy(newLoc)
                ol = self.energy(startLocation)

                self.setCoords(np.copy(newLoc))
                self.writeTS(transition,pnum)
                trd.write( str( tEnergy ) + '\t' + str(comm.Get_rank()) + '\t' + str(n) + '\t' + str(int(prevStruct)) + '\t' + str(int(pnum)) + '\n' )
                min.write( str(pnum) + '\t' + str( ne ) + '\t' + str( comm.Get_rank() ) + '\n' )
                vol.write( str(pnum) + '\t' + str( self.eval('vol') ) + '\n' )
                crv.write( str(pnum) + '\t' + str( curv ) + '\t' + str( comm.Get_rank() ) + '\n' )
                prevStruct = pnum

                trd.flush()
                min.flush()
                vol.flush()
                crv.flush()

                prev = self.energy(newLoc)

                # if comm.Get_rank() == 0:
                print ('\n')
                print (str(comm.Get_rank()) + ': Total Time for Transition: ' + str((time.time() - begin)/60.) + ' min')
                print (str(comm.Get_rank()) + ': Total Time since beginning: ' + str((time.time() - start)/3600.) + ' hrs')
                print ('\n')

                begin = time.time()
                self.setCoords(newLoc)

                suc +=1
                curr = len(glob.glob('./kMC.*.xyz'))

        trd.close()
        min.close()
        vol.close()
        crv.close()

def init(input):
    lmp = lammps(cmdargs=['-screen','screen.'+str(comm.Get_rank())+'dat'],comm=sub_comm)

    lmp.command("atom_modify map array sort 0 0.0")
    cmd = 'log none'
    lmp.command(cmd)
    lmp.file(input)

    lmp.command('run 0');

    return lmp

try:
    from lammps import PyLammps, lammps
except:
    print('')
    print('PyLammps not aviable: Please install PyLammpys, Numpy, Scipy, Matplotlib, Pandas, Glob')
    print('\tPyLammps also must be built with the Lammps Hessian Project: ')
    print('\thttps://bitbucket.org/numericalsolutions/lammps-hessian/src/master/')
    print('')
    exit()

file = ''
num = 100
timeLimit = 100
thresh = 0.0001
help = False
center = False
press = 999.
n = 0
freqMD = 0
freqSV = 0

custom = 0
fthresh = 0.
bthresh = 0.
stepSize = 0.

for a in range(1,len(sys.argv),2):
    if sys.argv[a] == '-stepsize':
        stepSize = float(sys.argv[a+1])
    if sys.argv[a] == '-force':
        fthresh = float(sys.argv[a+1])
    if sys.argv[a] == '-curve':
        bthresh = float(sys.argv[a+1])
    if sys.argv[a] == '-file':
        file = sys.argv[a+1]
    elif sys.argv[a] == '-press':
        press = float(sys.argv[a+1])
    elif sys.argv[a] == '-num':
        num = int(sys.argv[a+1])
    elif sys.argv[a] == '-time':
        timeLimit = float(sys.argv[a+1])
    elif sys.argv[a] == '-thresh':
        thresh = float(sys.argv[a+1])
    elif sys.argv[a] == '-help' or sys.argv[a] == '-h':
        help = True
    elif sys.argv[a] == '-center':
        center=False
    elif sys.argv[a] == '-cusom':
        custom = self.argv[a+1]
    elif sys.argv[a] == '-anchor':
        n = int(sys.argv[a+1])
    elif sys.argv[a] == '-jump' or sys.argv[a] == '-md':
        freqMD = int(sys.argv[a+1])
    elif sys.argv[a] == '-shove':
        freqSV = int(sys.argv[a+1])

if file == '' or help == True:
    print('')
    print('Error: Incorrect Usage input is incorrect')
    print('\tProper Usage is:')
    print('\tpython kMCNN.py -file [input] \n\n\tOptional:\n\t[-force  [force threshold]\n\t -stepsize [Size of each step in eigenvector following]\n\t -curve  [curvature (eigen value threshold)]\n\t -press  [pressure in atmosphere]\n\t -num    [number of runs]\n\t -time   [max time in hours]\n\t -thresh [energy threshold]\n\t -center [Should the system be centered]\n\t -md     [every n steps for MD]\n\t -shove  [every n steps for ET]\n\t -custom [custom script for MD exploration] ]')
    print('')
    exit()

if comm.Get_rank()==0 or not parallel:
    # try:
    #     os.system('clear')
    # except:
    #     print('Window Not Cleared\n')

    print ('\n')
    print ('*******************************************')
    print ('             Explorer.Py                   ')
    print (' A tool for generating Material Landscapes ')
    print ('       Enthalpy and Energy Mapping         ')
    print ('Eigenvector Following Method of Exploration')
    print ('')
    print ('    By: C.J. Wilkinson and J.C. Mauro      ')
    print ('\n')
    print ('[Run Options]')

    if press != 999.:
        print ('* Designated Pressure      : ' + str(round(press,3)))
    else:
        print ('No Specified Pressure  | Will Run NVT')

    if parallel:
        print ('* Parrallel Processors     : ' + str(comm.Get_size()) )
    else:
        print ('Running in Serial | Only One Process' )

    print ('* Target Number of Basin   : ' + str(num))
    print ('* Max Time                 : ' + str(timeLimit))
    print ('* Energy Threshold         : ' + str(thresh))
    print ('* MD Exploration [step]    : ' + str(freqMD))
    print ('* Shove Exploration [step] : ' + str(freqSV))
    print ('*******************************************')
    print ('\n')

initialTime = time.time()
sim = kMCNN(init(file),press,center,n,freqMD,freqSV)
comm.Barrier()
print ('Starting Simulation on Process: ' + str(comm.Get_rank()) + '\tStarting Energy: ' + str(sim.energy()))
sim.exploreLandscape(num,timeLimit,thresh,fthresh,bthresh,stepSize,custom)
print ('Exiting Process: '+ str(comm.Get_rank() ) + ' | Total Wall Time: ' + str((time.time()- initialTime)/3600.) + ' hrs.' )
