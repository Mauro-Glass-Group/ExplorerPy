import numpy as np
import glob
import os

a = []
b = []
c = []
t = []

for d in range( len(glob.glob('min.*.dat'))):
    if d == 0:
        a = np.loadtxt('min.'+str(d)+'.dat',comments='#')
        b = np.loadtxt('vol.'+str(d)+'.dat',comments='#')
        c = np.loadtxt('crv.'+str(d)+'.dat',comments='#')
        t = np.loadtxt('tsd.'+str(d)+'.dat',comments='#')

    else:
        aT = np.loadtxt('min.'+str(d)+'.dat',comments='#')
        bT = np.loadtxt('vol.'+str(d)+'.dat',comments='#')
        cT = np.loadtxt('crv.'+str(d)+'.dat',comments='#')
        tT = np.loadtxt('tsd.'+str(d)+'.dat',comments='#')

        a = np.concatenate([a,aT])
        b = np.concatenate([b,bT])
        c = np.concatenate([c,cT])
        t = np.concatenate([t,tT])


v = open('vol.dat','w')
r = open('crv.dat','w')
s = open('tsd.dat','w')

a = a[a[:,0].argsort()]

for k in range( len(t[:,3]) ):
    if t[k,3] == 1:
        index = np.where( (t[k,3] == a[:,0]) )[0]
    else:
        index = np.where( (t[k,3] == a[:,0]) & (t[k,1] == a[:,2]) )[0]
    if len(index) > 2:
        exit()

    t[k,3] = index[0]+1
    if t[k,4] == 1:
        index = np.where( (t[k,4] == a[:,0]) )[0]
    else:
        index = np.where( (t[k,4] == a[:,0]) & (t[k,1] == a[:,2]) )[0]
    if len(index) > 2:
        print(index)
        exit()
    t[k,4] = index[0]+1

# exit()
for j in range(len(a[:,2])):
    if a[j,0] != j+1:

        a[j,0] = j+1
        b[j,0] = j+1

    try:
        cmd = 'mv kMC.' + str(j+1) + '.' + str( int(a[j,2]) ) + '.xyz kMC.' + str(j+1) + '.xyz'
        os.system(cmd)
    except:
        pass

try:
    os.system('rm kMC.*.*.xyz')
except:
    pass

for l in range(len(b[:,0])):
    v.write(str(l+1) + ' ' + str(b[l,1]) + '\n')

for l in range(len(t[:,0])):
    r.write( str(int(c[l,0])) + ' ' + str(c[l,1]) + '\n')
    if t[l,0] < a[ int(t[l,3])-1 , 1] or t[l,0] < a[ int(t[l,4])-1 , 1] :
        t[l,0] = np.max([a[ int(t[l,3])-1 , 1],a[ int(t[l,4])-1 , 1]])

    msg = str(t[l,0])
    for m in range(4):
        msg += '\t' + str( int(t[l,1+m]) )
    msg += '\n'
    s.write(msg)


np.savetxt('minsort.dat',a[:,1])

s.close()
v.close()
r.close()
