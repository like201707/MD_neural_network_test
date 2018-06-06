import numpy as np
import Lammpstrj_reader
import sys
from numpy.linalg import norm
import time

def angle(B, A, C):
    """calculate the cos(theta)"""
    BA = A - B
    BC = C - B
    theta = np.dot(BA, BC)/(norm(BA)*norm(BC))
    return theta

def fc(Rij, Rc):
    """cutoff function"""
    if Rij > Rc:
        return 0.
    else:
        return 0.5 * (np.cos(np.pi * Rij / Rc) + 1.)

def g1(Rij, Rc):
    """type 1 of fingerprint"""
    return fc(Rij, Rc)

def g2(Rij, Rc, Rs, etha):
    """type 1 of fingerprint"""
    return np.exp(-etha * (Rij - Rs) ** 2 / Rc**2) * fc(Rij, Rc)

def g3(Rij, Rc, kappa):
    """type 3 of fingerprint"""
    return np.cos(kappa * Rij) * fc(Rij, Rc)

def g4(Rij, Rik, Rjk, Rc, Rs, zeta, lmb, theta, etha):
    """type 4 of fingerprint"""
    return (1 + lmb * theta) ** zeta * \
            np.exp(-etha * (Rij ** 2 + Rik ** 2 + Rjk ** 2)/ Rc ** 2) * \
            fc(Rij, Rc) * fc(Rik, Rc) * fc(Rjk, Rc)

def g5(Rij, Rik, Rjk, Rc, Rs, zeta, lmb, theta, etha):
    """type 4 of fingerprint"""
    return (1 + lmb * theta) ** zeta * \
            np.exp(-etha * (Rij ** 2 + Rik ** 2)) * fc(Rij, Rc) * fc(Rik, Rc)

def distance(Ri, Rj):
    RijV = Ri - Rj
    Rij = np.sqrt( RijV[0]**2 + RijV[1]**2 + RijV[2]**2 )
    return Rij

def dfcR(R, Rc):
    return -0.5 * np.pi/Rc * np.sin(np.pi*R/Rc)

def dg1(Ri, Rj, Rc):
    Rij = distance(Ri, Rj)
    return dfcR(Rij, Rc)*(Ri-Rj)/Rij


def dg2(Ri, Rj, Rc, Rs, etha):
    Rij = distance(Ri, Rj)
    return (-2*etha*(Rij-Rs)*fc(Rij, Rc)/Rc**2+dfcR(Rij, Rc))* \
            np.exp(-etha*(Rij-Rs)**2/Rc**2)*(Ri-Rj)/Rij

def dg3(Ri, Rj, Rc, kappa):
    Rij = distance(Ri, Rj)
    return (np.cos(kappa*Rij)*dg1(Rij, Rc)-kappa*np.sin(kappa*Rij)*g1(Rij, Rc))*(Ri-Rj)/Rij

def dg4(Ri, Rj, Rk, Rc, Rs, zeta, lmb, theta, etha):
    Rij = distance(Ri, Rj)
    Rik = distance(Ri, Rk)
    Rjk = distance(Rj, Rk)
    """
    dTheta = 1/(Rij*Rik)*(Rj-Ri)*(Rk-Ri)+1/(Rij*Rik)*(Rj-Ri)*(Rk-Ri)- \
            np.dot((Rj-Ri), (Rk-Ri))/(Rij**2*Rik)*(Ri-Rj)/Rij- \
            np.dot((Rj-Ri), (Rk-Ri))/(Rij*Rik**2)*(Ri-Rk)/Rik
    """
    dTheta = 1/(Rij*Rik)*(Rk-Ri)+1/(Rij*Rik)*(Rj-Ri)- \
            np.dot((Rj-Ri), (Rk-Ri))/(Rij**2*Rik)*(Ri-Rj)/Rij- \
            np.dot((Rj-Ri), (Rk-Ri))/(Rij*Rik**2)*(Ri-Rk)/Rik

    return (1+lmb*theta)**(zeta-1)*np.exp(-etha*(Rij**2+Rik**2+Rjk**2)/Rc**2)* \
            (fc(Rij, Rc)*fc(Rik, Rc)*fc(Rjk, Rc)*(lmb*zeta*dTheta-2*etha* \
            (1+lmb*theta)*2*(Ri-Rk)/Rc**2)+(1+lmb*theta)* \
            (dfcR(Rij, Rc)*(Ri-Rj)/Rij*fc(Rik, Rc)*fc(Rjk, Rc)+ \
            fc(Rij, Rc)*dfcR(Rik, Rc)*(Ri-Rk)/Rik*fc(Rjk, Rc)+ \
            fc(Rij, Rc)*fc(Rik, Rc)*dfcR(Rjk, Rc)*(Rj-Rk)/Rjk))


class FingerPrint(object):
    """class of calculating symmetry functions"""

    def __init__(self, trajectory):
        self.traj  = trajectory
        self.cut = 3.77118

    def neighbors(self, coords, BoxSize):
        """calculate neighbors"""
        n_atoms = self.traj.nAtoms
        n_frames = self.traj.nFrames
        dist = np.zeros([n_atoms, n_atoms], dtype=np.float)
        newCoords = np.zeros([n_atoms, n_atoms, 3], dtype=np.float)
        for i in range(n_atoms):
            atom1 = coords[i]
            for j in range(n_atoms):
                atom2 = coords[j]
                dr = atom2 - atom1
                if dr[0] > BoxSize[0]/2.0:
                    dr[0] -= BoxSize[0]
                elif dr[0] < -BoxSize[0]/2.0:
                    dr[0] += BoxSize[0]
                if dr[1] > BoxSize[1]/2.0:
                    dr[1] -= BoxSize[1]
                elif dr[1] < -BoxSize[1]/2.0:
                    dr[1] += BoxSize[1]
                if dr[2] > BoxSize[2]/2.0:
                    dr[2] -= BoxSize[2]
                elif dr[2] < -BoxSize[2]/2.0:
                    dr[2] += BoxSize[2]
                newCoords[i][j][0] = dr[0]
                newCoords[i][j][1] = dr[1]
                newCoords[i][j][2] = dr[2]
                dist[i][j] = np.sqrt( dr[0]**2 + dr[1]**2 + dr[2]**2 )
        return (newCoords, dist)

    def G1(self, coords, BoxSize, Rc):
        """ calculate g1 symmetry functions"""
        dist = self.neighbors(coords, BoxSize)[1]
        x, y = dist.shape
        finger = np.zeros(x).reshape(x,1)
        for i in range(x):
            for j in range(y):
                if i != j:
                    tmp = g1(dist[i][j], Rc)
                    finger[i] += tmp
        return finger

    def G2(self, coords, BoxSize, Rc, Rs, etha):
        """ calculate g2 symmetry functions"""
        dist = self.neighbors(coords, BoxSize)[1]
        x, y = dist.shape
        finger = np.zeros(x).reshape(x,1)
        for i in range(x):
            for j in range(y):
                if i != j:
                    tmp = g2(dist[i][j], Rc, Rs, etha)
                    finger[i] += tmp
        return finger

    def G3(self, coords, BoxSize, Rc, kappa):
        """ calculate g3 symmetry functions"""
        dist = self.neighbors(coords, BoxSize)[1]
        x, y = dist.shape
        finger = np.zeros(x).reshape(x,1)
        for i in range(x):
            for j in range(y):
             if i != j:
                    tmp = g3(dist[i][j], Rc, kappa)
                    finger[i] += tmp
        return finger


    def G4(self, coords, newCoords, BoxSize, Rc, Rs, zeta, lmb, etha):
        """ calculate g4 symmetry functions"""
        dist = self.neighbors(coords, BoxSize)[1]
        x, y = dist.shape
        finger = np.zeros(x).reshape(x,1)
        for i in range(x):
            for j in range(x):
                if dist[i][j] <= Rc:
                    for k in range(x):
                        if i != j and j != k and i != k:
                            if dist[i][k] <= Rc:
                                theta = angle((0., 0., 0.), newCoords[i][j], newCoords[i][k])
                                tmp = g4(dist[i][j],
                                         dist[i][k],
                                         dist[j][k],
                                         Rc, Rs, zeta, lmb,
                                         theta, etha)
                                finger[i] += tmp
        return 2 ** (1 - zeta) * finger

    def G5(self, coords, BoxSize, Rc, Rs, zeta, lmb, etha):
        """ calculate g5 symmetry functions"""
        dist = self.neighbors(coords, BoxSize)[1]
        x, y = dist.shape
        finger = np.zeros(x).reshape(x,1)
        for i in range(x):
            for j in range(x):
                if dist[i][j] <= Rc:
                    for k in range(x):
                        if i != j and j != k and i != k:
                            if dist[i][k] <= Rc:
                                theta = angle(coords[i], coords[j], coords[k])

                                tmp = g5(dist[i][j], dist[i][k], dist[j][k],
                                         Rc, Rs, zeta, lmb, theta, etha)

                                finger[i] += tmp
        return 2 ** (1 - zeta) * finger

class Gprime(FingerPrint):


    def dG2(self, coords, newCoods, BoxSize, Rc, Rs, etha):
        newCoords = self.neighbors(coords, BoxSize)[0]
        dist = self.neighbors(coords, BoxSize)[1]
        x, y = dist.shape
        df = np.zeros((x,3))
        for i in range(x):
            for j in range(y):
                if dist[i][j] <= Rc:
                    if i != j:
                        tmp = dg2((0,0,0), newCoords[i][j], Rc, Rs, etha)
                        df[i] += tmp
        return df

    def dG4(self, coords, newCoods, BoxSize, Rc, Rs, zeta, lmb, etha):
        newCoords = self.neighbors(coords, BoxSize)[0]
        dist = self.neighbors(coords, BoxSize)[1]
        x, y = dist.shape
        df = np.zeros((x,3))
        for i in range(x):
            for j in range(x):
                if dist[i][j] <= Rc:
                    for k in range(x):
                        if i != j and j != k and i != k:
                            if dist[i][k] <= Rc:
                                theta = angle((0., 0., 0.), newCoords[i][j], newCoords[i][k])
                                tmp = dg4((0. ,0. ,0.), newCoords[i][j], newCoords[i][k], \
                                Rc, Rs, zeta, lmb, theta, etha)
                                df[i] += tmp
        return 2 ** (1 - zeta) * df

def TrainDataPreparation(filename):
    """
    transfer Cartesian coordinates to symmetry functions input
    ready for nueral network training
    input is xyz file
    """

    a = Lammpstrj_reader.Lammpstrj_reader(filename)
    b = FingerPrint(a)
    c = Gprime(a)
    atomC = b.traj.atomC
#    Frames = b.traj.nFrames
    Frames = 1000
    Atoms = b.traj.nAtoms
    BoxSize = b.traj.box
    inputData = np.zeros((Frames, Atoms, 2))
    dG2 = np.zeros((Frames, Atoms, 3))
    dG4 = np.zeros((Frames, Atoms, 3))
    G2 = np.zeros((Frames, Atoms))
    G4 = np.zeros((Frames, Atoms))
    for i in range(Frames):
        newCoords = b.neighbors(atomC[i], BoxSize[i])[0]
        G4[i] = np.squeeze(b.G4(atomC[i], newCoords, BoxSize[i], 3.77118, 0., 1., -1, 1))
        G2[i] = np.squeeze(b.G2(atomC[i], BoxSize[i], 3.77118, 0., 1.))
        dG4[i] = np.squeeze(c.dG4(atomC[i], newCoords, BoxSize[i], 3.77118, 0., 1., -1, 1))
        dG2[i] = np.squeeze(c.dG2(atomC[i], newCoords, BoxSize[i], 3.77118, 0., 1.))
        for j in range(Atoms):
            inputData[i][j][0] = G2[i][j]
            inputData[i][j][1] = G4[i][j]
        #show progress
        sys.stdout.write("\r%2d %% ---data prepare complete---" % ((float(i)/Frames)*100))
        sys.stdout.flush()
    np.save('dG2.npy', dG2)
    np.save('dG4.npy', dG4)
    print ("\n\nWriting input data ...")
    with open (str(filename)+"Input.dat", "w") as g:
        for i in range(Frames):
            g.write(str(Atoms)+"\n")
            g.write("Frame number: "+str(i+1)+"\n")
            for j in range(Atoms):
                g.write('{0} {1} {2}\n'.format(inputData[i][j][0], inputData[i][j][1], 0))
    print ("Done! ..............\n\n")
    return inputData



if __name__ == '__main__':
#    TrainDataPreparation('Si.lammpstrj')
#    F = Lammpstrj_reader.Lammpstrj_reader("Si.lammpstrj").atomF
#    np.save('F.npy', F)
#    a = Lammpstrj_reader.Lammpstrj_reader("Si.lammpstrj")
#    b = FingerPrint(a)
#    c = Gprime(a)
#    coords = b.traj.atomC[0]
#    BoxSize = b.traj.box[0]
#    newCoords = b.neighbors(coords, BoxSize)[0]
#    t_start = time.clock()
#    testDG2 = c.dG2(coords, newCoords, 4., 0., 1.)
#    testDG4 = c.dG4(coords, newCoords, 4., 0., 1., -1, 1.)
#    testG4 =  b.G4(coords, newCoords, 4., 0., 1., -1, 1.)
#    testG2 = b.G2(b.traj.atomC[0], 4., 0., 1.)
#    testDist = b.neighbors(b.traj.atomC[0])[1]
#    t_end = time.clock()
#    print('\nTotal elapsed time = {} seconds.'.format(t_end - t_start))
#    np.save('dG4.npy', testDG4)
