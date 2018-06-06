import numpy as np


class XYZReader(object):
    """Reads from an XYZ file
    """
    format = "XYZ"
    # these are assumed!

    def __init__(self, filename, **kwargs):
        self.filename = filename
        self.xyzfile = open(self.filename, "r")
        self.nAtoms = self.n_atoms()
        self.nFrames = self.n_frames()
        self.atomC = np.empty([self.nFrames, self.nAtoms, 3], dtype=np.float)
        self.atomF = np.empty([self.nFrames, self.nAtoms, 3], dtype=np.float)
#        self.box = np.empty([self.nFrames, 3], dtype=np.float)
        self.energy = np.empty([self.nFrames, 1], dtype=np.float)
        print ("------------------------------------")
        print ("----------Initialization------------")
        print ("------------------------------------")
        print ("\n\nReading xyz file ...")
        self._read_all_frames()
        print ("Done! ..............\n\n")

    def n_atoms(self):
        """number of atoms in a frame"""
        with open(self.filename, 'r') as f:
            n = f.readline()
        return int(n)

    def n_frames(self):
        try:
            return self._read_xyz_n_frames()
        except IOError:
            return 0

    def _read_xyz_n_frames(self):
        # the number of lines in the XYZ file will be 2 greater than the
        # number of atoms
        linesPerFrame = self.nAtoms + 2
        counter = 0
        offsets = []

        with open(self.filename, 'r') as f:
            line = True
            while line:
                if not counter % linesPerFrame:
                    offsets.append(f.tell())
                line = f.readline()
                counter += 1

        # need to check this is an integer!
        n_frames = int(counter / linesPerFrame)
        self._offsets = offsets
        return n_frames

    def _read_all_frames(self):
        for frame in range(self.nFrames):
            self.xyzfile.seek(self._offsets[frame])
            self._read_next_timestep(frame)

    def _read_next_timestep(self, frame):
        f = self.xyzfile
        f.readline()
        f.readline()
#        line = f.readline().split()
#        self.box[frame] = np.array(list(map(float, line[0:3])), dtype=np.float)
#        self.energy[frame] = np.array(float(line[-1]), dtype=np.float)
        for i in range(self.nAtoms):
            lineInfo = f.readline().split()
            # atom rx ry rz vx vy vz
            self.atomC[frame][i] = np.array(lineInfo[0:3], dtype=np.float)
            try:
                self.atomF[frame][i] = np.array(lineInfo[4:7],
                                                dtype=np.float)
            except ValueError:
                pass

    def close(self):
        """Close xyz trajectory file if it was open."""
        if self.xyzfile is None:
            return
        self.xyzfile.close()
        self.xyzfile = None
