# -*- coding: utf-8 -*-

import numpy as np
import sys
import time


class Lammpstrj_reader(object):
    # format lammps dump trjectory
    def __init__(self, filename):
        self.filename = filename
        self.nAtoms = self.n_atoms()
        self.nFrames = self.n_frame()
        self.file = open(self.filename, "r")
        self.atomC = np.empty([self.nFrames, self.nAtoms, 3], dtype=np.float)
        self.atomF = np.empty([self.nFrames, self.nAtoms, 3], dtype=np.float)
        self.box = np.empty([self.nFrames, 3], dtype=np.float)
        print ("-------------Initializing-------------------")
        print ("""\
                         ._ o o
                        \_`-)|_
                    ,""       ~
                  ,"  ## |   ಠ ಠ.
                ," ##   ,-\__    `.
              ,"       /     `--._;)
            ,"     ## /
          ,"   ##    /
                    """)
        print ("-------------Initializing-------------------")
        print ("\n\nReading lammpstrj file ...")
        starttime = time.time()
        self.read_all_frames()
        print ("Done! ..............\n\n")
        endtime = time.time()
        print ('Reading file total time:%f') % (endtime-starttime)

    def n_atoms(self):
        with open(self.filename) as f:
            for i in range(3):
                f.readline()
            n = f.readline()
            return int(n)

    def n_frame(self):
        linesPerFrame = self.nAtoms + 9
        count = 0
        offsets = list()
        with open(self.filename) as f:
            line = True
            while line:
                if not count % linesPerFrame:
                    offsets.append(f.tell())
                line = f.readline()
                count += 1
        n_frames = int(count/linesPerFrame)
        self.offsets = offsets
        return n_frames

    def read_all_frames(self):
        for frame in range(self.nFrames):
            self.file.seek(self.offsets[frame])
            self.read_one_frame(frame)
            sys.stdout.write("\r%2d %% -----loading-----" % ((float(frame)/self.nFrames)*100))
            sys.stdout.flush()

    def read_one_frame(self, frame):
        f = self.file
        for i in range(5):
            f.readline()
        for i in range(3):
            line = f.readline().split()
            self.box[frame][i] = np.float(line[1]) - np.float(line[0])
        f.readline()
        for i in range(self.nAtoms):
            lineinfo = f.readline().split()
            self.atomC[frame][i] = np.array(lineinfo[1: 4], dtype=np.float)
            self.atomF[frame][i] = np.array(lineinfo[4: 7], dtype=np.float)

    def close(self):
        """Close trajectory file if it was open."""
        if self.file is None:
            return
        self.file.close()
        self.file = None
