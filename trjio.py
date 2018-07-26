"""trjio.py - DCD trajectory IO, manipulation and alignment"""

__package__ = "xstalpy"
__author__ = "James Gowdy" 
__email__ = "jamesgowdy1988@gmail.com"

import copy
import os
import struct
import numpy as np

from pdbio import ParsePDB, WritePDB
from progress import ProgressBar

class BinaryFile:

    def open(self, binfile):
        self._instream = open(binfile, mode='rb')
        self._close = self.instream.close
        self._seek = self.instream.seek
        self._read = self.instream.read
        self._tell = self.instream.tell

    def _read_data(self, fmt):
        size = struct.calcsize(fmt)
        buff = self.instream.read(size)
        if buff:
            data = struct.unpack(fmt, buff)
            return data # always a tuple
        else:
            return ['EOF']

################################################################################

# CHARMM DCD trajectory

################################################################################

class ParseDCD(BinaryFile, object):

    def __init__(self, dcd=None, ref=None):

        self.header = {}            # CHARMM DCD header
        self.data = []              # coordinate trajectory
        self.alignment = None       # alignment data (rmsds and euler angles)
        self._file = dcd            # file name
        self._aligned = False       # flag indicating if align() has been called
        self._calc_optrot = self._calc_optrot_kabsch

        if dcd:
            self.open(dcd)
            self.parse_header()
            self.parse_data()
            self._close()

        self.ref = None
        self._reftype = None
        self.set_ref(ref)

    #
    # methods to read and parse the DCD trajectory
    #

    def open(self, dcd):

        self._file = dcd
        self._instream = open(dcd, mode='rb')
        self._close = self._instream.close
        self._seek = self._instream.seek
        self._read = self._instream.read
        self._tell = self._instream.tell

        # get endianess of the machine used to write the trajectory
        byte = self._read(4)
        if byte[0] == 'T':
            self.end = '<'
        elif byte[-1] == 'T':
            self.end = '>'
        else:
            raise RuntimeError('DCD file has unrecognised magic number')

    def parse_header(self):

        # parse the header values from binary
        header = []
        self._seek(4)
        data = self._read_data(self.end+'cccc')
        header.append(''.join(data))
        data = self._read_data(self.end+'iiiiiiiiifiiiiiiiiiiiii')
        header.extend(list(data))
        ntitl = header[-1]
        fmt = self.end+'c'*80
        data = [''.join(self._read_data(fmt)) for i in range(ntitl)]
        header.append(data)
        data = self._read_data(self.end+'iiii')
        header.extend(data)

        keys = ['type','nset','istrt','nsavc','time','','','','dof','fixed',
                'delta','xstal','varpbc','','','','','','','fixed','chrmv',
                'tail','head','ntitl','titles','tail','head','nato','tail']

        self.header = {k: v for k, v in zip(keys, header) if k != ''}

        # covert the internal AKMA time units to femtoseconds
        convert_akma2si_time = lambda dt: (dt*488.8780249+0.5)//10.0
        self.header['delta'] = convert_akma2si_time(self.header['delta'])
        return self._tell()

    def parse_data(self, nframes=None, natoms=None, ntitles=None, offset=None):

        if isinstance(self.data, np.ndarray):
            self.data = self.data.tolist()

        # set defaults for number of atoms, frames, titles and byte offset
        if natoms is None:
            natoms = self.header.get('nato', 0)
        if nframes is None:
            nframes = self.header.get('nset', 0)
        if ntitles is None:
            ntitles = self.header.get('ntitl', 2)
        if offset is None:
            offset = 116 + (80*ntitles)

        # seek the start of the data
        self._seek(offset) 
        fmt = self.end + 'f'*natoms
        xyz = []
        while iframe < nframes:
            if len(xyz) == 3:
                # append the previous iterations xyz-coords
                self.data.append(xyz)
                xyz = []
            head, = self._read_data(self.end + 'i')
            if head == 'EOF':
                # no new data was read, then we've reached the end of the file
                break
            elif head == 4*natoms:
                # these are either x, y or z coordinates
                data = self._read_data(fmt)
                xyz.append(data)
            else:
                # its not a coordinate record (probs xstal), so we ignore
                self._read(head)
            # double check size of previous record
            tail, = self._read_data(self.end + 'i')
            assert head == tail

        self.data = np.array(self.data)
        return self._tell()

    #
    # methods to access, analyse and manipulate DCD trajectory
    #

    def set_ref(self, ref, check=False):
        # Will prefferentially set: 
        #  (1) passed PDB file or passed [x, y, z];
        #  (2) first frame of the trajectory if not already set
        if isinstance(ref, ParsePDB):
            self.ref = ref
            self._reftype = 'PDB({})'.format(ref._file or ref._code)
        elif ref is not None:
            x, y, z = ref
            self.ref = ParsePDB(str(WritePDB(x=x, y=y, z=z, n=[0]*len(x))))
            self._reftype = 'XYZ'
        elif self.ref is None and self.natoms:
            self.ref = self.toframe(0)
            self._reftype = 'DCD[0]'
        if check:
            assert self.natoms == len(self.ref)

    def __len__(self):
        return len(self.data)

    @property
    def natoms(self):
        return self.data.shape[2] if len(self.data) else 0

    @property
    def nframes(self):
        return self.data.shape[0]
 
    def copy(self):
        return copy.deepcopy(self)

    def selection(self, frames=None, atoms=None):
        if frames is None:
            frames = slice(None)
        if atoms is None:
            atoms = slice(None)
        new = self.copy()
        new.data = self.data[frames, :, atoms]
        new.set_ref(self.ref[atoms])
        return new

    def __getitem__(self, i):
        return self.selection(frames=i)

    def toframe(self, i):
        new = self.ref.copy() if isinstance(self.ref, ParsePDB) else ParsePDB('')
        new.x, new.y, new.z = self.data[i]
        return new

    def split_asus(self, ref=None):
        self.set_ref(ref)
        n = len(self.ref._asu)
        nsym = len(self.ref.R)
        trjs = [self.selection(range(n*i, n*(i+1))) for i in range(nsym)]
        return trjs

    def __repr__(self):
        fmt = 'ParseDCD(file={0._file:}, nframes={0.nframes:}, natoms={0.natoms:},'
        fmt += ' nset={nset:}, nsavc={nsavc:}, time={time:}, nato={nato:},'
        fmt += ' xstal={xstal:}, varpbc={varpbc:}, ref={0._reftype:},'
        fmt += ' aligned={0._aligned:})'
        kws = dict(nset='', nsavc='', time='', nato='', xstal='', varpbc='')
        kws.update(self.header)
        return fmt.format(self, **kws)

    def _calc_rmsd(self, X, Y):
        d = X - Y
        natoms = X.shape[-1]*1.0
        return np.sqrt(np.sum(d*d, axis=1)/natoms)

    def _calc_euler(self, R):
        thetax = np.arctan2(R[2, 1], R[2, 2])
        sq = np.sqrt(R[2, 1]*R[2, 1] + R[2, 2]*R[2, 2])
        thetay = np.arctan2(-R[2, 0], sq)
        thetaz = np.arctan2(R[1, 0], R[0, 0])
        return thetax, thetay, thetaz

    def _calc_axisangle(self, R):
        angle = np.degrees(np.arccos((np.trace(R)-1.0)/2.0))
        lam, V = np.linalg.eig(R)
        axis = V[:, 0]
        axis = np.around(axis/np.max(axis), 4)
        return angle, axis

    def _calc_optrot_kabsch(self, X0, Y0):
        # SVD implementation of Kabsch algorithm with determinant checking 
        assert X0.shape[0] == YO.shape[0] == 3, 'Transpose coordinate frames'
        XY = np.dot(X0, Y0.T)
        U, S, VT = np.linalg.svd(XY)
        I = np.eye(3)
        I[2, 2] = np.linalg.det(np.dot(U, VT))
        R = np.dot(np.dot(U, I), VT) 
        return R

    def _calc_optrot_quarternion(self, X0, Y0):
        raise NotImplementedError()

    def _calc_centroid(self, X):
        assert X.shape[0] == 3
        t = X.mean(axis=1)
        return t[:, np.newaxis]

    def centre(self, ref=None):
        ref = ref or self.ref # don't set for centre()
        new = self.copy()
        t = self.ref.mean(axis=1).reshape(1, 3, 1)
        new.data = self.data - t
        return new

    def _align(self, Ys, X, sel=None, record=True):

        pb = ProgressBar(final=len(Ys), label='align', tail_label='{i:}')
        if record:
            info = np.zeros((len(Ys), 11))

        # set defaults
        sel = sel or slice(None)

        X_sel = X[:, sel]
        t0 = self._calc_centroid(X_sel)
        X0 = X - t0                               # centre reference on origin
        X0_sel = X0[:, sel]

        for i, Y_ in enumerate(Ys):
            pb.update()            
            Y_sel = Y_[:, sel]
            t = self._calc_centroid(Y_sel)      
            Y0_ = Y_ - t                          # centre frame on origin
            Y0_sel = Y0_[:, sel]
            R = self._calc_optrot(X0_sel, Y0_sel) # get rotation matrix
            Y0 = np.dot(R, Y0_.T)                 # rotate
            Y = Y0 + t0                           # centre frame on t0
            Ys[i, :, :] = Y

            if record:
                # ref/frame comparison before alignment 
                info[i, 0] = self._calc_rmsd(X, Y_)   # all
                info[i, 2] = self._calc_rmsd(X0, Y0_) # rotational
                info[i, 3] = self._calc_rmsd(X0_sel, Y0_sel) # rotational selection 

                # ref/frame comparison after alignment
                info[i, 4] = self._calc_rmsd(X, Y)    # all
                info[i, 5] = self._calc_rmsd(X0, Y0)  # rotational

                # aligned/unaligned frame comparisons
                info[i, 6] = self._calc_rmsd(Y_, Y)   # all
                info[i, 7] = self._calc_rmsd(Y0, Y0_) # rotational

                # ref/frame and aligned/unaligned
                info[i, 8] = self._calc_rmsd(t, t0)   # centre difference

                # euler angles
                info[i, 8:11] = self._calc_euler(R)

        return (Ys, info) if record else Ys

    def align(self, sel=None, calpha=False, ref=None, record=True):
        # set reference and select atoms to align to
        self.set_ref(ref)
        if calpha and sel:
            sel = sorted(set(sel).intersection(self.ref.where('atom', 'CA')))
        elif calpha:
            sel = self.ref.where('atom', 'CA')

        # perform alignment 
        X = np.array([self.ref.x, self.ref.y, self.ref.z])
        data, alignment = self._align(self.data, X, sel=sel) 

        # save
        new = self.copy()
        new.data = data
        new.alignment = alignment 
        new._aligned = True
        return new

    def align_asus(self, calpha=True, ref=None, record=True, nsym=0):

        self.set_ref(ref, check=True)

        if len(self.ref) == len(self.ref._asu):
            warnings.warn('PDBs either not expanded or is in P1 spacegroup')
        if len(self.ref.R) == 0 and nsym == 0:
            warnings.warn('Space group operators not parsed, must provide nsym')

        n = len(self.ref._asu)
        X = np.array([self.ref.x, self.ref.y, self.ref.z])

        # align each ASU to its reference and update trajectory for that AS
        trjs = []
        for i in range(0, len(self.ref.R) or nsym):
            sel = np.arange(n*i, n*(i+1))
            trj = self.selection(atoms=sel)
            asu = self.ref.selection(sel)
            trj = trj.align(calpha=calpha, ref=asu, record=record)
            trjs.append(trj)
        return trjs

    def _combine(self, *others, **kws):
        new = self.copy()
        for other in others:
            new.data = np.append(new.data, other.data, **kws)
        return new

    def combine_atoms(self, *others):
        return self._combine(*others, axis=2)

    def combine_frames(self, *others):
        return self._combine(*others, axis=0)

    def rmsd(self, ref=None):
        ref = ref or self.ref  # don't set reference for rmsd() calls
        rmsd = self.data - ref # deviation
        rmsd *= rmsd           # square deviation
        rmsd = rmsd.sum(2)     # atom[0] + atom[1] ... + atom[n]
        rmsd = rmsd.sum(1)     # x + y + z
        rmsd /= self.natoms    # mean square deviation per frame
        return np.sqrt(rmsd)   # rmsd

    def rmsf(self, ref=None):
        ref = ref or self.ref   # don't set reference for rmsf() calls
        rmsf = self.data - ref  # deviation
        rmsf *= rmsf            # square deviation
        rmsf = rmsf.sum(1)      # x + y + z
        rmsf = rmsf.sum(0)      # frame[0] + frame[1] ... frame[n]
        rmsf /= self.nframes    # mean square deviation per atom
        return np.sqrt(rmsf)    # rmsf


################################################################################

# Write DCD class

################################################################################

class WriteDCD():

    def __init__(self, filename):
        assert not os.path.exists(filename)
        self.outstream = open(filename, mode='wb')

    def close(self):
        self.outstream.close()

    def _write_data(self, fmt, vals):
        assert len(fmt) == len(vals)
        size = struct.calcsize(fmt)
        self.outstream.write(struct.pack('i', size))
        self.outstream.write(struct.pack(fmt, *vals))
        self.outstream.write(struct.pack('i', size))

    def write_frame(self, r):
        natoms, three = r.shape
        assert three == 3, 'must transpose coordinates?'
        fmt = 'f'*natoms
        xyz = r.T.flatten().aslist()
        self._write_data(fmt, xyz)

    def write_header(self, nset, nsavc, natoms, delta, title):
        assert len(title) <= 80
        cord = ['C','O','R','D']
        #             n   i  n    t              f  d    x  v                    f  c
        #             s   s  s    i           d  i  e    s  p                    i  h
        #             e   t  a    m           o  x  l    t  b                    x  r
        #      CORD   t   r  v    e  -  -  -  f  d  t    a  c  -  -  -  -  -  -  d  m
        fmt = 'cccc''i'  'i''i'  'i''i''i''i''i''i''f'  'i''i''i''i''i''i''i''i''i''i'
        vals = cord+[nset,0,nsavc,0, 0, 0, 0, 0, 0,delta,0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self._write_data(fmt, vals)
        #               ntitl  titles
        self._write_data('i' + ('c'*80), [1, '{:<80s}'.format(title)])
        #               natoms
        self._write_data('i', [natoms])


class WriteDCDContext():

    def __init__(self, filename, natoms):
        self.filename = filename
        self.natoms = natoms
        if (self.filename is None):
            self.write = self.off
        else: 
            self.write = self.on

    def __enter__(self, *args):
        if self.filename is None:
            self.dcd = WriteDCD(self.filename)
            self.write_header(0, 0, self.natoms, 0, 'xstalpy')

    def __exit__(self, *args):
        if self.filename is not None: 
            self.dcd.outstream.close()

    def on(self, r):
        self.dcd.write_frame(r)

    def off(self, r):
        pass