"""pdbio.py - Download, parse, write and manipulate PDB coordinate files/data"""

from __future__ import division

import os
import copy
import warnings
import collections
import itertools
import functools
import types
 
import utils

__author__ = "James Gowdy" 
__email__ = "jamesgowdy1988@gmail.com"

################################################################################

# Download PDB files from the web

################################################################################

def get_pdb(pdbcode):
    # download a PDB file content using the 4 character PDB code
    template = 'http://files.rcsb.org/view/{}.pdb'
    u = urllib.urlopen(template.format(pdbcode))
    if u.code >= 400:
        raise IOError('No such URL [code = {}]'.format(u.code))
    return u.read()

def save_pdb(pdbcode, filename=None):
    # download a PDB file and save it to disk using the 4 character PDB code
    data = get_pdb(pdbcode)
    if filename is None:
        filename = pdbcode+'.pdb'
    with open(filename, 'w') as f: 
        f.write(data)

################################################################################

# Write coordinate models to a PDB format

################################################################################

class WritePDB():

    # http://www.wwpdb.org/documentation/file-format-content/format33/sect9.html
    # atom centering on the element symbol at column 13/14

    _MODEL_FMT = 'MODEL     {model:>4d}' + ' '*66

    _ATOM_FMT  = '{record:<6s}{atomid:>5d} {atom:>4s}{location:1s}{residue:<3s} '
    _ATOM_FMT += '{chain:1s}{resid:>4d}{insertion:1s}   {x:>8.3f}{y:>8.3f}{z:>8.3f}'
    _ATOM_FMT += '{n:>6.2f}{B:>6.2f}      {segment:<4s}{e:>2s}{charge:2}'

    _ANISOU_FMT  = 'ANISOU{atomid:>5d} {atom:>4s}{location:1s}{residue:<3s} '
    _ANISOU_FMT += '{chain:1s}{resid:>4d}{insertion:1s} {U[0][0]:>7d}{U[1][1]:>7d}'
    _ANISOU_FMT += '{U[2][2]:>7d}{U[0][1]:>7d}{U[0][2]:>7d}{U[1][2]:>7d}  '
    _ANISOU_FMT += '{segment:<4s}{e:>2s}{charge:2}'

    _TER_FMT  = 'TER   {terid:>5d}      {residue:<3s} {chain:1s}{resid:>4d}'
    _TER_FMT += '{insertion:1s}' + ' '*53

    _ENDMDL_FMT = 'ENDMDL' + ' '*74

    def __init__(self, *args, **kwargs):

        self.defaults = {'record':'ATOM  ',
                        'atomid':None,
                        'atom':'CA',
                        'location':'',
                        'residue':'ALA',
                        'chain':'A',
                        'resid':1,
                        'insertion':'',
                        'x':0.,
                        'y':0.,
                        'z':0.,
                        'n':1.,
                        'B':0.,
                        'segment':'',
                        'e':'  ',
                        'charge':'',
                        'U': [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.,]],
                        'isaniso': False,
                        'ister': False,
                        'model': 0
                        'isdummy': True, # isdummy won't be used
                        'i': -1} # i won't be used

        self._warn = 0
        self._curmodel = 0
        self._natoms = 0
        self.header = []
        self.lines = []
        self.final = []
        self.add_atoms(*args, **kwargs) 

    def _atom_formatter(self, atom=None, e=None, atomid=None, **tmp):

        # first two chr are elemnt name second two relative position code
        search = re.search(r'(^{})(.*)'.format(e), atom)
        if e and search:
            e, code = search.groups()
            code = '{:<2s}'.format(code)
            e = '{:>{width}s}'.format(e, width=min(4-len(code), 2))
            return e + code
        # otherwise take a guesss on the padded left and right whitespace
        else:
            if e != '  ' and self._warn < 5: # if not default 
                msg = 'ParsePDB: atomid={} atom={} element={}'
                warnings.warn(msg.format(atomid, atom, e))
                self._warn += 1
            width = 4 if tmp['record'] == 'HETATM' and len(atom) > 1 else 3
            return '{:<{width}s}'.format(atom, width=width)

    def add_atom(self, **kwargs):

        tmp = dict(self.defaults)
        tmp.update(kwargs)
        
        self._natoms *= bool(len(self.lines)) # reset?
        self._natoms += 1 # increment
       
        # add a MODEL record and potentially a previous ENDMDL record
        if tmp['model'] != self._curmodel:
            self._curmodel = tmp['model']
            if len(self.atoms) != 0:
                endmdl = self._ENDMDL_FMT.format(**tmp)
                self.atoms.append(endmdl)
            model = self._MODEL_FMT.format(**tmp)
            self.lines.append(model)

        # atom type formatting or best guess
        tmp['atom'] = self._atom_formatter(**tmp)

        # always add a ATOM record or HETATM record
        atom = self._ATOM_FMT.format(**tmp)
        self.lines.append(atom)

        # potentially add an ANISOU record
        if tmp['isaniso']:
            for i in range(3):
                for j in range(3):
                    tmp['U'][i][j] = int(tmp['U'][i][j] * 1e+4)
            anisou = self._ANISOU_FMT.format(**tmp)
            self.lines.append(anisou)

        # potentially add a TER record?
        if tmp['ister']:
            ter = self._TER_FMT.format(terid=tmp['atomid']+1, **tmp)
            self.lines.append(ter)

    def add_atoms(self, *args, **kwargs):

        for key in kwargs: 
            assert key in self.defaults, 'Unexpected keyword'

        if len(args):
            # deal with a ParsePDB template 
            tmp = args[0].copy()
            assert len(args) == 1, 'Add single PDBs at a time'
            # overwrite template data
            for key, vals in kwargs.items():
                assert len(vals) == len(tmp), 'Arrays should match PDB length'
                setattr(tmp, key, vals)
            tmp._init_data()
            # add each complete atom in turn
            for atom_data in tmp.data:
                d = dict(zip(tmp._data_vars, atom_data))
                self.add_atom(**d) 

        elif len(kwargs):
            # deal with kwargs without a ParsePDB template
            dataT = kwargs.values()
            keys = kwargs.keys()
            assert len(set([len(vs) for vs in dataT])) == 1, 'Arrays must match'
            # transpose the dictionary values
            data = zip(*dataT)
            # add each partially defined atom in turn (defaults may be required)
            for atom_data in data:
                d = dict(zip(keys, atom_data))
                self.add_atom(**d)

    def add_header(self, header):
        self.header.append(header)

    def finalize(self, end=True, endmodel=True, conect_record=None, master_record=None):
        if endmodel and self._curmodel != 0:
            self.final.append(self._ENDMDL_FMT)
        if isinstance(conect_record, list):
            self.final.extend(conect_record)
        elif isinstance(conect_record, dict):
            self.final.extend(conect_record['CONECT'])
        if isinstance(master_record, list):
            self.final.extend(master_record)
        elif isinstance(master_record, dict):
            self.final.extend(master_record['MASTER'])
        if end:
            self.final.append('END' + ' '*77)

    def __repr__(self):
        l = list(self.lines)
        ops = 'header={}, finalized={}, natoms={}'
        ops = ops.format(len(self.header), len(self.final), len(self))
        if len(l) > 10:
            l = l[:3] + ['...'] + l[-3:] + [ops]
        return 'WritePDB(' + '\n        '.join(l) + ')'

    def __str__(self):
        return '\n'.join(self.header + self.lines + self.final)

    def __len__(self):
        return self._natoms

    def save(self, filename='out.pdb'):
        assert filename.endswith('.pdb')
        with open(filename, mode='w') as f: 
            f.write(str(self))


################################################################################

# Parse and manipulate a PBD file

################################################################################ 

# PDB parsing only (150ish LOC), full supporting functionality (700ish LOC)

class ParsePDB_base(object):

    _using_interface = 0

    _data_vars = ['record', 'atomid',  'atom', 'location', 'residue', 'chain',
                  'resid', 'insertion', 'x', 'y', 'z', 'n', 'B', 'segment', 'e',
                  'charge', 'isaniso', 'U', 'ister', 'model', 'isdummy', 'i']

    def _init_data(self): pass # overwritten when mixin ParsePDB_interface
        
    def _init_parser(self):

        self.header = ''
        self.remarks = collections.defaultdict(list)
        self.records = collections.defaultdict(list) # don't confuse with record
        self.S = [] # parsed scale matrix
        self.offset = []
        self.symops = []
        self.R = collections.defaultdict(list) # space group symop R matrix 
        self.T = collections.defaultdict(list) # space group symop T vecs
        self.cryst1 = ''
        self.uc_abc = [None]*3 # unitcell lengths: A
        self.uc_angles = [None]*3 # unitcell angles: degrees
        self.spg_hm = '' # Hermann-Mauguin symbol
        self.uc_Z = None # (most populous) polymeric chains/unit cell
        self._expanded = False
        self._basis = 'world'
        self._file = None
        self._dtype = list
        self._update_flag = False
        self._ndel = 0
        self._asu = self # overwrite with a deepcopy after parsing
        self._update_asu = True # if equal length attrs set, update the asu

        # initialiation for ATOM/HETATM/ANISOU/MODEL/TER records
        for var in self._data_vars:
            l = []
            setattr(self, '_'+var, l)
            if not self._using_interface:
                setattr(self, var, l) # otherwise will be set as properties

    def _parse(self, inp):

        _upper = str.upper
        _strip = str.strip

        def _float(n)
            try: return float(n)
            except ValueError as e: return n

        def _int(n):
            try: return int(n)
            except ValueError as e: return n

        # state of parser
        curmodel = 0
        curidx = 0
        key2idx = {}
        key2U = {}
        isheader = True

        # parse MODEL/ATOM/HETATM/ANISOU/TER/CRYST1/SCALE/REMARK records
        for line in inp.split('\n'):

            line = line.rstrip('\r')

            # key to make sure only 1 ANISOU record per ATOM/HETATM record
            error = 'in model {:d}:\nANISOU{:s}{:^45s}{:s}\n'
            key = error.format(curmodel, line[6:27], '...', line[72:80])

            if line.startswith('MODEL'):
                curmodel = int(strip(line[11:14]))

            elif line.startswith('ATOM') or line.startswith('HETATM'):
                # Atom coordinate record
                self._i.append(curidx)
                self._model.append(curmodel)
                self._record.append(_strip(line[0:6]))
                self._atomid.append(_int(line[6:11]))
                self._atom.append(_upper(_strip(line[12:16])))
                self._location.append(line[16])
                self._residue.append(_upper(_strip(line[17:20])))
                self._chain.append(line[21])
                self._resid.append(_int(line[22:26]))
                self._insertion.append(line[26])
                self._x.append(_float(line[30:38]))
                self._y.append(_float(line[38:46]))
                self._z.append(_float(line[46:54]))
                self._n.append(_float(line[54:60]))
                B = _float(line[60:66])
                self._B.append(B)
                self._segment.append(_strip(line[72:76]))
                self._e.append(_upper(_strip(line[76:78])))
                self._charge.append(_int(line[78:80]))
                # defaults to be changed by subsquent records
                self._ister.append(False)
                self._isaniso.append(False)
                self._isdummy.append(False)
                B = B if isinstance(B, float) else 0.0
                self._U.append([[B, 0., 0.], [0., B, 0.], [0., 0., B]])
                key2idx[key] = curidx
                curidx += 1

            elif line.startswith('ANISOU'):
                # Anisotropic B-factor record
                U = [[0. , 0., 0.], [0., 0., 0.], [0., 0., 0.]]
                U[0][0] = float(line[28:35]) * 1e-4
                U[1][1] = float(line[35:42]) * 1e-4
                U[2][2] = float(line[42:49]) * 1e-4
                U[0][1] = U[1][0] = float(line[49:56]) * 1e-4
                U[0][2] = U[2][0] = float(line[56:63]) * 1e-4
                U[1][2] = U[2][1] = float(line[63:70]) * 1e-4
                # check ANISOU record is not dupicate for this model/atomid
                if key in key2U:
                    warnings.warn('\nParsePDB: duplicate ANISOU record ' + key)
                else:
                    key2U[key] = U
            
            elif line.startswith('TER'):
                # check that a terminal atom has the previous atom's atomid + 1
                if len(self.atomid) and self.atomid[-1]+1 == int(line[6:11]):
                    self.ister[-1] = True
                else:
                    self.ister[-1] = '?'
                    warnings.warn('ParsePDB: check TER record atomid:\n'+line)

            elif line.startswith('CRYST1'):
                # unit cell lengths and angles
                self.cryst1 = line
                self._cryst1 = line[:66]
                self.uc_abc = map(float, [line[6:15], line[15:24], line[24:33]])
                self.uc_angles = map(float, [line[33:40], line[40:47], line[47:54]])
                self.spg_hm = strip(line[55:66])
                self.uc_Z = int(strip(line[66:70]))

            elif line.startswith('SCALE'):
                # scale matrix converts xyz to fractional/crystallographic basis
                row = map(float, [line[10:20], line[20:30], line[30:40]])
                self.S.append(row)
                self.offset.append(float(line[45:55]))
                assert not any(self.offset), "Has offset don't know what to do?"

            elif line.startswith('REMARK 290   SMTRY'):
                # symmetry operators to convert PDB assymetric unit to unit cell
                lline = line.split()
                self.R[int(lline[3])].append(map(float, lline[4:7]))
                self.T[int(lline[3])].append(float(lline[7]))

            elif line.startswith('REMARK 290'):
                # symbolic symmetry operators
                lline = line.split()
                if len(lline) == 4 and lline[2].isdigit():
                    self.symops.append(lline[3])

            if len(line) >= 6:
                # get the record type
                record = strip(line[0:6]) # strip record name keys
                self.records[record].append(line)
                if record == 'REMARK':
                    number = strip(line[6:10])
                    if number.isdigit(): 
                        number = int(number)
                    self.remarks[number].append(line)
                elif record in ['ATOM', 'HETATM', 'ANISOU', 'MODEL', 'TER']:
                    isheader = False

            if isheader:
                self.header += line +'\n'

        # match ANISOU records to ATOM/HETATM records
        for key, U in key2U.items():
            idx = key2idx.pop(key, None)
            if idx is None:
                warnings.warn('ParsePDB: unmatched ANISOU record ' + key)
            else:
                self._U[idx] = U
                self._isaniso[idx] = True

# get/set factory functions for ParsePDB_interface properties with complex 
# notification patterns when attribute fields are updated

def _readonly(self, *val):
    raise AttributeError('Attribute is read only')

def _make_fget(var):
    def fget(self):
        return getattr(self, '_%s'%var)
    return fget

def _make_fget_flag(var):
    # set flag in case we do something like p.x[..] = .. or p.x.append(..)
    def fget(self):
        self._update_flag = True
        return getattr(self, '_%s'%var)
    return fget

# TODO: build AttributeList(list) AttributeArray(np.ndarray) classes for better
# notification of p.x.append etc.

def _make_fget_check(var):
    # get p.data/p.dataT update if p.x etc. have been accessed as above
    def fget(self):
        if self._update_flag:
            self._init_data()
        return getattr(self, '_%s'%var)
    return fget

@utils.requires_numpy
def _make_fset(var):
    def fset(self, val):
        # check if val needs to be changed
        if not isinstance(val, self._dtype):
            if isinstance(val, np.ndarray):
                val = val.tolist()
            else:
                val = self._dtype(val)
        # update interal attribute lists
        setattr(self, '_%s'%var, val)
        # if needed set dummy attributes, this requires ParsePDB_mutators mixin
        if len(val) != len(self):
            if isinstance(self, ParsePDB_mutators)
                self._fill_with_dummys()
            else:
                msg = "atom parameter '{:s}' must be of length {:d}"
                raise ValueError(msg.format(var, len(self)))
        # if no dummy atoms (not adding to original ASU) then update saved ASU?
        elif self._update_asu:
            setattr(self._asu, '_%s'%var, val)  
        # update the data matrix, this catches the obvious p.x = [..]
        self._init_data()
    return fset           

'''
class ParsePDB_MetaClass(type):     
    def __new__(cls, name, bases, dct):
        # if data_vars properties are re-set we will auto-update the .data array
        for var in dct['_data_vars']:
            dct[var] = property(_make_fget_flag(var), _make_fset(var), _readonly)
        return super(ParsePDB_MetaClass, cls).__new__(cls, name, bases, dct)
'''

# implementation of public and read-only data representation

def ParsePDB_interface(ParsePDB_parser):

    _using_interface = 1

    # alt to using locals()[] - intercept class creation to manipulate attributes
    # __metaclass__ = ParsePDB_MetaClass

    for _var in ParsePDB_parser._data_vars:
        locals()[_var] = property(_make_fget_flag(_var), _make_fset(_var), _readonly)
    del _var

    # make the .data and .dataT array read only
    data_vars = property(_make_fget('data_vars'), _readonly, _readonly)
    data = property(_make_fget_check('data'), _readonly, _readonly)
    dataT = property(_make_fget_check('dataT'), _readonly, _readonly)

    def _init_data(self):
        self._update_flag = False
        self._dataT = [getattr(self, var) for var in self._data_vars]
        self._data = zip(*self._dataT) # natoms x nvars
        if self._dtype is not list:
            self._dataT = self._dtype(self._dataT)
            self._data = self._dtype(self._data)

    def __init__(self, inp):

        self._init_parser()

        # if inp is empty string
        if inp is "":
            self._init_data()
            return
        # if inp is a file
        elif 0 < len(os.path.basename(inp)) <= 255:
            if os.path.isfile(inp):
                with open(inp, mode='r') as f:
                    lines = f.read()
                    lines.replace('\r\n', '\n')
                self._file = inp
                inp = lines
            else:
                abspath = os.path.abspath(inp)
                warnings.warn('Could not find {}'.format(abspath))
                self._init_data()
                return

        self._parse(inp)        # parse PDB coordinate data 
        self._init_data()       # initalise the read-only data structure
        self._asu = self.copy() # save a backup copy in case we expand later

    def __len__(self):
        return len(self.data)

    def copy(self):
        return copy.deepcopy(self)

    def __repr__(self):

        code, date, R, d = None, None, None, None
        if 'HEADER' in self.records:
            code = self.records['HEADER'][0][62:66]
            date = self.records['HEADER'][0][50:59].strip()
        if 3 in self.remarks:
            for l in self.remarks[3]:
                if 'R VALUE  ' in l:
                    R = l.split()[-1]; break
            for l in self.remarks[3]:
                if 'RESOLUTION RANGE HIGH' in l:
                    d = l.split()[-1]; break

        fmt = 'ParsePDB(file={}, code={}, spg_uc={}, nsymops={}, expanded={}, '
        fmt += 'a={:.2f}, b={:.2f}, c={:.2f}, scale={}, natoms={}, nunique_atoms={}, '
        fmt += 'nisanisou={}, nresidues={}, nchains={}, nsegments={}, '
        fmt += 'nmodels={}, nremarks={}, Rvalue={}, resolution={}, date={})'
        return fmt.format(self._file, code, repr(self.spg_hm), len(self.R),
                self._expanded, self.uc_abc[0], self.uc_abc[1], self.uc_abc[2],
                bool(self.S), len(self), len(set(zip(self.atom, self.resid,
                self.residue, self.chain, self.segment))), sum(self.isaniso), 
                len(self.get_residues()), len(set(self.chain)), len(set(self.model)), 
                len(set(self.segment)), len(self.remarks), R, d, date)

class ParsePDB_switchdtype(object):

    @utils.requires_numpy
    def toarray(self):
        new = self.copy()
        for var in new._data_vars:
            setattr(new, var, np.array(getattr(new, var)))
        new._dtype = np.array
        new._init_data()
        return new

    def tolist(self):
        new = self.copy()
        for var in new._data_vars:
            attr = getattr(new, var)
            if hasattr(attr, 'tolist'):
                setattr(new, var, attr.tolist())
        new._dtype = list
        new._init_data()
        return new

# low-level selection functionality based on data field values

class ParsePDB_selectors(object):

    def __getitem__(self, key):
        # pdb[i], pdb[i:j], pdb[(i,j,k)] etc
        if not isinstance(key, str):
            return self.selection(key)
        # pdb["atomid"], pdb["resid"] etc
        elif key in self._data_vars: 
            return getattr(self, key)
        else:
            raise KeyError(key)
    
    def selection(self, i):
        # does not renumber self.i
        new = self.copy()
        for var in self._data_vars:
            data = getattr(self, var)

            # new = pdb[i]
            if isinstance(i, int):
                setattr(new, var, [data[i]])
            # new = pdb[i:j]
            elif isinstance(i, slice):
                setattr(new, var, data[i])
            # new = pdb[(i, j)]
            elif hasattr(i, '__iter__'):
                setattr(new, var, [])
                for ii in i:
                    getattr(new, var).append(data[ii]) # vars(new)[var].append

            else:
                raise IndexError(i)

        new._init_data()
        return new

    def groupby(self, key, sort=False):

        # option 1 group using one of the PDB model parameters
        if key in self._data_vars:
            keyindex = self._data_vars.index(key)
            keyfunc = lambda l: l[keyindex]
            atoms = self.data # natoms x nvars
            if sort: 
                atoms = sorted(atoms, key=keyfunc) 
            iterator = itertools.groupby(atoms, keyfunc)
            groups = [[k, list(g)] for k, g in iterator]

        # option 2 group using user-supplied values for each atom    
        elif len(key) == len(self):
            keys = key
            keyfunc = lambda arg: arg[0]
            atoms = zip(keys, self.data) # natoms x nvars
            if sort: 
                atoms = sorted(zip(keys, atoms), key=keyfunc)
            iterator = itertools.groupby(atoms, keyfunc)
            groups = [[k, [l for k_, l in list(g)]] for k, g in iterator]
        else:
            raise TypeError('Argument must be list of len(self) or valid key')
        return groups

    def index(self, key):
        # get index of a data_vars attribute
        return self._data_vars.index(key)

    def where(self, key, val, get=None):
        # self.where('resid', 14)
        # self.where('resid', 14, get='atom')
        key = self.index(key)
        if get is None:
            return [i for i, v in enumerate(self.dataT[key]) if v == val]   
        else:
            key2 = self.index(get) if get != 'data' else slice(None)
            return [l[key2] for l in self.data if l[key] == val]

# split the PDB coordinate file into multiple models/chains/segments

class ParsePDB_splitters(ParsePDB_selectors):

    def _split(self, keys):
        pdbs = []
        for current in set(keys):
            include = [i for i, key in enumerate(keys) if key == current]
            pdbs.append(self.selection(include))
        return pdbs 

    def split_models(self):
        return self._split(self.model)

    def split_chains(self):
        if len(set(self.model)) > 1:
            warnings.warn('Multiple models - only using first')
            self = self.split_models()[0]
        return self._split(self.chain)

    def split_segments(self):
        if len(set(self.model)) > 1:
            warnings.warn('Multiple models - only using first')
            self = self.split_models()[0]
        return self._split(self.segment)

# selection based on residue types etc

class ParsePDB_getters(ParsePDB_selectors):

    def get_nonsolvent(self, maxresid=1000000):
        exclude_res = ['TIP', 'HOH']
        exclude_seg = ['SOLV', 'SOL']
        model = zip(self.residue, self.resid, self.segment)
        include = []
        for i, (r, ri, s) in enumerate(model):
            if r not in exclude_res and ri <= maxresid and s not in exclude_seg:
                include.append(i)
        return self.selection(include)

    def get_nonhetatm(self):
        return self.selection(self.where('record', 'ATOM'))

    def get_unique_atoms(self):
        keys = []
        idxs = []
        atoms = zip(self.atom, self.resid, self.residue, self.chain, self.segment)
        for i, key in enumerate(atoms):
            if key not in keys:
                keys.append(key)
                idxs.append(i)
        return self.selection(idxs)

    def get_consecutive_resid(self, start=0):
        ires = start-1
        _res = None
        resid = [] 
        original_res = {}
        for res in zip(self.residue, self.resid, self.chain, self.segment):
            if res != _res:
                ires += 1
            _res = res
            resid.append(ires)
            orignial[ires] = '{}-{}'.join(res[:2])
        return resid, original

    def get_residues(self, full=False):
        atoms = set(zip(self.residue, self.resid, self.chain, self.segment))
        return [(r, i) for r, i, c, s in atoms]

    def get_residue_counts(self):
        resnames, resids = zip(*self.get_residues())
        return {r: resnames.count(r) for r in set(resnames)}
        
    def get_residue_breakdown(self):

        """Return a matplotlib axes instance"""

        aa = ['ASP', 'GLU', 'ARG', 'LYS', 'HIS',
              'ASN', 'SER', 'THR', 'GLN', 'TYR', 'CYS', 'GLY',
              'ALA', 'LEU', 'VAL', 'ILE', 'TRP', 'PHE', 'MET', 'PRO']
        o = 'orange'
        cols = ['r', 'r', 'b', 'b', 'b', o, o, o, o, o, 'yellow', 'silver',
                'w', 'w', 'w', 'w', 'w', 'w', 'w', 'w', 'w']

        counts = self.get_residue_counts()
        indices = range(len(aa))

        import matplotlib.pyplot as plt

        fig, ax = plt.figure()
        ax.bar(indices, [counts.get(r, 0) for r in aa], color=cols)
        ax.set_xticks([i+0.5 for i in indices])
        ax.set_xticklabels(aa, rotation=90)
        ax.set_ylabel('Number of residues', fontsize=12)
        ax.set_xlabel('Amino acid', fontsize=12);
        return ax

# transform/alter the PDB object and calulctaion functions

class ParsePDB_transforms(object):
    
    @utils.requires_numpy
    def get_unitcell(self, force=False, relabel=False):

        if self._expanded: 
            warnings.warn('Already expanded UC')
            if not force:
                return self

        new = self.copy()
        nsym = len(self.R.keys())

        # apply symops
        uc_xyz = np.array([[], [], []])
        for j in self.R.keys():
            Rw = np.array(self.R[j])
            Tw = np.array(self.T[j]).reshape(3, 1)
            asu_xyz = np.dot(Rw, [self.x, self.y, self.z]) + Tw
            uc_xyz = np.column_stack((uc_xyz, asu_xyz))
        new.x, new.y, new.z = uc_xyz

        if relabel:
            lables = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            if nsym > len(labels):
                labels = ['%s%d'%(l, i) for i in range(10) for l in labels]
            new.segment = np.repeat(labels[:nsym], len(self))
        else:
            new.segment = np.tile(self.segment, nsym)

        # copy over everything else and extend lists as appropriate
        for var in self._data_vars:
            if var not in ['x', 'y', 'z', 'segment']:
                attr = getattr(self, var)
                attr = np.tile(attr, nsym)
                setattr(new, var, attr)
        new._init_data()
        new._expanded = True
        return new

    @utils.requires_numpy
    def get_crystal_box(self):
        # wrap structure into a parralelpiped box
        S = self.calc_invscale()
        SI = self.calc_invscale()
        new = self.copy()
        new.x, new.y, new.z = np.dot(SI, np.dot(S, [self.x, self.y, self.x])%1.0)
        return new

    @utils.requires_numpy
    def calc_volume(self):
        cos = np.cos
        assert all(self.uc_abc)
        a, b, c, al, be, ga = self.uc_abc + map(np.radians, self.uc_angles)
        V = a*b*c*(1-cos(al)**2-cos(be)**2-cos(ga)**2+2*cos(al)*cos(be)*cos(ga))**0.5
        return V

    @utils.requires_numpy
    def calc_scale(self):
        #   [1]    [0]    [0]     [ax]     [bx]     [cx]   [ax bx cx][xc]
        # xw[0]+ yw[1]+ zw[0] = xc[ay] + yc[by] + zc[cy] = [ay by cy][yc] = S xc
        #   [0]    [0]    [1]     [az]     [bz]     [cz]   [az bz cz][zc]

        #   [1]    [0]    [0]     [a*x]    [b*x]    [c*x]   [a*x b*x c*z][hc]
        # hw[0]+ hw[1]+ hw[0] = hc[a*y]+ kc[b*y]+ lc[c*y] = [a*y b*y c*y][kc] = ST hc
        #   [0]    [0]    [1]     [a*z]    [b*z]    [c*z]   [a*z b*z c*z][lc]
        sin = np.sin
        cos = np.cos
        assert any(self.uc_abc)
        a, b, c, al, be, ga = self.uc_abc + map(np.radians, self.uc_angles)
        V = self.calc_volume()
        S = np.zeros(3, 3)
        S[0, 0] = 1.0/a
        S[0, 1] = -cos(ga)/(a*sin(ga))
        S[0, 2] = (b*cos(ga)*c*(cos(al)-cos(be)*cos(ga))/sin(ga)-b*c*cos(be)*sin(ga))/V
        S[1, 0] = 0.
        S[1, 1] = 1.0/(b*sin(ga))
        S[1, 2] = -a*c*(cos(al)-cos(be)*cos(ga))/(V*sin(ga))
        S[2, 0] = 0.
        S[2, 1] = 0.
        S[2, 2] = a*b*sin(ga)/V
        return S

    @utils.requires_numpy
    def calc_invscale(self):
        sin = np.sin
        cos = np.cos
        assert any(self.uc_abc)
        a, b, c, al, be, ga = self.uc_abc + map(np.radians, self.uc_angles)
        V = self.calc_volume()
        SI = np.zeros(3, 3)
        SI[0, 0] = a
        SI[1, 0] = 0.
        SI[2, 0] = 0.
        SI[0, 1] = round(b*cos(ga), 9)
        SI[1, 1] = round(b*sin(ga), 9)
        SI[2, 1] = 0.
        SI[0, 2] = round(c*cos(be), 9)
        SI[1, 2] = round(c*(cos(al) - cos(be)*cos(ga))/sin(ga), 9)
        SI[2, 2] = round(V/(a*b*sin(ga)), 9) # np.sqrt(1 - SI[0][2]**2 - SI[1][2]**2) 
        return SI

# mutate atoms of residues and their fields

class ParsePDB_mutators(object):

    def _mutate_to_ala(self, sel, chg, lab, resid, residue, chain, segment):

        gamma = {'CG': 'HB1',  # R, N, D, Q, E, H, L, K, M, F, P, W, Y 
                 'SG': 'HB1',  # C
                 'OG': 'HB1',  # S
                 'OG1': 'HB1', # T
                 'CG1': 'HB1', # I, V         #   O   HA   H 
                 'CG2': 'HB2', # I, V, T      #   ||  |    | 
                 'HB': 'HB3'}  # I, V, T      # --C---CA---N-- Alanine
                                              #       |
        alanine = ['N', 'C', 'O', 'CA', 'HA', #       CB
                   'H1', 'H2', 'H3', 'H',     #      /|\
                   'CB', 'HB1', 'HB2', 'HB3'] #  HB1 HB2 HB3
        sel = sel[:]
        atoms = zip(self.resid, self.residue, self.atom, self.chain, self.segment)
        ndel = 0
        for ii, (i, r, a, c, s) in enumerate(atoms):
            if i == resid and r == residue and c == chain and s == segment:
                if r == 'GLY' or r == 'PRO':
                    warnings.warn('Cannot mutate {}{} to ALA'.format(r, i))
                    continue
                if a in alanine:
                    lab.append(ii) # keep
                elif a in gamma:
                    chg.append((ii, gamma[a])) 
                    lab.append(ii) # keep but re-purpose
                else:
                    ndel += 1
                    sel.remove(ii) # remove
        return sel, chg, lab, ndel

    def mutate_to_ala(self, resids, chain=None, residue=None, segment=None, force=False):

        # default atom selection - don't delete anything 
        sel = range(0, len(self), 1)
        chg = []
        lab = []
        new = self.copy()

        # deal with arguments when explicitly given exact amino acid
        if (chain is not None) and (residue is not None) and (segment is not None):
            sel, chg, lab, ndel = self._mutate_to_ala(sel, chg, lab, resids, 
                                                      residue, chain, segment)

        # deal with just resids
        else:
            if isinstance(resids, int):
                resids = [resids]

            # get all unique residues and arrange by resid
            atoms = set(zip(self.residue, self.resid, self.chain, self.segment))
            resdict = collections.defaultdict(list)
            for r, i, c, s in atoms: 
                resdict[i].append([r, c, s])
            redict = dict(resdict)

            # pass each resid to helper function for mutation
            for i in resids:
                group = resdict[i]
                if len(group) > 1 and not force:
                    msg = 'More than one residue/chain/segment for resid {}: {}'
                    warnings.warn(msg.format(i, group))
                else:
                    for r, c, s in group:
                        result = self._mutate_to_ala(sel, chg, lab, i, r, c, s)
                        sel, chg, lab, ndel = result
                        new._ndel += ndel

        # carry out mutations all at once and re-label
        for i, atm in chg:
            new.atom[i] = atm
            new.e[i] = atm[0]
        for i in lab:
            new.residue[i] = 'ALA'
        return new.selection(sel)

    @utils.requires_numpy
    def _fill_with_dummys(self, **kwargs):

        defaults = {'record':'ATOM  ',
                    'atomid':None,
                    'atom':'CA',
                    'location':'',
                    'residue':'ALA',
                    'chain':'A',
                    'resid':1,
                    'insertion':'',
                    'x':0.,
                    'y':0.,
                    'z':0.,
                    'n':0.,
                    'B':0.,
                    'segment':'',
                    'e':'  ',
                    'charge':'',
                    'U': [[0., 0., 0.], [0., 0., 0.], [0., 0., 0.,]],
                    'isaniso': False,
                    'ister': False,
                    'model': 0,
                    'isdummy': True,
                    'i': -1}

        defaults.update(kwargs)
        nmax = max([len(getattr(self, var)) for var in self._data_vars])

        for var in self._data_vars:
            attr = getattr(self, var)
            n = nmax - len(attr)
            if n > 0:
                # get extra attoms
                if var == 'U':
                    extra = [copy.deepcopy(defaults['U']) for _ in xrange(n)]
                elif var == 'i':
                    extra = range(len(attr), nmax)
                else:
                    extra = [defaults[var]]*n

                # extend the list or np.array
                if isinstance(attr, list):
                    attr.extend(extra)
                else:
                    setattr(self, var, np.append(attr, extra))

# ParsePDB class definition with full functionality - using all mixins
# for minimial parser can just use code for ParsePDB_base

class ParsePDB(ParsePDB_interface, # inherits ParsePDB_base
               ParsePDB_getters,
               ParsePDB_splitters,
               ParsePDB_selectors,
               ParsePDB_mutators,
               ParsePDB_switchdtype,
               ParsePDB_transforms): pass

amino_acid_makeup = {
    'CYS': ['C', 'CB', 'CA', 'O', 'N', 'H', 'HA', 'SG', 'HB3', 'HB2'], 
    'ASP': ['C', 'CB', 'CA', 'CG', 'O', 'N', 'H', 'OD1', 'HA', 'OD2', 'HB3', 
            'HB2'],
    'SER': ['C', 'OG', 'CB', 'CA', 'O', 'N', 'H', 'HA', 'HG', 'HB3', 'HB2'],
    'GLN': ['HG2', 'C', 'HA', 'CB', 'CA', 'CG', 'HG3', 'O', 'N', 'H', 'HE22',
            'HE21', 'CD', 'NE2', 'OE1', 'HB3', 'HB2'],
    'LYS': ['HA', 'HE2', 'HE3', 'HG2', 'HG3', 'NZ', 'HZ1', 'HZ3', 'HZ2', 'C',
            'CB', 'CA', 'CG', 'O', 'N', 'CE', 'CD', 'HD3', 'HD2', 'H', 'HB3',
            'HB2'],
    'ASN': ['C', 'HA', 'HD22', 'HD21', 'CB', 'CA', 'CG', 'O', 'N', 'H', 'OD1',
            'ND2', 'HB3', 'HB2'],
    'PRO': ['HG2', 'C', 'HA', 'HD3', 'HD2', 'CB', 'CA', 'CG', 'HG3', 'O', 'N',
            'CD', 'HB3', 'HB2'],
    'THR': ['C', 'HA', 'HG22', 'CB', 'CA', 'OG1', 'O', 'N', 'HG21', 'HG1', 'H',
            'HB', 'HG23', 'CG2'],
    'PHE': ['HZ', 'C', 'HE2', 'CE1', 'HD1', 'CB', 'CA', 'HB2', 'CG', 'O', 'N',
            'CZ', 'HD2', 'CD1', 'CD2', 'H', 'HE1', 'HA', 'HB3', 'CE2'],
    'ALA': ['C', 'CB', 'CA', 'O', 'N', 'H', 'HA', 'HB1', 'HB3', 'HB2'],
    'HIS': ['C', 'CE1', 'HD1', 'CB', 'CA', 'CG', 'O', 'N', 'HD2', 'CD2', 'H',
            'ND1', 'HE1', 'HA', 'NE2', 'HB3', 'HB2'],
    'GLY': ['C', 'H', 'CA', 'O', 'N', 'HA2', 'HA3'],
    'ILE': ['C', 'HA', 'HG22', 'CB', 'CA', 'HG21', 'O', 'N', 'HB', 'HG12', 
            'HG13', 'HD13', 'CD1', 'H', 'CG1', 'HD12', 'HG23', 'CG2', 'HD11'],
    'LEU': ['C', 'HD22', 'HD23', 'HD21', 'OXT', 'CB', 'CA', 'CG', 'O', 'N',
            'CD1', 'CD2', 'H', 'HD13', 'HD12', 'HD11', 'HA', 'HG', 'HB3', 'HB2'],
    'ARG': ['HG3', 'HA', 'HE', 'HG2', 'NE', 'HH22', 'HH21', 'H', 'C', 'CB',
            'CA', 'CG', 'O', 'N', 'CZ', 'NH1', 'NH2', 'CD', 'HD3', 'HD2',
            'HH12', 'HH11', 'HB3', 'HB2'],
    'TRP': ['HH2', 'CZ2', 'CZ3', 'CD1', 'CD2', 'HA', 'HE1', 'HE3', 'CH2', 'HZ3',
            'HZ2', 'C', 'CB', 'CA', 'CG', 'O', 'N', 'CE3', 'CE2', 'HD1', 'H',
            'NE1', 'HB3', 'HB2'],
    'VAL': ['C', 'HA', 'HG11', 'CB', 'CA', 'HG21', 'O', 'N', 'HB', 'HG12',
            'HG13', 'H', 'CG1', 'HG23', 'CG2', 'HG22'],
    'GLU': ['HG2', 'C', 'HA', 'CB', 'CA', 'CG', 'HG3', 'O', 'N', 'H', 'OE2',
            'CD', 'OE1', 'HB3', 'HB2'],
    'TYR': ['C', 'HE2', 'CE1', 'HD1', 'OH', 'CB', 'CA', 'HB2', 'CG', 'O', 'N',
            'CZ', 'HH', 'HD2', 'CD1', 'CD2', 'H', 'HE1', 'HA', 'HB3', 'CE2'],
    'MET': ['HG2', 'C', 'HE1', 'HE2', 'HE3', 'CB', 'CA', 'HB2', 'CG', 'HG3',
            'O', 'N', 'CE', 'H', 'HA', 'HB3', 'SD']}