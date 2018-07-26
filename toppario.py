"""toppario.py - parse CHARMM residue topology and forcefield parameter files"""

from __future__ import division

import warnings
import copy
import functools
import pickle
import math

import utils

__author__ = "James Gowdy" 
__email__ = "jamesgowdy1988@gmail.com"

################################################################################

# Parameters

################################################################################

class ParamDict(dict):
    def assign_name(self, name):
        self.name = name

class ParseParameters():

    data_vars = ['atoms', 'bonds', 'angles', 'ubs', 'dihedrals', 'impropers',
                 'cmaps', 'nonbonded', 'nonbonded_14', 'nbfix', 'hbonds', 
                 'cmaps_bicubic_coeffs'] # 'cmaps_grad']

    def __init__(self, prm_file, use_numpy=True):

        for var in data_vars:
            pd = ParamDict()
            setattr(self, var, pd)
            pd.assign_name(var)

        self._cmapn = 0
        self._parse = self._default

        # dispatch table
        parsers = {'ATOM': self._parse_atom,
                   'BOND': self._parse_bond,
                   'ANGL': self._parse_angle,
                   'DIHE': self._parse_dihedral,
                   'IMPR': self._parse_improper,
                   'CMAP': self._parse_cmap,
                   'NONB': self._parse_nonbonded,
                   'NBFI': self._parse_nbfix,
                   'HBON': self._parse_hbond}
        
        with open(prm_file, 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            if len(line) == 0:
                continue
            if line[0] in ['*', '!']:
                continue
            if line == 'END':
                continue
            if line[0:4] in self.parsers:
                self._parse = parsers[line[0:4]]
                continue
            line = line.split('!', 1) + ['']
            self._parse(line[0], line[1])

        # reverse direction for lookup 
        for (ai, aj), vals in self.bonds.items():
            self.bonds.setdefault((aj, ai), vals)
        for (ai, aj, ak), vals in self.angles.items():
            self.angles.setdefault((ak, aj, ai), vals)
        for (ai, aj, ak), vals in self.ub.items():
            self.ub.setdefault((ak, aj, ai), vals)
        for (ai, aj, ak, al), vals in self.dihedrals.items():
            self.dihedrals.setdefault((al, ak, aj, ai), vals)
        for (ai, aj, ak, al), vals in self.impropers.items():
            self.impropers.setdefault((ai, ak, aj, al), vals)

        # convert cmap to a 2D array and pre-calculate the 4 x 4 bicubic coeffs 
        # for each patch grid (i.e. a 24 x 24 x 4 x 4 tensor)
        if use_numpy:
            self._prepare_cmaps()
        else:
            self._prepare_cmaps_fallback()

    def _default(self, *args):
        warnings.warn("No parser set for line:\n{}".format(line))

    def _parse_atom(self, line, comment):
        ignore0, ignore1, ai, mass = line.split()
        self.atoms[ai] = float(mass), comment.strip()

    def _parse_bond(self, line, comment):
        # V(bond) = kb(b - b0)**2
        # kb: kcal/mol/Angstroms**2
        # b0: Angstoms
        ai, aj, kb, b0 = line.split()
        self.bonds[(ai, aj)] = float(kb), float(b0)
        self.bonds[(aj, ai)] = float(kb), float(b0)

    def _parse_angle(self, line, comment):
        # V(angle) = ktheta(theta - theta0)**2
        # ktheta: kcal/mol/radians**2
        # theta0: degrees
        line = line.split()
        ai, aj, ak, ktheta, theta0 = line[:5]
        self.angles[(ai, aj, ak)] = float(ktheta), float(theta0)
        self.angles[(ak, aj, ai)] = float(ktheta), float(theta0)
        # V(Urey-Bradley) = kub(b - b0)**2
        # kub: kcal/mole/Angstroms**2
        # b0: Angstroms
        if len(line) == 7:
            kub, b0 = line[5:7]
            self.ubs[(ai, aj, ak)] = float(kub), float(b0)
            self.ubs[(ak, aj, ai)] = float(kub), float(b0)

    def _parse_dihedral(self, line, comment):
        # V(dihedral) = kchi(1 + cos(n(chi) - delta))
        # kchi: kcal/mol
        # n: multiplicity
        # delta: degrees
        # chi: angle between planes (ai-aj-ak) and (aj-ak-al)
        ai, aj, ak, al, kchi, n, delta = line.split()
        params = float(kchi), float(n), float(delta)
        self.dihedrals.setdefault((ai, aj, ak, al), []).append(params)
        self.dihedrals.setdefault((al, ak, aj, ai), []).append(params)

    def _parse_improper(self, line, comment):
        # V(improper) = kpsi(psi - psi0)**2
        # kpsi: kcal/mol/rad**2
        # psi0: degrees
        # psi: angle between planes (ai, aj/ak, ak/aj) and (aj/ak, ak/aj, al)
        #      with ai at the center which differs from IC lines in RTF using ak

        # connectivity:
        #     i
        #   / | \ 
        # j   |  k
        #     |
        #     l

        # planes between which to calculate dihedral:
        #     i
        #   /   \ 
        # j ---- k
        #   \   /
        #     l

        # note: swapping a2 and a3 does not even change the sign of phi
        # phi = lambda a,b,c: arctan2(dot(x(x(a,b),x(b,c)),unit(b)),dot(x(a,b),x(b,c)))
        # +phi(-a1+a2, -a2+a3, -a3+a4) == 
        # +phi(-a1+a3, -a3+a2, -a2+a4) ==
        # -phi(-a4+a3, -a3+a2, -a2+a1) ==
        # -phi(-a4+a2, -a2+a3, -a3+a1) ==
        ai, aj, ak, al, kpsi, zero, psi0 = line.split()
        self.impropers[(ai, aj, ak, al)] = float(kpsi), float(psi0)
        self.impropers[(ai, ak, aj, al)] = float(kpsi), float(psi0) # TODO check if need -ve?

    def _parse_cmap(self, line, comment):

        # This is a bilinear interplation:
        # V(cmap) = V(QM)-V(CHARMM22)
        #         =  t(Y) * (t(X) * V[⌊X⌋,⌊Y⌋] + u(X) * V[⌈X⌉,⌊Y⌋]) 
        #          + u(Y) * (t(X) * V[⌊X⌋,⌈Y⌉] + u(X) * V[⌈X⌉,⌈Y⌉])

        # V[u,v]: n * n grid of V(phi, psi) cross-term corrections: kcal/mol
        # step: 360/n (e.g. n=24 equals step=15): degrees
        # X: phi(ai, aj, ak, al)/step 
        # Y: psi(am, an, ao, ap)/step
        # t(x): (⌈x⌉-x)/(⌈x⌉-⌊x⌋)
        # u(x): (x-⌊x⌋)/(⌈x⌉-⌊x⌋)

        if self._cmapn == 0:
            line = line.split()
            key = tuple(line[0:8]) # if always (?) consecutive only need 5 atoms
            n = int(line[8])
            self._cmapn = n*n
            self.cmaps[key] = self._cmaplist = [n]
        else:
            line = [float(v) for v in line.split()]
            self._cmapn -= len(line)
            self._cmaplist.extend(line)

    def _parse_nonbonded(self, line, comment): 
        # V(Lennard-Jones) = eps_ij[(rmin_ij/r_ij)**12 - 2(rmin_ij/r_ij)**6]
        # eps_ij = sqrt(eps_i * eps_j)
        # rmin_ij = rmin_i/2 + rmin_j/2
        # epsilon: kcal/mol
        # rmin/2: Angstrom

        # V(Lennard-Jones) = 4*eps_ij*[(sigma_ij/r_ij)**12 - (sigma_ij/r_ij)**6]
        # sigma_ij = rmin_ij/(2**(1./6.))

        # V(Lennard-Jones) = c12_ij/rij**12 - c6_ij/rij**6
        # c6_ij = 2.0*eps_ij*(rmin_ij**6) = 4.0*eps_ij*(sigma_ij**6)
        # c12_ij = eps_ij*(rmin_ij**12) = 4.0*eps_ij*(sigma_ij**12)

        if ('cutnb' in line.lower()) or ('wmin' in line.lower()):
            return
        line = line.split()
        ai, ignore, neg_epsilon, half_rmin = line[:4]
        eps = -float(neg_epsilon) # check?
        rmin = float(half_rmin)*2.0
        self.nonbonded[ai] = eps, rmin
        if len(line) == 7:
            # 1-4 fudge factor
            ignore, neg_epsilon_14, half_rmin_14 = line.split()[4:7]
            eps14 = -float(neg_epsilon_14) # check?
            rmin14 = float(half_rmin_14)*2.0
            self.nonbonded_14[ai] = eps14, rmin14 

    def _parse_nbfix(self, line, comment):
        # Emin: kcal/mol
        # rmin: Angstrom
        ai, aj, Emin, rmin = line.split()
        self.nbfix[(ai, aj)] = float(Emin), float(rmin)
        self.nbfix[(aj, ai)] = float(Emin), float(rmin)

    def _parse_hbond(self, line, comment):
        if ('cuthb' in line.lower()) or ('ctonha' in line.lower()):
            return
        ai, aj, prm1, prm2 = line.split()
        self.hbonds[(ai, aj)] = float(prm1), float(prm2)
        self.hbonds[(aj, ai)] = float(prm1), float(prm2)

    def __add__(self, other):
        assert isinstance(other, ParseParameters)
        new = copy.deepcopy(self)
        for attr in self.data_vars:
            getattr(new, attr).update(getattr(other, attr))

    def _prepare_cmaps_fallback(self):
        for key, cmap in self.cmaps.item()
            n = cmap.pop()
            cmap = [[cmap[i*n+j] for j in range(n)] for i in range(n)]
            self.cmaps[key] = cmap
            self.cmaps_bicubic_coeffs[key] = None
            #step = 2*(2.0*math.pi/n) # in radians
            #step2 = step*step
            #zeros = lambda : [[0 for j in range(n)] for i in range(n)]
            #dx, dy, dxy = zeros(), zeros(), zeros()
            #for x0 in xranage(n):
            #    xm, xp = x0 - 1, (x0 + 1)%n
            #    for y0 in xrange(n):
            #        ym, yp = y0 - 1, (y0 + 1)%n
            #        dx[x0][y0] = (cmap[xp][y0]-cmap[xm][y0]) / step
            #        dy[x0][y0] = (cmap[x0][yp]-cmap[x0][ym]) / step
            #        dxy[x0][y0] = (cmap[xp][yp]-cmap[xp][ym]-cmap[xm][yp]+cmap[xm][ym]) / step2          
            # self.cmaps_grad[key] = dx, dy, dxy

    @utils.requires_numpy
    def _prepare_cmaps(self):

        AI = np.array(
             [[ 1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
              [ 0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
              [-3,  3,  0,  0, -2, -1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
              [ 2, -2,  0,  0,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],
              [ 0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0],
              [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0],
              [ 0,  0,  0,  0,  0,  0,  0,  0, -3,  3,  0,  0, -2, -1,  0,  0],
              [ 0,  0,  0,  0,  0,  0,  0,  0,  2, -2,  0,  0,  1,  1,  0,  0],
              [-3,  0,  3,  0,  0,  0,  0,  0, -2,  0, -1,  0,  0,  0,  0,  0],
              [ 0,  0,  0,  0, -3,  0,  3,  0,  0,  0,  0,  0, -2,  0, -1,  0],
              [ 9, -9, -9,  9,  6,  3, -6, -3,  6, -6,  3, -3,  4,  2,  2,  1],
              [-6,  6,  6, -6, -3, -3,  3,  3, -4,  4, -2,  2, -2, -2, -1, -1],
              [ 2,  0, -2,  0,  0,  0,  0,  0,  1,  0,  1,  0,  0,  0,  0,  0],
              [ 0,  0,  0,  0,  2,  0, -2,  0,  0,  0,  0,  0,  1,  0,  1,  0],
              [-6,  6,  6, -6, -4, -2,  4,  2, -3,  3, -3,  3, -2, -1, -2, -1],
              [ 4, -4, -4,  4,  2,  2, -2, -2,  2, -2,  2, -2,  1,  1,  1,  1]])

        for key, cmap in self.cmaps.item()
            n = cmap.pop(0) 
            if n != 24:
                warnings.warn("The cmap is not of the expected size %d != 24"%n)
            cmap = np.array(cmap)
            self.cmaps[key] = cmap

            # get finite central differences
            # nb. LAMMPS uses bicubic splines to calc gradients - is it necessary??
            # https://github.com/lammps/lammps/blob/master/src/MOLECULE/fix_cmap.cpp
            dx, dy, dxy = np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n))
            for x0 in xrange(n):
                xm = (x0 - 1)%n
                xp = (x0 + 1)%n
                for y0 in xrange(n):
                    ym = (y0 - 1)%n
                    yp = (y0 + 1)%n
                    dx[x0, y0] = cmap[xp, y0]-cmap[xm, y0]
                    dy[x0, y0] = cmap[x0, yp]-cmap[x0, ym]
                    dxy[x0, y0] = cmap[xp, yp]-cmap[xp, ym]-cmap[xm, yp]+cmap[xm, ym]
            dx /= 2.0 # * step_rad
            dy /= 2.0 # * step_rad
            dxy /= 4.0 # * step_rad * step_rad
            # self.cmaps_grad[key] = map(np.array, [dx, dy, dxy])

            # calculate bicubic coefficents 
            aijs = np.empty((n, n, 4, 4))
            for i0 in xrange(n):
                for j0 in xrange(n):
                    _g, _dgdx, _dgdy, _dgdxy = (np.zeros((2,2)) for _ in range(4))
                    for t in [0, 1]:
                        i = (i0+t)%n
                        for u in [0, 1]:
                            j = (j0+u)%n
                            _g[t, u] = cmap[i, j]
                            _dgdx[t, u] = dgdx[i, j] # * step
                            _dgdy[t, u] = dgdy[i, j] # * step
                            _dgdxy[t, u] = dgdxy[i, j] # * step * step
                    vec = [mat.T.flatten() for mat in [_g, _dgdx, _dgdy, _dgdxy]]
                    vec = reduce(np.append, vec)
                    aij = np.dot(AI, vec)
                    aij = aij.reshape(4, 4).T
                    aijs[i0, j0, :, :] = aij
            self.cmaps_bicubic_coeffs[key] = aijs
            
################################################################################

# Residue topology

################################################################################

def append_if_absent(s, alist):
    rev = tuple(reversed(alist))
    if not (alist in s or rev in s):
        s.append(alist)

def delta(a): 
    return {'+': 1, '-': -1}.get(a[0], 0)

def strip(a):
    return a.lstrip('+-*')

class ParseTopology():

    data_vars = ['ispatch', 'isres', 'charge', 'structure', 'groups', 'atoms',
                  'bonds', 'doubles', 'impropers', 'cmaps', 'angles', 'dihedrals']

    def __init__(self, top_file):

        # parse a CHARMM topology file or unpickle a previously parsed file
        # for alternative search for cgenff_charmm2gmx.py by E. Prabhu Raman

        if isinstance(top_file, dict):
            self._from_dict(top_file)

        elif top_file.endswith('.pkl'):
            with open(top_file) as f:
                self._from_dict(pickle.load(f))

        else:
            for var in self.data_vars:
                setattr(self, var, {})

            self.mass = {}          # atoms to mass
            self.elem = {}          # atoms to elements
            self.desc = {}          # atoms to description
            self._decl = []         # DECLare default atoms in prev/next residue
            self._defa = ''         # DEFAult patches for first and last terminals
            self._auto = ''         # AUTOgenerate angle command
            self._ic = {}           # Internal Coordinate seeds
            self._group = []        # current group (reset by _parse_group)
            self._residues = []     # all parsed RES and PRES records

            parse = self._parse_atom
            parsers = {'MASS': self._parse_mass,
                       'DECL': self._parse_decl,
                       'DEFA': self._parse_defa,
                       'AUTO': self._parse_auto,
                       'RESI': self._parse_resi, # residue
                       'PRES': self._parse_resi, # patch residue
                       'GROU': self._parse_grou,
                       'ATOM': self._parse_atom,
                       'BOND': self._parse_bond,
                       'DOUB': self._parse_doub,
                       'IMPR': self._parse_impr,
                       'CMAP': self._parse_cmap,
                       'DONO': self._parse_donor,
                       'ACCE': self._parse_acceptor}

            with open(top_file, 'r') as f:
                lines = f.readlines()

            for line in lines:
                line = line.strip()
                if len(line) == 0:
                    continue
                elif line[0] in ['*', '!']:
                    continue
                elif line == 'END':
                    continue
                elif line[0:2] == 'IC':
                    parse = self._parse_ic # internal coordinates
                elif line[0:4] in parsers:
                    parse = parsers[line[0:4]]
                line = line.split('!', 1) + ['']
                parse(line[0], line[1])

            for res in self._residues:
                self._autogen(res)

    def _parse_mass(self, line, comment):
        idx, name, mass, element = line.split()[1:]
        self.mass[name] = float(mass)
        self.elem[name] = element
        self.desc[name] = comment.strip()

    def _parse_decl(self, line, comment):
        self._decl.append(line.split()[1])

    def _parse_defa(self, line, comment):
        self._defa = line[4:]

    def _parse_auto(self, line, comment):
        self._auto = line[4:]

    def _parse_resi(self, line, comment):
        typ, res, charge = line.split()
        self.ispatch[res] = typ == 'PRES'
        self.isres[res] = typ == 'RESI'
        self.charge[res] = float(charge)
        self.structure[res] = ''
        self.groups[res] = []
        self._residues.append(res)
        self._res = res
        self.atoms[res] = self._atoms =  []
        self.bonds[res] = self._bonds = []
        self.doubles[res] = self._doubles = []
        self.impropers[res] = self._impropers = []
        self.cmaps[res] = self._cmaps = []
        self.angles_ic[res] = self._angles_ic = []
        self.dihedrals_ic[res] = self._dihedrals_ic = []

    def _parse_group(self, line, comment):
        self._group = []
        self.groups[self._res].append(self._group)
        self._build_structure(comment)

    def _build_structure(self, comment):
        if comment:
            s = self.structure[self._res]
            self.structure[self._res] = s + comment.strip('!') + '\n'

    def _parse_atom(self, line, comment):
        name, typ, charge = line.split()[1:]
        charge = float(charge)
        # use typ for parameter lookup 
        atom = [name, typ, charge, self._group]
        self._atoms.append(atom)
        self._group.append(atom)
        self._build_structure(comment)

    def _parse_bond(self, line, comment):
        line = line.split()[1:]
        for i in range(len(line)//2):
            ai, aj = line[2*i], line[2*i+1]
            di, dj = delta(ai), delta(aj) # ai should be 0
            ai, aj = strip(ai), strip(aj) # remove +/-
            assert di == 0
            self._bonds.append([[ai, aj], [di, dj]])
        self._build_structure(comment)

    def _parse_doub(self, line, comment):
        line = line.split()[1:]
        for i in range(len(line)//2):
            ai, aj = line[2*i], line[2*i+1]
            di, dj = delta(ai), delta(aj) # both should be 0
            ai, aj = strip(ai), strip(aj) # remove +/-
            self._doubles.append([[ai, aj], [di, dj]])
        self._build_structure(comment)

    def _parse_impr(self, line, comment):
        # IMPR A B C D in the RTF defines the improper angle between two planes 
        # defined by (A,B,C) and (B,C,D) with first atom (A) in the center.

        # compare IMPR with IC lines (with third atom in the center):
        # IMPR  N  -C  CA  H     C  CA  +N  O    CA  N  C  CB
        # IC   -C  CA  *N  H    +N  CA  *C  O     N  C *CA CB

        # www.charmm.org/charmm/documentation/by-version/c37b1/params/doc/io/
        # note: phi will have different value for IC vs IMPR when using same
        # dihedral formula as identty of atoms in planes ((ABC) and (BCD)) differ

        # connectivity IMPR:        connectivity IC:
        #     A                           C
        #   / | \                       / | \
        #  C  |  B                     B  |  A
        #     |                           |  
        #     D                           D

        # planes IMPR:              planes IC:
        #     A                           C
        #   /   \                       / | \ 
        #  C-----B                     B  |--A
        #   \   /                       \ | 
        #     D                           D

        line = line.split()[1:]
        for i in range(len(line)//4):
            atoms = line[4*i], line[4*i+1], line[4*i+2], line[4*i+3]
            di, dj, dk, dl = [delta(a) for a in atoms] # di, dj, dl should be 0
            ai, aj, ak, al = [strip(a) for a in atoms] # remove +/-
            self._impropers.append([[ai, aj, ak, al], [di, dj, dk, dl]])
        self._build_structure(comment)

    def _parse_cmap(self, line, comment):
        atoms = line.split()[1:9]
        deltas = [delta(a) for a in atoms]
        atoms = [strip(a) for a in atoms]
        self._cmaps.append([atoms, deltas])
        self._build_structure(comment)

    def _parse_donor(self, line, comment):
        pass # not implemented

    def _parse_acceptor(self, line, comment):
        pass # not implemented

    def _parse_ic(self, line, comment):

        # IC A B *C D [bond(AC)] [angle(BCA)] [improper(ABCD)] [angle(BCD)] [bond(CD)]
        # IC A B  C D [bond(AB)] [angle(ABC)] [dihedral(ABCD)] [angle(BCD)] [bond(CD)]

        atoms = line.split()[1:5]
        di, dj, dk, dl = [delta(a) for a in atoms]
        ai, aj, ak, al = [strip(a) for a in atoms]
        if atoms[2].startwith('*'):
            append_if_absent(self._angles_ic, [[ai, ak, aj], [di, dk, dj]]) # A C B
            append_if_absent(self._angles_ic, [[aj, ak, al], [dj, dk, dl]]) # B C D
            append_if_absent(self._angles_ic, [[ai, ak, al], [di, dk, dl]]) # A C D
        else:
            append_if_absent(self._dihedrals_ic, [[ai, aj, ak, al], [di, dj, dk, dl]])
            append_if_absent(self._angles_ic, [[ai, aj, ak], [di, dj, dk]]) # A B C
            append_if_absent(self._angles_ic, [[aj, ak, al], [dj, dk, dl]]) # B C D

    # persistence and combination of ParseTopology objects

    def save(self, fname):
        assert fname.endswith('.pkl')
        pickle.dump(self._to_dict(), fname)

    def _from_dict(self, d):
        for attr in self.data_vars + ['mass', 'elem', 'desc']:
            setattr(self, attr, d[attr])

    def _to_dict(self):
        attrs = self.data_vars + ['mass', 'elem', 'desc']
        return {attr: getattr(self, attr) for attr in attrs}

    def __add__(self, other):
        new = copy.deepcopy(self)
        for attr in self.data_vars + ['mass', 'elem', 'desc']:
            getattr(new, attr).update(getattr(other, attr))
        return new

    def __contains__(self, res):
        return res in self._residues # checks res and pres

    def _autogen(self, res):

        bonds = self.bonds[res] + self.doubles[res]
        angles = self.angles[res] = []
        dihedrals = self.dihedrals[res] = []

        # generate angles when pairs share a common atom
        for ii, ((ai, aj), (di, dj)) in enumerate(bonds):
            for (ak, al), (dk, dl) in bonds[ii+1:]:
                if aj == ak:
                    angles.append([[ai, aj, al], [di, dj, dl]]) # i-jk-l
                elif ai == al:
                    angles.append([[ak, ai, aj], [dk, di, dj]]) # k-li-j       
                elif ai == ak:
                    angles.append([[aj, ai, al], [dj, di, dl]]) # j-ik-l
                elif aj == al:
                    angles.append([[ai, aj, ak], [di, dj, dk]]) # i-jl-k

        # add (-C, N, CA) and (-C, N, HN) angles
        append_if_absent(angles, ['C', 'N', 'CA'], [-1, 0, 0])
        append_if_absent(angles, ['C', 'N', 'HN'], [-1, 0, 0])

        # generate dihedrals when triples share two bonded atoms
        for ii, ((ai, aj, ak), (di, dj, dk)) in enumerate(angles):
            for (al, am, an), (dl, dm, dn) in angles[ii+1:]:
                if aj == al:
                    if ak == am:
                        # i-jl-km-n
                        dihedrals.append([[ai, aj, am, an], [di, dk, dm, dn]])
                    elif ai == am:
                        # k-jl-im-n
                        dihedrals.append([[ak, aj, am, an], [dk, dj, dm, dn]])
                elif aj == an:
                    if ak == am: 
                        # i-jn-km-l
                        dihedrals.append([[ai, aj, am, al], [di, dj, dm, dl]])
                    elif ai == am:
                        # k-jn-im-l
                        dihedrals.append([[ak, aj, am, al], [dk, dj, dm, dl]])

        # add (CA, C, N+, CA+) dihedral
        append_if_absent(dihedrals, [['CA', 'C', 'N', 'CA'], [0, 0, 1, 1]])
