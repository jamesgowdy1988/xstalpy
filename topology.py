"""topology.py - build a molecular topology by generating internal coordinates"""

from __future__ import division

import pickle
import collections
import copy
import math

import toppario
import terms
import nonbonded

__author__ = "James Gowdy" 
__email__ = "jamesgowdy1988@gmail.com"

################################################################################

# Generate a molecule and all its internal coordinates 

################################################################################

class TopologyError(Exception):
    pass

class Residues(object):

    def __init__(self, pdb):

        # re-number resids so that they're unique for segment gen
        resids, original = pdb.get_consecutive_resid(start=0)
        groups = pdb.groupby(zip(resids, pdb.residue))
        self.residues = zip(*groups)[0]
        self.nres = len(residues)

        # map atoms in each residues of this segment to unique atom index
        self.idxs = {}
        for (ires, res), group in groups:
            self.idxs[ires] = {}
            for atom in group:
                i = atom[pdb.data_vars.index('i')]
                ai = atom[pdb.data_vars.index('atom')]
                self.idxs[ires][ai] = i

        # indicate if a delta +1 or -1 linker atom is beyond the terminus
        for ai in set(pdb.atom):
            self.idxs[-1][ai] = None   # atom is beyond N-terminus
            self.idxs[nres][ai] = None # atom is beyond C-terminus

    def get_idx(self, ires, atoms, deltas):
        return [self.idxs[ires+d][a] for a, d in zip(atoms, deltas)]

    def loop(self, _internal_coords):
        # deltas are either -1/0/+1 indicating prev/this/next residue
        for ires, res in self.residues:
            if ires == 0 or ires == self.nres-1:
                for atoms, deltas in _internal_coords[res]:
                    if None not in idx:
                        yield self.get_idx(ires, atoms, deltas)
            else:
                for atoms, deltas in _internal_coords[res]:
                    yield self.get_idx(ires, atoms, deltas)

class Molecule(object):

    data_vars = ['atoms', 'bonds', 'ubs', 'nonbonded_14', 'angles', 'dihedrals', 
                'impropers', 'cmaps', 'nonbonded']

    def __init__(self, pdb, top=None, par=None):

        # default parameter and topology files (available for free online)
        self.top = top or toppario.ParseTopology('top_all22_prot.rtf')
        self.par = par or toppario.ParseParameters('par_all22_prot.prm')

        # re-number unique atoms so they are contiguous
        self.pdb = pdb.get_unique_atoms()
        self.pdb.i = range(0, len(self.pdb))

        # pairwise interactions to consider for Verlet list
        n = len(self.pdb)
        self.include = [[int(i < j) for j in xrange(n)] for i in xrange(n)]
        
        # initialise dicts to store bonds, angles, dihedral instances etc.
        for attr in self.data_vars:
            setattr(self, attr, [])
            serattr(self, attr.title(), {})

        # break up the chain/segments (we assume they're disconnected)
        self._residues = {}
        for chain in self.pdb.split_chains():
            for seg in chain.split_segments():
                self._generate_atoms(seg)
                self._generate_bonded(seg)

        # generate the atom pairs after building self.include for the VerletList
        self._generate_nonbonded()

        # order all internal coordinates for evaluation and energy calculation
        self._create_energy_namespace()
        self._traverse_dependence_hierachy()

    def _get_residues(self, pdb):
        nmaxkey = 1000000000
        unique_key = hash(repr(pdb.__dict__)[:nmaxkey]) + id(pdb)
        if unique_key not in self._residues:
            self._residues[unique_key] = Residues(pdb)
        return self._residues[unique_key]

    def _generate_atoms(self, pdb):

        # loop over all residues in the pdb segment
        res = self._get_residues(pdb)
        for ires, res in res.residues:
            # for each residue loop over its expected atoms
            for ai, typ, qi, group in self.top.atoms[res]:
                if ai not in res.idxs[ires]:
                    raise TopologyError("Missing atom in PDB resid = {}".format(ires))
                i = res.idxs[ires][ai]
                self.add_atom(i, ai, qi, typ)

        assert set(self.Atoms.keys()) == set(self.pdb.i)

    def _generate_bonded(self, pdb):

        res = self._get_residues(pdb)
        
        for i, j in res.loop(self.top.bonds):
            self.add_bond(i, j)

        for i, j in res.loop(self.top.doubles):
            self.add_bond(i, j)

        for i, j, k in res.loop(self.top.angles):
            self.add_angle(i, j, k)

        for i, j, k, l in res.loop(self.top.dihedrals):
            self.add_dihedral(i, j, k, l)

        for i, j, k, l in res.loop(self.top.impropers):
            self.add_improper(i, j, k, l)

        for ijklmnop in res.loop(self.top.cmaps):
            self.add_cmap(*ijklmnop)

    def _generate_nonbonded(self):
        # ignore pairs in bonds, angles, dihedrals, impropers, 1-3, 1-4 interactions
        for i, atom1 in enumerate(self.atoms):
            for atom2 in self.atoms[i:]:
                if self.include[atom1.i][atom2.i]:
                    # apply Lorentz-Berthelot combination rules
                    eps_ij = calc_eps(atom1.eps, atom2.eps)
                    rmin_ij = calc_rmin(atom1.rmin, atom2.rmin)
                    c6_ij, c12_ij = calc_c6c12(eps_ij, rmin_ij)
                    qiqj = qi*qj
                    pair = [atom1, atom2, qiqj, c6_ij, c12_ij]
                    self.nonbonded.append(pair)

        # these methods will require numpy - move out of __init__?
        S = self.pdb.calc_scale()
        SI = self.pdb.calc_invscale() 
        V = self.pdb.calc_volume()
        vl = nonbonded.VerletList(self.atoms, self.nonbonded, S, SI, V, rcut=12, kmax=10))
        self.verletlist = vl

    def _get_params(self, par, idxs, exlude_pair=None, register=True, errors_on=True):

        # get params if registered for energy calc
        key = tuple(self.Atoms[i].type if i != 'X' else 'X' for i in idxs)

        if exclude_pair is not None:
            i, j = exclude_pair
            self.include[i][j] = self.include[j][i] = 0

        if not register:
            return None, None
        elif key in par:
            return par[key]
        elif errors_on:
            error = "Params for {}({} = {}) not in param file"
            raise KeyError(error.format(par.name, idx, key))
        else:
            return None
            
    def add_atom(self, i, ai, q, typ):
        m = self.top.mass[typ]
        eps, rmin = self.par.nonbonded[typ]
        atom = terms.Atom(i, ai, q, typ, m, eps, rmin)
        self.Atoms[i] = atom
        self.atoms.append(atom)

    def add_bond(self, i, j, register=True):

        # create bond
        kb, b0 = self._get_params(self.par.bonds, (i, j), (i, j), register)
        A = self.Atoms[i]
        B = self.Atoms[j]
        bond = terms.Bond(A, B, kb, b0, register)

        # save to molecule
        self.Bonds[(i, j)] = bond
        self.Bonds[(j, i)] = -bond # RevBond paired to Bond
        self.bonds.append(bond)

    def add_angle(self, i, j, k, register=True):

        # create angle
        ktheta, theta0 = self._get_params(self.par.bonds, (i, j, k), (i, k), register)
        AB = self.Bonds[(i, j)]
        BC = self.Bonds[(j, k)]
        angle = terms.Angle(AB, BC, ktheta, theta0, register)

        # save
        self.Angles[(i, j, k)] = angle
        #self.Angles[(k, j, i)] = -angle
        self.angles.append((i, j, k))

        # check for (1,3) interactions
        if register:
            self.add_ub(i, j, k)

    def add_ub(self, i, j, k):

        # create Urey-Bradley bond
        kub, b0 = self._get_params(self.par.ubs, (i, j, k), (i, k))
        A = self.Atoms[i]
        C = self.Atoms[k]
        bond = terms.Ub(A, C, kub, b0, register=True)

        # save
        self.Ubs[(i, k)] = bond
        self.Ubs[(k, i)] = -bond # RevBond paired with bond
        self.ubs.append(bond)

    def add_dihedral(self, i, j, k, l, register=True):

        # get params
        X = 'X'
        p = self.par.dihedrals
        params = (self._get_params(p, (i, j, k, l), (i, l), errors_on=False) or 
                  self._get_params(p, (X, j, k, X), (i, l), errors_on=True))

        # create proper dihedral
        ABC = self.Angles[(i, j, k)]
        BCD = self.Angles[(j, k, l)]
        dihedral = terms.Dihedral(ABC, BCD, params, register)

        # save
        self.Dihedrals[(i, j, k, l)] = dihedral 
        self.Dihedrals[(l, k, j, i)] = -dihedral
        self.dihedrals.append(dihedral)

        # check for (1,4) interactions
        if register:
            self.add_nonbonded_14(i, l) 

   def add_cmap(self, i, j, k, l, m, n, o, p):

        # get params
        key = i, j, k, l, m, n, o, p
        cmap = self._get_params(self.par.cmaps, key)
        aij = self._get_params(self.par.cmaps_bicubic_coeffs, key)
        # dx, dy, dxy = self._get_params(self.par.cmaps_grad, key)

        # create and save CMAP correction - phi/psi should exist - TODO check
        phi = self.Dihedrals[(i, j, k, l)]
        psi = self.Dihedrals[(m, n, o, p)]
        cmap = terms.Cmap(phi, psi, cmap, aij, regitser=True)
        self.cmaps.append(cmap)

    def add_nonbonded_14(self, i, l):

        # get params
        A = self.Atoms[i]
        D = self.Atoms[l]
        par = self.par.nonbonded_14
        if (A.type in par) and (D.type in par):
            eps_ii, rmin_ii = par[A.type]
            eps_ll, rmin_ll = par[D.type]
            c6_il, c12_il = calc_c6c12(calc_eps(eps_ii, eps_ll), 
                                       calc_rmin(rmin_ii, rmin_ll))

            # create and save nonbonded (1,4) interaction
            nbond = terms.Nonbonded_14(A, D, c6_il, c12_il, register=True)
            self.nonbonded_14.append(nbond)

    def _add_dummy_bond(self, j, k):
        if (j, k) in self.Ubs:
            return self.Ubs[(j, k)]
        elif (j, k) not in self.Bond:
            self.add_bond(j, k, register=False) # make dummy
        return self.Bond[(j, k)]

    def _add_dummy_angle(self, i, j, k):
        if (i, j, k) not in self.Angles:
            self.add_angle(i, j, k, register=False) # make dummy
        return self.Angles[(i, j, k)]

    def add_improper(self, i, j, k, l):

        # get params
        X = 'X'
        p = self.par.impropers
        kpsi, psi0 = (self._get_params(p, (i, j, k, l), errors_on=False) or 
                      self._get_params(p, (i, X, X, j), errors_on=True))

        # check for normal bond
        # assert (i, j) in self.Bonds

        # create dummy bond or use through-space bond
        BC = self._add_dummy_bond(j, k)
        CD = self._add_dummy_bond(k, l)

        # create dummy angles (only need to be evaluated for grad of improper)
        ABC = self._add_dummy_angle(i, j, k) # uses AB, BC
        BCD = self._add_dummy_angle(j, k, l) # uses BC, CD

        # create/save improper dihedral
        improper = terms.Improper(ABC, BCD, kpsi, psi0, register=True)        
        self.impropers.append(improper)

    # methods for evaluation of molecules internal coordinates and energies across

    def _create_energy_namespace(self):

        class NameSpace:
            @classmethod
            def _reset(cls):
                for name in vars(cls):
                    if name[0] != '_': 
                        setattr(cls, name, 0.)
            @classmethod
            def _todict(cls)
                return {k: v for k, v in cls.__dict__.items() if k[0] != '_'}

        namespace = NameSpace.__dict__ # faster than using getattr and setattr??

        class Reference():
            def __init__(self, name):
                self.name = name
                namespace[name] = 0 
            def __iadd__(self, val):
                namespace[name] = val + namespace[name]
                return self

        self.energies = NameSpace
        self._make_energy_reference = Reference

    def _traverse_dependence_hierachy(self):

        self._for_energy = []
        self._for_calc = []

        def append_ics(typ, Vtyp)
            ics = getattr(self, typ)
            self._for_energy += [(ic, Vtyp) for ic in ics if ic.registered]
            self._for_calc += ics

        types = ['bonds', 'ubs', 'nonbonded_14', # 1st no dependecies
                 'angles',                       # 2nd dependent on bonds/ubs
                 'dihedrals', 'impropers',       # 3rd dependent on angles
                 'cmaps']:                       # 4th dependent on dihedrals

        # pass a reference to the relevant self.bonding_energies.Vtyp to be updated
        for typ in types:
            ref = self._make_energy_reference('V'+typ)
            append_ics(typ, ref)

    def calc(self):
        if not hasattr(self, '_for_calc'): 
            self._traverse_dependence_hierachy()
        for item in self._for_calc:
            item.calc()

    def _energy(self, update_force, Vwarn):
        if not hasattr(self, '_for_energy'):
            assert False, "have not called calc()"

        # set up
        self.bonded_energies._reset()
        self.bonded_energies.Vtotal = 0

        # compute energy for each registered internal coordinate
        if update_force:
            for item, Vtype in self._for_energy:
                Vitem = item.energy_update_forces()
                if Vitem > Vwarn:
                    warnings.warn("Energy of {} = {}".format(item, Vitem))
                Vtype += Vitem
                self.bonded_energies.Vtotal += Vitem
        else:
            for item, Vtype in self._for_energy:
                Vitem = item.energy()
                Vtype += Vitem
                self.bonded_energies.Vtotal += Vitem
        return self.bonded_energies

    def energy(self, Vwarn=1e9):
        self._energy(False, Vwarn)
        self.verletlist.energy()

    def energy_update_forces(self, Vwarn=1e9):
        self._energy(False, Vwarn)
        self.verletlist.energy_update_forces()

    # accessing and setting member atom positions, velocities and forces

    def _setter(self, attr, matrix):
        n, three = matrix.shape
        assert (n == len(self.atoms)) and (three == 3)
        for i, atom in enumerate(self.atoms)
            setattr(atom, attr, matrix[i, :])
        setattr(self, '_'+attr, matrix)

    def _getter(self, attr):
        return [getattr(atom, attr) for atom in self.atoms])

    def get_positions(self):
        return self._getter('r')

    def get_forces(self): 
        return self._getter('f')

    def get_velocities(self): 
        return self._getter('v')

    def set_positions(self, r): 
        self._setter('r', r)

    def set_velocities(self, v): 
        self._setter('v', v)

    def set_forces(self, f): 
        self._setter('f', f)

    r = property(fget=get_positions, fset=set_positions)
    v = property(fget=get_velocities, fset=set_velocities)
    f = property(fget=get_forces, fset=set_forces)

    def copy(self):
        return copy.deepcopy(self)
