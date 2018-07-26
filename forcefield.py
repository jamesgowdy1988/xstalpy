'''
class Bond():

    def energy(self):
        d = self.b - self.b0
        self.V = self.kb*d*d
        return self.V

    def energy_update_forces(self):
        d = self.b - self.b0
        self.V = self.kb*d*d
        dV = 2*self.kb*d
        self.a1.f -= self.e*dV # r21/b21*dV
        self.a2.f += self.e*dV # r12/b12*dV
        return self.V

class Angle():

    def energy(self):
        d = self.theta - self.theta0
        self.V = self.ktheta*d*d
        return self.V

    def energy_update_forces(self):
        d = self.theta - self.theta0
        self.V = self.ktheta*d*d
        dV = 2*self.ktheta*d
        self.r12.a1.f -= self.d1*dV
        self.r12.a2.f -= self.d2*dV
        self.r23.a2.f -= self.d3*dV
        return self.V

class DihedralEnergy():

    def energy(self):
        self.V = 0.0
        for kchi, n, delta in self.params:
            self.V += kchi * (1 + np.cos(n*self.chi - delta))
        return self.V

    def energy_update_forces(self):
        self.V = 0.0
        for kchi, n, delta in self.params:
            arg = n*self.chi - delta
            self.V += kchi * (1 + np.cos(arg))
            dV = - kchi*n*np.sin(arg)
            self.r12.a1.f -= self.d1 * dV
            self.r12.a2.f -= self.d2 * dV
            self.r24.a1.f -= self.d3 * dV
            self.r24.a2.f -= self.d4 * dV
        return self.V

class ImproperEnergy():

    def energy(self):
        d = self.chi - self.psi0
        self.V = self.kpsi*d*d
        return self.V

    def energy_update_forces(self):
        d = self.chi - self.psi0
        self.V = self.kpsi*d*d
        dV = 2*self.kpsi*d
        self.r12.a1.f -= self.d1 * dV
        self.r12.a2.f -= self.d2 * dV
        self.r34.a1.f -= self.d3 * dV
        self.r34.a2.f -= self.d4 * dV
        return self.V

class NonbondedEnergy():

    # for Lennard Jones 6-12 interactions only 

    def energy(self):
        self.V = self.c12/self.b**12 - self.c6/self.b**6
        return self.V

    def energy_update_forces(self):
        r6 = self.b**6
        V6 = self.c6/
        V12 = self.c12/(r6*r6)
        self.V = V12 - V6
        dV = (-12*V12 + 6*V6)/self.b
        self.a1.f += - self.e*dV
        self.a2.f += + self.e*dV
        return self.V

class CmapEnergy():

    # the CMAP energy correction is applied to the phi-psi backbone dihedrals
    # and depends on their paired value. It helps recreate the ramachandran plot

    def energy(self):
        phi = self.phi1234.chi
        psi = self.psi1234.chi
        i = self.angle_range[int(round(phi/15.)) + self.midpoint]
        j = self.angle_range[int(round(psi/15.)) + self.midpoint]
        self.V = self.cmap[i, j]
        return self.V
'''


################################################################################

# Potential energy and force field computation

################################################################################

'''
class BondedForceField():

    # implemented without vectorisation to test C code

    data_vars = ['Vbonds', 'Vangles', 'Vubs', 'Vdihedrals', 'Vimpropers', 
                 'Vcmaps', 'Vnonbonded_14']

    def __init__(self, molc):
        # get the representation of the molecule if not provided
        if isinstance(molc, topology.Molecule):
            self.molc = molc
        elif isinstance(molc, pdbio.ParsePDB):
            self.molc = topology.Molecule(molc)

        # initial coordinates
        self._r = np.array([self.molc.pdb.x, self.molc.pdb.y, self.molc.pdb.z]).T

    def energy_bonded(self, r=None):
        # calculate bonded energy terms without recalculation of intermediates
        r = r or self._r
        for var in self.data_vars[:-2]:
            setattr(self, var, 0.0)

        for bond in self.molc.Bonds.values():
            bond.calc(r[bond.a1.i], r[bond.a2.i])
            self.Vbonds += bond.energy()

        for ub in self.molc.Ubs.values():
            ub.calc(r[bond.a1.i], r[bond.a2.i])
            self.Vubs += ub.energy()

        for nbond in self.molc.Nonbonded_14.values():
            nbond.calc(r[bond.a1.i], r[bond.a2.i])
            self.Vnonbonded_14 += nbond.energy()

        for angle in self.molc.Angles.values():
            angle.calc()
            self.Vangles += angle.energy()

        for dihedral in self.molc.Dihedrals.values():
            dihedral.calc()
            self.Vdihedrals += dihedral.energy()

        for improper in self.molc.Impropers.values():
            improper.calc()
            self.Vimpropers += improper.energy()

        for cmap in self.molc.Cmaps.values():
            self.Vcmaps += cmap.energy()

        self.Vbonded = (self.Vbonds + self.Vubs + self.Vangles + self.Vcmaps +
                        self.Vdihedrals + self.Vimpropers + self.Vnonbonded_14)
        return self.Vbonded
'''
    def energy_and_force_bonded(self, r=None, clear_forces=True):

        data_vars = ['Vbonds', 'Vangles', 'Vubs', 'Vdihedrals', 'Vimpropers', 
                     'Vcmaps', 'Vnonbonded_14']

        r = r or self._r

        # initialise energy terms at zero
        for term in data_vars:
            setattr(self, term, 0.0)

        # initialise forces at zero for each Atom instance in molecule
        if clear_forces:
            self.molc.clear_forces()
        
        for (i, j), bond in self.molc.Bonds.items():
            b = bond.calc(r[i], r[j])
            self.Vbonds += bond.energy_update_forces()

        for (i, k), ub in self.molc.Ubs.items():
            b = ub.calc(r[i], r[k])
            self.Vubs += ub.energy_update_forces()

        for (i, l), nbond in self.molc.Nonbonded_14.items():
            b = nbond.calc(r[i], r[l])
            self.Vnonbonded_14 += nbond.energy_update_forces()

        for angle in self.molc.Angles.values():
            theta = angle.calc()
            angle.grad()
            self.Vangles += angle.energy_update_forces()

        for dihedral in self.molc.Dihedrals.values():
            dihedral.grad()
            chi = dihedral.calc_with_grad() # check
            self.Vdihedrals += dihedral.energy_update_forces()

        for improper in self.molc.Impropers.values():
            improper.grad()
            chi = improper.calc_with_grad()
            self.Vimpropers += improper.energy_update_forces()

        for cmap in self.molc.Cmaps.values():
            # TODO: force terms due to CMAP correction not accounted for
            self.Vcmaps += cmap.energy()

        # copy forces on individual atoms to a single array
        F = np.zero_like(r)
        for i, a in self.Atoms.items():
            F[a.i, :] = a.f

        # sum all bonded energy terms calculated above
        self.Vbonded = sum(getattr(self, term) for term in data_vars)

        return self.Vbonded, F

class NonBondedForceField(topology.VerletList,
                          NonbondedForceField_QEnergy,
                          NonbondedForceField_QEnergyForce,
                          NonbondedForceField_QLJEnergy,
                          NonbondedForceField_QLJEnergyForce,
                          NonbondedForceField_NaiveQLJEnergy):

    def __init__(self, molc):

        if isinstance(molc, topology.Molecule):
            self.molc = molc
        elif isinstance(molc, pdbio.ParsePDB):
            self.molc = topology.Molecule(molc)

        self._init_nb(molc)

        # initial coordinates
        self._r = np.array([self.molc.pdb.x, self.molc.pdb.y, self.molc.pdb.z]).T

    def energy(self, r=None, **kwargs):

        # get keyword arguments 
        kmax = kwargs.get('kmax', 4)
        r6 = kwargs.get('r6', 'direct')
        r1 = kwargs.get('r1', 'pme')
        scale_Vq = kwargs.get('scale_Vq', False)

        r = r or self._r
        a = self._alpha_q
        b = self._alpha_lj

        elif r1 == r6 == 'pme':
            Vsel = self._calc_ewald_self(a, b)
            Vdir = self._calc_ewald_real_energy_only(r, a, b)
            Vrec = self._calc_pme_recip_energy_only(r, a, b, kmax)
            self.Vq = Vsel[0] + Vdir[0] + Vrec[0] 
            self.VLJ = Vdir[2] - (Vsel[1] + Vdir[1] + Vrec[1]) # signs?

        elif method_r1 == include_r6 == 'ewald':
            Vsel = self._calc_ewald_self(a, b)
            Vdir = self._calc_ewald_real_energy_only(r, a, b)
            Vrec = self._calc_ewald_recip_energy_only(r, a, b, kmax)
            Vq = Vsel[0] + Vdir[0] + Vrec[0] 
            VLJ = Vdir[2] - (Vsel[1] + Vdir[1] + Vrec[1]) # signs?

        elif method_r1 == 'pme' and method_r6 == 'direct':
            Vq = self._calc_qpme_energy_only(r, a)
            VLJ = self._calc_ljnaive_energy_only(r)

        elif method_r1 == 'ewald' and method_r6 == 'direct':
            Vq = self._calc_qewald_energy_only(r, a, kmax)
            VLJ = self._calc_ljnaive_energy_only(r)

        elif method_r1 == method_r6 == 'direct':
            warnings.warn('Computing coulombic interactions without PME/Ewald')
            Vq, VLJ = self._calc_naive_energy_only(r)

        else:
            raise ValueError('Unsupported method_r1 and method_r6 combination')

        if scale_Vq:
            Vq *= 33.2/78.0 # 33.2 vacuum when [V] = kcal/mol and [r] = Angstrom

        self.Vq = Vq
        self.VLJ = VLJ
        self.Vnonbonded = self.Vq + self.VLJ
        return self.Vnonbonded

    def energy_and_force_nonbonded(self, r=None, **kwargs):

        r = r or self._r
        kmax = kwargs.get('kmax', 4)
        method_r6 = kwargs.get('method_r6', 'direct')
        method_r1 = kwargs.get('method_r1', 'pme')
        scale_Vq = kwargs.get('scale_Vq', False)

        a = self._alpha_q
        b = self._alpha_lj

        def _combine(q, c6, c12):
            # Vq, VLJ, 0, 0 = Vsel
            # Vq, VLJ, Fq, FLJ = Vrec
            # Vq, VLJ, Fq, FLJ, VLJ, FLJ = Vdir
            _Vq = Vsel[q] + Vdir[q] + Vrec[q]
            _VLJ = Vdir[c12] - (Vsel[c6] + Vdir[c6] + Vrec[c6]) # sign???
            return _Vq, _VLJ

        if method_r1 == method_r6 == 'pme':
            Vsel = self._calc_ewald_self(a, b)
            Vdir = self._calc_ewald_real(r, a, b)
            Vrec = self._calc_pme_recip(r, a, b, kmax)
            Vq, VLJ = _combine(0, 1, 4)
            Fq, FLJ = _combine(2, 3, 5)

        elif method_r1 == include_r6 == 'ewald':
            Vsel = self._calc_ewald_self(a, b)
            Vdir = self._calc_ewald_real(r, a, b)
            Vrec = self._calc_ewald_recip(r, a, b, kmax)
            Vq, VLJ = _combine(0, 1, 4)
            Fq, FLJ = _combine(2, 3, 5)

        elif method_r1 == 'pme' and method_r6 == 'direct':
            Vq, Fq = self._calc_qpme(r, a)
            VLJ, FLJ = self._calc_ljnaive(r)

        elif method_r1 == 'ewald' and method_r6 == 'direct':
            Vq, Fq = self._calc_qewald(r, a, kmax)
            VLJ, FLJ = self._calc_ljnaive(r)

        elif method_r1 == method_r6 == 'direct':
            warnings.warn('Computing coulombic interactions without PME/Ewald')
            Vq, VLJ, Fq, FLJ = self._calc_naive(r)

        else:
            raise ValueError('Unsupported method_r1 and method_r6 combination')

        if scale_Vq:
            Vq *= 33.2/78.0 # 33.2 vacuum when [V] = kcal/mol and [r] = Angstrom
            Fq *= 33.2/78.0 # 78.0 dielectric constant of water

        self.Vq = Vq
        self.VLJ = VLJ
        self.Vnonbonded = self.Vq + self.VLJ
        F = Fq + FLJ
        return self.Vnonbonded, F

class Forcefield(BondedForceField, NonBondedForceField):

    def __init__(self, molc)

        if isinstance(molc, topology.Molecule):
            self.molc = molc
        elif isinstance(molc, pdbio.ParsePDB):
            self.molc = topology.Molecule(molc)

        self._init_nb(molc)
        self._r = np.array([self.molc.pdb.x, self.molc.pdb.y, self.molc.pdb.z]).T

    def energy(self, r=None, **kwargs):
        # use with opt functionality when called 
        Vb = energy_and_force_bonded(r)
        Vnb = energy_and_force_nonbonded(r, **kwargs)
        return Vb + Vnb


