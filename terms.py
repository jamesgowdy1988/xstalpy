"""terms.py - bonding-type internal coordinates, their energies and force terms"""

from __future__ import division

import math
import warnings
import operator

import utils

try:
    import numpy as np
    TURN_OFF_NUMPY = False
except ImportError as NUMPY_ERROR:
    warnings.warn("numpy is required for full functionality")
    NUMPY_ERROR = utils.edit_error(NUMPY_ERROR, "install needed (www.scipy.org/install.html)")
    TURN_OFF_NUMPY = True
    
__author__ = "James Gowdy" 
__email__ = "jamesgowdy1988@gmail.com"

################################################################################

# Atom representation for molecular dynamics 

################################################################################

class Atom(object):

    _pascal = [math.factorial(6)/(math.factorial(k)*math.factorial(6-k))
               for k in range(0, 7)]
    del k
    
    def __init__(self, i, ai, q, typ, m, eps, rmin, register=True, S=None, SI=None,
                 molecule_id=0):
        self.i = i                     # unique atom index
        self.a = ai                    # atom name/type in PDB
        self.q = q                     # charge
        self.m = m                     # mass
        self.type = typ                # atom type in CHARMM param
        self.eps = eps                 # eps_ii LJ parameter
        self.rmin = rmin               # rmin_ii LJ parameter
        self.sigma = rmin*(2**(1./6.)) # r0_ii LJ parameter
        self.SI = SI                   # invese scale matrix - unit cell vector
        self.S = S

        # The following is for treatment of the LJ dispersion (r^-6) term by PME:
        # - Essmann et al (1995) eq. 5.4 on page 8582
        # - Perrman et al (1988) pg 881 halfway down page
        # we can either use a geometric approximation - account for -4 later
        # Vdisp = -4*eps_ij*(sigma_ij/r_ij)^6 
        #       = -4*sqrt(eps_i)sqrt(eps_j)*((sigma_i+sigma_j)/2r_ij)^6
        #       ≈ -4*sqrt(eps_i)sqrt(eps_j)*(sqrt(sigma_i)sqrt(sigma_j)/rij)^6
        #       ≈ -4 * (sqrt(eps_i)*sigma_i^3) * (sqrt(eps_j)*sigma_j^3)
        self.sqrtc6 = math.sqrt(self.eps)*(self.sigma**3)

        # or an expansion of the LB combination rules - account for -1/16 later
        # Vdisp = -4*eps_ij*(sigma_ij/r_ij)^6 
        #       = -4*sqrt(eps_i)sqrt(eps_j)*((sigma_i+sigma_j)/2r_ij)^6
        #       = -2^2/2^6*sqrt(eps_i)sqrt(eps_j)*((sigma_i+sigma_j)/r_ij)^6
        #       ≈ -1/2^4 * sum_k 6!/(k!(6-k)!) sqrt(eps_i)*sigma_i^k sqrt(eps_j)*sigma_j^(6-k)
        self.c6terms = [self._pascal[k]*math.sqrt(eps)*self.sigma**k for k in range(0, 7)]

        self.registered = register    # True unless dummy atoms for constraints?
        self.molecule_id = molecule_id# will be given to all ics
        self._my_dependents = set()   # will use for directly dependent Bonds
        self._my_dependencies = set()

        # values to be overwritten by friend functions:

        self.r = None # cartesian xyz position
        self.v = None # cartesian velocity vector
        self.f = None # total cartesian force vector
        self._vec_mic = None # cartesian correction for minimum image convention

    @property
    def r_mic(self):
        # cartesian xyz position corrected for minimum image convention
        return self.r + self._vec_mic

    def update_mic(self):
        # need to check more
        self._vec_mic = np.zeros(3)
        for i in [2, 1, 0]:
            diff = abs(self.r[i]) - (self.SI[i, i]/2)
            if diff > 0:
                n = math.ceil(diff/self.SI[i, i])
                if self.r[i] < 0:
                    self._vec_mic += n*self.SI[:, i]
                else:
                    self._vec_mic -= n*self.SI[:, i]
        return self._vec_mic

    def zero_force(self):
        self.f = np.zeros((3, 3))

    def __repr__(self):
        return "Atom({}, {})".format(self.i, self.type)     

################################################################################

# Internal coordinate ABCs

################################################################################

class InternalCoordinateError(Exception):
    pass

class InternalCoordinate(object):

    def _get_dependency_set(self, item):
        if isinstance(item, Atom):
            return self._atoms
        elif isinstance(item, Bond_base): # includes RevBond
            return self._bonds 
        elif isinstance(item, Angle_base): # includes RevAngle
            return self._angles
        elif isinstance(item, Dihedral_base):
            return self._dihedrals

    def _set_dependencies(self, *items):
        # initalise for all internal coordinates in dependence hierarchy
        terms = ("_atoms", "_bonds", "_angles", "_dihedrals")
        for term in terms:
            setattr(self, term, set())
        # initialise for direct dependents/dependencies
        self._my_dependents = set()
        self._my_dependencies = set()
        for item in items:
            # build the dependence hierarchy of internal coordinates
            item._my_dependents.add(self)
            self._my_dependencies.add(item)
            # collapsed sets of internal coordinates in the dependence hierarchy
            self._get_dependency_set(item).add(item)
            if isinstance(item, InternalCoordinate):
                for term in terms:
                    getattr(self, term).update(getattr(item, term))
        s = {atom.molecule_id for atom in self._atoms}
        if len(s) > 1:
            raise ValueError("{} has atoms from different molecules")


    def __contains__(self, item):
        s = self._get_dependency_set(item)
        return (item in s) if (s is not None) else (False)

    def __neg__(self):
        return self._reversed

    def __repr__(self):
        typ = type(self).__name__
        group = ", ".join([repr(atom) for atom in self._atoms])
        return "{}({})".format(typ, group)

    # this is the exposed interface which should be implemented for each class

    def calc(self): 
        raise NotImplementedError()
    def grad(self): 
        raise NotImplementedError()
    def energy(self): 
        raise NotImplementedError()
    def energy_update_forces(self): 
        raise NotImplementedError()


################################################################################

# Bond classes

################################################################################

class Bond(InternalCoordinate):
    
    def __init__(self, atom1, atom2, kb, b0, register=True):
        self._set_dependencies(atom1, atom2)
        self._init_param(atom1, atom2, kb, b0, register)
        self._reversed = RevBond(self) # paired bond in reverse

    def _init_param(atom1, atom2, kb, b0, register):
        self.registered = register # for inclusion in energy calc
        self.a1 = atom1
        self.a2 = atom2
        self.kb = kb
        self.b0 = b0

    def calc(self):
        # length of bond vector, b = ||r12|| = ||r2-r1||
        self.r = - self.a1.r + self.a2.r
        self.b2 = sum(self.r*self.r)
        self.b = np.sqrt(self.b2) 
        self.e = self.r/self.b
        return self.b

    def grad(self):
        return self.e, -self.e

    def energy(self):
        d = self.b - self.b0
        self.V = self.kb*d*d
        return self.V

    def energy_update_forces(self):
        # energy
        d = self.b - self.b0
        self.V = self.kb*d*d
        # forces
        dV = 2*self.kb*d
        self.a1.f -= self.e * dV # r21/b21*dV
        self.a2.f += self.e * dV # r12/b12*dV
        return self.V

    if TURN_OFF_NUMPY:
        def calc(self):
            raise NUMPY_ERROR
        def grad(self):
            raise NUMPY_ERROR
        def energy(self):
            raise NUMPY_ERROR
        def energy_update_forces(self):
            raise NUMPY_ERROR

class Ub(Bond):

    def __init__(self, atom1, atom2, kb, b0, register):
        self._set_dependencies(atom1, atom2)
        self._init_param(atom1, atom2, kb, b0, register)
        self._reversed = RevUb(self)

class Nonbonded_14(Bond): 

    # for Lennard Jones 6-12 interactions only 
    def __init__(self, atom1, atom2, c6, c12, register=True):
        self._set_dependencies(atom1, atom2)
        self._init_param(atom1, atom2, c6, c12, register)
        self._reversed = RevNonbonded_14(self)

    def _init_param(self, atom1, atom2, c6, c12, register):
        self.registered = register
        self.a1 = atom1
        self.a2 = atom2
        self.c6 = c6
        self.c12 = c12

    def energy(self):
        self.V = self.c12/self.b**12 - self.c6/self.b**6
        return self.V

    def energy_update_forces(self):
        # energy
        r6 = self.b**6
        V6 = self.c6/
        V12 = self.c12/(r6*r6)
        self.V = V12 - V6
        # forces
        dV = (-12*V12 + 6*V6)/self.b
        self.a1.f += - self.e * dV
        self.a2.f += + self.e * dV
        return self.V

    if TURN_OFF_NUMPY:
        def energy(self):
            raise NUMPY_ERROR
        def energy_update_forces(self):
            raise NUMPY_ERROR

################################################################################

# Angle

################################################################################

class _Angle(InternalCoordinate):
    
    def __init__(self, r12, r23, ktheta, theta0, register=True):
        self._set_dependencies(r12, r23)
        self._init_param(r12, r23, ktheta, theta0, register)
        self._check()
        self._reversed = RevAngle(self)

    def _init_param(self, r12, r23, ktheta, theta0, register):
        self.r12 = r12
        self.r23 = r23
        self.a1 = r12.a1
        self.a2 = r12.a2
        self.a3 = r23.a2
        self.ktheta = ktheta
        self.theta0 = theta0
        self.registered = register 

    def _check(self):
        assert len(self._atoms) == 3
        assert self.r12.a2 == self.r23.a1

    def calc(self):
        # angle between two vectors using the dot product formula
        self.cos = np.dot(-self.r12.e, self.r23.e)
        self.theta = np.arccos(self.cos)
        return self.theta

    def grad(self):
        s = - 1.0/np.sin(self.theta)
        self.d1 = s * (+self.r23.e + self.cos*self.r12.e)/self.r12.b
        self.d3 = s * (-self.r12.e - self.cos*self.r23.e)/self.r23.b
        self.d2 = s * (- self.d1 - self.d2) # CHECK?
        return self.d1, self.d2, self.d3

    def energy(self):
        d = self.theta - self.theta0
        self.V = self.ktheta*d*d
        return self.V

    def energy_update_forces(self):
        # energy
        d = self.theta - self.theta0
        self.V = self.ktheta*d*d
        # forces
        dV = 2*self.ktheta*d
        self.a1.f -= self.d1 * dV
        self.a2.f -= self.d2 * dV
        self.a3.f -= self.d3 * dV
        return self.V

    if TURN_OFF_NUMPY:
        def calc(self):
            raise NUMPY_ERROR
        def grad(self):
            raise NUMPY_ERROR
        def energy(self):
            raise NUMPY_ERROR
        def energy_update_forces(self):
            raise NUMPY_ERROR

################################################################################

# Dihedral classes (proper and improper)

################################################################################

class Dihedral(InternalCoordinate):

    def __init__(self, theta123, theta234, params, register=True):
        # note: user should use self._bonds as dependency list if not using grad
        self._set_dependencies(theta123, theta234)
        self._init_param(theta123, theta234, params, register)
        self._check()
        self._reversed = RevDihedral(self) 

        # will only need reversed if autogenerated in the wrong direction (?)

    def _init_param(self, theta123, theta234, params, register)
        self.t123 = theta123
        self.t234 = theta234
        self.r12 = theta123.r12
        self.r23 = theta123.r23
        self.r34 = theta234.r23
        self.a1 = theta123.a1
        self.a2 = theta123.a2 # spine of two planes
        self.a3 = theta123.a3 # spine of two planes
        self.a4 = theta234.a4
        self.params = params
        self.registered = register

    def _check(self):
        assert len(self._atoms) == 4
        assert self.t123.r23 == self.t234.r12

    def _calc_with_cos(self):
        # the dihedral between 2 planes is angle between the unit normal vectors
        # cross(a, b) == cross(-a, -b) == -cross(b, a) == -cross(-b, -a)
        self.n123 = np.cross(self.r12.r, self.r23.r)
        self.n234 = np.cross(self.r23.r, self.r34.r)
        n123 /= np.linalg.norm(n123)
        n234 /= np.linalg.norm(n234)
        self.cos = np.dot(n123, n234)
        self.phi = np.arcos(self.cos)
        return self.phi

    def _calc_with_tan(self):
        # alternative dihedral angle calculation using arctangent
        self.n123 = np.cross(self.r12.r, self.r23.r)
        self.n234 = np.cross(self.r23.r, self.r34.r)
        s = np.dot(np.cross(n123, n234), self.r23.e)
        c = np.dot(n123, n234)
        self.phi = np.arctan2(s, c)
        return self.phi

    def _calc_with_grad(self):
        # grad must be called first
        self.n123 = self.n234 = None
        self.phi = self.r23.b * self.r23.b * np.dot(self.t123.d3, self.t234.d2)
        return self.phi

    calc = _calc_with_tan # default choise 

    def grad(self):
        # if used _calc_with_grad will have to calc n123 & n234 otherwise reuse 
        n123 = self.n123 or np.cross(self.r12.e, self.r23.e)
        n234 = self.n234 or np.cross(self.r23.e, self.r34.e)
        b123 = np.linalg.norm(n123)
        b234 = np.linalg.norm(n234)
        self.d1 = -n123/(b123*b123*self.r12.b)
        self.d4 = +n234/(b234*b234*self.r34.b)
        self.d2 = ((self.r12.b*self.t123.theta/self.r23.b - 1)*self.d1 - 
                   (self.r34.b*self.t234.theta/self.r23.b)*self.d4)
        self.d3 = ((self.r34.b*self.t234.theta/self.r23.b - 1)*self.d4 -
                   (self.r12.b*self.t123.theta/self.r23.b)*self.d1)
        return self.d1, self.d2, self.d3, self.d4

    def energy(self):
        self.V = 0.0
        for kchi, n, delta in self.params:
            self.V += kchi * (1 + np.cos(n*self.chi - delta))
        return self.V

    def energy_update_forces(self):
        self.V = 0.0
        for kchi, n, delta in self.params:
            # energy
            arg = n*self.chi - delta
            self.V += kchi * (1 + np.cos(arg))
            # forces
            dV = - kchi*n*np.sin(arg)
            self.a1.f -= self.d1 * dV
            self.a2.f -= self.d2 * dV
            self.a3.f -= self.d3 * dV
            self.a4.f -= self.d4 * dV
        return self.V

    if TURN_OFF_NUMPY:
        def _calc_with_grad(self):
            raise NUMPY_ERROR
        def _calc_with_tan(self):
            raise NUMPY_ERROR
        def _calc_with_cos(self): 
            raise NUMPY_ERROR
        def grad(self):
            raise NUMPY_ERROR
        def energy(self):
            raise NUMPY_ERROR
        def energy_update_forces(self):
            raise NUMPY_ERROR

class Improper(Dihedral_base):

    # TODO do we need angles for grad equation

    # connectivity:
    #     a1
    #   / | \ 
    # a2  |  a3
    #     |
    #     a4
    # planes between which to calculate dihedral:
    #     a1
    #   /   \ 
    # a2---- a3
    #   \   /
    #     a4

    def __init__(self, theta123, theta234, kpsi, psi0, register=True):
        self._set_dependencies(theta123, theta234)
        self._init_param(theta123, theta234, kpsi, psi0, register)
        self._check()
        self._reversed = None # no dependents so no need for a paired RevImproper

    @property
    def psi(self):
        return self.phi

    def _init_param(self, theta123, theta234, kpsi, psi0, register)
        self.t123 = theta123
        self.t234 = theta234
        self.r12 = theta123.r12
        self.r23 = theta123.r23
        self.r34 = theta234.r23
        self.a1 = theta123.a1 # central atom
        self.a2 = theta123.a2 # spine of two planes
        self.a3 = theta123.a3 # spine of two planes
        self.a4 = theta234.a4
        self.kpsi = kpsi
        self.psi0 = psi0
        self.registered = register

    def energy(self):
        d = self.psi - self.psi0
        self.V = self.kpsi*d*d
        return self.V

    def energy_update_forces(self):
        # energy
        d = self.chi - self.psi0
        self.V = self.kpsi*d*d
        # forces
        dV = 2*self.kpsi*d
        self.a1.f -= self.d1 * dV
        self.a2.f -= self.d2 * dV
        self.a3.f -= self.d3 * dV
        self.a4.f -= self.d4 * dV
        return self.V

    if TURN_OFF_NUMPY:
        def energy(self):
            raise NUMPY_ERROR
        def energy_update_forces(self):
            raise NUMPY_ERROR

################################################################################

# CMAP class 

################################################################################

class Cmap(InternalCoordinate):

    # the CMAP energy correction to pairs of (proper or improper) dihedrals
    # is applied as a bilinear interpolation using phi1234 and psi2345

    __refs = []

    def __init__(self, phi1234, psi2345, cmap, aij, register=True):
        self._set_dependencies(phi1234, psi2345)
        self._init_param(phi1234, psi2345, cmap, aij, register)
        self._check()
        self._reversed = None # no dependents so no need for a paired RevCmap
        self.__refs.append(self)

    @classmethod
    def set_interpolator(cls, method):
        for obj in cls.__refs:
            obj.energy = dict(nearest=obj._energy_nearest, 
                              bilinear=obj._energy_bilinear,
                              bicubic=obj._energy_bicubic)[method]

    def _init_param(self, phi1234, psi2345, cmap, aij, register):
        n, m = cmap.shape
        self.n = n
        self.step_deg = 360.0 / self.n
        self.step_rad = (2 * math.pi) / self.n
        self.cmap = cmap      # cmap (nxn grid)
        self.aij = aij        # bicubic interpolation coefficents (nxnx4x4 grid)
        self.phi = phi1234
        self.psi = psi2345
        self.a1 = phi1234.a1
        self.a2 = phi1234.a2
        self.a3 = phi1234.a3
        self.a4 = phi1234.a4
        self.a5 = psi2345.a4
        self.registered = register
        use_bicubic = if aij is not None:
        self.energy = self._energy_bicubic if use_bicubic else self._energy_bilinear

    def _check(self):
        # check phi/psi are consecutive - not checked when parsing topology file
        
        #   |---------phi---------|      
        #          |---phi.t234---|
        #   a1-----a2-----a3-----a4-----a5
        #          |---psi.t123---|       
        #          |---------psi---------|                

        assert len(self._atoms) == 5
        assert self.phi.t234 == self.psi.t123

    def calc(self):
        # desired point in grid at which to interpolate
        # self.x_grid = (np.degrees(self.phi.phi) + 180.)/self.step_deg
        # self.y_grid = (np.degrees(self.psi.psi) + 180.)/self.step_deg
        self.x_grid = (self.phi.phi + np.pi)/self.step_rad
        self.y_grid = (self.psi.psi + np.pi)/self.step_rad

    def grad(self): 
        # - self.phi.d1 
        # - self.phi.d2 - self.psi.d1 
        # + self.phi.d3 + self.psi.d2
        # + self.phi.d4 + self.psi.d3
        # + self.psi.d4
        pass

    def _energy_nearest(self):
        i = int(round(self.x_grid)) % self.n
        j = int(round(self.y_grid)) % self.n
        self.V = self.cmap[i, j]
        return self.V

    def _energy_bilinear(self):

        # these are interger indicies
        i0 = int(self.x_grid) % self.n
        j0 = int(self.y_grid) % self.n
        j1 = (i0 + 1) % self.n
        j1 = (j0 + 1) % self.n

        # equivalent to self.xgrid - int(self.xgrid)
        t = self.x_grid % 1
        u = self.y_grid % 1

        # get values at unit square grid points
        V00 = self.cmap[i0, j0]
        V01 = self.cmap[i0, j1]
        V10 = self.cmap[i1, j0]
        V11 = self.cmap[i1, j1]

        # perform bilinear interpolation for desired point using
        self.V = V00 + (V10-V00)*t + (V01-V00)*u + (V00-V10-V01+V11)*u*t
        return self.V

    def _energy_bicubic(self):

        # get grid patch indices
        i0 = int(self.x_grid) % self.n
        j0 = int(self.y_grid) % self.n

        # calculate fractional cartesian coordinate on patch
        t = self.x_grid % 1
        u = self.x_grid % 1
        tvec = np.array([1, t, t*t, t*t*t])
        uvec = np.array([1, u, u*u, u*u*u])

        # retrive pre-calculated bicubic coefficents for this specific patch
        assert self.aij is not None, "bicubic interpolation coefficents not found"
        aij = self.aij[i0, j0, :, :]

        # perform bicubic interpolation: V(t, u) = sum_i sum_j (aij*t^i*u^j)
        self.V = np.dot(np.dot(tvec, aij), uvec)
        return self.V

    def energy_update_forces(self):

        ## answers would be E/grid so to convert to E/rad = E/grid * grid/rad:
        # dVdphi = np.dot(np.dot(_tvec, aij), uvec) / step_rad
        # dVpsi = np.dot(np.dot(tvec, aij), _uvec) / step_rad
        # d2Vdphidpsi = np.dot(np.dot(_tvec, aij), _uvec) / step_rad**2

        # get grid patch indices
        i0 = int(self.x_grid) % self.n
        j0 = int(self.y_grid) % self.n

        # compute fractional point on unit grid patch
        t = self.x_grid % 1
        u = self.x_grid % 1

        # calculate higher-order terms
        t2 = t*t
        u2 = u*u
        tvec = np.array([1, t, t2, t*t2])
        uvec = np.array([1, u, u2, u*u2])
        _tvec = np.array([0, 1, 2*t, 3*t2])
        _uvec = np.array([0, 1, 2*u, 3*u2])

        # retrive pre-calculated bicubic coefficents for this patch
        # see toppario.py for calc and interplation.ipynb for explanation
        assert self.aij is not None, "bicubic interpolation coefficents not found"
        aij = self.aij[i0, j0, :, :]

        # perform interpolation:
        # V(t, u) = sum_i sum_j (aij*t^i*u^j)
        # dV(t, u)/dt = sum_i sum_j (i*aij*t^(i-1)*u^j)
        # dV(t, u)/du = sum_i sum_j (j*aij*t^i*u^(j-1))
        # dV/dphi = dV/dt * dt/dx_grid * dx_grid/dphi = dV/dt * 1/1 * 1/step_rad
        # dV/dpsi = dV/du * du/dy_grid * dy_grid/dpsi = dV/du * 1/1 * 1/step_rad
        self.V = np.dot(np.dot(tvec, aij), uvec)
        dVdphi = np.dot(np.dot(_tvec, aij), uvec) / self.step_rad
        dVdpsi = np.dot(np.dot(tvec, aij), _uvec) / self.step_rad

        # update forces - TODO check signs 
        self.a1.f += dVdphi*self.phi.d1
        self.a2.f += dVdphi*self.phi.d2 + dVdpsi*self.psi.d1
        self.a3.f -= dVdphi*self.phi.d3 + dVdpsi*self.psi.d2 # deduct
        self.a4.f -= dVdphi*self.phi.d4 + dVdpsi*self.psi.d3 # deduct
        self.a5.f -=                      dVdpsi*self.psi.d4 # deduct
        return self.V

    if TURN_OFF_NUMPY:
        def calc(self):
            raise NUMPY_ERROR
        def _energy_bilinear(self):
            raise NUMPY_ERROR
        def _energy_bicubic(self):
            raise NUMPY_ERROR
        def energy_update_forces(self):
            raise NUMPY_ERROR


################################################################################

# Reversed classes

################################################################################

class Reversed(object):

    __error_msg = "This is a Reversed subclass - are you double counting the energy?"

    def energy(self): 
        raise InternalCoordinateError(__error_msg)
    def energy_update_forces(self):
        raise InternalCoordinateError(__error_msg)

def make_fget(attr):
    return operator.attrgetter('_reversed.'+attr)

class RevBond(Reversed, Bond_base):

    def __init__(self, bond):
        self._set_dependencies(bond)
        self._init_param(bond.a2, bond.a1, bond.kb, bond.k0, register=False)
        self._reversed = bond

    # evalutated as needed
    @property
    def r(self): 
        return -self._reversed.r
    @property
    def e(self):
        return -self._reversed.e

    b = property(make_fget('b'))
    b2 = property(make_fget('b2'))

    def calc(self): 
        return self.b

class RevUb(RevBond):
    pass

class RevNonbonded_14(RevBond, Nonbonded_14):

    # has "diamond" inheritance pattern using new-style MRO, compare:
    # new-style MRO: RevNonbonded_14, RevBond(Reversed, Bond), Reversed, Nonbonded_14(Bond), Bond
    # legacy MRO: RevNonbonded_14, RevBond(Reversed, Bond), Reversed, Bond, Nonbonded_14(Bond), Bond
    # thus will inherit RevBond.calc and Nonbonded_14._init_param

    def __init__(self, bond):
        self._set_dependencies(bond)
        self._init_param(bond.a2, bond.a1, bond.c6, bond.c12, register=False)
        self._reversed = bond

class RevAngle(Reversed, Angle_base):

    def __init__(self, angle):
        self._set_dependencies(angle)
        self._init_param(-angle.r23, -angle.r12, angle.ktheta, angle.theta0, register=False)
        self._reversed = angle

    cos = property(make_fget('cos')) # operator.attrgetter
    theta = property(make_fget('theta'))

    # just swap d3 and d1
    d3 = property(make_fget('d1'))
    d2 = property(make_fget('d2'))
    d1 = property(make_fget('d3'))

    def calc(self): 
        return self.theta

    def grad(self): 
        return self.d1, self.d2, self.d3 # see properties above

class RevDihedral(Reversed, Dihedral):

    # phi = lambda a,b,c: arctan2(dot(x(x(a,b),x(b,c)),unit(b)),dot(x(a,b),x(b,c)))
    # +phi(-a1+a2, -a2+a3, -a3+a4) ==
    # -phi(-a4+a3, -a3+a2, -a2+a1) == 
    # +phi(-a1+a3, -a3+a2, -a2+a4) ==
    # -phi(-a4+a2, -a2+a3, -a3+a1) ==

    def __init__(self, dihedral):
        self._set_dependencies(dihedral)
        self._init_param(-dihedral.t234, -dihedral.t123, dihedral.params, register=False)
        self._check()
        self._reversed = dihedral

    @property
    def n123(self): 
        return -self._reversed.n234
    @property
    def n234(self): 
        return -self._reversed.n123
    @property
    def phi(self):
        return -self._reversed.phi

    d4 = property(make_fget('d1'))
    d3 = property(make_fget('d2'))
    d2 = property(make_fget('d3'))
    d1 = property(make_fget('d4'))

    def calc(self): 
        return self.phi

    def grad(self): 
        return self.d1, self.d2, self.d3, self.d4



