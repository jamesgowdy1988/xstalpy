"""nonbonded.py - Verlet list and Ewald summation/PME code"""

from __future__ import division

try:
    import numpy as np
except ImportError as NUMPY_ERROR:
    warnings.warn("numpy is required for full functionality")

__author__ = "James Gowdy" 
__email__ = "jamesgowdy1988@gmail.com"

################################################################################

# Lennard Jones parameters and functional forms

################################################################################

# LB combination

def combine_eps(eps_ii, eps_jj):
    # VLJ = eps_ij*[(rmin_ij/rij)^12 - 2*(rmin_ij/rij)^6]
    eps_ij = math.sqrt(eps_ii*eps_jj)
    return eps_ij

def combine_rmin(rmin_ii, rmin_jj):
    # VLJ = eps_ij*[(rmin_ij/rij)^12 - 2*(rmin_ij/rij)^6]
    rmin_ij = 0.5*(rmin_ii+rmin_jj)
    return rmin_ij

def combine_sigma(sigma_ii, sigma_jj):
    # VLJ = 4*eps_ij*[(sigma_ij/rij)^12 - (sigma_ij/rij)^6]
    sigma_ij = 0.5*(sigma_ii+sigma_jj)
    return sigma_ij

# convert params

def calc_sigma_from_rmin(rmin_ij):
    # note: VLJ(sigma_ij) = 0 vs. VLJ(rmin_ij) = min(VLJ)
    # i.e: sigma = r0
    sigma_ij = rmin_ij/(2**(1./6.))
    return sigma_ij

def calc_rmin_from_sigma(sigma_ij):
    rmin_ij = sigma_ij*2**(1./6.)
    return rmin_ij

# conversion to Mie parameter form

def calc_c6c12_from_rmin(eps_ij, rmin_ij):
    # VLJ = c12_ij/rij^12 - c6_ij/rij^6
    c6_ij = 2.0*eps_ij*(rmin_ij**6)
    c12_ij = eps_ij*(rmin_ij**12)
    return c6_ij, c12_ij

def calc_c6c12_from_sigma(eps_ij, sigma_ij):
    # VLJ = c12_ij/rij^12 - c6_ij/rij^6
    c6_ij = 4.0*eps_ij*(sigma_ij**6)
    c12_ij = 4.0*eps_ij*(sigma_ij**12)
    return c6_ij, c12_ij

# functional form

def VLJ_with_epssigma(eps_ij, sigma_ij, rij)
    return 4*eps_ij*[(sigma_ij/rij)**12 - (sigma_ij/rij)**6]

def VLJ_with_epsrmin(eps_ij, sigma_ij, rij)
    return eps_ij*[(rmin_ij/rij)**12 - 2*(rmin_ij/rij)**6]

def VLJ_with_c6c12(c12_ij, c6_ij, rij):
    return c12_ij/rij**12 - c6_ij/rij**6

################################################################################

# Non-bonding pair list creation and update functionality

################################################################################

class VerletList(object):

    def __init__(self, atoms, pairs_params, S, SI, volume, uc_abc, rcut=12, kmax=10):

        self.atoms = atoms
        self.all_pairs = pairs_params
        
        # precalculated matrices & volume for use during Q/LJ PME computation
        self._S = S
        self._SI = SI
        self._ST = np.transpose(S)
        self._V = volume # check not inv
        self._uc_abc = uc_abc

        # default rcut kcut and kmax values for interactions 
        self.set_rcut(rcut=rcut, rshift=12, rlist=14, rsafe=10)
        self.kmax = kmax

        # get Ewald coefficents for the set cutoff values
        self._alpha_q = self._calc_alpha(rcut, "q")
        self._alpha_lj = self._calc_alpha(rcut, "lj")

        self.update_verletlist()

        # precalculate constants for pme
        self._pi2 = np.pi*np.pi
        self._alpha2_q = alpha_q*alpha_q
        self._negpi2_over_alpha2_q = -self._pi2/alpha2_q
        self._factor = 2.0*alpha_q/np.sqrt(np.pi)

        for ai, aj, qq, c6, c12 in pairs_params:
            self.Vq_self += qq
        self.Vq_self *= alpha_q/np.sqrt(np.pi)

    def set_rcut(self, rcut, rlist, rsafe):

        # r0                   rsafe          rcut          rlist
        # |                       |             |             |
        # |<-----safe zone------->|<-------danger zone------->|<---not checked--
        # |<-------Vq in direct space---------->|<-------Vq in k space--------->
        # |<-----------VLJ included------------>|<--VLJ only included if use PME 
                     
        # set rcut and ewald coefficents (alpha)
        assert rsafe < rcut < rlist
        assert all(np.dot(self._SI, e*0.5) < rcut for e in np.eye(3)), 'rcut too big?'
        self._rcut = rcut
        self._rcut2 = rcut*rcut
        self._rshift2 = rshift*rshift
        self._rlist2 = rlist*rlist
        self._rsafe2 = rsafe*rsafe
        self._kcut2 = 1.0/(rcut*rcut)

    def _calc_alpha(self, rcut, typ="lj", tolerance=1e-4):

        # http://archive.is/TABrQ 
        # http://ambermd.org/Questions/ewald.htm
        # http://micro.stanford.edu/mediawiki/images/4/46/Ewald_notes.pdf

        def g6r6(al, rc):
            x = al*rc
            x2 = x*x
            x4 = x2*x2
            g6 = math.exp(-x2)*(1 + x2 + x4/2.0) # see Essmann near eq A7
            return g6/rc**6

        def g1r1(al, rc):
            g1 = math.erfc(al*rc)
            return g1/rc

        func = {"lj": g6r6, "q": g1r1}[typ.lower()]

        alpha_min = 0.0
        alpha_max = 1.0
        while func(alpha_max, rcut) >= tolerance:
            alpha_min = alpha_max
            alpha_max *= 2.0
        for i in range(50):
            alpha = (alpha_min+alpha_max)/2.0
            if func(alpha, rcut) >= tolerance:
                alpha_min = alpha
            else:
                alpha_max = alpha
        return alpha # alpha = 1.0/(sigma*np.sqrt(2)) # see Stanford notes

    def update_zones(self):
        self._danger_zone = []
        self._safe_zone = []
        self.edges = []
        for pair in self.all_pairs:
            aj, ai, qq, c6, c12 = pair
            rij = rj - ri
            rij = self._min_image(rij)
            r2 = np.dot(rij, rij)
            if r2 < self._rlist2:
                if r2 < self._rlist2:
                    if self._rsafe2 < r2:
                        self._danger_zone.append(pair)
                    else:
                        self._safe_zone.append(pair)

    def _min_image_alt(self, r):
        f = np.dot(self._S, r)
        f -= np.floor(f+0.5)
        return np.dot(self._SI, f)

    def _min_image(self, r):
        r = r.copy()
        for i in [2, 1, 0]:
            diff = abs(r[i]) - (self._SI[i, i]/2)
            if diff > 0:
                n = np.ceil(diff/self._SI[i, i])
                if r[i] < 0:
                    vec = n*self._SI[:, i]
                else:
                    vec = -n*self._SI[:, i]
                r += vec
        return r

    def get_pairs(self):
        for ai, aj, qq, c6, c12 in self._danger_zone:
            rij = rj - ri
            rij = self._min_image(rij)
            bij2 = sum(rij*rij)
            if bij2 < self._rcut2:
                yield ai, aj, bij2, rij, qq, c6, c12
        for ai, aj, qq, c6, c12 in self._safe_zone:
            rij = rj - ri
            rij = self._min_image(rij)
            bij2 = sum(rij*rij)
            yield ai, aj, bij2, rij, qq, c6, c12

    def _energy_direct(self):
        self.Vq_real = 0.0
        self.Vlj = 0.0
        for ai, aj, r2, rij, qq, c6, c12 in self._get_pairs(r):
            b = np.sqrt(r2)
            r6 = r2*r2*r2
            self.Vlj += (c12/r6 - c6)/r6
            self.Vq_real += qq*math.erfc(self.alpha_q*b)/b

    def _energy_update_forces_direct(self):
        self.Vq_real = 0.0
        self.Vlj = 0.0
        for ai, aj, r2, rij, qq, c6, c12 in self._get_pairs(r):
            # lennard jones energy
            r6 = r2*r2*r2
            self.Vlj += (c12/r6 - c6)/r6
            # coulomb energy
            b = np.sqrt(r2)
            Vq = qq*math.erfc(self.alpha_q*b)/b
            self.Vq_real += Vq
            # lennard jones force
            flj = (12*c12/r6 - 6*c6)/(r6*r2)*rij
            ai.f += flj
            aj.f -= flj
            # coulomb force
            fq = (qq*self._factor*np.exp(-self._alpha2_q*r2) + Vq) * rij/r2
            ai.f -= Fq
            aj.f += Fq

    def _loop(self):
        for hc in xrange(-self.kmax, self.kmax):
            for kc in xrange(-self.kmax, self.kmax):
                for lc in xrange(-self.kmax, self.kmax):
                    # get recip lattice point distance
                    k = np.dot(self._ST, [hc, kc, lc]) # cartesian
                    k2 = np.dot(k, k)
                    if k2 == 0 or (k2 > self._kcut2):
                        continue
                    # carry out FT (better to use FFT)
                    k *= 2j*np.pi
                    yield k, k2

    def _energy_recip_ewald(self):
        self.Vq_recip = 0.0
        # loop over recip grid
        for k, k2 in self._loop()
            Sk = 0.0
            Sk_summands = []
            for ai in self.atoms:
                qekrij = ai.q*np.exp(np.dot(ai.r, k))
                Sk += qekrij
                Sk_summands.append(qekrij)
            f1 = np.exp(self._negpi2_over_alpha2_q*k2)/k2
            self.Vq_recip += f1*np.abs(Sk)**2
        # corrections
        self.Vq_recip *= 1.0/(np.pi*2.0*self._volume)

    def _energy_update_forces_recip_ewald(self):
        self.Vq_recip = 0.0
        Fq_recip = np.zeros_like((len(self.atoms), 3))
        # loop over recip grid
        for k, k2 in self._loop()
            Sk = 0.0
            Sk_summands = []
            for ai in self.atoms:
                qekrij = ai.q*np.exp(np.dot(ai.r, k))
                Sk += qekrij
                Sk_summands.append(qekrij)
            f1 = np.exp(self._negpi2_over_alpha2_q*k2)/k2
            self.Vq_recip += f1*np.abs(Sk)**2
            qekrij = np.conj(Sk_summands)*Sk
            Fq_recip += np.outer(qekrij, k*f1)
        # corrections
        self.Vq_recip *= 1.0/(np.pi*2.0*self._volume)
        Fq_recip *= 2.0/(np.pi*2.0*self._volume)
        # update forces
        for ai, f in zip(self.atoms, Fq_recip):
            ai.f += f

    def energy(self):
        self._energy_direct()
        self._energy_recip_ewald()
        self.Vq = self.Vq_recip + self.Vq_self + self.Vq_direct
        return self.Vq, self.Vlj

    def energy_update_forces(self):
        self._energy_update_forces_direct()
        self._energy_update_forces_recip_ewald()
        self.Vq = self.Vq_recip + self.Vq_self + self.Vq_direct
        return self.Vq, self.Vlj


class PME(object):

    def M2(self, u):
        if u < 0 or u > 2:
            return 0.0
        return 1.0 - np.abs(u-1.0)

    M3s = {}
    def M3(sekf, u):
        if u in self.M3s:
            return self.M3s[u]
        M3s[u] = M3 = (u*self.M2(u) + (2.0-u)*self.M2(u-1))/2.0
        return M3

    # 4th order cardinal B-spline using recursion (see Essmann eq. 4.1)
    M4s = {}
    def M4(u):
        if u in self.M4s:
            return self.M4s[u]
        M3u = self.M3(u)
        M3um1 = self.M3(u-1)
        M4 = (u*M3u + (3.0-u)*M3um1)/3.0
        dM4 = M3u - M3um1
        self.M4s[u] = M4, dM4
        return M4, dM4

    # explonential terms for Euler/cardinal B-splines

    def bi2(mi_over_Ki):
        sum_Mn_exp = 0.0
        for k in range(0, n-2):
            M4, dM4 = M4(k+1)
            sum_Mn_exp += M4*np.exp(2j*np.pi*mi_over_Ki*k)
        bi = np.exp(twojpi*(n-1)*mi_over_Ki)/sum_Mn_exp
        bi2 = np.abs(bi)**2
        return bi2

    def _loop_over_local_grids(self, xgs, grid, d=3):
        na, nb, nc = grid

        # note: the localgrids do not sample space evenly unless the crystal 
        # system is cubic, but this is how the PME method is described by Essmann

        drange = localgrid(+d, -d-1, -1)
        for (xg, yg, zg), (ai, q, c6, c12) in zip(xgs, atoms):
            Mnys = [[yg-dyg, dyg, self.M4(dyg)] for dyg in localgrid]
            Mnzs = [[zg-dzg, dzg, self.M4(dzg)] for dzg in localgrid]
            for dxg in localgrid:
                rxg = xg-dxg
                Mnx, dMnx = self.M4(dxg)
                for ryg, dyg, (Mny, dMny) in Mnys:
                    for rzg, dzg, (Mnz, dMnz) in Mnzs:
                        #v = np.dot(SI_grid, [dxg, dyg, dzg])
                        #if np.dot(v, v) > self.rcut2:
                        #   continue
                        ixg = rxg%na
                        iyg = ryg%nb
                        izg = rzg%nc
                        M = Mnx*Mny*Mnz
                        dM = np.array([dMx, dMy, dMz])
                        yield ixg, iyg, izg, q, c6, M, dM

    def _loop_over_hkl(self, kmax, grid, bi2):
        # calculate spline coeffs once before triple hkl loop
        # maybe B should be vectorized so B[h, k, l] is an array???
        na, nb, nc = grid
        bas = {}
        bbs = {}
        bcs = {}
        for m in xrange(-kmax, kmax):
            bas[m] = bi2(m/na)
            bbs[m] = bi2(m/nb)
            bcs[m] = bi2(m/nc)
        for hc in xrange(-kmax, kmax):
            ba = bas[hc]
            for kc in xrange(-kmax, kmax):
                bb = bbs[kc]
                for lc in xrange(-kmax, kmax):
                    if hc or kc or lc:
                        # product of cardinal-B spline coefficents
                        bc = bcs[lc]
                        B = ba*bb*bc
                        mw = np.dot(self._ST, [hc, kc, lc])
                        ksq = np.dot(mw, mw)
                        yield, hc, kc, lc, B, ksq

    def _energy_recip_pme(self):
        self.Vq_recip = 0.0
        d = 3 # half length of local cube of (d*2)**3 points to calc spline
        g = 1.0/3.0 # grid resolution
        grid = np.floor(np.around(np.array(self._uc_abc)/10.0)*10.0/g)
        xcs = np.zeros((len(self.atoms, 3)))
        for ai in enumerate(self.atoms):
            xcs[i, :] = np.dot(self._S, ai.r) % 1
        xgs = (xcs*grid).astype(int)
        Mn, bi2 = self._get_recursive_M4()
        Q = np.zeros(grid)
        for x, y, z, q, c6, M, dM in self._loop_over_local_grids(xgs, grid, d):
            Q[x, y, z] += q*M
        iFQ2 = np.abs(np.fft.ifftn(Q))**2
        for hc, kc, lc, Bk, k2 in self._loop_over_hkl(kmax, grid bi2): 
            Sk2_q = iFQ2[hc, kc, lc]
            f1 = np.exp(-pi2*k2/alpha2_q)/k2
            self.Vq_recip += f1*Bk*Sk2_q
        self.Vq_recip *= 1.0/(2.0*self._V*np.pi)






