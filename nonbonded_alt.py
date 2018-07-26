__package__ = "xstalpy"
__author__ = "James Gowdy" 
__email__ = "jamesgowdy1988@gmail.com"

import numpy
import topology

################################################################################

# Potential energy and force field computation for non-bonded atoms (migrate to C++)

################################################################################

class NonbondedForceField_QEnergy(topology.VerletList):

    # Coulomb ewald summation for energy only

    def _calc_qewald_energy_only(self, r, alpha_q, kmax):
        Vq_real = 0.0
        Vq_self = 0.0
        Vq_recip = 0.0
        alpha2_q = alpha_q*alphq_q
        negpi2_over_alpha2_q = -np.pi**2/alpha2_q
        factor = 2.0*alpha_q/np.sqrt(np.pi)
        for i, qi, qiqi, c6ii in self._coeffs_iqc6:
            Vq_self += qiqi
        Vq_self *= alpha_q/np.sqrt(np.pi)
        for i, j, qiqj, c6, c12, r2, rij in self._get_pairs(r):
            bij = np.sqrt(r2)
            Vq_real += qiqj*math.erfc(alpha_q*bij)/bij
        for hc in xrange(-kmax, kmax):
            for kc in xrange(-kmax, kmax):
                for lc in xrange(-kmax, kmax):
                    if hc == 0 and kc == 0 and lc == 0:
                        continue
                    k = np.dot(self._ST, [hc, kc, lc]) # cartesian
                    k2 = np.dot(k, k)
                    k *= 2j*np.pi
                    Sk = 0.0
                    for i, qi, qiqi, c6ii in self._coeffs_iqc6:
                        Sk += qi*np.exp(np.dot(r[i], k))
                    f1 = np.exp(negpi2_over_alpha2_q*k2)/k2
                    Vq_recip += f1*np.abs(Sk)**2
        Vq_recip *= 1.0/(np.pi*2.0*self._V)
        return Vq_self + Vq_recip + Vq_real

    def _calc_qpme_energy_only(self, r, alpha_q, kmax):
        Vq_real = 0.0
        Vq_self = 0.0
        Vq_recip = 0.0
        alpha2_q = alpha_q*alpha_q
        pi2 = np.pi*np.pi
        negpi2_over_alpha2_q = -pi2/alpha2_q
        factor = 2.0*alpha_q/np.sqrt(np.pi)
        for i, qi, qiqi, c6ii in self._coeffs_iqc6:
            Vq_self += qiqi
        Vq_self *= alpha_q/np.sqrt(np.pi)
        for i, j, qiqj, c6, c12, r2, rij in self._get_pairs(r):
            r1 = np.sqrt(r2)
            Vq_real += qiqj*math.erfc(alpha_q*r1)/r1
        d = 3 # half length of local cube of (d*2)**3 points to calc spline
        g = 1.0/3.0 # grid resolution
        grid = np.floor(np.around(np.array(self.molc.pdb.uc_abc)/10.0)*10.0/g)
        xcs = np.dot(self.molc.pdb.S, r.T).T % 1
        xgs = (xcs*grid).astype(int)
        Mn, bi2 = self._get_recursive_M4()
        Q = np.zeros(grid)
        for x, y, z, q, c6, M, dM in self._loop_over_local_grids(xgs, grid, Mn, d):
            Q[x, y, z] += q*M
        iFQ2 = np.abs(np.fft.ifftn(Q))**2
        for hc, kc, lc, Bk, k2 in self._loop_over_hkl(kmax, grid bi2): 
            Sk2_q = iFQ2[hc, kc, lc]
            f1 = np.exp(-pi2*k2/alpha2_q)/k2
            Vq_recip += f1*Bk*Sk2_q
        Vq_recip *= 1.0/(2.0*self._V*np.pi)
        return Vq_self + Vq_recip + Vq_real

    # Coulomb ewald summation for energy AND force

class NonbondedForceField_QEnergyForce(topology.VerletList):

    def _calc_qewald(self, r, alpha_q, kmax):

        # Ewald latiice summation for coulombic energy
        Vq_real = 0.0
        Vq_self = 0.0
        Vq_recip = 0.0

        alpha2_q = alpha_q*alphq_q
        negpi2_over_alpha2_q = -np.pi**2/alpha2_q
        factor = 2.0*alpha_q/np.sqrt(np.pi)

        for i, qi, qiqi, c6ii in self._coeffs_iqc6:
            Vq_self += qiqi
        Vq_self *= alpha_q/np.sqrt(np.pi)

        for i, j, qiqj, c6, c12, r2, rij in self._get_pairs(r):
            bij = np.sqrt(r2)
            Vij = qiqj*math.erfc(alpha_q*bij)/bij
            Vq_real += Vij
            Fij = (qiqj*factor*np.exp(-alpha2_q*r2) + Vij) * rij/r2
            Fq_real[i, :] -= Fij
            Fq_real[j, :] += Fij

        for hc in xrange(-kmax, kmax):
            for kc in xrange(-kmax, kmax):
                for lc in xrange(-kmax, kmax):
                    if hc == 0 and kc == 0 and lc == 0:
                        continue
                    k = np.dot(self._ST, [hc, kc, lc]) # cartesian
                    k2 = np.dot(k, k)
                    k *= 2j*np.pi
                    Sk = 0.0
                    Sk_summands = []
                    for i, qi, qiqi, c6ii in self._coeffs_iqc6:
                        qekrij = qi*np.exp(np.dot(r[i], k))
                        Sk += qekrij
                        Sk_summands.append(qekrij)
                    f1 = np.exp(negpi2_over_alpha2_q*k2)/k2
                    Vq_recip += f1*np.abs(Sk)**2
                    qekrij = np.conj(Sk_summands)*Sk
                    Fq_recip += np.outer(qekrij, k*f1)

        Vq_recip *= 1.0/(np.pi*2.0*self._V)
        Fq_recip *= 2.0/(np.pi*2.0*self._V)
        Vq = Vq_self + Vq_recip + Vq_real
        Fq = Fq_recip + Fq_real
        return Vq, Fq

    def _calc_qpme(self, r, alpha_q, kmax):

        # PME latiice summation for coulombic energy and forces

        Vq_real = 0.0
        Vq_self = 0.0
        Vq_recip = 0.0

        Fq_recip = np.zeros_like(r)
        Fq_real = np.zeros_like(r)

        alpha2_q = alpha_q*alpha_q
        pi2 = np.pi*np.pi
        negpi2_over_alpha2_q = -pi2/alpha2_q
        factor = 2.0*alpha_q/np.sqrt(np.pi)

        # self interaction
        for i, qi, qiqi, c6ii in self._coeffs_iqc6:
            Vq_self += qiqi
        Vq_self *= alpha_q/np.sqrt(np.pi)

        # direct space part of interaction
        for i, j, qiqj, c6, c12, r2, rij in self._get_pairs(r):
            r1 = np.sqrt(r2)
            Vij = qiqj*math.erfc(alpha_q*r1)/r1
            Fij = (qiqj*factor*np.exp(-alpha2_q*r2) + Vij) * rij/r2
            Vq_real += Vij
            Fq_real[i, :] += Fij
            Fq_real[j, :] -= Fij

        d = 3 # half length of local cube of (d*2)**3 points to calc spline
        g = 1.0/3.0 # grid resolution

        grid = np.floor(np.around(np.array(self.molc.pdb.uc_abc)/10.0)*10.0/g)
        xcs = np.dot(self.molc.pdb.S, r.T).T % 1
        xgs = (xcs*grid).astype(int)
        Mn, bi2 = self._get_recursive_M4()
        Q = np.zeros(grid)
        dQ = np.zeros((na, nb, nc, 3))
        for x, y, z, q, c6, M, dM in self._loop_over_local_grids(xgs, grid, Mn, d):
            Q[x, y, z] += q*M
            dQ[x, y, z, :] += q*dM
        iFQ = np.fft.ifftn(Q)
        iFQ2 = np.abs(iFQ)**2

        for hc, kc, lc, Bk, k2 in self._loop_over_hkl(kmax, grid bi2): 
            Sk2_q = iFQ2[hc, kc, lc]
            f1 = np.exp(-pi2*k2/alpha2_q)/k2
            iFtheta = f1*Bk 
            Vq_recip += iFtheta*Sk2_q
            iFQ[hc, kc, lc] *= iFtheta

        thetaxQ = np.fft.fftn(iFQ)[:, :, :, np.newaxis] 
        Fq_recip = -(dQ*thetaxQ)[xs, ys, zs, :] 
        Fq_recip *= 1.0/(2.0*self._V*np.pi)
        Vq_recip *= 1.0/(2.0*self._V*np.pi)
            
        Vq = Vq_self + Vq_recip + Vq_real
        Fq = Fq_recip + Fq_real
        return Vq, Fq

class NonbondedForceField_QLJEnergy():

    # Coulomb AND Lennard-Jones ewald summation for energy only

    def _calc_ewald_real_energy_only(self, r, alpha_q, alpha_lj):
        Vq_real = 0.0
        VLJ_real = 0.0
        VLJ_c12 = 0.0
        alpha2_lj = alpha_lj*alpha_lj
        factor = 2.0*alpha_q/np.sqrt(np.pi)
        for i, j, qiqj, c6, c12, r2, rij in self._get_pairs(r):
            r1 = np.sqrt(r2)
            g1 = math.erfc(alpha_q*r1)
            Vq_real += qiqj*g1/r1
            x2 = alpha2_lj*r2
            g6 = np.exp(-x2)*(1 + x2 + x2*x2/2.0)
            r6 = r2*r2*r2
            VLJ_real -= c6*g6/r6
            VLJ_c12 += c12/r12
        return Vq_real, VLJ_real, VLJ_c12
    
    def _calc_ewald_recip_energy_only(self, r, alpha_q, alpha_lj, kmax, mixing_rules='geometric'):

        twojpi = 2j*np.pi
        sqrtpi = np.sqrt(np.pi)
        pi_over_alpha_lj = np.pi/alpha_lj
        negpi2_over_alpha2_q = -np.pi*np.pi/(alpha_q*alpha_q)

        Vq_recip = 0.0
        VLJ_recip = 0.0
        factorial = lambda v: float(math.factorial(v))
        Pn = [factorial(6)/(factorial(n)*factorial(6-n)) for n in range(0, 7)]

        if mixing_rules == 'geometric':

            def calc_Sks(k):
                Sk_q = 0.0
                Sk_lj = 0.0
                k *= twojpi
                for i, qi, qiqi, c6ii in self._coeffs_iqc6:
                    ex = np.exp(np.dot(r[i], k))
                    Sk_q += qi*ex
                    Sk_lj += c6ii*ex
                return Sk_q, Sk_lj

        elif mixing_rules == 'lorentz bertholot':

            # summands not collected so forces not returned for LB rules

            def calc_Sks(k):
                c6terms = self._coeffs_iqc6terms
                Sk_q = Sk_lj = 0.0
                Zn = Zn_6 = 0.0
                k *= 2.0j*np.pi

                # 0th loop: precalc fourier basis functions for next loops
                waves = []
                negwaves = []
                loop = zip(c6terms[0], c6terms[6])
                for (i, qi, c6), (i_, qi_, c6_) in loop:
                    ex = np.exp(np.dot(r[i], k))
                    waves.append(ex)
                    Zn += c6*ex
                    Zn_6 += c6_*ex
                    Sk_q += qi*ex
                Sk_lj += Pn[0]*Zn*np.conj(Zn_6)

                # next 1 - 6 loops: only calculate expansion terms and add to Sk
                for n in range(1, 7):
                    Zn = Zn_6 = 0.0
                    loop = zip(c6terms[n], c6terms[6-n], waves)
                    for (i, qi, c6), (i_, qi_, c6_), ex in loop:
                        Zn += c6*ex
                        Zn_6 += c6_*ex
                    Sk_lj += Pn[n]*Zn*np.conj(Zn_6)

                Sk_lj *= 2**-4 #sqrt(4)*sqrt(4)*(2**n)*(2**(6-n)) for n in [0,6]
                return Sk_q, Sk_lj, 0, 0

        # sum_i(sum_j(qi*qj*exp(2j*pi*dot(ri-rj, k))))
        # sum_j(sum_i(qi*exp(2j*pi*dot(ri, k)) * qj*exp(2j*pi*dot(rj, -k))))
        # sum_i(qi*exp(2j*pi*dot(ri, k))) * sum_i(qi*exp(2j*pi*dot(ri,-k)))
        # S(k)S(-k) = |S(k)|^2

        for hc in range(-kmax, kmax):
            for kc in range(-kmax, kmax):
                for lc in range(-kmax, kmax):

                    # convert k to the cartesian basis 
                    # TODO: check its the same when r is in the fractional basis
                    k = np.dot(self._ST, [hc, kc, lc])
                    k2 = np.dot(k, k)

                    if k2 == 0 or (k2 > self._kcut2 and not self._ignore_kcut):
                        continue

                    # charge- and c6ii-weighted structure factors, S(k)
                    # S(k)S(-k) = |S(k)|^2 = sum_ij (qi*qj*exp(2jpik*rij))
                    Sk_q, Sk_lj = calc_Sks(k)

                    # Coulomb recip energy
                    Sk2_q = np.abs(Sk_q)**2
                    f1 = np.exp(negpi2_over_alpha2_q*k2)/k2
                    Vq_recip += f1*Sk2_q

                    # LJ recip dispersion energy
                    # Essmann (1995) eq 5.2, A5 and A11 pg 8589 & gromacs 5.0.1 
                    # page 112 (note we apply final corrections below)
                    Sk2_lj = np.abs(Sk_lj)**2
                    x = np.sqrt(k2)*pi_over_alpha_lj 
                    x2 = x*x
                    f6 = ((1-2*x2)*np.exp(-x2)+2*x2*x*sqrtpi*math.erfc(x))/3.
                    VLJ_recip += f6*Sk2_lj

        Vq_recip *= 1.0/(np.pi*2.0*self._V)
        VLJ_recip *= (sqrtpi*alpha_lj)**3/(2.0*self._V)
        return Vq_recip, VLJ_recip

    def _calc_pme_recip_energy_only(self, r, alpha_q, alpha_lj, kmax):

        d = 3 # half length of local cube of (d*2)**3 points to calc spline
        g = 1.0/3.0 # grid resolution

        # grid dimensions
        uc_abc = np.array(self.molc.pdb.uc_abc)
        grid = na, nb, nc = np.floor(np.around(uc_abc/10.0)*10.0/g)
              
        # recursive 4th order cardinal B spline function and coefficent function
        M4, bi2 = self._get_recursive_M4()

        # coordinate transforms
        xws = r                                    # cartesian world coordinates
        xcs = np.dot(self.molc.pdb.S, xws.T).T % 1 # fractional crystal coordinates
        xgs = (xcs*grid).astype(int)               # grid coordinates

        # initalise grids
        Q = np.zeros((na, nb, nc))
        C6 = np.zeros((na, nb, nc))
        dQ = np.zeros((na, nb, nc, 3))
        dC6 = np.zeros((na, nb, nc, 3))

        # populate charge/c6 grid by looping over atoms' surrounding grid points
        # see Harvey et al. section C: charge spreading and Orac manual
        # compare to make_ljpme_c6grid in gromacs/mdlib/forcerec.cpp - eh?
        for x, y, z, q, c6, M, dM in self._loop_over_local_grids(xgs, grid, M4, d):
            Q[x, y, z] += q*M
            dQ[x, y, z, :] += q*dM

            # we use geometric mixing rules, alt. is to use LB mixing rules
            C6[x, y, z] += c6*M
            dC6[x, y, z, :] += c6*dM

        # fourier transform charge grid 
        # |S(k)|^2 = |F(Q)(k)|^2 = F(Q)(k)*F(Q)(-k) = S(k)S(-k)
        iFQ = np.fft.ifftn(Q)
        iFQ2 = np.abs(iFQ)**2

        # fourier transform geometric c6 coefficent grid
        iFC6 = np.fft.ifftn(C6)
        iFC62 = np.abs(iFC6)**2

        # pre-caclulate to save time
        alpha2_q = alpha_q*alpha_q
        alpha2_lj = alpha_lj*alpha_lj
        pi2 = np.pi*np.pi
        sqrtpi = np.sqrt(np.pi)

        # reciprocal energy
        Vq_recip = 0.0
        VLJ_recip = 0.0

        # loop over fourier Q and C6 grids
        for hc, kc, lc, B, k2 in self._loop_over_hkl(kmax, grid, bi2):

            # charge- and LJ C6 parameter-weighted structure factors
            Sk2_q = iFQ2[hc, kc, lc]
            Sk2_lj = iFC62[hc, kc, lc]

            # in Essmann theta = fft(C*B) = fft(B*f1*pi*V)
            f1 = np.exp(-pi2*k2/alpha2_q)/k2
            Vq_recip += f1*B*Sk2_q

            # f6 function defined by Essmann for LJ ewald sum              
            x2 = pi2*k2/alpha2_lj
            x = np.sqrt(x2)
            f6 = ((1-2*x2)*np.exp(-x2) + 2*x2*x*sqrtpi*math.erfc(x))/3.0
            VLJ_recip += f6*B*Sk2_lj

        # apply multiplicative corrections not included in for loop
        Vq_recip *= 1.0/(2.0*self._V*np.pi)
        VLJ_recip *= (sqrtpi*alpha_lj)**3/(2*self._V)
        return Vq_recip, VLJ_recip


class NonbondedForceField_QLJEnergyForce():

    # Coulomb AND Lennard-Jones ewald summation for energy AND forces

    def _calc_ewald_self(self, alpha_q, alpha_lj):
        # self interaction of gaussian clouds & point charges/C6s
        # note the self interaction does not depend on r so dV_self/dr = 0
        Vq_self = 0.0
        VLJ_self = 0.0
        for i, qi, qiqi, c6ii in self._coeffs_iqc6:
            Vq_self += qiqi
            VLJ_self += c6_ii
        Vq_self *= alpha_q/np.sqrt(np.pi)
        VLJ_self *= (alpha_lj**6)/12.0
        return Vq_self, VLJ_self, 0, 0

    def _calc_ewald_real(self, r, alpha_q, alpha_lj):
        # interactions in real space (point charges/C6s + shielding potentials)
        Vq_real = 0.0
        VLJ_real = 0.0
        VLJ_c12 = 0.0
        Fq_real = np.zeros_like(r)
        FLJ_real = np.zeros_like(r)
        FLJ_c12 = np.zeros_like(r)

        alpha2_lj = alpha_lj*alpha_lj
        alpha2_q = alpha_q*alpha_q
        factor = 2.0*alpha_q/np.sqrt(np.pi)

        # pairs given explitily so no double counting correction
        for i, j, qiqj, c6, c12, r2, rij in self._get_pairs(r):

            # Coulombic direct-space r^-1 energy terms
            r1 = np.sqrt(r2)
            g1 = math.erfc(alpha_q*r1)
            Vij = qiqj*g1/r1
            Vq_real += Vij

            # Coulombic direct-space forces - i.e. negative gradient of Vq_real
            Fij = (qiqj*factor*np.exp(-alpha2_q*r2) + Vij) * rij/r2
            Fq_real[i, :] -= Fij
            Fq_real[j, :] += Fij

            # note we use LB mixing rules here but geometric for recip space
            # LB: c6 = 2*eps_ij*(rmin_ij**6) = 2*eps_ij*((rmin_ii+rmin_jj)/2)**6
            # geometric: c6 = sqrt(c6_ii*c6_jj) = 2*eps_ij*(rmin_ii*rmin_jj)**3
            # g6 is defined in gromacs manual 5.0.1 page 112 with a negative (?)

            # Lennard-Jones direct-space r^-6 London dispersion energy terms
            x2 = alpha2_lj*r2
            x4 = x2*x2
            g6 = np.exp(-x2)*(1 + x2 + x4/2.0)
            r6 = r2*r2*r2
            VLJ_real -= c6*g6/r6 # -c6*a6*exp(-x2)*(1/x6 + 1/x4 + 0.5/x2)

            # VLJ_real = -c6*a6*(u*v)
            # y2 = 1.0/x2
            # u = np.exp(-x2); v = ((y2 + 1)*y2 + 0.5)*y2
            # du = -2*r*np.exp(-a2*r2)
            # dv = (-6/(r6*a6) - 4/(r4*a4) - 1/(r2*a2))*a6/r
            # dVLJ_real/dr = -c6*a6*(u*dv + v*du)
            
            # Lennard-Jones diresct-space r^-6 London dispersion forces (Gao D.51)
            a8 = alpha2_lj**4
            y2 = 1.0/x2
            Fij = c6*a8*rij*np.exp(-x2)*(((6*y2 + 6)*y2 + 3)*y2 + 1)*y2
            FLJ_real[i, :] += Fij
            FLJ_real[j, :] -= Fij

            # Lennard-Jones r^-12 Pauli repulsion energy terms
            VLJ_c12 += c12/r12

            # Lennard-Jones r^-12 Pauli repulsion forces
            Fij = 12.0*VLJ_c12 * rij/r2
            FLJ_c12[i, :] += Fij
            FLJ_c12[j, :] -= Fij

        return Vq_real, VLJ_real, Fq_real, FLJ_real, VLJ_c12, FLJ_c12
    
    def _calc_ewald_recip(self, r, alpha_q, alpha_lj, kmax, mixing_rules):
        # long range interaction in reciprocal space

        # precalculate constants
        twojpi = 2j*np.pi
        sqrtpi = np.sqrt(np.pi)
        pi_over_alpha_lj = np.pi/alpha_lj
        negpi2_over_alpha2_q = -np.pi*np.pi/(alpha_q*alpha_q)

        # initalise energy and force arrays
        Vq_recip = 0.0
        VLJ_recip = 0.0
        Fk = np.zeros(len(r), dtype=complex)
        Fq_recip = np.zeros_like(r)
        FLJ_recip = np.zeros_like(r)

        # precalc binomial coefficent for 6th row
        factorial = lambda v: float(math.factorial(v))
        Pn = [factorial(6)/(factorial(n)*factorial(6-n)) for n in range(0, 7)]

        if mixing_rules == 'geometric':

            def calc_Sks(k):
                Sk_q = 0.0
                Sk_lj = 0.0
                k *= 2.0j*np.pi
                Sk_q_summands = []
                for i, qi, qiqi, c6ii in self._coeffs_iqc6:
                    ex = np.exp(np.dot(r[i], k))
                    Sk_q_summands.append(qi*ex)
                    Sk_lj_summands.append(c6ii*ex)
                Sk_q = sum(Sk_q_summands)
                Sk_lj = sum(Sk_lj_summands)
                return Sk_q, Sk_lj, Sk_q_summands, Sk_lj_summands

        elif mixing_rules == 'lorentz bertholot':

            # summands not collected so forces not returned for LB rules

            def calc_Sks(k):
                c6terms = self._coeffs_iqc6terms
                Sk_q = Sk_lj = 0.0
                Zn = Zn_6 = 0.0
                k *= 2.0j*np.pi

                # 0th loop: precalc fourier basis functions for next loops
                waves = []
                negwaves = []
                loop = zip(c6terms[0], c6terms[6])
                for (i, qi, c6), (i_, qi_, c6_) in loop:
                    ex = np.exp(np.dot(r[i], k))
                    waves.append(ex)
                    Zn += c6*ex
                    Zn_6 += c6_*ex
                    Sk_q += qi*ex
                Sk_lj += Pn[0]*Zn*np.conj(Zn_6)

                # next 1 - 6 loops: only calculate expansion terms and add to Sk
                for n in range(1, 7):
                    Zn = Zn_6 = 0.0
                    loop = zip(c6terms[n], c6terms[6-n], waves)
                    for (i, qi, c6), (i_, qi_, c6_), ex in loop:
                        Zn += c6*ex
                        Zn_6 += c6_*ex
                    Sk_lj += Pn[n]*Zn*np.conj(Zn_6)

                Sk_lj *= 2**-4 #sqrt(4)*sqrt(4)*(2**n)*(2**(6-n)) for n in [0,6]
                return Sk_q, Sk_lj, 0, 0

        # sum_i(sum_j(qi*qj*exp(2j*pi*dot(ri-rj, k))))
        # sum_j(sum_i(qi*exp(2j*pi*dot(ri, k)) * qj*exp(2j*pi*dot(rj, -k))))
        # sum_i(qi*exp(2j*pi*dot(ri, k))) * sum_i(qi*exp(2j*pi*dot(ri,-k)))
        # S(k)S(-k) = |S(k)|^2

        for hc in range(-kmax, kmax):
            for kc in range(-kmax, kmax):
                for lc in range(-kmax, kmax):

                    # convert k to the cartesian basis 
                    # TODO: check its the same when r is in the fractional basis
                    k = np.dot(self._ST, [hc, kc, lc])
                    k2 = np.dot(k, k)

                    if k2 == 0 or (k2 > self._kcut2 and not self._ignore_kcut):
                        continue

                    # charge- and c6ii-weighted structure factors, S(k)
                    # S(k)S(-k) = |S(k)|^2 = sum_ij (qi*qj*exp(2jpik*rij))
                    Sk_q, Sk_lj, Sk_q_summands, Sk_lj_summands = calc_Sks(k)

                    # Coulomb recip energy 
                    Sk2_q = np.abs(Sk_q)**2
                    f1 = np.exp(negpi2_over_alpha2_q*k2)/k2
                    Vq_recip += f1*Sk2_q

                    # Coulomb recip forces
                    # app6.pdf Appendix F eq. F.52b and Toukmaji96.pdf eq 9
                    # array of n scalars multiplied by 1 vector
                    ekrij = 1j*np.conj(Sk_q_summands)*Sk_q
                    vec = k*f1 
                    Fq_recip += np.outer(ekrij, vec)

                    # LJ recip dispersion energy
                    # Essmann (1995) eq 5.2, A5 and A11 pg 8589 & gromacs 5.0.1 
                    # page 112 (note we apply final corrections below)
                    Sk2_lj = np.abs(Sk_lj)**2
                    x = np.sqrt(k2)*pi_over_alpha_lj 
                    x2 = x*x
                    f6 = ((1-2*x2)*np.exp(-x2)+2*x2*x*sqrtpi*math.erfc(x))/3.
                    VLJ_recip += f6*Sk2_lj

                    # LJ recip dispersion forces
                    # Gao thesis appendix eq D.51
                    ekrij = 1j*np.conj(Sk_lj_summands)*Sk_lj
                    vec = k*f6
                    FLJ_recip += np.outer(ekrij, vec)


        correction_q = 1.0/(np.pi*2.0*self._V)
        correction_lj = (sqrtpi*alpha_lj)**3/(2.0*self._V)

        Vq_recip *= correction_q
        VLJ_recip *= correction_lj

        Fq_recip *= 2*(2*np.pi)*correction_q
        FLJ_recip *= 2*(2*np.pi)*correction_lj

        return Vq_recip, VLJ_recip, Fq_recip, FLJ_recip

    def _calc_pme_recip(self, r, alpha_q, alpha_lj, kmax):

        d = 3 # half length of local cube of (d*2)**3 points to calc spline
        g = 1.0/3.0 # grid resolution

        # grid dimensions
        uc_abc = np.array(self.molc.pdb.uc_abc)
        grid = na, nb, nc = np.floor(np.around(uc_abc/10.0)*10.0/g)
              
        # recursive 4th order cardinal B spline function and coefficent function
        M4, bi2 = self._get_recursive_M4()

        # coordinate transforms
        xws = r                                    # cartesian world coordinates
        xcs = np.dot(self.molc.pdb.S, xws.T).T % 1 # fractional crystal coordinates
        xgs = (xcs*grid).astype(int)               # grid coordinates

        # initalise grids
        Q = np.zeros((na, nb, nc))
        C6 = np.zeros((na, nb, nc))
        dQ = np.zeros((na, nb, nc, 3))
        dC6 = np.zeros((na, nb, nc, 3))

        # populate charge/c6 grid by looping over atoms' surrounding grid points
        # see Harvey et al. section C: charge spreading and Orac manual
        # compare to make_ljpme_c6grid in gromacs/mdlib/forcerec.cpp - eh?
        for x, y, z, q, c6, M, dM in self._loop_over_local_grids(xgs, grid, M4, d):
            Q[x, y, z] += q*M
            dQ[x, y, z, :] += q*dM

            # we use geometric mixing rules, alt. is to use LB mixing rules
            C6[x, y, z] += c6*M
            dC6[x, y, z, :] += c6*dM

        # fourier transform charge grid 
        # |S(k)|^2 = |F(Q)(k)|^2 = F(Q)(k)*F(Q)(-k) = S(k)S(-k)
        iFQ = np.fft.ifftn(Q)
        iFQ2 = np.abs(iFQ)**2

        # fourier transform geometric c6 coefficent grid
        iFC6 = np.fft.ifftn(C6)
        iFC62 = np.abs(iFC6)**2

        # pre-caclulate to save time
        alpha2_q = alpha_q*alpha_q
        alpha2_lj = alpha_lj*alpha_lj
        pi2 = np.pi*np.pi
        sqrtpi = np.sqrt(np.pi)

        # reciprocal energy
        Vq_recip = 0.0
        VLJ_recip = 0.0

        # loop over fourier Q and C6 grids
        for hc, kc, lc, B, k2 in self._loop_over_hkl(kmax, grid, bi2):
            # charge- and LJ C6 parameter-weighted structure factors
            Sk2_q = iFQ2[hc, kc, lc]
            Sk2_lj = iFC62[hc, kc, lc]

            # in Essmann theta = fft(C*B) = fft(B*f1*pi*V)
            f1 = np.exp(-pi2*k2/alpha2_q)/ksq
            iFtheta_q = f1*B
            Vq_recip += iFtheta_q*Sk2_q

            # multipliy so iFQ is no longer ifft(Q) but ifft(Q)*ifft(f1*B)            
            iFQ[hc, kc, lc] *= iFtheta_q

            # f6 function defined by Essmann for LJ ewald sum              
            x2 = pi2*k2/alpha2_lj
            x = np.sqrt(x2)
            f6 = ((1-2*x2)*np.exp(-x2) + 2*x2*x*sqrtpi*math.erfc(x))/3.0
            iFtheta_lj = f6*B
            VLJ_recip += iFtheta_lj*Sk2_lj

            # multipliy so iFC6 is no longer ifft(C6) but ifft(C6)*ifft(f6*B)
            iFC6[hc, kc, lc] *= iFtheta_lj

        # grid coordinates for atom positions 
        xs, ys, zs = map(tuple, xgs.T)

        # force computation see Essmann eq. 4.9 - evaluation at atom positions
        # fft(ifft(theta)*ifft(Q)) = conv(theta, Q)
        thetaxQ = np.fft.fftn(iFQ)[:, :, :, np.newaxis] 
        Fq_recip = -(dQ*thetaxQ)[xs, ys, zs, :] 

        thetaxC6 = np.fft.fftn(iFC6)[:, :, :, np.newaxis]
        FLJ_recip = -(dC6*thetaxC6)[xs, ys, zs, :]

        # apply multiplicative corrections not included in for loop
        correction_q = 1.0/(2.0*self._V*np.pi)
        Vq_recip *= correction_q
        Fq_recip *= correction_q

        correction_lj = (sqrtpi*alpha_lj)**3/(2*self._V)
        VLJ_recip *= correction_lj
        FLJ_recip *= correction_lj

        return Vq_recip, VLJ_recip, Fq_recip, FLJ_recip

    # helper functions for PME

    def _get_nonrecursive_Mn(self, n=4):
        # Non-recursive func for nth order cardinal B-spline (see Essmann eq. C1)
        fac = lambda v: float(math.factorial(v))
        vals = []
        for k in range(0, n):
            sign = (-1)**k
            Pn = fac(n)/(fac(k)*fac(n-k)) # self._Pn might not go high enough
            val = sign*Pn/fac(n-1)
            vals.append(val) 

        def Mn(u):
            Mn = 0.0
            for k in range(0, n+1):
                if u < k: break
                Mn += vals[k]*(u-k)**(n-1)
            return Mn

        return Mn

    def _get_recursive_M4(self):

        def M2(u):
            if u < 0 or u > 2:
                return 0.0
            return 1.0 - np.abs(u-1.0)

        M3s = {}
        def M3(u):
            if u in M3s:
                return M3s[u]
            M3s[u] = M3 = (u*M2(u) + (2.0-u)*M2(u-1))/2.0
            return M3

        # 4th order cardinal B-spline using recursion (see Essmann eq. 4.1)
        M4s = {}
        def M4(u):
            if u in M4s:
                return M4s[u]
            M3u = M3(u)
            M3um1 = M3(u-1)
            M4 = (u*M3u + (3.0-u)*M3um1)/3.0
            dM4 = M3u - M3um1
            M4s[u] = M4, dM4
            return M4, dM4

        # explonential terms for Euler/cardinal B-splines
        twojpi = 2j*np.pi
        def bi2(mi_over_Ki):
            sum_Mn_exp = 0.0
            for k in range(0, n-2):
                M4, dM4 = M4(k+1)
                sum_Mn_exp += M4*np.exp(twojpi*mi_over_Ki*k)
            bi = np.exp(twojpi*(n-1)*mi_over_Ki)/sum_Mn_exp
            bi2 = np.abs(bi)**2
            return bi2

        return M4, bi2

    def _loop_over_local_grids(self, xgs, grid, Mn, d=3):
        na, nb, nc = grid
        # see self._init_nonbonded for how c6 coefficents are calculated

        # note: the localgrids do not sample space evenly unless the crystal 
        # system is cubic, but this is how the PME method is described by Essmann

        drange = localgrid(+d, -d-1, -1)
        for (xg, yg, zg), (i, qi, qiqi, c6) in zip(xgs, self._coeffs_iqc6):
            Mnys = [[yg-dyg, dyg, Mn(dyg)] for dyg in localgrid]
            Mnzs = [[zg-dzg, dzg, Mn(dzg)] for dzg in localgrid]
            for dxg in localgrid:
                rxg = xg-dxg
                Mnx, dMnx = Mn(dxg)
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
                        yield ixg, iyg, izg, qi, c6, M, dM

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







class NonbondedForceField_NaiveQLJEnergy():

    # naive implementations for energy only

    def _calc_naive_energy_only(self, r):
        Vq = 0.0
        VLJ = 0.0
        for i, j, qiqj, cij6, cij12, bij2, rij in self._get_pairs(r):
            Vq += qiqj/bij2
            r6 = r2*r2*r2
            VLJ += (cij12/r6 - cij6)/r6
        return Vq, VLJ

    def _calc_ljnaive_energy_only(self, r):
        VLJ = 0.0
        for i, j, qiqj, cij6, cij12, bij2, rij in self._get_pairs(r):
            r6 = r2*r2*r2
            VLJ += (cij12/r6 - cij6)/r6
        return VLJ

    # naive implementations for energy and forces

    def _calc_naive(self, r):
        # see physical chemistry: quanta, matter and change by Atkins eq 35.14
        Vq = 0.0
        VLJ = 0.0
        Fq = np.zeros_like(r)
        FLJ = np.zeros_like(r)
        for i, j, qiqj, cij6, cij12, bij2, rij in self._get_pairs(r):

            # Coulombic interaction energy
            Vq += qiqj/bij2
            # Coulombic force on both atoms as the negative energy gradient:
            # Fij = -dUdr = -dUdb*dbdr = -(-2*qiqj/bij3 * rij/bij)
            Fij = 2*qiqj*rij/(r2*r2) 
            Fq[i, :] += Fij
            Fq[j, :] -= Fij
            # 12-6 Lennard-Jones interaction energy
            r6 = r2*r2*r2
            VLJ += (cij12/r6 - cij6)/r6
            # 12-6 Lennard-Jones force on both atoms:
            # Fij = -dUdr = -dUdb*dbdr = -(-12*c12/bij13 + 6*c6/bij7)*rij/bij
            Fij = (12*cij12/r6 - 6*cij6)/(r6*r2)*rij
            FLJ[i, :] += Fij
            FLJ[j, :] -= Fij

        return Vq, VLJ, Fq, FLJ

    def _calc_ljnaive(self, r):
        VLJ = 0.0
        FLJ = np.zeros_like(r)
        for i, j, qiqj, cij6, cij12, bij2, rij in self._get_pairs(r):
            r6 = r2*r2*r2
            VLJ += (cij12/r6 - cij6)/r6
            Fij = (12*cij12/r6 - 6*cij6)/(r6*r2)*rij
            FLJ[i, :] += Fij
            FLJ[j, :] -= Fij
        return VLJ, FLJ
    
