
"""core.py - density sampling and structure factor calculation"""

__package__ = "xstalpy"
__author__ = "James Gowdy" 
__email__ = "jamesgowdy1988@gmail.com"

from progress import ProgressBar

import numpy as np

################################################################################

# Structure factor calculation - direct summation algorithm implementations

################################################################################

def fhkl_printer(results, sort=True):

    # printer Fhkl results in a table

    fmt = '{:2d} {:2d} {:2d} {:9.2f} {:9.2f} '
    if isinstance(results[0], int):
        results = [results]
    if sort:
        if isinstance(results, np.ndarray):
            results = results[(-results[:, 3]).argsort()]
        else:
            results = sorted(results, key=lambda a: -a[3])
    for h, k, l, d, A, phi in results:
        h, k, l = map(int, [h, k, l])
        if A < 0.05: 
            print((fmt+'{:>9}').format(h, k, l, d, A, '-'))
        else: 
            print((fmt+'{:9.2f}').format(h, k, l, d, A, phi))

def check_np(*req_version):
    cur_version = tuple(map(int, np.__version__.split('.')))
    if cur_version <= tuple(req_version):
        msg = 'Must upgrade numpy > {}'.format(req_version) # for new meshgrid
        raise ImportError(msg)

def directsum_single(pdb, h, k, l, expand=False):

    # Fhkl calculation using direct summation for one-off calcs and reference
    if expand:
        pdb = pdb.get_unitcell()

    # calculate resolution (inverse of recip lattice vector)
    ST = np.array(pdb.S).T
    hw, kw, lw = np.dot(ST, [h, k, l])
    dstar2 = hw*hw + kw*kw + lw*lw
    stol2 = dstar2/4.0
    d = 1.0/np.sqrt(dstar2)

    # load Cromer-Mann coefficients and calculate atomic scattering factors
    with open('cm.pkl', 'rb') as _f: 
        _A, _B, _C = pickle.load(_f)

    f0 = {}
    for e in set(pdb.e):
        f0[e] = _C[e] + sum(_A[e][i]*np.exp(-_B[e][i]*stol2) for i in range(4))

    # direct summation using isotropic B-factors
    F = 0 + 0j
    for x, y, z, e, n, B in zip(pdb.x, pdb.y, pdb.z, pdb.e, pdb.n, pdb.B):
        #hx = np.dot([h, k, l], np.dot(pdb.S, [x, y, z]))
        hx = np.dot([hw, kw, lw], [x, y, z])
        F += n * f0[e] * np.exp(-B*stol2) * np.exp(2.0j*np.pi*hx)

    A = abs(F)
    phi = np.round(np.degrees(np.arctan2(F.imag, F.real)), 2)
    return [h, k, l, d, A, phi]

def directsum_numpy(pdb, dmin, dmax, expand=False):

    # direct sum Fhkl algorithm - vectorised implementation

    # apply symmetry operators and expand to full unit cell
    if expand:
        pdb = pdb.get_unitcell()

    # convert to crystallographic/fractional basis 
    S = np.array(pdb.S)
    xyzc = np.dot(S, [pdb.x, pdb.y, pdb.z])

    ####################### enumerate hkls

    # get maximum hkl indicies possible for a resolution range
    hmax = int(1.0/(dmin*np.linalg.norm(S[0]))) + 1
    kmax = int(1.0/(dmin*np.linalg.norm(S[1]))) + 1
    lmax = int(1.0/(dmin*np.linalg.norm(S[2]))) + 1

    # build cube in reciprocal space
    check_np(1, 6, 2)
    H, K, L = np.meshgrid(range(hmax), range(-kmax, kmax), range(-lmax, lmax))
    H = H.transpose(1, 0, 2).flatten()
    K = K.transpose(1, 0, 2).flatten()
    L = L.flatten()
    hkls = np.row_stack((H, K, L))
    
    # calculate resolution (inverse of recip lattice vector)
    dstar = np.dot(S.T, hkls)
    dstar = np.sqrt(np.sum(dstar*dstar, axis=0))
    d = 1.0/np.where(dstar != 0.0, dstar, 1e10)

    # trim to sphere in recip space
    mask = (dmin <= d) & (d <= dmax)
    hkls = hkls[:, mask].T
    d = d[mask]
    stol2 = 1.0/(4.0*d*d)

    #######################

    # load Cromer-Mann coefficients from ITC tables vol C ch 6.1 
    assert os.path.isfile('cm.pkl')
    with open('cm.pkl', 'rb') as _f:
        _A, _B, _C = pickle.load(_f)

    # calculate Cromer-Mann atomic scattering factors
    f0 = {}
    for e in set(pdb.e):
        f0[e] = _C[e]
        f0[e] += sum(_A[e][i]*np.exp(-_B[e][i]*stol2) for i in range(4))

    # direct summation while printing progress
    F = np.zeros(len(hkls), dtype=complex)
    p = ProgressBar(final=len(pdb), label='directsum', tail_label='{f:>10}')
    for xyzc, e, n, b in zip(xyzc.T, pdb.e, pdb.n, pdb.B):
        p.update()
        hx = np.sum(hkls * xyzc, axis=1)
        F += n * f0[e] * np.exp(-b * stol2) * np.exp(np.pi * 2j *hx)

    A = np.abs(F)
    phi = np.degrees(np.arctan2(F.imag, F.real))
    F = np.around(np.column_stack((hkls, d, A, phi)), 6)
    # F = F[(-F[:, 3]).argsort()]
    return F

def directsum_python(pdb, dmin, dmax, expand=False):

    # directsum Fhkl calculation - reference implementation to check c code

    # load Cromer-Mann coefficient dictionaries from ITC tables vol C ch 6.1
    assert os.path.isfile('cm.pkl')
    with open('cm.pkl', 'rb') as _f:
        _A, _B, _C = pickle.load(_f)

    # apply symmetry operators and expand to full unit cell
    if expand: 
        pdb = pdb.get_unitcell()

    S = pdb.S
    ST = [[S[0][0], S[1][0], S[2][0]],
          [S[0][1], S[1][1], S[2][1]],
          [S[0][2], S[1][2], S[2][2]]]

    # convert to crystallographic/fractional basis 
    xcs, ycs, zcs = [], [], []
    for x, y, z in zip(pdb.x, pdb.y, pdb.z):
        xcs.append(S[0][0]*x + S[0][1]*y + S[0][2]*z)
        ycs.append(S[1][0]*x + S[1][1]*y + S[1][2]*z)
        zcs.append(S[2][0]*x + S[2][1]*y + S[2][2]*z)

    es = set(pdb.e)
    atoms = zip(pdb.e, xcs, ycs, zcs, pdb.n, pdb.B)
    F = []

    ####################### enumerate hkls

    # rearrange d*max <= hmax|a*| to d*max/|a*| <= hmax etc.
    astar = math.sqrt(S[0][0]*S[0][0] + S[0][1]*S[0][1] + S[0][2]*S[0][2])
    bstar = math.sqrt(S[1][0]*S[1][0] + S[1][1]*S[1][1] + S[1][2]*S[1][2])
    cstar = math.sqrt(S[2][0]*S[2][0] + S[2][1]*S[2][1] + S[2][2]*S[2][2])
    hmax = int(1.0/(dmin*astar)) + 1
    kmax = int(1.0/(dmin*bstar)) + 1
    lmax = int(1.0/(dmin*cstar)) + 1

    p = ProgressBar(final=hmax*2*kmax, label='directsum', tail_label='{t:>10}')

    for h in range(0, hmax):
        hw = ST[0][0]*h # + ST[0][1]*k + ST[0][2]*l
        for k in range(-kmax, kmax):
            p.update()
            kw = ST[1][0]*h + ST[1][1]*k # + ST[1][2]*l
            for l in range(-lmax, lmax):
                lw = ST[2][0]*h + ST[2][1]*k + ST[2][2]*l
                dstar2 = hw*hw + kw*kw + lw*lw
                d = 1.0/math.sqrt(dstar2+1e-20)
                if d < dmin or d > dmax:
                    continue

                stol2 = dstar2/4.0

                #######################

                # calculate Cromer-Mann atomic scattering factors
                f0 = {}
                for e in es:
                    f0[e] = _C[e]
                    f0[e] += sum(_A[e][i]*np.exp(-_B[e][i]*stol2) for i in range(4))

                # direct summation using isotropic B-factors
                Freal, Fimag = 0.0, 0.0
                for e, x, y, z, n, b in atoms:
                    alpha = 2.0 * np.pi * (h*x + k*y + l*z)
                    tmp = n * f0[e] * math.exp(-b*stol2)
                    Freal += tmp * math.cos(alpha)
                    Fimag += tmp * math.sin(alpha)

                # convert to amplitude and phase
                A = math.sqrt(Freal*Freal + Fimag*Fimag)
                phi = round(math.degrees(math.atan2(Fimag, Freal)), 3)
                F.append([h, k, l, d, A, phi])

    # sort according to resolution
    # F = sorted(F, key=lambda a: -a[3])
    return F

################################################################################

# Structure factor calculation - density sampling algorithm implementations

################################################################################

class Mixin():

    def _enumerate_hkl_python(self, dmin, dmax, S=None):
        return self.enumerate_hkl_python(dmin, dmax, S or self.pdb.S)

    @staticmethod
    def enumerate_hkl_python(dmin, dmax, S):

        dstarmin2 = 1.0/(dmax*dmax)
        dstarmax2 = 1.0/(dmin*dmin)

        # rearrange d*max <= hmax|a*| to d*max/|a*| <= hmax etc
        astar = math.sqrt(S[0][0]*S[0][0] + S[0][1]*S[0][1] + S[0][2]*S[0][2])
        bstar = math.sqrt(S[1][0]*S[1][0] + S[1][1]*S[1][1] + S[1][2]*S[1][2])
        cstar = math.sqrt(S[2][0]*S[2][0] + S[2][1]*S[2][1] + S[2][2]*S[2][2])

        hmax = int(1.0/(dmin*astar)) + 1
        kmax = int(1.0/(dmin*bstar)) + 1
        lmax = int(1.0/(dmin*cstar)) + 1

        hkls = []
        ds = []
        for h in range(0, hmax):
            hw = S[0][0]*h
            hw2 = hw*hw
            for k in range(-kmax, kmax):
                # kw = ST[1][0]*h + ST[1][1]*k
                kw = S[0][1]*h + S[1][1]*k
                kw2 = kw*kw
                # if hw2+kw2 > dstarmax2:
                #   break
                for l in range(-lmax, lmax):
                    # lw = ST[2][0]*h + ST[2][1]*k + ST[2][2]*l
                    lw = S[0][2]*h + S[1][2]*k + S[2][2]*l
                    dstar2 = hw2 + kw2 + lw*lw
                    #if dstar2 > dstarmax2:
                    #    break
                    if dstar2 < dstarmin2:
                        continue
                    d = 1.0/math.sqrt(dstar2)
                    hkls.append([h, k, l])
                    ds.append(d)
        hkls = zip(*hkls)
        return hkls, ds

    def _enumerate_hkl_numpy(self, dmin, dmax, S=None):
        return self.enumerate_hkl_numpy(dmin, dmax, S or self.pdb.S)

    @staticmethod
    def enumerate_hkl_numpy(dmin, dmax, S):

        # get maximum hkl indicies possible for a resolution range
        hmax = int(1.0/(dmin*np.linalg.norm(S[0]))) + 1
        kmax = int(1.0/(dmin*np.linalg.norm(S[1]))) + 1
        lmax = int(1.0/(dmin*np.linalg.norm(S[2]))) + 1

        # meshgrid takes only 2 args in numpy 1.6.2, need later version
        check_np(1, 6, 2)

        # build cube in reciprocal space
        H, K, L = np.meshgrid(range(hmax), range(-kmax, kmax), range(-lmax, lmax))
        H = H.transpose(1, 0, 2).flatten()
        K = K.transpose(1, 0, 2).flatten()
        L = L.flatten()
        hkls = np.row_stack((H, K, L))

        # calculate resolution (inverse of recip lattice vector)
        S = np.array(S)
        dstar = np.dot(S.T, hkls)
        dstar = np.sqrt(np.sum(dstar*dstar, axis=0))
        d = 1.0/np.where(dstar != 0.0, dstar, 1e10)

        # trim to sphere in recip space
        mask = (dmin <= d) & (d <= dmax)
        hkls = hkls[:, mask].T
        d = d[mask]
        return hkls, d

    def _isabsent_python(self, h, k, l):

        hR = []
        hTmod1 = []
        for R, T in zip(self._R, self._T):
            _h = round(h*R[0][0] + k*R[1][0] + l*R[2][0], 4)
            _k = round(h*R[0][1] + k*R[1][1] + l*R[2][1], 4)
            _l = round(h*R[0][2] + k*R[1][2] + l*R[2][2], 4)
            hT = round(h*T[0] + k*T[1] + l*T[2], 4)
            hR.append([_h, _k, _l])
            hTmod1.append(hT%1.0)

        # If hR == h and hT%1 != 0 then the reflection h is absent
        # TODO: I do not think we need the outer loop / pairwise check 
        for i in range(len(hR)):
            cancelsout = False
            for j in range(len(hR)):
                if i != j and hR[i] == hR[j] and (hTmod1[i] or hTmod1[j]):
                    cancelsout = True
                    break
            if not cancelsout: # if loop did not break
                return False
        return True

    def _isabsent_numpy(self, hkls):

        # next two lines slowish
        hR = np.around([np.dot(hkls, R) for R in self._R], 4)
        hT = np.around([np.dot(hkls, T) for T in self._T], 4)
        assert hT.dtype.type is np.float_

        rem = hT-np.round(hT)
        rem = np.where(abs(rem) < 1e-10, 0, rem)
        notabsent = np.ones(len(hkls))

        for i in range(len(self._R)):
            for j in range(len(self._R)):
                if i == j: continue
                
                # If hR == h and hT%1 != 0 then the reflection h is absent
                areequal = np.all(np.equal(hR[i], hR[j], axis=1))
                remainder = np.logical_or(rem[i], rem[j])
                partnered = np.logical_and(areequal, remainder)
                notabsent[partnered] = 0 # where partnered hkl is absent

        return np.logical_not(notabsent)

    def _make_FhklArray(self):

        def _slice2list(s, i):
            if isinstance(s, slice):
                start = s.start or 0
                stop = s.stop or self._grid[i] # fixed
                step = s.step or 1
                return range(start, stop, step)
            elif isinstance(s, int):
                return [s]
            return s

        class FhklArray:
            def __getitem__(self_, hkl):
                H, K, L = hkl
                hkls = []
                for h in _slice2list(H, 0):
                    for k in _slice2list(K, 1):
                        for l in _slice2list(L, 2):
                            hkls.append([h, k, l])
                return self[hkls] # fixed

        return FhklArray() # return FhklArray instance which acts as a facade

class DensitySampling_python(Mixin):

    # density sampling Fhkl calc - reference implementation to debug c/c++ code

    def __init__(self, pdb, dmin=1.0, g=1.0/3.0, rho_cutoff=0.01, rpad=2, Q=100.,
                 expand=False):

        self._expand = expand

        # inherit from Mixin
        self.enumerate_hkl = self._enumerate_hkl_python
        self.isabsent = self._isabsent_python

        cos = math.cos
        sin = math.sin

        # unit cell parameters
        a, b, c = pdb.uc_abc
        V = self._V = pdb.calc_volume()
        S = self._S = pdb.calc_scale()
        SI = pdb.calc_invscale()
        
        def matrixmult(A, B):
            AB = [[0]*3, [0]*3, [0]*3]
            for i in range(3):
                for j in range(3):
                    for k in range(3): 
                        AB[i][j] += A[i][k]*B[k][j]
                    AB[i][j] = round(AB[i][j], 4)
            return AB

        # convert symops to the crystallographic basis for reciprocal expansion
        self._R = []
        self._T = []
        S = pdb.S
        for i in pdb.R.keys():
            Rw = pdb.R[i]
            Rc = matrixmult(matrixmult(S, Rw), SI)
            Tw = pdb.T[i]
            Tc = [S[0][0]*Tw[0] + S[0][1]*Tw[1] + S[0][1]*Tw[2],
                  S[1][0]*Tw[1] + S[1][1]*Tw[1] + S[1][1]*Tw[2],
                  S[2][0]*Tw[2] + S[2][1]*Tw[2] + S[2][1]*Tw[2]]

            Tc = np.around(np.dot(S, Tw), 4)

            # self._T = np.around([np.dot(S, pdb.T[i]) for i in pdb.R.keys()], 4)
            self._T.append(Tc)
            self._R.append(Rc)

        # baseline Biso and Ueq = Biso/(8*pi^2)
        sigma = 0.5/g
        bbase = (math.log10(Q)*dmin*dmin)/(sigma*(sigma-1))
        bmin = min(pdb.B)
        bsmear = min(bbase-bmin, 100*(8*math.pi**2))
        self._bsmear = bsmear

        # grid dimensions
        na = int(round(a/10.0)*10.0/g)
        nb = int(round(b/10.0)*10.0/g)
        nc = int(round(c/10.0)*10.0/g)
        self._grid = na, nb, nc

        # build 3D array for density
        rpad = 2
        self.rho = [None]*na
        for i in xrange(na):
            self.rho[i] = [None]*nb
            for j in xrange(nb):
                self.rho[i][j] = [0.0]*(nc+rpad)

        # ensure true division
        _na = 1.*na
        _nb = 1.*nb
        _nc = 1.*nc

        # initalise progress bar
        progressbar = ProgressBar(len(pdb), label='sampling', tail_label='{t:>10}')

        # load Cromer-Mann coefficient dictionaries from ITC tables vol C ch 6.1
        with open('cm.pkl', 'rb') as _f: 
            _A, _B, _C = pickle.load(_f)

        A = {e: _A[e] + [_C[e]] for e in set(pdb.e)}
        B = {e: _B[e] + [0] for e in set(pdb.e)}

        # loop over all atoms
        for e, xw, yw, zw, n, b in zip(pdb.e, pdb.x, pdb.y, pdb.z, pdb.n, pdb.B):

            progressbar.update()

            # atom crystallographic fractional coordinates
            xc = S[0][0]*xw + S[0][1]*yw + S[0][2]*zw
            yc = S[1][1]*yw + S[1][2]*zw
            zc = S[2][2]*zw

            # atom crystallographic grid coordinate
            xg = int(round(xc*na))
            yg = int(round(yc*nb))
            zg = int(round(zc*nc))
            
            # prepare stuff for density calculation
            b += bsmear
            prs = [n*A[e][i]*(4.0*math.pi/(B[e][i]+b))**(3./2.) for i in range(5)]
            exs = [4.0*math.pi*math.pi/(B[e][i]+b) for i in range(5)]

            def density(dsq):
                return sum(prs[i]*math.exp(-exs[i]*dsq) for i in range(5))

            # get cutoff radius for the (1-rho_cutoff) isosurface
            dcut = 0
            rho_origin = density(0)
            while density(dcut*dcut)/rho_origin > rho_cutoff: 
                dcut += g
            assert dcut > 0
            dcutsq = dcut*dcut

            # cutoff radius in crystallographic fractional cordinates
            dmax_xc = S[0][0]*dcut + S[0][1]*dcut + S[0][2]*dcut
            dmax_yc = S[1][1]*dcut + S[1][2]*dcut
            dmax_zc = S[2][2]*dcut

            # local sampling box grid coordinate limits
            dmax_xg = int(round(dmax_xc*na))
            dmax_yg = int(round(dmax_yc*nb))
            dmax_zg = int(round(dmax_zc*nc))

            # local sampling box grid limits centered on atoms
            rxg_min = xg-dmax_xg
            rxg_max = xg+dmax_xg
            ryg_min = yg-dmax_yg
            ryg_max = yg+dmax_yg
            rzg_min = zg-dmax_zg
            rzg_max = zg+dmax_zg

            # loop over local voxel grid coordinates
            for rzg in range(rzg_min, rzg_max):
                rzc = rzg/_nc
                rzw = SI[2][2]*rzc
                dzw = rzw-zw
                dzwdzw = dzw*dzw
                rzg %= nc

                for ryg in range(ryg_min, ryg_max):
                    ryc = ryg/_nb
                    ryw = SI[1][1]*ryc + SI[1][2]*rzc
                    dyw = ryw-yw
                    dywdyw = dyw*dyw
                    ryg %= nb

                    for rxg in range(rxg_min, rxg_max):
                        rxc = rxg/_na
                        rxw = SI[0][0]*rxc + SI[0][1]*ryc + SI[0][2]*rzc
                        dxw = rxw-xw
                        rxg %= na

                        # distance^2 between atom and grid voxels
                        dsq = dxw*dxw + dywdyw + dzwdzw

                        # trim edges of box exceeding radial cutoff
                        if dsq > dcutsq:
                            continue


                        # add current atom's density constribution to voxel
                        self.rho[rxg][ryg][rzg] += density(dsq)

        # complex conjugate of fourier coeffs is equiv to iFFT
        self._F = np.conj(np.fft.fftpack.fftn(self.rho, s=self._grid))
        self._F *= self._V/(na*nb*nc)

    def _equivalent(self, h, k, l):
        j = 1j
        if l < 0:
            h, k, l, j = -h, -k, -l, -1j
        if h < 0:
            h += self._grid[0]
        if k < 0:
            k += self._grid[1]
        return h, k, l, j

    def __getitem__(self, hkl):

        h, k, l = hkl
        if self._expand:
            F = 0 + 0j
            # enumerate symmetry-related reflections
            for R, T in zip(self._R, self._T):
                hR = int(round(h*R[0][0] + k*R[1][0] + l*R[2][0]))
                kR = int(round(h*R[0][1] + k*R[1][1] + l*R[2][1]))
                lR = int(round(h*R[0][2] + k*R[1][2] + l*R[2][2]))
                hT = h*T[0] + k*T[1] + l*T[2]
                shift = cmath.exp(2j*math.pi*hT)

                # get the equivalent refletion in the grid
                hR, kR, lR, j = self._equivalent(hR, kR, lR)
                _ = self._F[hR][kR][lR]
                F += (_.real + _.imag*j) * shift
        else:
            h, k, l, j = self._equivalent(h, k, l)
            _ = self._F[h][k][l]
            F = _.real + _.imag*j

        # apply bsmear correction
        S = self._S
        hw = S[0][0]*h
        kw = S[0][1]*h + S[1][1]*k
        lw = S[0][2]*h + S[1][2]*k + S[2][2]*l
        d2 = 1.0/(hw*hw + kw*kw + lw*lw)
        F *= math.exp(self._bsmear/(4.0*d2))

        # return resolution, amplitude and phase
        d = math.sqrt(d2)
        A = abs(F)
        phi = round(math.degrees(cmath.phase(F)), 3)
        if A < 0.05: phi = 0
        return h, k, l, d, A, phi

class DensitySampling_numpy(Mixin):

    # density sampling Fhkl calculation - this implementation is vectorised

    def __init__(self, pdb, dmin=1.0, g=0.333333333333, rho_cutoff=0.01, rpad=2,
                 Q=100., expand=False):

        self._expand = expand

        # inherit from Mixin
        self.enumerate_hkl = self._enumerate_hkl_numpy
        self.isabsent = self._isabsent_numpy

        # enable access to self.F[h, k, l] with slicing support
        self.F = self._make_FhklArray()

        check_np(1, 6, 2) # for meshgrid

        V  = self._V = pdb.calc_volume()
        S  = self._S = np.array(pdb.calc_scale())
        SI = self._SI = np.array(pdb.calc_invscale())

        keys = pdb.R.keys()
        self._R = np.around([np.dot(np.dot(S, pdb.R[i]), SI) for i in keys], 4)
        self._T = np.around([np.dot(S, pdb.T[i]) for i in keys], 4)
        
        # baseline Biso and Ueq = Biso/(8*pi^2)
        sigma = 0.5/g
        bbase = (np.log10(Q)*dmin*dmin)/(sigma*(sigma-1))
        bmin = min(pdb.B)
        bsmear = bbase-bmin
        self._bsmear = bsmear

        # grid dimensions and build zeros
        gridf = np.floor(np.around(np.array(pdb.uc_abc, dtype=float)/10.0)*10.0/g)
        gridi = gridf.astype(int)
        self._grid = gridi
        self._rho = np.zeros((gridi[0], gridi[1], gridi[2]+rpad))

        # initalise progress bar
        progressbar = ProgressBar(len(pdb), label='sampling', tail_label='{f:>10}')

        # load Cromer-Mann coefficient dictionaries from ITC tables vol C ch 6.1
        with open('cm.pkl', 'rb') as _f: 
            _A, _B, _C = pickle.load(_f)

        A = {e: _A[e] + [_C[e]] for e in set(pdb.e)}
        B = {e: _B[e] + [0] for e in set(pdb.e)}

        # coordinates of point atoms
        xws = np.row_stack((pdb.x, pdb.y, pdb.z))
        xcs = np.dot(S, xws)
        xgs = np.round(xcs*gridf.reshape(3, 1)).astype(int)

        for e, xw, xc, xg, b, n in zip(pdb.e, xws.T, xcs.T, xgs.T, pdb.B, pdb.n):

            progressbar.update()

            # model electron density as inverse scattering factor
            b += bsmear
            prs = [n*A[e][i]*(4*math.pi/(B[e][i]+b))**(3./2.) for i in range(5)]
            exs = [4*math.pi*math.pi/(B[e][i]+b) for i in range(5)]

            def density(dsq):
                return sum(prs[i]*np.exp(-exs[i]*dsq) for i in range(5))

            # get cutoff radius for the (1-rho_cutoff) isosurface
            dcut = 0
            rho0 = density(0.0)
            while density(dcut*dcut)/rho0 > rho_cutoff: 
                dcut += g
            assert dcut > 0

            # cube limits in crystallographic and grid coordinates
            dmax_c = np.dot(S, [dcut, dcut, dcut])
            dmax_g = np.round(dmax_c*gridf).astype(int)

            # build cube 
            X, Y, Z = np.meshgrid(*[range(-dg, dg) for dg in dmax_g])
            X = X.transpose(1, 0, 2).flatten()
            Y = Y.transpose(1, 0, 2).flatten()
            Z = Z.flatten()
            dgs = np.column_stack((X, Y, Z))

            # translate to atom centre and change coordinates
            rgs = dgs + xg
            rcs = rgs/gridf 
            rws = np.dot(SI, rcs.T).T

            # distance^2 between atom and grid voxels
            dws = rws - xw 
            dsqs = np.sum(dws*dws, axis=1)

            # trim edges of box exceeding radial cutoff
            mask = dsqs <= dcut*dcut
            dsqs = dsqs[mask]
            rgs = rgs[mask]
            rcs = rcs[mask]

            # bring grid points into unit cell and add electron density
            rgs = (rgs%gridf).astype(int)
            idx = tuple(rgs.T)

            self._rho[idx] += density(dsqs)
        
        # conj(fft()) is equivalent to ifft()
        self._F = np.conj(np.fft.fftpack.fftn(self._rho, s=self._grid))

        # volume/ngrid factor correction
        self._F *= self._V/np.prod(self._grid)

    def _equivalent(self, hkls):
        islneg = np.where(hkls[:, 2] < 0, -1, 1)
        hkls = hkls * islneg.reshape(-1, 1) # *= would update beyond func scope
        hkls = np.where(hkls < 0, hkls+self._grid, hkls).astype(int)
        h, k, l = hkls.T
        return h, k, l, islneg*1j 

    def __getitem__(self, hkls):

        hkls = np.array(hkls).reshape(-1, 3)

        if self._expand:
            F = np.zeros(len(hkls), dtype=complex)
            for R, T in zip(self._R, self._T):
                hRs = np.dot(hkls, R)
                shifts = np.exp(2j*np.pi*np.dot(hkls, T))
                h, k, l, j = self._equivalent(hRs)
                _ = self._F[h, k, l]
                F += (_.real + _.imag*j) * shifts

        else:
            h, k, l, j = self._equivalent(hkls)
            _ = self._F[h, k, l]
            F = _.real + _.imag*j

        # bsmear correction
        d2 = 1.0/(np.dot(self._S.T, hkls.T)**2).sum(0)
        F *= np.exp(self._bsmear * 1.0/(4.0*d2))

        # return the hkl, resolution, amplitude and phase
        d = np.sqrt(d2)
        A = abs(F)
        phi = np.degrees(np.arctan2(F.imag, F.real))
        phi = np.where(A > 0.05, phi, 0)

        # sort by resolution
        F = np.around(np.column_stack([hkls, d, A, phi]), 6)
        # F = F[(-F[:, 3]).argsort()]
        return F

class DirectSum_breakdown(Mixin):

    # directsum Fhkl calc - this implementation keeps track of atom contributions

    def __init__(self, pdb, dmin, dmax, expand=True):

        # inherit from Mixin
        self.enumerate_hkl = self._enumerate_hkl_numpy

        # enable access to self.F[h, k, l] with slicing support
        self.F = self._make_FhklArray()

        # coordinates
        if expand:
            pdb = pdb.get_unitcell()

        uc_xyzc = np.dot(pdb.S, [pdb.x, pdb.y, pdb.z])
        self.n = len(pdb)

        # reflections 
        hkls, d = self._enumerate_hkl_numpy(dmin, dmax, S=pdb.S)
        stol2 = 1.0/(4.0*d*d)

        # Cromer-Mann scattering factors 
        assert os.path.isfile('cm.pkl')
        with open('cm.pkl', 'rb') as _f: _A, _B, _C = pickle.load(_f)
        f0 = {e: _C[e] + sum(_A[e][i]*np.exp(-_B[e][i]*stol2) for i in range(4)) 
              for e in set(pdb.e)}

        # direct summation for each summand
        self.Fatoms = np.zeros((len(pdb), len(hkls)), dtype=complex)
        p = ProgressBar(final=len(pdb), label='directsum', tail_label='{f:>10}')

        for i, (xyzc, e, n, b) in enumerate(zip(uc_xyzc.T, pdb.e, pdb.n, pdb.B)):
            p.update()
            hx = np.sum(hkls * xyzc, axis=1)
            self.Fatoms[i, :] = n*f0[e]*np.exp(-b*stol2) * np.exp(np.pi*2j*hx)

        # save drange
        self._hkls = hkls
        self._d = d
        self.selection()

    def selection(self, sel=None):

        # get only the selected atoms
        if sel is None: 
            sel = slice(0, self.n, 1)
        F = self.Fatoms[sel, :].sum(0)

        # normal conversion to A, phi
        A = abs(F)
        phi = np.degrees(np.arctan2(F.imag, F.real))
        phi = np.where(A > 0.05, phi, 0)
        F = np.around(np.column_stack((self._hkls, self._d, A, phi)), 6)

        self._lookup = {(h, k, l): [h, k, l, d, A, phi] for h, k, l, d, A, phi in F} 
        # F = F[(-F[:, 3]).argsort()]
        return F

    def __getitem__(self, hkls):
        # to choose another atom selection must call self.selection()
        hkls = np.array(hkls).reshape(-1, 3)
        F = []
        for h, k, l in hkls:
            assert self._lookup[(h, k, l)], '%d %d %d not in drange?'%(h, k, l)
            F.append(self._lookup[(h, k, l)])
        F = np.array(F)
        # F = F[(-F[:, 3]).argsort()]
        return F
