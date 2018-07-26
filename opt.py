"""opt.py - optimisation routines"""

__package__ = "xstalpy"
__author__ = "James Gowdy" 
__email__ = "jamesgowdy1988@gmail.com"

try:
    import numpy as np
except ImportError as err:
    err.args = err.args[0] + ' - install required (www.scipy.org/install.html)',
    raise err


def line_search_wolfe_bisection(func, grad, x, direction, a=0, b=1000, maxn=10000):

    step = 5.0
    c1 = 0.15 
    c2 = 0.3

    fx = func(x)
    gx = grad(x)
    f = lambda s: func(x + s*direction)
    g = lambda s: grad(x + s*direction)

    for i in xrange(maxn):
        
        leftf = f(step)
        gx_d= np.dot(gx, direction)
        rightf = fx + step*c1*gx_d

        if leftf > rightf:
            # first condition not met
            b = step
            step = 0.5*(a+b)

        elif np.dot(g(step), direction) < c2*gx_d:
            # second condition not met
            a = step
            if b > 100:
                step = 2.0*a
            else:
                step = 0.5*(a+b)

        else:
            # step size meets wolfe conditions
            break


        i += 1
        stop_val = valf + step*direction
        stop_iter = i
        minima = func(stop_val)
    print(val, objectf)
    return stop_val, minima, stop_iter, step, val, objectf

def get_initial_bracket(func, x, direction, nmax=1000):

    f = lambda s: func(x + s*direction)
    f0 = func(x)

    # check initial bracket
    a = -10
    b = +10
    for i in xrange(1000):
        a *= 2
        if f0 < f(a):
            break

    for j in xrange(1000):
        b *= 2
        if f0 < f(b):
            break

    return a, b


def line_search_golden(func, x, direction, a=-1000, b=1000, maxn=10000, tol=1e-5):

    """perform a golden section line search"""

    # 9780387243481-c2.pdf
    # https://en.wikipedia.org/wiki/Golden-section_search#Iterative_algorithm
    # http://www.aip.de/groups/soe/local/numres/bookcpdf/c10-1.pdf
    # http://adl.stanford.edu/aa222/Lecture_Notes_files/AA222-Lecture2.pdf

    a, b = min(a, b), max(a, b)

    f = lambda s: func(x + s*direction)


    h = b - a
    if h <= tol: return (a+b)/2.0

    # required steps to achieve desired tolerance
    n = np.ceil(np.log(tol/h)/np.log(invphi))
    n = int(min(n, maxn))

    # define two midpoints to test
    p = a + h*invphi2 # b - h/phi
    q = a + h*invphi  # b - h/phi**2
    fp = f(p)
    fq = f(q)

    for i in xrange(n-1):
        # h decreases by fixed ratio (equiv to q-a and b-p)
        h = invphi*h 

        if fp < fq:
            # move bracket to left [a--?--p--q]--b
            b = q
            q, fq = p, fp
            p = a + invphi2*h
            fp = f(p)
        else:
            # move bracket to right a--[p--q--?--b]
            a = p
            p, fp = q, fq
            q = a + invphi*h
            fq = f(q)

    return (a+q if fp < fq else p+b)/2.0
