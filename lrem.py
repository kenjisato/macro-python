import numpy as np
import scipy.linalg as LA


def SS(E, F=None, G=None, H=None):
    def f(x0, u):
        x1 = E @ x0 + F @ u
        y1 = G @ x1 + H @ u
        return x1, y1
    f.nx = E.shape[1]  # dim(x)
    f.ny = G.shape[0]  # dim(y)
    f.nu = F.shape[1]  # dim(u)
    return f


def initial(ss, x0, T):
    x0 = np.asarray(x0)
    x = np.empty((T, ss.nx))
    y = np.empty((T, ss.ny))
    u = np.zeros(ss.nu)
    x[0, :], y[0, :] = ss(x0, u)
    for t in range(T - 1):
        x[t + 1, :], y[t + 1, :] = ss(x[t, :], u)
    return x, y


def impulse(ss, e, T):
    x = np.empty((T, ss.nx))
    y = np.empty((T, ss.ny))
    x[0, :], y[0, :] = ss(x0=np.zeros(ss.nx), u=np.asarray(e))
    for t in range(T - 1):
        x[t + 1, :], y[t + 1, :] = ss(x[t, :], u=np.zeros(ss.nu))
    return x, y



def eig(a):
    """A thin wrapper for scipy.linalg.eig to have eigenvalues/vectors sorted"""
    
    E, V = LA.eig(a)
    idx = np.abs(E).argsort()[::]   
    E = E[idx]
    V = V[:, idx]
    
    return E, V


def ordqz(a, b):
    """A wrapper function for scipy.linalg.ordqz
    
    Usage:
    T, S, tii, sii, Q, Z = ordqz(B, A)
    """
    
    def iouc(a, b): 
        return (np.abs(a) / np.abs(b)) <= 1.0    
    
    return LA.ordqz(a, b, sort=iouc)


def make_slices(n1, ns):

    _1s = np.s_[:n1, :ns]
    _2s = np.s_[n1:, :ns]
    _1u = np.s_[:n1, ns:]
    _2u = np.s_[n1:, ns:]
    _ss = np.s_[:ns, :ns]
    _su = np.s_[:ns, ns:]
    _uu = np.s_[ns:, ns:]
    _s = np.s_[:ns]
    _u = np.s_[ns:]
    
    return _1s, _2s, _1u, _2u, _ss, _su, _su, _uu, _s, _u


def solve_lrem(a, b, c, n1):

    T, S, tii, sii, Q, Z = ordqz(b, a)
    ns = sum(abs(tii / sii) <= 1) 
    assert n1 == ns

    _1s, _2s, _1u, _2u, _ss, _su, _su, _uu, _s, _u = make_slices(n1, ns)
    
    E = Z[_2s] @ LA.inv(Z[_1s])
    G = Z[_1s] @ LA.solve(S[_ss], T[_ss]) @ LA.inv(Z[_1s])
    
    return E, G


def solve_lrem_ar(a, b, c, p, n1):
    
    T, S, tii, sii, Q, Z = ordqz(b, a)
    U = Q.T @ c
    
    ns = sum(abs(tii / sii) <= 1) 
    assert n1 == ns
    
    _1s, _2s, _1u, _2u, _ss, _su, _su, _uu, _s, _u = make_slices(n1, ns)
    
    M = LA.solve_sylvester(- LA.solve(T[_uu], S[_uu]), LA.inv(p), 
                           - LA.solve(T[_uu], U[_u]) @ LA.inv(p))

    E = Z[_1s] @ LA.solve(S[_ss], T[_ss]) @ LA.inv(Z[_1s])
    F = (- Z[_1s] @ LA.solve(S[_ss], T[_ss]) @ LA.solve(Z[_1s], Z[_1u]) @ M
         + Z[_1s] @ LA.solve(S[_ss], T[_su] @ M - S[_su] @ M @ p + U[_s])
         + Z[_1u] @ M @ p)
    G = Z[_2s] @ LA.inv(Z[_1s])
    H = (Z[_2u] - Z[_2s] @ LA.solve(Z[_1s], Z[_1u])) @ M
    
    return SS(E, F, G, H)



if __name__ == "__main__":
    pass