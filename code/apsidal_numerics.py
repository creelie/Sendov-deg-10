"""Numerical apsidal angle Psi(k) and period T_1(k) of the extremal orbits (E=1),
and a direct check of the mode-2 expansion. Requires mpmath and numpy."""
import numpy as np
from mpmath import mp, quad, sqrt, polyroots, mpf, pi, cos, sin, re
mp.dps = 25
a = mpf(3)/4*mpf(4)**(-mpf(1)/3)
g = lambda r: r - r**4

def orbit(k):
    rs = sorted(r.real for r in polyroots([-1, 0, 0, 1, -k]) if abs(r.imag) < 1e-18 and r.real > 0)
    lo, hi = rs[0], rs[1]; cc, h = (lo+hi)/2, (hi-lo)/2
    F = lambda f: re(2*quad(lambda th: f(cc-h*cos(th))*h*sin(th)/sqrt(g(cc-h*cos(th))**2-k**2), [0, pi]))
    return F(lambda r: k/r), F(g)

for k in [a*mpf(j)/20 for j in range(1, 20)] + [a*(1-mpf('1e-6'))]:
    P, T = orbit(k)
    print(f"k/a={float(k/a):.6f}  Psi/pi={float(P/pi):.6f}  T1={float(T):.6f}")

# mode-2 deformation: mean |chord|^p over circle value, compared with 1 + pS(pS-3)eps^2/9
N = 1 << 14; th = 2*np.pi*np.arange(N)/N
def ratio(eps, u_idx, p):
    f = np.exp(1j*(th + eps*np.sin(2*th)))
    G = np.concatenate([[0], np.cumsum((f+np.roll(f, -1))/2)])*2*np.pi/N
    i = np.arange(N); idx = i+u_idx
    ch = np.abs(G[idx % N] + (idx//N)*G[N] - G[i])
    u = 2*np.pi*u_idx/N
    return np.mean(ch**p)/(2*np.sin(u/2))**p
for p in [2.5, 3.0, 3.5]:
    for u_idx in [N//2, N//3]:
        u = 2*np.pi*u_idx/N; S = np.sin(u/2)**2; e = 0.02
        print(f"p={p} u={u:.4f}: numeric {(ratio(e,u_idx,p)-1)/e**2:.5f}  formula {p*S*(p*S-3)/9:.5f}")
