"""Exact checks of the algebraic identities in Section 5 (apsidal angle lemma)
and of the second-order expansion in Section 7. Requires sympy."""
import sympy as sp

sg, d, x = sp.symbols('sigma delta xi', positive=True)
d2 = (1 - 4*sg**3)/(4*sg)
s, t = sg - d, sg + d

def red(e):
    """Reduce a polynomial in delta modulo delta^2 = d2."""
    p = sp.Poly(sp.expand(e), d)
    return sp.simplify(sum(c*d2**(k//2)*d**(k % 2) for (k,), c in p.terms()))

# relation (s+t)(s^2+t^2)=1 is equivalent to s - s^4 = t - t^4
assert sp.simplify(red((s+t)*(s**2+t**2)) - 1) == 0
assert sp.simplify(red(s*t) - (8*sg**3-1)/(4*sg)) == 0
assert sp.simplify(red((1-4*s**3)*(4*t**3-1)) - (1-4*sg**3)*(128*sg**6+1)/(4*sg**3)) == 0
assert sp.simplify(red(s - s**4) - (64*sg**6-1)/(16*sg**2)) == 0
assert sp.simplify(red(t*(1-4*t**3) - s*(1-4*s**3)) + 6*d) == 0

# X^2 = (yD)^2 and Y^2 in the variable xi = 4^(1/3) sigma
y = (64*sg**6-1)/(16*sg**2)
D = 96*sg**4*d/((8*sg**3-1)*(1-4*sg**3)*(128*sg**6+1))
X2 = sp.simplify(red((y*D)**2))
X2xi = 9*x**3*(2*x**3+1)**2/(4*(1-x**3)*(8*x**6+1)**2)
c = sp.Rational(4)**sp.Rational(1, 3)
assert sp.simplify(X2.subs(sg, x/c) - X2xi) == 0
a = sp.Rational(3, 4)/c
assert sp.simplify((y/a).subs(sg, x/c) - (4*x**6-1)/(3*x**2)) == 0
Y2xi = 9*x**4/(9*x**4-(4*x**6-1)**2)
assert sp.factor(9*x**4-(4*x**6-1)**2) == sp.factor((1-x**2)*(2*x**2+1)**2*(4*x**6+3*x**2-1))

M = 1+6*x+7*x**2-2*x**3-4*x**4+18*x**5+36*x**6+16*x**7-16*x**8-8*x**9
den = 4*(1+x)*(2*x**2+1)**2*(1+x+x**2)*(4*x**4-2*x**2+1)**2*(4*x**6+3*x**2-1)
assert sp.simplify(X2xi - Y2xi - 9*x**3*(2*x**3-1)*M/den) == 0
x0 = 2**sp.Rational(-1, 3)
print("denominator at 2^(-1/3):", sp.N(den.subs(x, x0)))
print("M has no root in [0,1]:", sp.Poly(M, x).count_roots(0, 1) == 0)

# second-order coefficient of Proposition 7.1
p, b1 = sp.symbols('p beta1')
coef = p*(1-b1)/16*((p-2)*(1-b1)-2*(1+b1))
S = sp.symbols('S')
assert sp.simplify(coef.subs(b1, 1-sp.Rational(4, 3)*S) - p*S*(p*S-3)/9) == 0
print("all identities verified")
