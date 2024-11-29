# Exercises for Variational methods

## Lecture 8

Experiment with the Galerkin and collocation methods and approximate the global functions
1. $u(x) = |x|, \quad x \in [-1, 1]$,
2. $u(x) = \exp(\sin(x)), \quad x \in [0, 2]$,
3. $u(x) = x^{10}, \quad x \in [0, 1]$,
4. $u(x) = \exp(-(x-0.5)^2) - \exp(-0.25) \quad x \in [0, 1]$,
5. $u(x) = J_0(x), \quad x \in [0, 100]$,

where $J_0(x)$ is the [Bessel function]() of the first kind. The Bessel function is available both in [Scipy](https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.jv.html#scipy.special.jv) and [Sympy](https://docs.sympy.org/latest/modules/functions/special.html#sympy.functions.special.bessel.besselj).

Experiment using either Legendre polynomials or sines as basis functions for the Galerkin method and Lagrange polynomials for the collocation method. Use both exact and numerical integration for the Galerkin method. Measure the error as a function of $N$ by computing an $L^2$ error norm.

The approximation of the Bessel function with Legendre polynomials is shown below for $N=(20, 40, 60)$. For $N=60$ there is no visible difference from the exact solution.