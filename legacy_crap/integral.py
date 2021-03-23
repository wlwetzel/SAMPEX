import numpy as np
import sympy as sym

offset = sym.Symbol('offset')
height = sym.Symbol('height')
center = sym.Symbol('center')
sig = sym.Symbol('sig')
e = 2.7182818284590452353602874713527
def f(pitch):
    return offset + height * e**( (pitch - center)**2 / (sig**2))
