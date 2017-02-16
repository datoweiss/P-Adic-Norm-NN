from fractions import Fraction
from PIL import Image
import matplotlib.pyplot as plt
from scipy import signal
from scipy import misc

def prime_factors(n, b):
    i = 2
    nuples = np.array([])
    factors = np.array([])
    if n == 1:
        return float(1)
    elif n == 0:
        return float(0)
    else:
        while i * i <= n:
            if n % i:
                i += 1
            else:
                n //= i
                factors = np.append(factors,i)
        if n > 1:
            factors = np.append(factors,n)
        for x in np.nditer(factors):
            if x % b == 0:
                nuples = np.append(nuples,[x])
        y = 1/np.prod(nuples)
    
        return float(y)

def pAdicNorm(x,y,base):
    if y == 0:
        return float(0)
    a = abs(x-y)
    q = Fraction('{}'.format(a))
    num = prime_factors(q.numerator,base)
    den = (prime_factors(q.denominator,base))**-1
    return float((num*den))
        
vpAdicNorm = np.vectorize(pAdicNorm,otypes=[np.float32])
