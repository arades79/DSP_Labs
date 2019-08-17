"""
Skyelar Craver
DSPs Lab 2
Summer 2019
"""

#%% imports
from math import sqrt, e
import numpy as np
from time import perf_counter
import matplotlib.pyplot as plt
from functools import lru_cache
#%% functions
def recursive_fib(n):
    if n < 2: 
        return n
    else: 
        return recursive_fib(n-1) + recursive_fib(n-2)

def equation_fib(n):
    r5 = sqrt(5)
    c1 = 1 / r5
    c2 = (1 + r5) / 2
    c3 = (1 - r5) / 2
    return c1 * (c2**n - c3**n)

@lru_cache()
def cached_fib(n):
    if n < 2:
        return n
    else:
        return cached_fib(n - 1) + cached_fib (n - 2)

def timer(func, n):
    start = perf_counter()
    func(n)
    return (perf_counter() - start)


#%% run tests
nums = range(1, 20)
recursive_times = [timer(recursive_fib, n) for n in nums]
equation_times = [timer(equation_fib, n) for n in nums]
cached_times = [timer(cached_fib, n) for n in nums]

plt.style.use('seaborn')
f, ax = plt.subplots()
ax.plot(nums, recursive_times)
ax.plot(nums, equation_times)
ax.plot(nums, cached_times)
ax.set_xlabel('n')
ax.set_ylabel('time')
plt.savefig("timing.png")
plt.close(f)


#%% convolution
n = np.arange(start=0, stop=(1e-5), step=1e-8)
ht = [(1e6 * e ** (-1e6 * t)) for t in n]
step = [(1 if t >= 0 else 0) for t in n]
conv =  np.convolve(ht, step)
plt.plot(conv)
plt.savefig("convolution.png")
plt.close()