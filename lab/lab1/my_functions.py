from math import sqrt
from itertools import chain, combinations
import numpy as np
import scipy.stats as st

def effect(x1, x2)->float:
    '''
    Cohen's d size of effect
    :param x1: first array-like series
    :param x2: second aray-like series
    :return: size of effect, float
    '''
    s = np.sqrt(((len(x1) - 1) * np.std(x1, ddof=1)**2 + (len(x2) - 1) * np.std(x2, ddof=1)**2) \
                /(len(x1) + len(x2) - 2))
    out = np.abs((np.mean(x1) - np.mean(x2))/s)
    return out


def effect_descr(effect)->str:
    '''
    as there is description for sizes of effect in wiki let's get it
    :param effect: size of effect
    :return: description for size of effect
    ''' 
    if (effect <= 0.01):
        return 'Very small'
    elif (effect <= 0.2):
        return 'Small'
    elif (effect <= 0.5):
        return 'Medium'
    elif (effect <= 0.8):
        return 'Large'
    elif (effect <= 1.2):
        return 'Huge'
    else:
        return 'Very huge'
    
    
def quadratic_eq(a=0, b=0, c=0)->list:
    '''
    simple function for solving quadratic equasions
    :param a: coefficient for x^2
    :param b: coefficient for x
    :param c: coefficient for 1
    :return: list with 2 roots, if they exist, else None
    '''
    r = b**2 - 4*a*c

    if r > 0:
        num_roots = 2
        x1 = (((-b) + sqrt(r))/(2*a))     
        x2 = (((-b) - sqrt(r))/(2*a))
        return x2, x1
    elif r == 0:
        num_roots = 1
        x = (-b) / 2*a
        return x, x
    else:
        return None
    
    
def get_geom_var(p)->float:
    '''
    variance for geometric distribution
    :param p: the only one parametr for geometric ditribution - probability of success
    :return: variance
    '''
    out = (1 - p) / (p ** 2)
    return out


def powerset(iterable):
    '''
    https://stackoverflow.com/questions/1482308/how-to-get-all-subsets-of-a-set-powerset
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    '''
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


def get_mult_corr_p(x, y):
    R = np.corrcoef(x.T)
    c = np.array([st.pearsonr(feat, y)[0] for feat in x.T])
    if len(c) == 1:
        return c[0]
    out = np.dot(c, np.linalg.inv(R))
    out = np.dot(out, c)
    out = np.sqrt(out)
    return out