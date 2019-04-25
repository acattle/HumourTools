'''
    Created on Jan 24, 2018
    
    @author: Andrew Cattle <acattle@connect.ust.hk>
    
    This module contains miscellaneous utility functions 
'''
from __future__ import division #maintain Python 2.7 compatibility

def mean(l):
    """
        Calculates the arithmetic mean of the elements in list l using built-in
        functions.
        
        Quicker than numpy.mean for short lists since l does not need to be
        converted to a numpy.array first.
        
        :param l: the list of values to be averaged
        :type l: Iterable[float]
        
        :returns: the arithmetic mean of l
        :rtype: float
    """
    
    return sum(l)/max(len(l), 1) #use of max() is to prevent divide by 0 errors if l == []

