'''
Created on Feb 5, 2018

@author: Andrew Cattle <acattle@cse.ust.hk>

Utilities related to logging. Useful because logging.Logger instances can't be
pickled natively.

See https://stackoverflow.com/a/35022654/1369712 for more details
'''

import logging

class LoggerMixin():
    
    @property #using property lets us call this function like a variable https://www.programiz.com/python-programming/property
    def logger(self):
        component = "{}.{}".format(type(self).__module__, type(self).__name__)
        return logging.getLogger(component)
