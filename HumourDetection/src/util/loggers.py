'''
Created on Feb 5, 2018

@author: Andrew Cattle <acattle@cse.ust.hk>

Utilities related to logging. Useful because logging.Logger instances can't be
pickled natively.

See https://stackoverflow.com/a/35022654/1369712 for more details
'''

import logging

# https://docs.python.org/3/howto/logging-cookbook.html
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
# tell the handler to use this format
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
console.setFormatter(formatter)
# add the handler to the root logger
logging.getLogger().addHandler(console)

class LoggerMixin():
    
    @property #using property lets us call this function like a variable https://www.programiz.com/python-programming/property
    def logger(self):
        component = "{}.{}".format(type(self).__module__, type(self).__name__)
        logger = logging.getLogger(component)
#         if not len(logger.handlers):
#             #if we haven't already set up a handler, add one that prints to console
#             ch = logging.StreamHandler()
#             logger.addHandler(ch)
        return logger
