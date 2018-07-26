"""utils.py - common utility functions shared across the package"""

import functools

__author__ = "James Gowdy" 
__email__ = "jamesgowdy1988@gmail.com"

def edit_error(err, added_string):
    err.args = err.args[0] + " - " + added_string,
    return err

def requires_numpy(fn):

    """The @requires_numpy decorator will import numpy as np to the global scope of 
    the module of the decorated function as needed. Beware of the overhead 
    involved (checking if imported) when calling a function many the times!"""

    @functools.wraps(fn) # preserve fn metadata
    def new_fn(*args, **kwargs):
        try:
            import numpy as np
        except ImportError as err:
            # catch and re-raise the error with instruction to user
            err = edit_error("install needed (www.scipy.org/install.html)")
            raise err
        else:
            # use instead of 'global np', it wouldn't work outside this module
            fn.__globals__["np"] = np 
        return fn(*args, **kwargs)
    return new_fn