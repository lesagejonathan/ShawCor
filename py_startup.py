import os
import os.path
if os.path.exists('/Users/jlesage/Dropbox/python'):
	os.chdir('/Users/jlesage/Dropbox/python')
elif os.path.exists('/home/jonathan/Dropbox/python'):
	os.chdir('/home/jonathan/Dropbox/python')
elif os.path.exists('/home/jlesage/Dropbox/python'):
	os.chdir('/home/jlesage/Dropbox/python')

from numpy import *
from numpy.fft import *
from numpy.linalg import *
from numpy import savetxt,loadtxt
import os.path
import sys
from matplotlib.pyplot import *
import scipy as sp
import scipy.special as sf
import scipy.signal as ss
import readline
import rlcompleter
import atexit
import os
histfile = os.path.join(os.path.expanduser("~"), ".pyhist")
try:
    readline.read_history_file(histfile)
except IOError:
    pass
import atexit
atexit.register(readline.write_history_file, histfile)
del os, histfile
