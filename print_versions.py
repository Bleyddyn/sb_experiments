import os
import sys

def pythonVersionString():
    """Current system python version as string major.minor.micro [(alpha|beta|etc)]"""
    vstring = "{0}.{1}.{2}".format(sys.version_info.major, sys.version_info.minor, sys.version_info.micro)
    if sys.version_info.releaselevel != "final":
        vstring += " ({})".format( sys.version_info.releaselevel )
    if sys.version_info.serial != 0:
        vstring += " (serial: {})".format( sys.version_info.serial )
    return vstring

def printVersion(module):
    if str(module) == "Python":
        print( "{}: {}".format( "Python", pythonVersionString() ) )
    else:
        print( "{}: {}".format( module.__name__, module.__version__) )

def printVersions( modules ):
    for module in modules:
        printVersion(module)

if __name__ == "__main__":        
    import tensorflow as tf
    import numpy as np
    import gym
    import stable_baselines
    printVersions( ["Python", tf, np, gym, stable_baselines] )
