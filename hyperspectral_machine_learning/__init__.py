import pkgutil
import os, sys; sys.path.append(os.path.dirname(os.path.realpath(__file__)))

__path__ = pkgutil.extend_path(__path__, __name__)
for importer, modname, ispkg in pkgutil.walk_packages(path=__path__, prefix=__name__+'.'):
    __import__(modname)