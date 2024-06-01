import logging
import os
from setuptools import setup, find_namespace_packages

log_format = "%(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
log = logging.getLogger("Point Net Suite")

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name="point_net_suite",
    version="0.1.0",
    description="Point Net Suite",
    url="https://github.com/jeferal/point_net_suite",
    packages=find_namespace_packages(include=["models.*", "data_utils.*"]),
    long_description=read("README.md"),
)
