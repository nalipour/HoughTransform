from setuptools import setup, find_packages
import glob

setup(name='HoughTransform',
      version='0.1.0',
      description='HEP - Track Reco with Hough Transforms ',
      author='Niloufar Tehrani',
      author_email='niloufar.alipour.tehrani@cern.ch',
      url='https://github.com/HEP-FCC/HoughTransform',
      requires=['numpy', 'pandas', 'scipy', 'sklearn'], 
      packages=find_packages(),
      package_dir={"HoughTransfrom": "../HoughTransform"},
      scripts=glob.glob('scripts/*')
)
