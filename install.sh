#!/bin/bash

echo "Install libraries"
sudo apt update
sudo apt install -y gfortran
sudo apt install -y gcc
sudo apt install -y g++
sudo apt install -y build-essential
sudo apt install -y libopenmpi-dev
sudo apt install -y libscalapack-openmpi-dev
sudo apt install -y libscalapack-openmpi2.0
sudo apt install -y libblas-dev 
sudo apt install -y liblapack-dev
sudo apt install -y libopenblas-dev
sudo apt install -y libarpack2-dev
sudo apt install -y libfftw3-dev
sudo apt install -y libxc-dev
sudo apt install -y git
sudo apt install -y wget
sudo apt install -y make
sudo apt install -y cmake
sudo apt install -y python-dev
sudo apt install -y python-distutils
sudo apt install -y python-setuptools
sudo apt install -y python-numpy
sudo apt install -y python-scipy
sudo apt install -y python-f2py
sudo apt install -y python-mpmath
sudo apt install -y python-matplotlib
sudo apt install -y python-sympy
#sudo apt install -y grace
#sudo apt install -y jmol
#sudo apt install -y gnuplot

echo " "
echo "DFTBaby install"
python setup.py install --user

echo " "
echo "Extensions"
cd ~/DFTBaby-0.1.0/DFTB/extensions
make clean
make
cd -

ecbo " "
echo "The rudimentary DREIDING force field for QM/MM calculation"
cd ~/DFTBaby-0.1.0/DFTB/ForceField/src/
make clean
make
cd -

echo "Installation, END"
