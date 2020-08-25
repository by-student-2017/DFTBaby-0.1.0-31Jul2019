#!/bin/bash

echo "Install libraries"
sudo apt update
sudo apt install -y gfortran
sudo apt install -y gcc g++ build-essential
sudo apt install -y libopenmpi-dev
sudo apt install -y libscalapack-openmpi-dev libscalapack-openmpi2.0
sudo apt install -y libblas-dev liblapack-dev libopenblas-dev libarpack2-dev
sudo apt install -y libfftw3-dev
sudo apt install -y libxc-dev
sudo apt install -y git wget make cmake
sudo apt install -y python-dev python-distutils  python-setuptools
sudo apt install -y python-numpy python-scipy python-f2py
sudo apt install -y python-mpmath python-matplotlib
sudo apt install -y python-sympy
sudo apt install -y grace jmol gnuplot

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
