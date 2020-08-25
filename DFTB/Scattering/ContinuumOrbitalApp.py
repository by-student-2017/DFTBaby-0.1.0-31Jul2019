#!/usr/bin/env python
"""
shows DFTB molecular orbitals and dipole prepared continuum orbitals
and computes molecular frame and orientation averaged photoangular distributions (PADs)
"""
from DFTB.Analyse.mayavi.CubeViewerWidget import QCubeViewerWidget, CubeData
from DFTB.LR_TDDFTB import LR_TDDFTB
from DFTB import XYZ, AtomicData
from DFTB.BasisSets import load_pseudo_atoms, AtomicBasisSet
from DFTB.Analyse import Cube
from DFTB.Modeling import MolecularCoords as MolCo
from DFTB.Scattering import slako_tables_scattering
from DFTB.Scattering.SlakoScattering import AtomicScatteringBasisSet, load_slako_scattering, ScatteringDipoleMatrix
from DFTB.Scattering.SlakoScattering import load_dyson_orbitals
from DFTB.Scattering import PAD
import DFTB

from pyface.qt import QtGui, QtCore

from matplotlib.figure import Figure
from matplotlib import patches
from matplotlib.backends.backend_qt4agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar)
from mpl_toolkits.mplot3d import Axes3D, proj3d
from matplotlib.patches import FancyArrowPatch

import numpy as np
import numpy.linalg as la
import copy
import operator

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs
        
    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        FancyArrowPatch.draw(self, renderer)

def busy(func):
    def wrapper(self, *args, **kwds):
        self.statusBar().showMessage("... PLEASE WAIT ..." * 10)
        self.repaint()
        func(self, *args, **kwds)
        self.statusBar().showMessage(self.default_message)
    return wrapper


class FigurePopup(QtGui.QMainWindow):
    def __init__(self, pke, sigma, beta2):
        QtGui.QMainWindow.__init__(self)
        self.pke = pke
        self.sigma = sigma
        self.beta2 = beta2
        self.frame = QtGui.QFrame()
        layout = QtGui.QVBoxLayout(self.frame)
        # tabs
        tabs = QtGui.QTabWidget()
        layout.addWidget(tabs)
        # sigma
        sigmaFrame = QtGui.QFrame()
        sigmaLayout = QtGui.QVBoxLayout(sigmaFrame)
        tabs.addTab(sigmaFrame, "sigma")
        self.sigmaFig = Figure()
        self.sigmaCanvas = FigureCanvas(self.sigmaFig)
        sigmaLayout.addWidget(self.sigmaCanvas)
        self.sigmaCanvas.draw()
        NavigationToolbar(self.sigmaCanvas, sigmaFrame, coordinates=True)

        ax = self.sigmaFig.add_subplot(111)
        ax.set_xlabel("PKE / eV", fontsize=15)
        ax.set_ylabel("$\sigma$", fontsize=15)
        ax.plot(self.pke, self.sigma, "o", markersize=10, color="black")

        # betas
        betaFrame = QtGui.QFrame()
        betaLayout = QtGui.QVBoxLayout(betaFrame)
        tabs.addTab(betaFrame, "beta")
        self.betaFig = Figure()
        self.betaCanvas = FigureCanvas(self.betaFig)
        betaLayout.addWidget(self.betaCanvas)
        self.betaCanvas.draw()
        NavigationToolbar(self.betaCanvas, betaFrame, coordinates=True)

        ax = self.betaFig.add_subplot(111)
        ax.set_xlabel("PKE / eV", fontsize=15)
        ax.set_ylabel("$\\beta_2$", fontsize=15)
        ax.set_ylim((-1.0, 2.0))
        ax.plot(self.pke, self.beta2, "o", markersize=10, color="blue")

        
        self.setCentralWidget(self.frame)
        self.show()

class Settings(dict):
    def getOption(self, group, option):
        v = self[group][option]
        if type(v) == list:
            o = v[1][ v[0] ]
        else:
            o = v
        return o
        
class SettingsPopup(QtGui.QMainWindow):
    def __init__(self, settings):
        QtGui.QMainWindow.__init__(self)
        self.frame = QtGui.QFrame()
        layout = QtGui.QVBoxLayout(self.frame)
        # tabs
        tabs = QtGui.QTabWidget()
        layout.addWidget(tabs)
        self.settings = settings
        self.copied_settings = copy.deepcopy(self.settings)
        self.fields = {}
        # settings
        for title,dic in self.copied_settings.iteritems():
            self.fields[title] = {}
            tabFrame = QtGui.QFrame()
            tabLayout = QtGui.QFormLayout(tabFrame)
            tabs.addTab(tabFrame, "%s" % title)
            for k,v in dic.iteritems():
                if type(v) == int:
                    field = QtGui.QSpinBox()
                    field.setValue(v)
                    field.setMaximum(1000)
                    field.setMinimum(1)
                    field.valueChanged.connect(self.updateSettings)
                elif type(v) == float:
                    field = QtGui.QLineEdit()
                    field.setText(str(v))
                    field.editingFinished.connect(self.updateSettings)
                elif type(v) == list:
                    field = QtGui.QComboBox()
                    activeItem = v[0]
                    l = v[1]
                    assert type(l) == list
                    for item in l:
                        field.addItem(str(item))
                    field.setCurrentIndex(activeItem)
                    field.currentIndexChanged.connect(self.updateSettings)
                else:
                    raise Exception("BUG ??")    
                    
                self.fields[title][k] = field

                tabLayout.addRow(QtGui.QLabel("%s:" % k), field)
        
        # "OK" and "Cancel" buttons
        buttonFrame = QtGui.QFrame()
        layout.addWidget(buttonFrame)
        buttonLayout = QtGui.QHBoxLayout(buttonFrame)
        okButton = QtGui.QPushButton("Apply")
        okButton.clicked.connect(self.ok)
        buttonLayout.addWidget(okButton)
        closeButton = QtGui.QPushButton("Close")
        closeButton.clicked.connect(self.cancel)
        buttonLayout.addWidget(closeButton)
        
        self.setCentralWidget(self.frame)
        self.setWindowTitle("Settings")
        self.show()
    def updateSettings(self):
        print "update settings"
        for title, dic in self.settings.iteritems():
            for k,v in dic.iteritems():
                if type(v) == int:
                    self.copied_settings[title][k] = self.fields[title][k].value()
                elif type(v) == float:
                    self.copied_settings[title][k] = float(str(self.fields[title][k].text()))
                elif type(v) == list:
                    self.copied_settings[title][k][0] = self.fields[title][k].currentIndex()
        
    def cancel(self):
        self.close()
    def ok(self):
        # copied setting back
        for title,dic in self.copied_settings.iteritems():
            self.settings[title] = copy.deepcopy(dic)
        print self.settings
        #self.close()
        
class Main(QtGui.QMainWindow):
    def __init__(self, xyz_file, dyson_file=None):
        super(Main, self).__init__()
        self.settings = Settings(
            {
                "Continuum Orbital":
                {"Ionization transitions": [0, ["only intra-atomic", "inter-atomic"]]
                 },
                "Averaging":
                { "Euler angle grid points": 5,
                  "polar angle grid points": 1000,
                  "sphere radius Rmax": 300.0,
                 },
                "Scan":
                { "nr. points": 20},
                "Cube":
                { "extra space / bohr": 15.0,
                  "points per bohr" : 3.0}
            })
        # perform DFTB calculation
        
        # BOUND ORBITAL = HOMO
        self.atomlist = XYZ.read_xyz(xyz_file)[0]
        # shift molecule to center of mass
        print "shift molecule to center of mass"
        pos = XYZ.atomlist2vector(self.atomlist)
        masses = AtomicData.atomlist2masses(self.atomlist)
        pos_com = MolCo.shift_to_com(pos, masses)
        self.atomlist = XYZ.vector2atomlist(pos_com, self.atomlist)
        
        self.tddftb = LR_TDDFTB(self.atomlist)
        self.tddftb.setGeometry(self.atomlist, charge=0)
        options={"nstates": 1}
        try:
            self.tddftb.getEnergies(**options)
        except DFTB.Solver.ExcitedStatesNotConverged:
            pass

        self.valorbs, radial_val = load_pseudo_atoms(self.atomlist)

        if dyson_file == None:
            # Kohn-Sham orbitals are taken as Dyson orbitals
            self.HOMO, self.LUMO = self.tddftb.dftb2.getFrontierOrbitals()
            self.bound_orbs = self.tddftb.dftb2.getKSCoefficients()
            self.orbe = self.tddftb.dftb2.getKSEnergies()
            orbital_names = []
            norb = len(self.orbe)
            for o in range(0, norb):
                if o < self.HOMO:
                    name = "occup."
                elif o == self.HOMO:
                    name = "HOMO"
                elif o == self.LUMO:
                    name = "LUMO "
                else:
                    name = "virtual"
                name = name + "  " + str(o).rjust(4) + ("   %+10.3f eV" % (self.orbe[o] * 27.211))
                orbital_names.append(name)
            initially_selected = self.HOMO
        else:
            # load coefficients of Dyson orbitals from file
            names, ionization_energies, self.bound_orbs = load_dyson_orbitals(dyson_file)
            self.orbe = np.array(ionization_energies) / 27.211
            orbital_names = []
            norb = len(self.orbe)
            for o in range(0, norb):
                name = names[o] + "  " + str(o).rjust(4) + ("   %4.2f eV" % (self.orbe[o] * 27.211))
                orbital_names.append(name)
            initially_selected = 0
            
        self.photo_kinetic_energy = slako_tables_scattering.energies[0]
        self.epol = np.array([15.0, 0.0, 0.0])
        
        # Build Graphical User Interface
        main = QtGui.QWidget()
        mainLayout = QtGui.QHBoxLayout(main)
        #
        selectionFrame = QtGui.QFrame()
        selectionFrame.setSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Preferred)
        mainLayout.addWidget(selectionFrame)
        selectionLayout = QtGui.QVBoxLayout(selectionFrame)
        # 
        label = QtGui.QLabel(selectionFrame)
        label.setText("Select bound MO:")
        selectionLayout.addWidget(label)
        
        # bound orbitals
        self.orbitalSelection = QtGui.QListWidget(selectionFrame)
        self.orbitalSelection.itemSelectionChanged.connect(self.selectBoundOrbital)
        norb = len(self.orbe)
        self.orbital_dict = {}
        for o in range(0, norb):
            name = orbital_names[o]
            self.orbital_dict[name] = o
            item = QtGui.QListWidgetItem(name, self.orbitalSelection)
            if o == initially_selected:
                selected_orbital_item = item
            selectionLayout.addWidget(self.orbitalSelection)
            
        ### VIEWS
        center = QtGui.QWidget()
        mainLayout.addWidget(center)
        centerLayout = QtGui.QGridLayout(center)
        #
        boundFrame = QtGui.QFrame()
        
        centerLayout.addWidget(boundFrame, 1, 1)
        boundLayout = QtGui.QVBoxLayout(boundFrame)
        # "Bound Orbital"
        label = QtGui.QLabel(boundFrame)
        label.setText("Bound Orbital")
        boundLayout.addWidget(label)
        #
        self.boundOrbitalViewer = QCubeViewerWidget(boundFrame)
        boundLayout.addWidget(self.boundOrbitalViewer)
        
        # continuum orbital
        continuumFrame = QtGui.QFrame()
        centerLayout.addWidget(continuumFrame, 1, 2)
        continuumLayout = QtGui.QVBoxLayout(continuumFrame)
        # "Dipole-Prepared Continuum Orbital"
        label = QtGui.QLabel(continuumFrame)
        label.setText("Dipole-Prepared Continuum Orbital")
        continuumLayout.addWidget(label)

        self.continuumOrbitalViewer = QCubeViewerWidget(continuumFrame)
        continuumLayout.addWidget(self.continuumOrbitalViewer)

        self.efield_objects = []
        self.efield_actors = []
        self.selected = None
        # picker
        self.picker = self.continuumOrbitalViewer.visualization.scene.mayavi_scene.on_mouse_pick(self.picker_callback)
        self.picker.tolerance = 0.01

        # PHOTO KINETIC ENERGY
        sliderFrame = QtGui.QFrame(continuumFrame)
        continuumLayout.addWidget(sliderFrame)
        sliderLayout = QtGui.QHBoxLayout(sliderFrame)
        # label
        self.pke_label = QtGui.QLabel()
        self.pke_label.setText("PKE: %6.4f eV" % (self.photo_kinetic_energy * 27.211))
        sliderLayout.addWidget(self.pke_label)
        # Slider for changing the PKE
        self.pke_slider = QtGui.QSlider(QtCore.Qt.Horizontal)
        self.pke_slider.setMinimum(0)
        self.pke_slider.setMaximum(len(slako_tables_scattering.energies)-1)
        self.pke_slider.setValue(0)
        self.pke_slider.sliderReleased.connect(self.changePKE)
        self.pke_slider.valueChanged.connect(self.searchPKE)
        sliderLayout.addWidget(self.pke_slider)

        #

        # molecular frame photoangular distribution
        mfpadFrame = QtGui.QFrame()
        centerLayout.addWidget(mfpadFrame, 2,1)
        mfpadLayout = QtGui.QVBoxLayout(mfpadFrame)
        mfpadLayout.addWidget(QtGui.QLabel("Molecular Frame PAD"))
        mfpadTabs = QtGui.QTabWidget()
        mfpadLayout.addWidget(mfpadTabs)
        # 2D map
        mfpadFrame2D = QtGui.QFrame()
        mfpadTabs.addTab(mfpadFrame2D, "2D")
        mfpadLayout2D = QtGui.QVBoxLayout(mfpadFrame2D)
        self.MFPADfig2D = Figure()
        self.MFPADCanvas2D = FigureCanvas(self.MFPADfig2D)
        mfpadLayout2D.addWidget(self.MFPADCanvas2D)
        self.MFPADCanvas2D.draw()
        NavigationToolbar(self.MFPADCanvas2D, mfpadFrame2D, coordinates=True)
        # 3D 
        mfpadFrame3D = QtGui.QFrame()
        mfpadTabs.addTab(mfpadFrame3D, "3D")
        mfpadLayout3D = QtGui.QVBoxLayout(mfpadFrame3D)
        self.MFPADfig3D = Figure()
        self.MFPADCanvas3D = FigureCanvas(self.MFPADfig3D)
        mfpadLayout3D.addWidget(self.MFPADCanvas3D)
        self.MFPADCanvas3D.draw()
        NavigationToolbar(self.MFPADCanvas3D, mfpadFrame3D, coordinates=True)

        
        # orientation averaged photoangular distribution
        avgpadFrame = QtGui.QFrame()
        centerLayout.addWidget(avgpadFrame, 2, 2)
        avgpadLayout = QtGui.QVBoxLayout(avgpadFrame)
        self.activate_average = QtGui.QCheckBox("Orientation Averaged PAD")
        self.activate_average.setToolTip("Check this box to start averaging of the molecular frame PADs over all orientations. This can take a while.")
        self.activate_average.setCheckState(QtCore.Qt.Unchecked)
        self.activate_average.stateChanged.connect(self.activateAveragedPAD)
        avgpadLayout.addWidget(self.activate_average)
        
        avgpadTabs = QtGui.QTabWidget()
        avgpadLayout.addWidget(avgpadTabs)
        # 1D map
        avgpadFrame1D = QtGui.QFrame()
        avgpadTabs.addTab(avgpadFrame1D, "1D")
        avgpadLayout1D = QtGui.QVBoxLayout(avgpadFrame1D)
        self.AvgPADfig1D = Figure()
        self.AvgPADCanvas1D = FigureCanvas(self.AvgPADfig1D)
        avgpadLayout1D.addWidget(self.AvgPADCanvas1D)
        self.AvgPADCanvas1D.draw()
        NavigationToolbar(self.AvgPADCanvas1D, avgpadFrame1D, coordinates=True)
        # 2D map
        avgpadFrame2D = QtGui.QFrame()
        avgpadFrame2D.setToolTip("The averaged PAD should have no phi-dependence anymore. A phi-dependence is a sign of incomplete averaging.")
        avgpadTabs.addTab(avgpadFrame2D, "2D")
        avgpadLayout2D = QtGui.QVBoxLayout(avgpadFrame2D)
        self.AvgPADfig2D = Figure()
        self.AvgPADCanvas2D = FigureCanvas(self.AvgPADfig2D)
        avgpadLayout2D.addWidget(self.AvgPADCanvas2D)
        self.AvgPADCanvas2D.draw()
        NavigationToolbar(self.AvgPADCanvas2D, avgpadFrame2D, coordinates=True)
        # Table
        avgpadFrameTable = QtGui.QFrame()
        avgpadTabs.addTab(avgpadFrameTable, "Table")
        avgpadLayoutTable = QtGui.QVBoxLayout(avgpadFrameTable)
        self.avgpadTable = QtGui.QTableWidget(0,6)
        self.avgpadTable.setToolTip("Activate averaging and move the PKE slider above to add a new row with beta values. After collecting betas for different energies you can save the table or plot a curve beta(PKE) for the selected orbital.")
        self.avgpadTable.setHorizontalHeaderLabels(["PKE / eV", "sigma", "beta1", "beta2", "beta3", "beta4"])
        avgpadLayoutTable.addWidget(self.avgpadTable)
        # Buttons
        buttonFrame = QtGui.QFrame()
        avgpadLayoutTable.addWidget(buttonFrame)
        buttonLayout = QtGui.QHBoxLayout(buttonFrame)
        deleteButton = QtGui.QPushButton("Delete")
        deleteButton.setToolTip("clear table")
        deleteButton.clicked.connect(self.deletePADTable)
        buttonLayout.addWidget(deleteButton)
        buttonLayout.addSpacing(3)
        scanButton = QtGui.QPushButton("Scan")
        scanButton.setToolTip("fill table by scanning automatically through all PKE values")
        scanButton.clicked.connect(self.scanPADTable)
        buttonLayout.addWidget(scanButton)
        saveButton = QtGui.QPushButton("Save")
        saveButton.setToolTip("save table as a text file")
        saveButton.clicked.connect(self.savePADTable)
        buttonLayout.addWidget(saveButton)
        plotButton = QtGui.QPushButton("Plot")
        plotButton.setToolTip("plot beta2 column as a function of PKE")
        plotButton.clicked.connect(self.plotPADTable)
        buttonLayout.addWidget(plotButton)
        """
        # DOCKS
        self.setDockOptions(QtGui.QMainWindow.AnimatedDocks | QtGui.QMainWindow.AllowNestedDocks)
        #
        selectionDock = QtGui.QDockWidget(self)
        selectionDock.setWidget(selectionFrame)
        selectionDock.setFeatures(QtGui.QDockWidget.DockWidgetFloatable | QtGui.QDockWidget.DockWidgetMovable)
        self.addDockWidget(QtCore.Qt.DockWidgetArea(1), selectionDock)
        #
        boundDock = QtGui.QDockWidget(self)
        boundDock.setWidget(boundFrame)
        boundDock.setFeatures(QtGui.QDockWidget.DockWidgetFloatable | QtGui.QDockWidget.DockWidgetMovable)
        boundDock.setSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        self.addDockWidget(QtCore.Qt.DockWidgetArea(2), boundDock)
        # 
        continuumDock = QtGui.QDockWidget(self)
        continuumDock.setWidget(continuumFrame)
        continuumDock.setFeatures(QtGui.QDockWidget.DockWidgetFloatable | QtGui.QDockWidget.DockWidgetMovable)
        continuumDock.setSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Preferred)
        self.addDockWidget(QtCore.Qt.DockWidgetArea(2), continuumDock)
        """
        self.setCentralWidget(main)

        self.status_bar = QtGui.QStatusBar(main)
        self.setStatusBar(self.status_bar)
        self.default_message = "Click on the tip of the green arrow in the top right figure to change the orientation of the E-field"
        self.statusBar().showMessage(self.default_message)

        # Menu bar
        menubar = self.menuBar()
        exitAction = QtGui.QAction('&Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit program')
        exitAction.triggered.connect(exit)
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(exitAction)
        
        settingsMenu = menubar.addMenu('&Edit')
        settingsAction = QtGui.QAction('&Settings...', self)
        settingsAction.setStatusTip('Edit settings')
        settingsAction.triggered.connect(self.editSettings)
        settingsMenu.addAction(settingsAction)
        
        self.loadContinuum()
        # select HOMO
        selected_orbital_item.setSelected(True)
    def loadContinuum(self):
        E = self.photo_kinetic_energy
        k = np.sqrt(2*E)
        wavelength = 2.0 * np.pi/k
        # determine the radius of the sphere where the angular distribution is calculated. It should be
        # much larger than the extent of the molecule
        (xmin,xmax),(ymin,ymax),(zmin,zmax) = Cube.get_bbox(self.atomlist, dbuff=0.0)
        dx,dy,dz = xmax-xmin,ymax-ymin,zmax-zmin
        Rmax0 = self.settings.getOption("Averaging", "sphere radius Rmax")
        Rmax = max([dx,dy,dz]) + Rmax0
        Npts = max(int(Rmax),1) * 50
        print "Radius of sphere around molecule, Rmax = %s bohr" % Rmax
        print "Points on radial grid, Npts = %d" % Npts

        self.bs_free = AtomicScatteringBasisSet(self.atomlist, E, rmin=0.0, rmax=Rmax+2*wavelength, Npts=Npts) 
        self.SKT_bf, SKT_ff = load_slako_scattering(self.atomlist, E)
        if self.settings.getOption("Continuum Orbital", "Ionization transitions") == "inter-atomic":
            inter_atomic = True
        else:
            inter_atomic = False
        print "inter-atomic transitions: %s" % inter_atomic
        self.Dipole = ScatteringDipoleMatrix(self.atomlist, self.valorbs, self.SKT_bf,
                                             inter_atomic=inter_atomic).real
        #
        if self.activate_average.isChecked():
            print "ORIENTATION AVERAGING"
            npts_euler = self.settings.getOption("Averaging", "Euler angle grid points")
            npts_theta = self.settings.getOption("Averaging", "polar angle grid points")
            self.orientation_averaging = PAD.OrientationAveraging_small_memory(self.Dipole, self.bs_free, Rmax, E, npts_euler=npts_euler, npts_theta=npts_theta)
        else:
            print "NO AVERAGING"
    def searchPKE(self):
        self.photo_kinetic_energy = slako_tables_scattering.energies[self.pke_slider.value()]
        self.pke_label.setText("PKE: %6.4f eV" % (self.photo_kinetic_energy * 27.211))
        #self.pke_label.update()
    @busy
    def changePKE(self):
        self.photo_kinetic_energy = slako_tables_scattering.energies[self.pke_slider.value()]
        self.pke_label.setText("PKE: %6.4f eV" % (self.photo_kinetic_energy * 27.211))
        self.loadContinuum()
        self.plotContinuumOrbital()
        self.plotMFPAD()
        self.plotAveragedPAD()
    @busy
    def selectBoundOrbital(self):
        self.plotBoundOrbital()
        self.plotContinuumOrbital()
        self.plotMFPAD()
        self.deletePADTable()
        self.plotAveragedPAD()
    def plotBoundOrbital(self):
        selected = self.orbitalSelection.selectedItems()
        assert len(selected) == 1
        selected_orbital = self.orbital_dict[str(selected[0].text())]

        self.mo_bound = self.bound_orbs[:,selected_orbital]
        # shift geometry so that the expectation value of the dipole operator vanishes
        # dipole matrix
        dipole = np.tensordot(self.mo_bound, np.tensordot(self.tddftb.dftb2.D, self.mo_bound, axes=(1,0)), axes=(0,0))
        print "expectation value of dipole: %s" % dipole
        # shift molecule R -> R - dipole
        self.atomlist = MolCo.transform_molecule(self.tddftb.dftb2.getGeometry(), (0,0,0), -dipole)
        # load bound basis functions
        self.bs_bound = AtomicBasisSet(self.atomlist)
        # plot selected orbital
        (xmin,xmax),(ymin,ymax),(zmin,zmax) = Cube.get_bbox(self.atomlist, dbuff=5.0)
        dx,dy,dz = xmax-xmin,ymax-ymin,zmax-zmin
        ppb = 3.0 # Points per bohr
        nx,ny,nz = int(dx*ppb),int(dy*ppb),int(dz*ppb)
        x,y,z = np.mgrid[xmin:xmax:nx*1j, ymin:ymax:ny*1j, zmin:zmax:nz*1j]
        grid = (x,y,z)
        amplitude_bound = Cube.orbital_amplitude(grid, self.bs_bound.bfs, self.mo_bound, cache=False)

        bound_cube = CubeData()
        bound_cube.data = amplitude_bound.real
        bound_cube.grid = grid
        bound_cube.atomlist = self.atomlist

        self.boundOrbitalViewer.setCubes([bound_cube])

    def plotContinuumOrbital(self):
        dbuff = self.settings["Cube"]["extra space / bohr"]
        ppb = self.settings["Cube"]["points per bohr"]  
        (xmin,xmax),(ymin,ymax),(zmin,zmax) = Cube.get_bbox(self.atomlist, dbuff=dbuff)
        dx,dy,dz = xmax-xmin,ymax-ymin,zmax-zmin
        nx,ny,nz = int(dx*ppb),int(dy*ppb),int(dz*ppb)
        x,y,z = np.mgrid[xmin:xmax:nx*1j, ymin:ymax:ny*1j, zmin:zmax:nz*1j]
        grid = (x,y,z)
        
        # plot continuum orbital
        # projection of dipoles onto polarization direction
        Dipole_projected = np.zeros((self.Dipole.shape[0], self.Dipole.shape[1]))
        # normalize polarization
        epol_unit = self.epol / np.sqrt(np.dot(self.epol, self.epol))
        print "Polarization direction of E-field:"
        print "E (normalized) = %s" % epol_unit
        for xyz in [0,1,2]:
            Dipole_projected += self.Dipole[:,:,xyz] * epol_unit[xyz]
        #print "Dipole projected"
        #print Dipole_projected
        # unnormalized coefficients of dipole-prepared continuum orbitals
        self.mo_free = np.dot(self.mo_bound, Dipole_projected)
        nrm2 = np.dot(self.mo_free.conjugate(), self.mo_free)
        #print "nrm2 = %s" % nrm2
        # normalized coefficients
        self.mo_free /= np.sqrt(nrm2)

        amplitude_continuum = Cube.orbital_amplitude(grid, self.bs_free.bfs, self.mo_free, cache=False)

        continuum_cube = CubeData()
        continuum_cube.data = amplitude_continuum.real
        continuum_cube.grid = grid
        continuum_cube.atomlist = self.atomlist

        self.continuumOrbitalViewer.setCubes([continuum_cube])

        # plot E-field
        for o in self.efield_objects:
            o.remove()
        mlab = self.continuumOrbitalViewer.visualization.scene.mlab
#        self.efield_arrow = mlab.quiver3d(0,0,0, self.epol[0], self.epol[1], self.epol[2],
        self.efield_arrow = mlab.quiver3d(0,0,0, float(self.epol[0]),float(self.epol[1]),float(self.epol[2]),
                                          color=(0.0,1.0,0.0), scale_factor=1.0, mode='arrow', resolution=20,
                                          figure=self.continuumOrbitalViewer.visualization.scene.mayavi_scene)
        self.efield_text = mlab.text(self.epol[0], self.epol[1],"E-field",z=self.epol[2],
                                     figure=self.continuumOrbitalViewer.visualization.scene.mayavi_scene)
        self.efield_text.actor.set(text_scale_mode='none', width=0.05, height=0.1)
        self.efield_text.property.set(justification='centered', vertical_justification='centered')

        self.efield_head = mlab.points3d([self.epol[0]], [self.epol[1]], [self.epol[2]],
                                         scale_factor=0.5, mode='cube', resolution=20, color=(0.0,1.0,0.0),
                                         figure=self.continuumOrbitalViewer.visualization.scene.mayavi_scene)
        self.efield_head.glyph.glyph_source.glyph_source.center = [0,0,0]
        self.efield_outline = mlab.outline(line_width=3,
                                    figure=self.continuumOrbitalViewer.visualization.scene.mayavi_scene)
        self.efield_outline.outline_mode = 'cornered'
        w = 0.1
        self.efield_outline.bounds = (self.epol[0]-w, self.epol[0]+w,
                                      self.epol[1]-w, self.epol[1]+w,
                                      self.epol[2]-w, self.epol[2]+w)
        
        self.efield_objects = [self.efield_arrow, self.efield_text, self.efield_head, self.efield_outline]
        self.efield_actors = [self.efield_head.actor.actors]
    def picker_callback(self, picker):
        for actors in self.efield_actors:
            if picker.actor in actors:
                mlab = self.continuumOrbitalViewer.visualization.scene.mlab
                self.selected = "arrow"
                w = 1.0
                self.efield_outline.bounds = (self.epol[0]-w, self.epol[0]+w,
                                              self.epol[1]-w, self.epol[1]+w,
                                              self.epol[2]-w, self.epol[2]+w)
                break
        else:
            #
            if self.selected != None:
                w = 0.1
                self.efield_outline.bounds = (self.epol[0]-w, self.epol[0]+w,
                                              self.epol[1]-w, self.epol[1]+w,
                                              self.epol[2]-w, self.epol[2]+w)

                self.epol = np.array(picker.pick_position)
                self.plotContinuumOrbital()
                self.plotMFPAD()
                self.selected = None
                
    def plotMFPAD(self):
        Rmax = 80.0
        npts = 30
        
        E = self.photo_kinetic_energy
        k = np.sqrt(2*E)
        wavelength = 2.0 * np.pi/k

        # spherical grid
        rs, thetas, phis = np.mgrid[Rmax:(Rmax+wavelength):30j, 0.0:np.pi:npts*1j, 0.0:2*np.pi:npts*1j]
        # transformed into cartesian coordinates
        xs = rs * np.sin(thetas) * np.cos(phis)
        ys = rs * np.sin(thetas) * np.sin(phis)
        zs = rs * np.cos(thetas)

        grid = (xs,ys,zs)
        amplitude_continuum = Cube.orbital_amplitude(grid, self.bs_free.bfs, self.mo_free, cache=False)
        # integrate continuum orbital along radial-direction for 1 wavelength
        wfn2 = abs(amplitude_continuum)**2
        # 
        dr = wavelength / 30.0
        wfn2_angular = np.sum(wfn2 * dr, axis=0)

        # SPHERICAL PLOT
        self.MFPADfig3D.clf()

        xs = wfn2_angular * np.sin(thetas[0,:,:]) * np.cos(phis[0,:,:])
        ys = wfn2_angular * np.sin(thetas[0,:,:]) * np.sin(phis[0,:,:])
        zs = wfn2_angular * np.cos(thetas[0,:,:])

        ax = self.MFPADfig3D.add_subplot(111, projection='3d')

        rmax = wfn2_angular.max()*1.5
        ax.set_xlim((-rmax,rmax))
        ax.set_ylim((-rmax,rmax))
        ax.set_zlim((-rmax,rmax))

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        # draw efield
        arrow_vec = wfn2_angular.max()*1.2 * self.epol / la.norm(self.epol)
        arrow = Arrow3D([0.0, arrow_vec[0]], [0.0, arrow_vec[1]], [0.0, arrow_vec[2]],
                        color=(0.0, 1.0, 0.0), mutation_scale=20, lw=2, arrowstyle="-|>")
        ax.add_artist(arrow)
        ax.text(arrow_vec[0], arrow_vec[1], arrow_vec[2], "E-field", color=(0.0, 1.0, 0.0))
        
        ax.plot_surface(xs, ys, zs, rstride=1, cstride=1)
        ax.scatter(xs, ys, zs, color="k", s=20)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.w_zaxis.set_ticks([])
        
        self.MFPADCanvas3D.draw()

        # 2D PLOT
        self.MFPADfig2D.clf()

        ax = self.MFPADfig2D.add_subplot(111)

        image = ax.imshow(np.fliplr(wfn2_angular.transpose()),
                          extent=[0.0, np.pi, 0.0, 2*np.pi], aspect=0.5, origin='lower')
        ax.set_xlim((0.0, np.pi))
        ax.set_ylim((0.0, 2*np.pi))
        ax.set_xlabel("$\\theta$")
        ax.set_ylabel("$\phi$")
        self.MFPADfig2D.colorbar(image)
        
        # show piercing points of E-field vector
        # ... coming out of the plane
        r = la.norm(self.epol)
        th_efield = np.arccos(self.epol[2]/r) 
        phi_efield = np.arctan2(self.epol[1],self.epol[0]) + np.pi
        print "th, phi = %s %s" % (th_efield, phi_efield)
        ax.plot([th_efield],[phi_efield], "o", markersize=10, color=(0.0,1.0, 0.0))
        ax.text(th_efield, phi_efield, "E-field", color=(0.0,1.0,0.0), ha="center", va="top")
        # ... going into the plane
        th_efield = np.arccos(-self.epol[2]/r) 
        phi_efield = np.arctan2(-self.epol[1],-self.epol[0]) + np.pi
        print "- th, phi = %s %s" % (th_efield, phi_efield)
        ax.plot([th_efield],[phi_efield], "x", markersize=10, color=(0.0,1.0, 0.0))
        
        self.MFPADCanvas2D.draw()
    @busy
    def activateAveragedPAD(self,s):
        print "s = %s" % s
        self.loadContinuum()
        self.plotAveragedPAD()
    @busy
    def plotAveragedPAD(self):
        if self.activate_average.isChecked() == False:
            for fig in [self.AvgPADfig1D, self.AvgPADfig2D]:
                fig.clf()
                ax = fig.add_subplot(111)
                ax.set_xlim((0.0, 1.0))
                ax.set_ylim((0.0, 1.0))
                ax.text(0.2, 0.6, "click checkbox to", fontsize=20)
                ax.text(0.2, 0.3, "activate averaging", fontsize=20)
                #self.AvgPADfig1D.draw()
        else:
            pad,betas = self.orientation_averaging.averaged_pad(self.mo_bound)

            # 1D plot
            # cut along any phi
            self.AvgPADfig1D.clf()
            ax1d = self.AvgPADfig1D.add_subplot(111, projection='polar')
            nx,ny = pad.shape
            thetas = np.linspace(0.0, np.pi, nx)
            for i in range(0, ny):
                ax1d.plot(thetas, pad[:,i], color="black")
                ax1d.plot(thetas+np.pi, pad[::-1,i], color="black")
            # plot PAD = sigma/4pi * (1 + beta2 * P2(cos(theta))) in read
            pad2 = betas[0]/(4.0*np.pi) * (1.0 + betas[2] * 0.5 * (3*np.cos(thetas)**2 - 1.0))
            print "max(pad) / max(pad2) = %s" % (pad.max() / pad2.max())
            ax1d.plot(thetas, pad2, color="red", ls="-.")
            ax1d.plot(thetas+np.pi, pad2[::-1], color="red", ls="-.")
            
            self.AvgPADCanvas1D.draw()
            # 2D plot
            self.AvgPADfig2D.clf()
            ax2d = self.AvgPADfig2D.add_subplot(111)
            image = ax2d.imshow(np.fliplr(pad.transpose()),
                                extent=[0.0, np.pi, 0.0, 2*np.pi], aspect=0.5, origin='lower')
            ax2d.set_xlim((0.0, np.pi))
            ax2d.set_ylim((0.0, 2*np.pi))
            ax2d.set_xlabel("$\\theta$")
            ax2d.set_ylabel("$\phi$")
            self.AvgPADfig2D.colorbar(image)
            self.AvgPADCanvas2D.draw()
            # Table
            n = self.avgpadTable.rowCount()
            self.avgpadTable.insertRow(n)
            self.avgpadTable.setItem(n,0,QtGui.QTableWidgetItem("%6.4f" % (self.photo_kinetic_energy * 27.211)))
            for i,b in enumerate(betas):
                if i == 0:
                    # sigma
                    self.avgpadTable.setItem(n,i+1, QtGui.QTableWidgetItem("%e" % betas[i]))
                else:
                    self.avgpadTable.setItem(n,i+1, QtGui.QTableWidgetItem("%6.4f" % betas[i]))
    def deletePADTable(self):
        n = self.avgpadTable.rowCount()
        for i in range(0, n):
            self.avgpadTable.removeRow(0)
    def scanPADTable(self):
        if self.activate_average.isChecked() == False:
            self.statusBar().showMessage("You have to activate averaging first before performing a scan!")
            return
        self.deletePADTable()
        # scan through all PKE's and fill table
        print "SCAN"
        nskip = len(slako_tables_scattering.energies)/int(self.settings.getOption("Scan", "nr. points"))
        nskip = max(1, nskip)
        print "nskip = %s" % nskip
        for i,pke in enumerate(slako_tables_scattering.energies):
            if i % nskip != 0:
                continue
            print "*** PKE = %s Hartree ***" % pke
            self.pke_slider.setValue(i)
            self.changePKE()
    def getPADTable(self):
        m, n = self.avgpadTable.rowCount(), self.avgpadTable.columnCount()
        pad_data = np.zeros((m,n))
        for i in range(0, m):
            for j in range(0, n):
                item = self.avgpadTable.item(i,j)
                if item is not None:
                    pad_data[i,j] = float(item.text())
        return pad_data
    def savePADTable(self):
        data_file = QtGui.QFileDialog.getSaveFileName(self, 'Save Table', '', '*')[0]
        pad_data = self.getPADTable()
        if str(data_file) != "":
            fh = open(data_file, "w")
            print>>fh, "# PKE/eV     sigma    beta1   beta2   beta3   beta4"
            np.savetxt(fh, pad_data)
            fh.close()
            print "Wrote table with betas to %s" % data_file
    def plotPADTable(self):
        pad_data = self.getPADTable()
        self.plot_window = FigurePopup(pad_data[:,0], pad_data[:,1], pad_data[:,3])
    def editSettings(self):
        self.settings_window = SettingsPopup(self.settings)
        
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print "Usage: %s <xyz-file> [<file with Dyson orbitals>]" % sys.argv[0]
        print "  Perform DFTB calculation and show Kohn-Sham orbitals"
        print "  If you specify the optional file with Dyson orbitals, those"
        print "  orbitals will replace the tight-binding Kohn-Sham orbitals."
        exit(-1)
    xyz_file = sys.argv[1]
    if len(sys.argv) > 2:
        dyson_file = sys.argv[2]
    else:
        dyson_file = None
    app = QtGui.QApplication.instance()
    window = Main(xyz_file, dyson_file)
    window.show()
    app.exec_()
    
    
