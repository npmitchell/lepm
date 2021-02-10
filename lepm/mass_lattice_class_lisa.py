import numpy as np
import pickle
import lepm.plotting.plotting as leplt
import matplotlib.pyplot as plt
import lepm.lattice_elasticity as le

'''file that contains the Gyro_Lattice class'''


class mass_lattice:
    def __init__(self, R, Ni, Nk, val_array, simulation_type, PVx=None, PVy=None, PVxydict=None):
        """Initializes the class
        Parameters
        R : matrix of dimension nx3
            Equilibrium positions of all the gyroscopes
        ngyro : int
            Number of gyroscopes
        Ni : matrix of dimension n x (max number of neighbors)
            Each row corresponds to a gyroscope.  The entries tell the numbers of the neighboring gyroscopes
        Nk : matrix of dimension n x (max number of neighbors)
            Correponds to Ni matrix.  1 corresponds to a true connection while 0 signifies that there is not a connection
        val_array :
            contains [spring, pin, other things?] Currently just holds spring and pinning values, but you could add
            more things here, which would fall under self.labels

        Class members
        ----------
        R : matrix of dimension nx3
            Equilibrium positions of all the gyroscopes
        ngyro : int
            Number of gyroscopes
        Ni : matrix of dimension n x (max number of neighbors)
            Each row corresponds to a gyroscope.  The entries tell the numbers of the neighboring gyroscopes
        Nk : matrix of dimension n x (max number of neighbors)
            Correponds to Ni matrix.  1 corresponds to a true connection while 0 signifies that there is not a connection
        spring : float
            spring constant
        pin : float
            gravitational spring constant
        matrix : matrix of dimension 2n x 2n
            Linearized matrix for finding normal modes of system
        eigval : array of dimension 2nx1
            Eigenvalues of self.matrix
        eigvect : array of dimension 2n x 2n
            Eigenvectors of self.matrix
        current_pos :matrix of dimensions nx3 (n = number of gyros)
            Current positions of the gyroscopes
        """
        
        self.R = R  # equilibrium R positions
        self.Ni = Ni 
        self.Nk = Nk
        self.PVx = PVx
        self.PVy = PVy
        self.PVxydict = PVxydict
        self.spring = val_array[0]
        self.pin = val_array[1]
        self.labels = val_array[2:]
        self.ngyro = len(R)
        self.sim_type = simulation_type
        
        # things that have to be calculated
        self.matrix = None
        self.eigval = None
        self.eigvect = None
        
        # #will be updated in the time domain simulation
        # if shape(initial_pos) !=  shape(R):
        #     self.current_pos = R
        # else:
        #     self.current_pos = initial_pos #starts the simulations with the inital pos of the gyros, Will be updated

    def calc_matrix(self):
        """calculates the matrix for finding the normal modes of the system"""
        
        NP, NN = self.Ni.shape
        M1 = np.zeros((2*NP, 2*NP))
        M2 = np.zeros((2*NP, 2*NP))

        m2_shape = M2.shape

        # Unpack periodic boundary vectors
        if self.PVx and self.PVy:
            PVx = self.PVx
            PVy = self.PVy
        elif self.PVxydict:
            PVx, PVy = le.PVxydict2PVxPVy(self.PVxydict, NL)
        else:
            PVx = np.zeros((NP, NN), dtype=float)
            PVy = np.zeros((NP, NN), dtype=float)

        for i in range(NP):
            for nn in range(NN):
                ni = self.Ni[i, nn]
                k = np.abs(self.Nk[i, nn])  # true connection?

                diffx = self.R[ni, 0] - self.R[i, 0] + PVx[i, nn]
                diffy = self.R[ni, 1] - self.R[i, 1] + PVy[i, nn]

                # This is Lisa's original version
                # rij_mag = np.sqrt(diffx**2+diffy**2)
                # if k!=0:
                #     alphaij = np.arccos( diffx /rij_mag)
                # else: alphaij=0
                # if diffy<0 :
                #    alphaij=2*np.pi-alphaij
                # if self.Nk[i,nn] < 0  : alphaij = (np.pi + alphaij)%(2*np.pi)

                # This is my version (05-28-16)
                if abs(k) > 0:
                    alphaij = np.arctan2(diffy, diffx)
     
                Cos = np.cos(alphaij)
                Sin = np.sin(alphaij)
     
                if abs(Cos) < 10E-3:
                    Cos = 0
                else:
                    Cos = Cos
     
                if abs(Sin) < 10E-3:
                    Sin = 0
                else:
                    Sin = Sin
     
                Cos2 = Cos**2
                Sin2 = Sin**2
                CosSin = Cos*Sin
     
                # Real equations (x components)
                M1[2*i, 2*i] += k*Cos2
                M1[2*i, 2*i+1] += k*CosSin
                M1[2*i, 2*ni] += -k*Cos2
                M1[2*i, 2*ni+1] += -k*CosSin
     
                # Imaginary equations (y components)
                M1[2*i+1, 2*i] += k*CosSin
                M1[2*i+1, 2*i+1] += k*Sin2
                M1[2*i+1, 2*ni] += -k*CosSin
                M1[2*i+1, 2*ni+1] += -k*Sin2

                # pinning
                M2[2*i, 2*i] = 1
                M2[2*i+1, 2*i+1] = 1  
    
        self.matrix = (-float(self.spring) * M1 - float(self.pin) * M2)
        return self.matrix
        
    def eig_vals_vects(self):
        """finds the eigenvalues and eigenvectors of self.matrix"""
        eigval, eigvect = np.linalg.eig(self.matrix)
        eigval = -eigval
        si = np.argsort(np.abs(np.real(eigval)))
        eigvect = np.array(eigvect)
        eigvect = eigvect.T[si]
        eigval = np.sqrt(eigval[si])
        return eigval, eigvect

    def save_eigval_eigvect(self, infodir):
        """"""
        if len(np.nonzero(self.eigvect)) == 0:
            self.eig_vals_vects()

        if self.matrix is None:
            self.calc_matrix()
            self.eigval, self.eigvect = self.eig_vals_vects()
        elif self.eigval is None:
            self.eigval, self.eigvect = self.eig_vals_vects()

        output = open(infodir + 'eigval_mass.pkl', 'wb')
        pickle.dump(self.eigval, output)
        output.close()
        output = open(infodir + 'eigvect_mass.pkl', 'wb')
        pickle.dump(self.eigvect, output)
        output.close()

        # Plot histogram of eigenvalues
        self.plot_eigval_hist(infodir)

    def plot_eigval_hist(self, infodir=None, show=False):
        print 'mass: self.eigval = ', self.eigval
        fig, DOS_ax = leplt.initialize_DOS_plot(self.eigval, 'mass', pin=-5000)
        if infodir is not None:
            infodir = le.prepdir(infodir)
            plt.savefig(infodir + 'eigval_mass_hist.png')
        if show:
            plt.show()
        plt.clf()

    def save_DOSmovie(self, infodir=None, attribute=True, save_DOS_if_missing=True):
        pass
        print 'save_DOSmovie for mass_lattice_class IS UNFINISHED. BELOW is code from gyro_lattice_class.py'
        # if infodir is None:
        #     infodir = self.lattice.lp['meshfn'] + '/'
        #
        # # Obtain eigval and eigvect, and matrix if necessary
        # if self.eigval is None or self.eigvect is None:
        #     # check if we can load the DOS info
        #     if glob.glob(infodir + 'eigval.pkl') and glob.glob(infodir + 'eigvect.pkl'):
        #         print "Loading eigval and eigvect from " + self.lattice.lp['meshfn']
        #         eigval = pickle.load(open(infodir + "eigval.pkl", "rb"))
        #         eigvect = pickle.load(open(infodir + "eigvect.pkl", "rb"))
        #     else:
        #         if self.matrix is None:
        #             matrix = self.calc_matrix(attribute=attribute)
        #             eigval, eigvect = self.eig_vals_vects(matrix=matrix, attribute=attribute)
        #         else:
        #             eigval, eigvect = self.eig_vals_vects(attribute=attribute)
        #
        #         if save_DOS_if_missing:
        #             output = open(infodir + 'eigval.pkl', 'wb')
        #             pickle.dump(self.eigval, output)
        #             output.close()
        #
        #             output = open(infodir + 'eigvect.pkl', 'wb')
        #             pickle.dump(self.eigvect, output)
        #             output.close()
        #
        #             print 'Saved gyro DOS to ' + infodir + 'eigvect(val).pkl\n'
        #
        # if not glob.glob(infodir + 'eigval_gyro_hist.png'):
        #     fig, DOS_ax = leplt.initialize_DOS_plot(self.eigval, 'gyro', pin=- 5000)
        #     plt.savefig(infodir + 'eigval_gyro_hist.png')
        #     plt.clf()
        #
        # le.plot_movie_normal_modes_Nashgyro(infodir, self.lattice.xy, self.lattice.NL, self.lattice.KL, self.OmK,
        #                                     self.Omg, params={}, dispersion=[], sim_type='gyro',
        #                                     rm_images=True, gapims_only=False, save_into_subdir=True)
        le.plot_movie_normal_modes_mass(meshfn+'/', xy, NL, KL, OmK, Omg, sim_type='mass',
                                        rm_images=True, save_ims=True, gapims_only=False)
