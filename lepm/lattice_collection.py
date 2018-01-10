import lepm.lattice_class as lattice_class
import numpy as np
##################
import os
import glob

'''
Generate lattices using the lattice class: hexagonal, deformed kagome, triangular, square, hexagonalperBC, etc.

Example usage:
import lepm.lattice_collection as latcoll
lc = latcoll.lattice_collection()
lc.add_meshfn('/Users/npmitchell/Dropbox/Soft_Matter/GPU/networks/iscentroid/iscentroid_square_hexner_size8192_conf*30_x_*30')

'''
    
class LatticeCollection():
    """Create a collection of lattices. Lattices can exist in memory, on hard disk, or both.
    When a lattice exists only in memory, then its corresponding meshfn is 'in_memory'.
    
    Attributes
    ----------
    self.lattices : list of instances of lattice_class.lattice() 
    self.meshfns : file paths of lattices, or 'in_memory' for those not saved on hard disk
    """
    def __init__(self):
        """Create an instance of a lattice_collection."""
        self.lattices = []
        self.meshfns = []
    
    def add_meshfn(self,meshfn):
        """Add one or more lattice file paths to self.meshfns.

        Parameters
        ----------
        meshfn : string or list of strings
            The paths of lattices, to add to self.meshfns 
        """
        if isinstance(meshfn,list):
            for fn in meshfn:
                ind = 0
                # check that it exists or get list of matching lattices
                fnglob = sorted(glob.glob(fn))
                print 'fnglob = ', fnglob
                is_a_dir = np.where(np.array([os.path.isdir(ii) for ii in fnglob]))[0]
                addfn = fnglob[is_a_dir]
                if len(addfn) > 1:
                    for eachfn in addfn:
                        self.meshfns.append(eachfn)
                        ind += 1
                else:
                    self.meshfns.append(addfn)
                    ind += 1
                print 'Added '+str(ind)+' lattice filenames to lattice collection'
        elif isinstance(meshfn,str):
            fnglob = sorted(glob.glob(meshfn))
            print 'fnglob = ', fnglob
            is_a_dir = np.where(np.array([os.path.isdir(ii) for ii in fnglob]))[0]
            print 'is_a_dir = ', is_a_dir
            addfn = fnglob[is_a_dir]
            if np.size(is_a_dir) > 1:
                for eachfn in addfn:
                    self.meshfns.append(eachfn)
                print 'Added ' + str(len(addfn)) + ' lattice filenames to lattice collection'
            else:
                self.meshfns.append(addfn)
                print 'Added lattice filename to lattice collection'
        else:
            print RuntimeError('Argument to lattice_collection instance method add_meshfns() must be string or list of strings.')
        return self.meshfns
    
    def get_meshfns(self):
        try:
            return self.meshfns
        except NameError:
            # Load meshfns from each lattice in the collection
            meshfns = []
            for lat in lattice_collection.lattice:
                try:
                    meshfns.append(lattice.lp['meshfn'])
                except NameError:
                    meshfns.append('in_memory')
            
            self.meshfns = meshfns
            return self.meshfns
    
    def build_and_add(self,*args):
        """Create an instance of a lattice_collection.
        Parameters
        ----------
        *args : arguments sufficient to build a lattice
        
        Returns
        ----------
        self : lattice_collection instance
            now with a newly built lattice appended to list self.lattices
        """
        lattice = lattice_class.lattice(*args)
        self.lattices.append()
        self.meshfns.append(['in_memory'])
    
    def load_from_meshfns(self):
        # Load one or more saved lattices based on meshfn string
        pass
    
    def load_and_add(self, meshfn):
        # Load one or more saved lattices based on meshfn string
        lattice = lattice_class.lattice()
        lattice.load(meshfn=meshfn)
        self.lattices.append(lattice)
        self.add_meshfn(meshfn)
        
    def collect_meshfns(self, meshfns='none'):
        """Get list of mesh filenames (meshfns) or assign it to lattice collection.
        
        Parameters
        ----------
        meshfns : string or list
            path and name of lattice saved on harddrive, or list of paths saved on drive, or string 'none'
            If 'none', assigns list of paths of all the lattices in self.lattices to self.meshfns.
        """
        # Only use the NV, NH, LatticeTop, and lp parameters
        if isinstance(meshfns,list):
            meshfn_list = meshfns
        elif meshfns != 'none':
            '''Meshfns must be given as a glob string to search for'''
            meshfn_list = glob.glob(meshfns)
        elif meshfns == 'none':
            '''Meshfns must be stored as attribute of instance'''
            meshfn_list = self.meshfns
        
        for fn in meshfn_list:
            if fn == 'in_memory':
                print RuntimeWarning('Mesh could not be collected since only exists in RAM, not saved on harddrive')
            else:
                '''Add code here to check if directory, etc'''
                pass

    def gxy_gr(self, outdir=None, show=False):
        """Calc crude (using 1/9 of sample) gxy and gr for all lattices in collection
        
        """
        xgrid = []
        ygrid = []
        gxy_grid = []
        grV = []
        for ii in range(len(self.lattices)):
            if self.meshfns[ii] == 'in_memory':
                (xgrid[ii], ygrid[ii], gxy_grid[ii]), grV[ii] = self.lattices[ii].gxy_gr
            else:
                lat = lattice_class.lattice()
                lat.load(meshfn=self.meshfns[ii])
                (xgrid[ii], ygrid[ii], gxy_grid[ii]), grV[ii] = lat.gxy_gr()
        
        # For now, assume all gxy are same size --> FIX LATER
        
        xgridarr = np.array(xgrid)
        ygridarr = np.array(ygrid)
        gxy_garr = np.array(gxy_grid)
        
        # Now average each bin in all grids
        print 'gxy = ', gxy_garr
        gxy = np.mean(gxy_garr, axis=0)

    def characterize_collection(self):
        """This is unfinished, but would do a statistical characterization summary for a bunch of lattices
        """
        # If the lattices are not collected, look for matching saved lattices.
        # This allows the option of characterizing without holding all lattices in memory
        if self.lattices != []:
            matching = self.lattices
            if len(self.lattices) < len(self.meshfns):
                # Add other meshfns to self.lattices
                pass
        else:
            if not hasattr(self,'meshfns'):
                self.get_meshfns()
            
            # List the meshfn (paths) to the matching saved lattices 
            matching = glob.glob(self.lp['rootdir']+'networks/'+self.LatticeTop+'/'+self.LatticeTop+'_'+self.shape+'*')
        

if __name__ == '__main__':
    '''Perform an example of using the lattice_collection class'''

    rootdir = '/Users/npmitchell/Dropbox/Soft_Matter/GPU/'
    meshfn = rootdir + 'networks/hucentroid/hucentroid_square_d01_originX0p00Y30p00_000030_x_000030'
    lat = LatticeCollection()
    lat.load_and_add(meshfn)
    lat.gxy_gr(show=True)
    # lat.characterize_structures()
