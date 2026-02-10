import qutip as qt
import numpy as np
import errors

sx = qt.sigmax()
sy = qt.sigmay()
sz = qt.sigmaz()



#couplings in statndard encoding
def mirror_symmetric_terms(size, factor):
    strengths = np.zeros(size)
    for i in range(0,size):
        strengths[i] = -0.5*factor*np.sqrt((i+1)*(size-i))
    return strengths


class Hamiltonian:
    '''Hamiltonian object. It is first initialized, then chosen the type    
    '''
    def __init__(self, system_size, lambda_factor, global_J, barrier_size, barrier_location, H_type = 'diffusion',
                 j_error = None, z_error = None, l_error = None):
        '''
        Args:
        
        system_size:(int) Length of chain
        lambda_factor:(float) Global prefactor that controls error and simulation speed
        global_J:(float) Domain wall coupling (not needed for standard)
        barrier_size: (float) Size of the potential barrier
        barrier_location: (int) spin which contains the barrier potential
        
        '''
        self.n_spins = system_size
        self.lambda_factor = lambda_factor
        self.J = global_J
        self.V = barrier_size
        self.V_index = barrier_location  
        self.H_type = H_type  

        self.j_err = j_error
        self.l_err = l_error
        self.z_err = z_error
    

        self.sx_list, self.sy_list, self.sz_list = self._initialize_operators()
        self.couplings = self._calculate_couplings()
        self.ham = self._build_hamiltonian()
    
    def _initialize_operators(self):
        '''Setup operators for individual qubits
           for each value of i it puts the paulis in different positions of the list, 
           then does IxIxI...sigma_ixIxI...xI
        '''
        sx_list, sy_list, sz_list = [], [], []
        for i in range(self.n_spins):
            #list of 2x2 identity matrices
            op_list = [qt.qeye(2)] * self.n_spins
            #replace i-th element with sigma_x
            op_list[i] = sx
            #create matrices of 2^Nx2^N
            sx_list.append(qt.tensor(op_list))
            #do the same for sigma_y and sigma_z
            op_list[i] = sy
            sy_list.append(qt.tensor(op_list))
            op_list[i] = sz
            sz_list.append(qt.tensor(op_list))

        return sx_list, sy_list, sz_list
    

    def calculate_two_body_operators(self):
        '''
        Make list of two body operators xx, yy, and zz
        '''
        self.sxx_list, self.syy_list, self.szz_list = [], [], []
        for i in range(self.n_spins -1):
            #initialize list of identity
            op_list = [qt.qeye(2)] * self.n_spins
            #replace operators on 2 neighboring spins
            op_list[i], op_list[i+1] = sx, sx
            #append tensor prod of chain to list
            self.sxx_list.append(qt.tensor(op_list))
            #repeat for yy and zz
            op_list[i], op_list[i+1] = sy, sy
            self.syy_list.append(qt.tensor(op_list))
            op_list[i], op_list[i+1] = sz, sz
            self.szz_list.append(qt.tensor(op_list))

        return
    
    def _calculate_couplings(self):
        """
        Calculate couplings for lambda, J and z fields for different situations (standard or domain walls)
        If errors are zeros we avoid calling the errors.py script

        Returns: 
            couplings: Dictionary containing all necesary values for the hamiltonian, indexed by their name
                       with errors added if requested

        """
        #Define all Hamiltonian couplings as dictionary
        couplings = {}

        #Define ideal transverse fields
        if self.H_type == 'diffusion':
            error_free_l = [self.lambda_factor]*(self.n_spins)
        elif self.H_type == 'transport':
            error_free_l = error_free_l = mirror_symmetric_terms(self.n_spins, self.lambda_factor)
        else:
            raise ValueError(f"{self.H_type} is not a valid Hamiltonian type. Enter 'diffusion' or 'transport'")

        #errors in transverse fields
        if self.l_err and self.l_err != 0.0:
            couplings["lambda"] = errors.apply_gaussian_rel_error(error_free_l, self.l_err)
        else:
            couplings["lambda"] = error_free_l

        #errors in domain wall couplings
        if self.J:  #not used in standard encoding
            error_free_j = [self.J]*(self.n_spins) #two virtual spins
            #J at ends without error, correspond to local fields at chain ends
            if self.j_err and self.j_err != 0.0:
                couplings["J"] = [self.J] + errors.apply_gaussian_rel_error(error_free_j, self.j_err) + [self.J]
            else:
                couplings["J"] = 2*[self.J] + error_free_j

        if self.z_err and self.z_err != 0.0:
            couplings["z"] = errors.apply_gaussian_abs_error([0]*self.n_spins, self.z_err)
        else:
            couplings["z"] = [0]*self.n_spins

        return couplings
    
    def _build_hamiltonian(self):
        '''Create a different type of hamiltonian depending on the string passed'''

        H = 0

        l_terms = self.couplings["lambda"]
        z_terms = self.couplings["z"]
        j_terms = self.couplings["J"]

        #Transverse field but not in first spin
        for i in range(0, self.n_spins):          #all spins with transverse field
            H += -l_terms[i] * self.sx_list[i]

        #Virtual qubit up at the start of chain
        H += -j_terms[0]*self.sz_list[0]

        #Virtual qubit down at end of chain
        H += j_terms[-1]*self.sz_list[self.n_spins-1]

        #Interaction terms with the rest of the spins except for the last one
        for i in range(0, self.n_spins-1):
            H += j_terms[i+1]*self.sz_list[i]*self.sz_list[i+1]

        #Potential barrier
        H += -0.5*self.V*self.sz_list[self.V_index]*self.sz_list[self.V_index +1]
        # H += -0.5*self.V*self.sz_list[self.V_index +1]*self.sz_list[self.V_index +2]
        
        #Residual z fields
        for i in range(0,self.n_spins):
            H += z_terms[i] * self.sz_list[i]

        return H