import qutip as qt
import numpy as np
import errors

sx = qt.sigmax()
sy = qt.sigmay()
sz = qt.sigmaz()


class Hamiltonian:
    '''Hamiltonian object. It is first initialized, then chosen the type    
    '''
    def __init__(self, system_size, lambda_factor, global_J, barrier_size, barrier_location,
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
    
    def _build_hamiltonian(self):
        '''Create a different type of hamiltonian depending on the string passed'''

        H = 0

        return H