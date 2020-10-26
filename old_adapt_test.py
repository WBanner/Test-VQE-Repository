import scipy
import os
import numpy as np
import openfermion
import openfermionpsi4
import copy
import random 
import sys
import copy,itertools
import time 
from qisresearch.utils.random_hamiltonians import get_h_4_hamiltonian
from qiskit.tools import parallel_map

from qisresearch.utils.ofer import openfermion_to_qiskit
from qiskit.aqua.operators import WeightedPauliOperator
from qiskit.quantum_info import Pauli
import psutil

import warnings
warnings.simplefilter("ignore")

import openfermionpyscf

import operator_pools

from qisresearch.adapt.operator_pool import PauliPool

from openfermion import *
from openfermionpsi4 import *

def convert_to_wpauli_list(term, *args):
    num_qubits = args[0]
    if term[0] == complex(0):
        separated_ham = 0*WeightedPauliOperator.from_list(paulis = [Pauli.from_label('I'*num_qubits)])
    else:
        separated_ham = WeightedPauliOperator.from_list([term[1]],[1])
    return separated_ham


def split_into_paulis_minus_weight(ham):
    """
        Used to split the given hamiltonian into lists of constituent parts:
        weight_list: list of complex floats that are the weights of terms in the hamiltonian
        pauli_list: a list of the Weighted_pauli_operator representations of the pauli strings in the hamiltonian
                    (does not include weights)
        name_list: a list of strings that correspond to the names of the operators in the hamiltonian
    """
    args = [ham.num_qubits]
    ham_list = ham.paulis
    separated_ham_list = parallel_map(convert_to_wpauli_list, ham_list, args,num_processes = len(psutil.Process().cpu_affinity()))

    return separated_ham_list

geometry = [('H',(0, 0, 0)),('H',(0,0,1)), ('H',(0, 0, 2)), ('H',(0, 0, 3)), ('H',(0, 0, 4)), ('H',(0, 0, 5))]

def old_setup(geometry,
        basis           = "sto-3g",
        multiplicity    = 1, #same as george, non-degenerate orbitals
        charge          = 1, #doesn't seem to matter in hamiltonian (weird)
        adapt_conver    = 'norm', #doesn't matter for setup, but looks like daniel used a different method to determine convergence, but only in energy-case
        adapt_thresh    = 1e-3, #gradient threshold
        theta_thresh    = 1e-7,
        adapt_maxiter   = 200, #maximum number of iterations
        pool            = operator_pools.singlet_GS(), #generalized singles and doubles as default, though he's not sure if he changed it- nick seems to think that he used GS
        reference       = 'rhf', #is default in MolecularData anyway
        brueckner       = 0, #mattered for psi4 but is 0 as default anyway
        ref_state       = None, 
        spin_adapt      = True,
        fci_nat_orb     = 0,
        cisd_nat_orb    = 0,
        hf_stability    = 'none',
        single_vqe      = False, #will be true in our case
        chk_ops         = [],
        energy_thresh   = 1e-6, #threshold for single_vqe
        fci_overlap     = False,
        psi4_filename   = "psi4_%12.12f"%random.random()
        ):
# designed to set up the problem in the same way as Daniel, did have to substitute openfermionpyscf for psi4 but shouldn't make a difference
#still need to investiage versioning for the pool init and singlet GSD methods as these require the current version of MolecularData objects to have:
# n_orbitals (y)
# get_n_alpha_electrons (y)
# get_n_beta_electrons (y)
# FermionOperator class (y)
#    requried attributes? (~)
#
#
# hermitian_conjugate function (~)
# normal_ordered function (~)
# {{{
    start_time = time.time()
    molecule = openfermion.hamiltonians.MolecularData(geometry, basis, multiplicity)
    
    if cisd_nat_orb == 1:
        cisd = 1
    else:
        cisd = 0
    molecule = openfermionpyscf.run_pyscf(molecule, run_fci = True)
    pool.init(molecule)

    if fci_overlap == False:
        print(" Basis: ", basis)
        print(' HF energy      %20.16f au' %(molecule.hf_energy))
        #print(' MP2 energy     %20.16f au' %(molecule.mp2_energy))
        #print(' CISD energy    %20.16f au' %(molecule.cisd_energy))
        #print(' CCSD energy    %20.16f au' %(molecule.ccsd_energy))
        if brueckner == 1:
            print(' BCCD energy     %20.16f au' %(molecule.bccd_energy))
        if cisd == 1:
            print(' CISD energy     %20.16f au' %(molecule.cisd_energy))
        if reference == 'rhf':
            print(' FCI energy     %20.16f au' %(molecule.fci_energy))

    # if we are going to transform to FCI NOs, it doesn't make sense to transform to CISD NOs
    if cisd_nat_orb == 1 and fci_nat_orb == 0:
        print(' Basis transformed to the CISD natural orbitals')
    if fci_nat_orb == 1:
        print(' Basis transformed to the FCI natural orbitals')

    #Build p-h reference and map it to JW transform
    if ref_state == None:
        ref_state = list(range(0,molecule.n_electrons))

    reference_ket = scipy.sparse.csc_matrix(openfermion.jw_configuration_state(ref_state, molecule.n_qubits)).transpose()
    reference_bra = reference_ket.transpose().conj()

    #JW transform Hamiltonian computed classically with OFPsi4
    hamiltonian_op = molecule.get_molecular_hamiltonian()
    hamiltonian = openfermion.transforms.get_sparse_operator(hamiltonian_op)

    if fci_overlap:
        e, fci_vec = openfermion.get_ground_state(hamiltonian)
        fci_state = scipy.sparse.csc_matrix(fci_vec).transpose()
        index = scipy.sparse.find(reference_ket)[0]
        print(" Basis: ", basis)
        print(' HF energy      %20.16f au' %(molecule.hf_energy))
        if brueckner == 1:
            print(' BCCD energy     %20.16f au' %(molecule.bccd_energy))
        print(' FCI energy     %20.16f au' %e)
        print(' <FCI|HF>       %20.16f' % np.absolute(fci_vec[index]))

    print(' Orbitals')
    print(molecule.canonical_orbitals)
    #Thetas
    parameters = []

    #pool.generate_SparseMatrix()
    pool.gradient_print_thresh = theta_thresh
    
    ansatz_ops = []     #SQ operator strings in the ansatz
    ansatz_mat = []     #Sparse Matrices for operators in ansatz
        
    op_indices = []
    parameters = []
    #curr_state = 1.0*reference_ket
    curr_energy = molecule.hf_energy

    molecule = openfermion.transforms.jordan_wigner(hamiltonian_op)

    pool_list = []
    for op in pool.fermi_ops:
        pool_list.append(1j*openfermion.transforms.jordan_wigner(op))

    pool_list = PauliPool().from_openfermion_qubit_operator_list(pool_list)

    qiskit_molecule = openfermion_to_qiskit(molecule)

    return pool_list, qiskit_molecule
"""

op_pool_list, qiskit_molecule = old_setup(geometry)

print('num ops', len(op_pool_list))

#print(qiskit_molecule.print_details())
op_list = []
for op in op_pool_list.pool:
    print('op', op.print_details())
    split_op = split_into_paulis_minus_weight(op)
    op_list = op_list + split_op

print(len(op_list))
def fill_I(op, **kwargs):
    ham_num = kwargs['ham_num']
    num_qubits = op.num_qubits
    if num_qubits < ham_num:
        op_string = op.print_details()[:num_qubits]
        while num_qubits < ham_num:
            op_string = op_string + 'I'
    elif num_qubits > ham_num:
        op_string =  op.print_details()[:ham_num]
    else:
        return op

    return WeightedPauliOperator.from_list(paulis = [Pauli.from_label(op_string)])

#kwargs = {'ham_num': qiskit_molecule.num_qubits}
#fill_op_list = parallel_map(fill_I, op_list, task_kwargs = kwargs, num_processes =  len(psutil.Process().cpu_affinity()))

#print('finished fill op list')
#print(other_ham.print_details())
#print(new_ham.print_details())
#almost all weights in new ham are on order of e-16, many on e-17, a few on e-15 so these hams are practically identical.
new_op = 0*WeightedPauliOperator.from_list(paulis = [Pauli.from_label('I'*qiskit_molecule.num_qubits)])
#print('ham num qubits', qiskit_molecule.num_qubits)
for op in op_list:
    new_op = new_op + op

new_op_pool_list = split_into_paulis_minus_weight(new_op)

print('done generating pool from singles')

"""


