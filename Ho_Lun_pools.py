import openfermion
import numpy as np
import copy as cp
import re
import scipy

import random

from openfermion import *



class OperatorPool:
    def __init__(self):
        self.n_orb = 0
        self.n_occ_a = 0
        self.n_occ_b = 0
        self.n_vir_a = 0
        self.n_vir_b = 0

        self.n_spin_orb = 0

    def init(self,molecule):

        self.n_orb = molecule.n_orbitals
        # print(self.n_orb)
        # print(molecule.n_electrons)
        self.n_spin_orb = 2*self.n_orb 
        self.n_occ_a = molecule.get_n_alpha_electrons()
        # print(self.n_occ_a)
        self.n_occ_b = molecule.get_n_beta_electrons()
        # print(self.n_occ_b)
    
        self.n_vir_a = self.n_orb - self.n_occ_a
        self.n_vir_b = self.n_orb - self.n_occ_b
        
        self.n_occ = self.n_occ_a
        self.n_vir = self.n_vir_a
        self.n_ops = 0

        self.generate_SQ_Operators()

    def generate_SQ_Operators(self):
        print("Virtual: Reimplement")
        exit()

    def generate_SparseMatrix(self):
        self.spmat_ops = []
        print(" Generate Sparse Matrices for operators in pool")
        xx = 0
        for op in self.fermi_ops:
            # print(op)
            # print("")
            self.spmat_ops.append(transforms.get_sparse_operator(op, n_qubits = self.n_spin_orb))
        assert(len(self.spmat_ops) == self.n_ops)
        # print(xx)
        # print(self.spmat_ops[1])
        return

class singlet_GSD(OperatorPool):
    def generate_SQ_Operators(self):
        """
        n_orb is number of spatial orbitals assuming that spin orbitals are labelled
        0a,0b,1a,1b,2a,2b,3a,3b,....  -> 0,1,2,3,...
        """
        
        print(" Form singlet GSD operators")
        
        self.fermi_ops = []
        for p in range(0,self.n_orb):
            pa = 2*p
            pb = 2*p+1
 
            for q in range(p,self.n_orb):
                qa = 2*q
                qb = 2*q+1
        
                termA =  FermionOperator(((pa,1),(qa,0)))
                termA += FermionOperator(((pb,1),(qb,0)))
 
                termA -= hermitian_conjugated(termA)
               
                termA = normal_ordered(termA)
                
                #Normalize
                coeffA = 0
                for t in termA.terms:
                    coeff_t = termA.terms[t]
                    coeffA += coeff_t * coeff_t
            
                if termA.many_body_order() > 0:
                    termA = termA/np.sqrt(coeffA)
                    self.fermi_ops.append(termA)
                       
      
        pq = -1 
        for p in range(0,self.n_orb):
            pa = 2*p
            pb = 2*p+1
 
            for q in range(p,self.n_orb):
                qa = 2*q
                qb = 2*q+1
        
                pq += 1
        
                rs = -1 
                for r in range(0,self.n_orb):
                    ra = 2*r
                    rb = 2*r+1
                    
                    for s in range(r,self.n_orb):
                        sa = 2*s
                        sb = 2*s+1
                    
                        rs += 1
                    
                        if(pq > rs):
                            continue

                        termA =  FermionOperator(((ra,1),(pa,0),(sa,1),(qa,0)), 2/np.sqrt(12))
                        termA += FermionOperator(((rb,1),(pb,0),(sb,1),(qb,0)), 2/np.sqrt(12))
                        termA += FermionOperator(((ra,1),(pa,0),(sb,1),(qb,0)), 1/np.sqrt(12))
                        termA += FermionOperator(((rb,1),(pb,0),(sa,1),(qa,0)), 1/np.sqrt(12))
                        termA += FermionOperator(((ra,1),(pb,0),(sb,1),(qa,0)), 1/np.sqrt(12))
                        termA += FermionOperator(((rb,1),(pa,0),(sa,1),(qb,0)), 1/np.sqrt(12))

                        termB =  FermionOperator(((ra,1),(pa,0),(sb,1),(qb,0)),  1/2.0)
                        termB += FermionOperator(((rb,1),(pb,0),(sa,1),(qa,0)),  1/2.0)
                        termB += FermionOperator(((ra,1),(pb,0),(sb,1),(qa,0)), -1/2.0)
                        termB += FermionOperator(((rb,1),(pa,0),(sa,1),(qb,0)), -1/2.0)
 
                        termA -= hermitian_conjugated(termA)
                        termB -= hermitian_conjugated(termB)
               
                        termA = normal_ordered(termA)
                        termB = normal_ordered(termB)
                        
                        #Normalize
                        coeffA = 0

                        for t in termA.terms:
                            coeff_t = termA.terms[t]
                            coeffA += coeff_t * coeff_t

                        coeffB = 0

                        for t in termB.terms:
                            coeff_t = termB.terms[t]
                            coeffB += coeff_t * coeff_t

                        if termA.many_body_order() > 0:
                            termA = termA/np.sqrt(coeffA)
                            self.fermi_ops.append(termA)
                        
                        if termB.many_body_order() > 0:
                            termB = termB/np.sqrt(coeffB)
                            self.fermi_ops.append(termB)

        self.n_ops = len(self.fermi_ops)
        print(" Number of operators: ", self.n_ops)
        return

class custom(OperatorPool):
    def generate_SQ_Operators(self):
        """
        n_orb is number of spatial orbitals assuming that spin orbitals are labelled
        0a,0b,1a,1b,2a,2b,3a,3b,....  -> 0,1,2,3,...
        """
        
        print(" Form singlet GSD operators")

        self.n_spin_orb = 3
        
        self.fermi_ops = []
        gen_pool =[]

        # termA = QubitOperator('X0 X1 Y2', 1j)
        # self.fermi_ops.append(termA)
        # termA = QubitOperator('X0 Y1', 1j)
        # self.fermi_ops.append(termA)
        # termA = QubitOperator('X0 Z1 Y2', 1j)
        # self.fermi_ops.append(termA)
        # termA = QubitOperator('Y0', 1j)
        # self.fermi_ops.append(termA)

        termA = QubitOperator('Y1 X2', 1j)
        self.fermi_ops.append(termA)
        termA = QubitOperator('Y1 Z2', 1j)
        self.fermi_ops.append(termA)
        termA = QubitOperator('X0 Y2', 1j)
        self.fermi_ops.append(termA)
        termA = QubitOperator('Y0 X1', 1j)
        self.fermi_ops.append(termA)

        first_picked = []
        pool_vec = np.zeros((4 ** self.n_spin_orb,))

        for i in self.fermi_ops:
            for line in i.terms:
                line = str(line)
                # print(line)
                Bin = np.zeros((2 * self.n_spin_orb,), dtype=int)
                X_pat_1 = re.compile("(\d{,2}), 'X'")
                X_1 = X_pat_1.findall(line)
                if X_1:
                    for i in X_1:
                        k = int(i)
                        Bin[self.n_spin_orb + k] = 1
                Y_pat_1 = re.compile("(\d{,2}), 'Y'")
                Y_1 = Y_pat_1.findall(line)
                if Y_1:
                    for i in Y_1:
                        k = int(i)
                        Bin[self.n_spin_orb + k] = 1
                        Bin[k] = 1
                Z_pat_1 = re.compile("(\d{,2}), 'Z'")
                Z_1 = Z_pat_1.findall(line)
                if Z_1:
                    for i in Z_1:
                        k = int(i)
                        Bin[k] = 1
                # print(Bin)
                index = int("".join(str(x) for x in Bin), 2)
                # print("index", index)
                first_picked.append(Bin)
                pool_vec[index] = 1

        for i in first_picked:
            gen_pool.append(i)

        # termA = QubitOperator('Z1 Y2', 1j)
        # self.fermi_ops.append(termA)
        # termA = QubitOperator('Y0 Z1', 1j)
        # self.fermi_ops.append(termA)
        # termA = QubitOperator('X0 Y1', 1j)
        # self.fermi_ops.append(termA)
        # termA = QubitOperator('Z0 Y1 Z2', 1j)
        # self.fermi_ops.append(termA)

        # termA = QubitOperator('Y0 Z1 Z2', 1j)
        # self.fermi_ops.append(termA)
        # termA = QubitOperator('Y0 Z1', 1j)
        # self.fermi_ops.append(termA)
        # termA = QubitOperator('X0 Y2', 1j)
        # self.fermi_ops.append(termA)
        # termA = QubitOperator('X0 Y1 Z2', 1j)
        # self.fermi_ops.append(termA)

        print("initial picked pool size:", len(self.fermi_ops))

        length = 0

        while len(gen_pool) < 4 ** self.n_spin_orb:
            new = []
            if length == 0:
                end = len(gen_pool)
            else:
                end = length
            for i in range(end):
                for j in range(i + 1, len(gen_pool)):
                    Bin = [(gen_pool[i][k] + gen_pool[j][k]) % 2 for k in range(2 * self.n_spin_orb)]
                    index = int("".join(str(x) for x in Bin), 2)
                    if pool_vec[index] == 0:
                        new.append(Bin)
                        pool_vec[index] = 1
            length = len(new)
            for i in new:
                gen_pool.insert(0, i)
            if length == 0:
                break

        print('size of set %12i' % (len(gen_pool)))

        self.generated_ops = []

        for i in range(len(gen_pool)):
            pauli_string = ''
            for j in range(self.n_spin_orb):
                if gen_pool[i][j] == 0:
                    if gen_pool[i][j + self.n_spin_orb] == 1:
                        pauli_string += 'X%d ' % j
                if gen_pool[i][j] == 1:
                    if gen_pool[i][j + self.n_spin_orb] == 0:
                        pauli_string += 'Z%d ' % j
                    else:
                        pauli_string += 'Y%d ' % j
            A = QubitOperator(pauli_string, 0 + 1j)
            self.generated_ops.append(A)

        # for i in range(len(first_picked)):
        #     pauli_string = ''
        #     for j in range(self.n_spin_orb):
        #         if first_picked[i][j] == 0:
        #             if first_picked[i][j + ii] == 1:
        #                 pauli_string += 'X%d ' % j
        #         if first_picked[i][j] == 1:
        #             if first_picked[i][j + ii] == 0:
        #                 pauli_string += 'Z%d ' % j
        #             else:
        #                 pauli_string += 'Y%d ' % j
        #     A = QubitOperator(pauli_string, 0 + 1j)
        #     self.fermi_ops.append(A)

        self.n_ops = len(self.fermi_ops)

        self.n_ops = len(self.fermi_ops)
        print(" Number of operators: ", self.n_ops)
        return

class com_gen(OperatorPool):
    def generate_SQ_Operators(self):

        self.bin_pool = []
        self.fermi_ops = []

        ii = 4

        self.n_spin_orb = ii

        for i in range(2 ** (2 * ii)):
            b_string = [int(j) for j in bin(i)[2:].zfill(2 * ii)]
            self.bin_pool.append(b_string)

        self.odd_string = []

        for i in self.bin_pool:
            if sum(i[k] * i[k + ii] for k in range(ii)) % 2 == 1:
                self.odd_string.append(i)

        print("total number of antisymmetric ops :"+str(len(self.odd_string)))

        rank = 0

        while rank < 2 ** ii -1:
            random.shuffle(self.odd_string)
            first_picked = self.odd_string[:(2 * ii - 2)]
            picked = self.odd_string[:(2 * ii - 2)]
            
    
            pool_vec = np.zeros((4 ** ii,))
    
            for i in picked:
                index = int("".join(str(x) for x in i), 2)
                pool_vec[index] = 1
    
            length = 0
    
            while len(picked) < 4 ** ii:
                new = []
                if length == 0:
                    end = len(picked)
                else:
                    end = length
                for i in range(end):
                    for j in range(i+1, len(picked)):
                        Bin = [(picked[i][k] + picked[j][k]) % 2 for k in range(2 * ii)]
                        index = int("".join(str(x) for x in Bin), 2)
                        if pool_vec[index] == 0:
                            if sum(Bin[k] * Bin[k + ii] for k in range(ii)) % 2 == 1:
                                new.append(Bin)
                                pool_vec[index] = 1
                length = len(new)
                for i in new:
                    picked.insert(0, i)
                if length == 0:
                    break

            print('size of pool %12i' % (len(first_picked)))
            print('size of commutator set %12i' % (len(picked)))
    
            self.generated_ops = []
    
            for i in range(len(picked)):
                pauli_string = ''
                for j in range(ii):
                    if picked[i][j] == 0:
                        if picked[i][j + ii] == 1:
                            pauli_string += 'X%d ' % j
                    if picked[i][j] == 1:
                        if picked[i][j + ii] == 0:
                            pauli_string += 'Z%d ' % j
                        else:
                            pauli_string += 'Y%d ' % j
                A = QubitOperator(pauli_string, 0+1j)
                self.generated_ops.append(A)

            generated_pool = []
    
            for op in self.generated_ops:
                generated_pool.append(transforms.get_sparse_operator(op, n_qubits=ii))
    
            over_mat = np.zeros(shape=(len(generated_pool), len(generated_pool)))
            self.vec = np.zeros((2 ** ii, 1))
            self.vec[0] = 1
            print(self.vec)
            # print(vec)
            norm = 0
        
            for i in self.vec:
                norm += i * i
        
            self.vec = np.true_divide(self.vec, np.sqrt(norm))
            self.vec = scipy.sparse.csc_matrix(self.vec)
        
            for i in range(len(self.generated_ops)):
                print(self.generated_ops[i])
                for j in range(len(self.generated_ops)):
                    element = self.vec.transpose().conjugate().dot(generated_pool[i].transpose().conjugate().dot(generated_pool[j].dot(self.vec)))[0, 0]
                    over_mat[i, j] = element.real
    
            rank = np.linalg.matrix_rank(over_mat, tol=1e-12)

        print("initial picked pool size:", len(picked))
    
        print("rank =", rank)

        for i in range(len(first_picked)):
            pauli_string = ''
            for j in range(ii):
                if first_picked[i][j] == 0:
                    if first_picked[i][j + ii] == 1:
                        pauli_string += 'X%d ' % j
                if first_picked[i][j] == 1:
                    if first_picked[i][j + ii] == 0:
                        pauli_string += 'Z%d ' % j
                    else:
                        pauli_string += 'Y%d ' % j
            A = QubitOperator(pauli_string, 0+1j)
            self.fermi_ops.append(A)

        self.n_ops = len(self.fermi_ops)

class GSD_extract(OperatorPool):
    def generate_SQ_Operators(self):
        """
        n_orb is number of spatial orbitals assuming that spin orbitals are labelled
        0a,0b,1a,1b,2a,2b,3a,3b,....  -> 0,1,2,3,...
        """

        print(" Form singlet GSD operators")

        self.fermi_ops = []
        for p in range(0, self.n_orb):
            pa = 2 * p
            pb = 2 * p + 1

            for q in range(p, self.n_orb):
                qa = 2 * q
                qb = 2 * q + 1

                termA = FermionOperator(((pa, 1), (qa, 0)))
                termA += FermionOperator(((pb, 1), (qb, 0)))

                termA -= hermitian_conjugated(termA)

                termA = normal_ordered(termA)

                # Normalize
                coeffA = 0
                for t in termA.terms:
                    coeff_t = termA.terms[t]
                    coeffA += coeff_t * coeff_t

                if termA.many_body_order() > 0:
                    termA = termA / np.sqrt(coeffA)
                    self.fermi_ops.append(termA)

        pq = -1
        for p in range(0, self.n_orb):
            pa = 2 * p
            pb = 2 * p + 1

            for q in range(p, self.n_orb):
                qa = 2 * q
                qb = 2 * q + 1

                pq += 1

                rs = -1
                for r in range(0, self.n_orb):
                    ra = 2 * r
                    rb = 2 * r + 1

                    for s in range(r, self.n_orb):
                        sa = 2 * s
                        sb = 2 * s + 1

                        rs += 1

                        if (pq > rs):
                            continue

                        termA = FermionOperator(((ra, 1), (pa, 0), (sa, 1), (qa, 0)), 2 / np.sqrt(12))
                        termA += FermionOperator(((rb, 1), (pb, 0), (sb, 1), (qb, 0)), 2 / np.sqrt(12))
                        termA += FermionOperator(((ra, 1), (pa, 0), (sb, 1), (qb, 0)), 1 / np.sqrt(12))
                        termA += FermionOperator(((rb, 1), (pb, 0), (sa, 1), (qa, 0)), 1 / np.sqrt(12))
                        termA += FermionOperator(((ra, 1), (pb, 0), (sb, 1), (qa, 0)), 1 / np.sqrt(12))
                        termA += FermionOperator(((rb, 1), (pa, 0), (sa, 1), (qb, 0)), 1 / np.sqrt(12))

                        termB = FermionOperator(((ra, 1), (pa, 0), (sb, 1), (qb, 0)), 1 / 2.0)
                        termB += FermionOperator(((rb, 1), (pb, 0), (sa, 1), (qa, 0)), 1 / 2.0)
                        termB += FermionOperator(((ra, 1), (pb, 0), (sb, 1), (qa, 0)), -1 / 2.0)
                        termB += FermionOperator(((rb, 1), (pa, 0), (sa, 1), (qb, 0)), -1 / 2.0)

                        termA -= hermitian_conjugated(termA)
                        termB -= hermitian_conjugated(termB)

                        termA = normal_ordered(termA)
                        termB = normal_ordered(termB)

                        # Normalize
                        coeffA = 0

                        for t in termA.terms:
                            coeff_t = termA.terms[t]
                            coeffA += coeff_t * coeff_t

                        coeffB = 0

                        for t in termB.terms:
                            coeff_t = termB.terms[t]
                            coeffB += coeff_t * coeff_t

                        if termA.many_body_order() > 0:
                            termA = termA / np.sqrt(coeffA)
                            self.fermi_ops.append(termA)

                        if termB.many_body_order() > 0:
                            termB = termB / np.sqrt(coeffB)
                            self.fermi_ops.append(termB)

        self.n_ops = len(self.fermi_ops)
        print(" Number of fermionic operators: ", self.n_ops)

        n = self.n_spin_orb

        pool_vec = np.zeros((4 ** n,))

        for i in self.fermi_ops:
            pauli = openfermion.transforms.jordan_wigner(i)
            for line in pauli.terms:
                line = str(line)
                # print(line)
                Bin = np.zeros((2 * n,), dtype=int)
                X_pat_1 = re.compile("(\d{,2}), 'X'")
                X_1 = X_pat_1.findall(line)
                if X_1:
                    for i in X_1:
                        k = int(i)
                        Bin[n + k] = 1
                Y_pat_1 = re.compile("(\d{,2}), 'Y'")
                Y_1 = Y_pat_1.findall(line)
                if Y_1:
                    for i in Y_1:
                        k = int(i)
                        Bin[n + k] = 1
                        Bin[k] = 1
                Z_pat_1 = re.compile("(\d{,2}), 'Z'")
                Z_1 = Z_pat_1.findall(line)
                if Z_1:
                    for i in Z_1:
                        k = int(i)
                        Bin[k] = 1
                # print(Bin)
                index = int("".join(str(x) for x in Bin), 2)
                # print("index", index)

                pool_vec[index] = int(1)

        nz = np.nonzero(pool_vec)[0]

        print("pauli pool size:", len(pool_vec[nz]))

        self.fermi_ops = []

        m = 2*n

        for i in nz:
            p = int(i)
            bi = bin(p)
            b_string = [int(j) for j in bi[2:].zfill(m)]
            pauli_string = ''
            flip = []
            for k in range(n):
                if b_string[k] == 0:
                    if b_string[k + n] == 1:
                        pauli_string += 'X%d ' % k
                        flip.append(k)
                if b_string[k] == 1:
                    if b_string[k + n] == 1:
                        pauli_string += 'Y%d ' % k
                        flip.append(k)
            flip.sort()
            z_string = list(range(flip[0] + 1,flip[1]))
            if len(flip) == 4:
                for i in range(flip[2] + 1, flip[3]):
                    z_string.append(i)
            print("Z string:", z_string)
            for i in z_string:
                b_string[i] += 1
                b_string[i] = b_string[i] % 2
            for k in range(n):
                if b_string[k] == 1:
                    if b_string[k + n] == 0:
                        pauli_string += 'Z%d ' % k
            A = QubitOperator(pauli_string, 0 + 1j)
            print("Pauli:", pauli_string)
            self.fermi_ops.append(A)

        self.n_ops = len(self.fermi_ops)
        print(" Number of pauli operators: ", self.n_ops)

        return

class GSD_extract_bk(OperatorPool):
    def generate_SQ_Operators(self):
        """
        n_orb is number of spatial orbitals assuming that spin orbitals are labelled
        0a,0b,1a,1b,2a,2b,3a,3b,....  -> 0,1,2,3,...
        """

        print(" Form singlet GSD operators")

        self.fermi_ops = []
        for p in range(0, self.n_orb):
            pa = 2 * p
            pb = 2 * p + 1

            for q in range(p, self.n_orb):
                qa = 2 * q
                qb = 2 * q + 1

                termA = FermionOperator(((pa, 1), (qa, 0)))
                termA += FermionOperator(((pb, 1), (qb, 0)))

                termA -= hermitian_conjugated(termA)

                termA = normal_ordered(termA)

                # Normalize
                coeffA = 0
                for t in termA.terms:
                    coeff_t = termA.terms[t]
                    coeffA += coeff_t * coeff_t

                if termA.many_body_order() > 0:
                    termA = termA / np.sqrt(coeffA)
                    self.fermi_ops.append(termA)

        pq = -1
        for p in range(0, self.n_orb):
            pa = 2 * p
            pb = 2 * p + 1

            for q in range(p, self.n_orb):
                qa = 2 * q
                qb = 2 * q + 1

                pq += 1

                rs = -1
                for r in range(0, self.n_orb):
                    ra = 2 * r
                    rb = 2 * r + 1

                    for s in range(r, self.n_orb):
                        sa = 2 * s
                        sb = 2 * s + 1

                        rs += 1

                        if (pq > rs):
                            continue

                        termA = FermionOperator(((ra, 1), (pa, 0), (sa, 1), (qa, 0)), 2 / np.sqrt(12))
                        termA += FermionOperator(((rb, 1), (pb, 0), (sb, 1), (qb, 0)), 2 / np.sqrt(12))
                        termA += FermionOperator(((ra, 1), (pa, 0), (sb, 1), (qb, 0)), 1 / np.sqrt(12))
                        termA += FermionOperator(((rb, 1), (pb, 0), (sa, 1), (qa, 0)), 1 / np.sqrt(12))
                        termA += FermionOperator(((ra, 1), (pb, 0), (sb, 1), (qa, 0)), 1 / np.sqrt(12))
                        termA += FermionOperator(((rb, 1), (pa, 0), (sa, 1), (qb, 0)), 1 / np.sqrt(12))

                        termB = FermionOperator(((ra, 1), (pa, 0), (sb, 1), (qb, 0)), 1 / 2.0)
                        termB += FermionOperator(((rb, 1), (pb, 0), (sa, 1), (qa, 0)), 1 / 2.0)
                        termB += FermionOperator(((ra, 1), (pb, 0), (sb, 1), (qa, 0)), -1 / 2.0)
                        termB += FermionOperator(((rb, 1), (pa, 0), (sa, 1), (qb, 0)), -1 / 2.0)

                        termA -= hermitian_conjugated(termA)
                        termB -= hermitian_conjugated(termB)

                        termA = normal_ordered(termA)
                        termB = normal_ordered(termB)

                        # Normalize
                        coeffA = 0

                        for t in termA.terms:
                            coeff_t = termA.terms[t]
                            coeffA += coeff_t * coeff_t

                        coeffB = 0

                        for t in termB.terms:
                            coeff_t = termB.terms[t]
                            coeffB += coeff_t * coeff_t

                        if termA.many_body_order() > 0:
                            termA = termA / np.sqrt(coeffA)
                            self.fermi_ops.append(termA)

                        if termB.many_body_order() > 0:
                            termB = termB / np.sqrt(coeffB)
                            self.fermi_ops.append(termB)

        self.n_ops = len(self.fermi_ops)
        print(" Number of fermionic operators: ", self.n_ops)

        n = self.n_spin_orb

        pool_vec = np.zeros((4 ** n,))

        for i in self.fermi_ops:
            pauli = openfermion.transforms.bravyi_kitaev(i)
            for line in pauli.terms:
                line = str(line)
                # print(line)
                Bin = np.zeros((2 * n,), dtype=int)
                X_pat_1 = re.compile("(\d{,2}), 'X'")
                X_1 = X_pat_1.findall(line)
                if X_1:
                    for i in X_1:
                        k = int(i)
                        Bin[n + k] = 1
                Y_pat_1 = re.compile("(\d{,2}), 'Y'")
                Y_1 = Y_pat_1.findall(line)
                if Y_1:
                    for i in Y_1:
                        k = int(i)
                        Bin[n + k] = 1
                        Bin[k] = 1
                Z_pat_1 = re.compile("(\d{,2}), 'Z'")
                Z_1 = Z_pat_1.findall(line)
                if Z_1:
                    for i in Z_1:
                        k = int(i)
                        Bin[k] = 1
                # print(Bin)
                index = int("".join(str(x) for x in Bin), 2)
                # print("index", index)

                pool_vec[index] = int(1)

        nz = np.nonzero(pool_vec)[0]

        print("pauli pool size:", len(pool_vec[nz]))

        self.fermi_ops = []

        m = 2*n

        for i in nz:
            p = int(i)
            bi = bin(p)
            b_string = [int(j) for j in bi[2:].zfill(m)]
            pauli_string = ''
            flip = []
            for k in range(n):
                if b_string[k] == 0:
                    if b_string[k + n] == 1:
                        pauli_string += 'X%d ' % k
                        flip.append(k)
                if b_string[k] == 1:
                    if b_string[k + n] == 1:
                        pauli_string += 'Y%d ' % k
                        flip.append(k)
            flip.sort()
            # z_string = list(range(flip[0] + 1,flip[1]))
            # if len(flip) == 4:
            #     for i in range(flip[2] + 1, flip[3]):
            #         z_string.append(i)
            # print("Z string:", z_string)
            # for i in z_string:    % removing z-string
            #     b_string[i] += 1
            #     b_string[i] = b_string[i] % 2
            for k in range(n):
                if b_string[k] == 1:
                    if b_string[k + n] == 0:
                        pauli_string += 'Z%d ' % k
            A = QubitOperator(pauli_string, 0 + 1j)
            print("Pauli:", pauli_string)
            self.fermi_ops.append(A)

        self.n_ops = len(self.fermi_ops)
        print(" Number of pauli operators: ", self.n_ops)

        return

class GSD_extract_hh(OperatorPool):
    def generate_SQ_Operators(self):
        """
        n_orb is number of spatial orbitals assuming that spin orbitals are labelled
        0a,0b,1a,1b,2a,2b,3a,3b,....  -> 0,1,2,3,...
        """

        print(" Form singlet GSD operators")

        self.fermi_ops = []
        for p in range(0, self.n_orb):
            pa = 2 * p
            pb = 2 * p + 1

            for q in range(p, self.n_orb):
                qa = 2 * q
                qb = 2 * q + 1

                termA = FermionOperator(((pa, 1), (qa, 0)))
                termA += FermionOperator(((pb, 1), (qb, 0)))

                termA -= hermitian_conjugated(termA)

                termA = normal_ordered(termA)

                # Normalize
                coeffA = 0
                for t in termA.terms:
                    coeff_t = termA.terms[t]
                    coeffA += coeff_t * coeff_t

                if termA.many_body_order() > 0:
                    termA = termA / np.sqrt(coeffA)
                    self.fermi_ops.append(termA)

        pq = -1
        for p in range(0, self.n_orb):
            pa = 2 * p
            pb = 2 * p + 1

            for q in range(p, self.n_orb):
                qa = 2 * q
                qb = 2 * q + 1

                pq += 1

                rs = -1
                for r in range(0, self.n_orb):
                    ra = 2 * r
                    rb = 2 * r + 1

                    for s in range(r, self.n_orb):
                        sa = 2 * s
                        sb = 2 * s + 1

                        rs += 1

                        if (pq > rs):
                            continue

                        termA = FermionOperator(((ra, 1), (pa, 0), (sa, 1), (qa, 0)), 2 / np.sqrt(12))
                        termA += FermionOperator(((rb, 1), (pb, 0), (sb, 1), (qb, 0)), 2 / np.sqrt(12))
                        termA += FermionOperator(((ra, 1), (pa, 0), (sb, 1), (qb, 0)), 1 / np.sqrt(12))
                        termA += FermionOperator(((rb, 1), (pb, 0), (sa, 1), (qa, 0)), 1 / np.sqrt(12))
                        termA += FermionOperator(((ra, 1), (pb, 0), (sb, 1), (qa, 0)), 1 / np.sqrt(12))
                        termA += FermionOperator(((rb, 1), (pa, 0), (sa, 1), (qb, 0)), 1 / np.sqrt(12))

                        termB = FermionOperator(((ra, 1), (pa, 0), (sb, 1), (qb, 0)), 1 / 2.0)
                        termB += FermionOperator(((rb, 1), (pb, 0), (sa, 1), (qa, 0)), 1 / 2.0)
                        termB += FermionOperator(((ra, 1), (pb, 0), (sb, 1), (qa, 0)), -1 / 2.0)
                        termB += FermionOperator(((rb, 1), (pa, 0), (sa, 1), (qb, 0)), -1 / 2.0)

                        termA -= hermitian_conjugated(termA)
                        termB -= hermitian_conjugated(termB)

                        termA = normal_ordered(termA)
                        termB = normal_ordered(termB)

                        # Normalize
                        coeffA = 0

                        for t in termA.terms:
                            coeff_t = termA.terms[t]
                            coeffA += coeff_t * coeff_t

                        coeffB = 0

                        for t in termB.terms:
                            coeff_t = termB.terms[t]
                            coeffB += coeff_t * coeff_t

                        if termA.many_body_order() > 0:
                            termA = termA / np.sqrt(coeffA)
                            self.fermi_ops.append(termA)

                        if termB.many_body_order() > 0:
                            termB = termB / np.sqrt(coeffB)
                            self.fermi_ops.append(termB)

        self.n_ops = len(self.fermi_ops)
        print(" Number of fermionic operators: ", self.n_ops)

        n = self.n_spin_orb

        pool_vec = np.zeros((4 ** n,))

        for i in self.fermi_ops:
            pauli = openfermion.transforms.jordan_wigner(i)
            for line in pauli.terms:
                line = str(line)
                # print(line)
                Bin = np.zeros((2 * n,), dtype=int)
                X_pat_1 = re.compile("(\d{,2}), 'X'")
                X_1 = X_pat_1.findall(line)
                if X_1:
                    for i in X_1:
                        k = int(i)
                        Bin[n + k] = 1
                Y_pat_1 = re.compile("(\d{,2}), 'Y'")
                Y_1 = Y_pat_1.findall(line)
                if Y_1:
                    for i in Y_1:
                        k = int(i)
                        Bin[n + k] = 1
                        Bin[k] = 1
                Z_pat_1 = re.compile("(\d{,2}), 'Z'")
                Z_1 = Z_pat_1.findall(line)
                if Z_1:
                    for i in Z_1:
                        k = int(i)
                        Bin[k] = 1
                # print(Bin)
                index = int("".join(str(x) for x in Bin), 2)
                # print("index", index)

                pool_vec[index] = int(1)

        nz = np.nonzero(pool_vec)[0]

        print("pauli pool size:", len(pool_vec[nz]))

        self.fermi_ops = []

        m = 2*n

        for i in nz:
            p = int(i)
            bi = bin(p)
            b_string = [int(j) for j in bi[2:].zfill(m)]
            pauli_string = ''
            flip = []
            for k in range(n):
                if b_string[k] == 0:
                    if b_string[k + n] == 1:
                        pauli_string += 'X%d ' % k
                        flip.append(k)
                if b_string[k] == 1:
                    if b_string[k + n] == 1:
                        pauli_string += 'Y%d ' % k
                        flip.append(k)
            flip.sort()
            z_string = list(range(flip[0] + 1,flip[1]))
            if len(flip) == 4:
                for i in range(flip[2] + 1, flip[3]):
                    z_string.append(i)
            print("Z string:", z_string)
            for i in z_string:
                b_string[i] += 1
                b_string[i] = b_string[i] % 2
            for k in range(n):
                if b_string[k] == 1:
                    if b_string[k + n] == 0:
                        pauli_string += 'Z%d ' % k
            A = QubitOperator(pauli_string, 0 + 1j)
            print("Pauli:", pauli_string)
            self.fermi_ops.append(A)

        print("original size", len(self.fermi_ops))

        self.fermi_ops_1 = self.fermi_ops
        self.fermi_ops = []

        np.random.shuffle(self.fermi_ops_1)
        for op in self.fermi_ops_1[:len(self.fermi_ops_1)//4]:
            self.fermi_ops.append(op)

        for op in self.fermi_ops:
            print(op)

        self.n_ops = len(self.fermi_ops)
        print(" Number of operators: ", self.n_ops)
        return

class GSD_extract_hhhh(OperatorPool):
    def generate_SQ_Operators(self):
        """
        n_orb is number of spatial orbitals assuming that spin orbitals are labelled
        0a,0b,1a,1b,2a,2b,3a,3b,....  -> 0,1,2,3,...
        """

        print(" Form singlet GSD operators")

        self.fermi_ops = []
        for p in range(0, self.n_orb):
            pa = 2 * p
            pb = 2 * p + 1

            for q in range(p, self.n_orb):
                qa = 2 * q
                qb = 2 * q + 1

                termA = FermionOperator(((pa, 1), (qa, 0)))
                termA += FermionOperator(((pb, 1), (qb, 0)))

                termA -= hermitian_conjugated(termA)

                termA = normal_ordered(termA)

                # Normalize
                coeffA = 0
                for t in termA.terms:
                    coeff_t = termA.terms[t]
                    coeffA += coeff_t * coeff_t

                if termA.many_body_order() > 0:
                    termA = termA / np.sqrt(coeffA)
                    self.fermi_ops.append(termA)

        pq = -1
        for p in range(0, self.n_orb):
            pa = 2 * p
            pb = 2 * p + 1

            for q in range(p, self.n_orb):
                qa = 2 * q
                qb = 2 * q + 1

                pq += 1

                rs = -1
                for r in range(0, self.n_orb):
                    ra = 2 * r
                    rb = 2 * r + 1

                    for s in range(r, self.n_orb):
                        sa = 2 * s
                        sb = 2 * s + 1

                        rs += 1

                        if (pq > rs):
                            continue

                        termA = FermionOperator(((ra, 1), (pa, 0), (sa, 1), (qa, 0)), 2 / np.sqrt(12))
                        termA += FermionOperator(((rb, 1), (pb, 0), (sb, 1), (qb, 0)), 2 / np.sqrt(12))
                        termA += FermionOperator(((ra, 1), (pa, 0), (sb, 1), (qb, 0)), 1 / np.sqrt(12))
                        termA += FermionOperator(((rb, 1), (pb, 0), (sa, 1), (qa, 0)), 1 / np.sqrt(12))
                        termA += FermionOperator(((ra, 1), (pb, 0), (sb, 1), (qa, 0)), 1 / np.sqrt(12))
                        termA += FermionOperator(((rb, 1), (pa, 0), (sa, 1), (qb, 0)), 1 / np.sqrt(12))

                        termB = FermionOperator(((ra, 1), (pa, 0), (sb, 1), (qb, 0)), 1 / 2.0)
                        termB += FermionOperator(((rb, 1), (pb, 0), (sa, 1), (qa, 0)), 1 / 2.0)
                        termB += FermionOperator(((ra, 1), (pb, 0), (sb, 1), (qa, 0)), -1 / 2.0)
                        termB += FermionOperator(((rb, 1), (pa, 0), (sa, 1), (qb, 0)), -1 / 2.0)

                        termA -= hermitian_conjugated(termA)
                        termB -= hermitian_conjugated(termB)

                        termA = normal_ordered(termA)
                        termB = normal_ordered(termB)

                        # Normalize
                        coeffA = 0

                        for t in termA.terms:
                            coeff_t = termA.terms[t]
                            coeffA += coeff_t * coeff_t

                        coeffB = 0

                        for t in termB.terms:
                            coeff_t = termB.terms[t]
                            coeffB += coeff_t * coeff_t

                        if termA.many_body_order() > 0:
                            termA = termA / np.sqrt(coeffA)
                            self.fermi_ops.append(termA)

                        if termB.many_body_order() > 0:
                            termB = termB / np.sqrt(coeffB)
                            self.fermi_ops.append(termB)

        self.n_ops = len(self.fermi_ops)
        print(" Number of fermionic operators: ", self.n_ops)

        n = self.n_spin_orb

        pool_vec = np.zeros((4 ** n,))

        for i in self.fermi_ops:
            pauli = openfermion.transforms.jordan_wigner(i)
            for line in pauli.terms:
                line = str(line)
                # print(line)
                Bin = np.zeros((2 * n,), dtype=int)
                X_pat_1 = re.compile("(\d{,2}), 'X'")
                X_1 = X_pat_1.findall(line)
                if X_1:
                    for i in X_1:
                        k = int(i)
                        Bin[n + k] = 1
                Y_pat_1 = re.compile("(\d{,2}), 'Y'")
                Y_1 = Y_pat_1.findall(line)
                if Y_1:
                    for i in Y_1:
                        k = int(i)
                        Bin[n + k] = 1
                        Bin[k] = 1
                Z_pat_1 = re.compile("(\d{,2}), 'Z'")
                Z_1 = Z_pat_1.findall(line)
                if Z_1:
                    for i in Z_1:
                        k = int(i)
                        Bin[k] = 1
                # print(Bin)
                index = int("".join(str(x) for x in Bin), 2)
                # print("index", index)

                pool_vec[index] = int(1)

        nz = np.nonzero(pool_vec)[0]

        print("pauli pool size:", len(pool_vec[nz]))

        self.fermi_ops = []

        m = 2*n

        for i in nz:
            p = int(i)
            bi = bin(p)
            b_string = [int(j) for j in bi[2:].zfill(m)]
            pauli_string = ''
            flip = []
            for k in range(n):
                if b_string[k] == 0:
                    if b_string[k + n] == 1:
                        pauli_string += 'X%d ' % k
                        flip.append(k)
                if b_string[k] == 1:
                    if b_string[k + n] == 1:
                        pauli_string += 'Y%d ' % k
                        flip.append(k)
            flip.sort()
            z_string = list(range(flip[0] + 1,flip[1]))
            if len(flip) == 4:
                for i in range(flip[2] + 1, flip[3]):
                    z_string.append(i)
            print("Z string:", z_string)
            for i in z_string:
                b_string[i] += 1
                b_string[i] = b_string[i] % 2
            for k in range(n):
                if b_string[k] == 1:
                    if b_string[k + n] == 0:
                        pauli_string += 'Z%d ' % k
            A = QubitOperator(pauli_string, 0 + 1j)
            print("Pauli:", pauli_string)
            self.fermi_ops.append(A)

        print("original size", len(self.fermi_ops))

        self.fermi_ops_1 = self.fermi_ops
        self.fermi_ops = []

        np.random.shuffle(self.fermi_ops_1)
        for op in self.fermi_ops_1[:len(self.fermi_ops_1) // 16]:
            self.fermi_ops.append(op)

        for op in self.fermi_ops:
            print(op)

        self.n_ops = len(self.fermi_ops)
        print(" Number of operators: ", self.n_ops)
        return

class GSD_extract_hhhhh(OperatorPool):
    def generate_SQ_Operators(self):
        """
        n_orb is number of spatial orbitals assuming that spin orbitals are labelled
        0a,0b,1a,1b,2a,2b,3a,3b,....  -> 0,1,2,3,...
        """

        print(" Form singlet GSD operators")

        self.fermi_ops = []
        for p in range(0, self.n_orb):
            pa = 2 * p
            pb = 2 * p + 1

            for q in range(p, self.n_orb):
                qa = 2 * q
                qb = 2 * q + 1

                termA = FermionOperator(((pa, 1), (qa, 0)))
                termA += FermionOperator(((pb, 1), (qb, 0)))

                termA -= hermitian_conjugated(termA)

                termA = normal_ordered(termA)

                # Normalize
                coeffA = 0
                for t in termA.terms:
                    coeff_t = termA.terms[t]
                    coeffA += coeff_t * coeff_t

                if termA.many_body_order() > 0:
                    termA = termA / np.sqrt(coeffA)
                    self.fermi_ops.append(termA)

        pq = -1
        for p in range(0, self.n_orb):
            pa = 2 * p
            pb = 2 * p + 1

            for q in range(p, self.n_orb):
                qa = 2 * q
                qb = 2 * q + 1

                pq += 1

                rs = -1
                for r in range(0, self.n_orb):
                    ra = 2 * r
                    rb = 2 * r + 1

                    for s in range(r, self.n_orb):
                        sa = 2 * s
                        sb = 2 * s + 1

                        rs += 1

                        if (pq > rs):
                            continue

                        termA = FermionOperator(((ra, 1), (pa, 0), (sa, 1), (qa, 0)), 2 / np.sqrt(12))
                        termA += FermionOperator(((rb, 1), (pb, 0), (sb, 1), (qb, 0)), 2 / np.sqrt(12))
                        termA += FermionOperator(((ra, 1), (pa, 0), (sb, 1), (qb, 0)), 1 / np.sqrt(12))
                        termA += FermionOperator(((rb, 1), (pb, 0), (sa, 1), (qa, 0)), 1 / np.sqrt(12))
                        termA += FermionOperator(((ra, 1), (pb, 0), (sb, 1), (qa, 0)), 1 / np.sqrt(12))
                        termA += FermionOperator(((rb, 1), (pa, 0), (sa, 1), (qb, 0)), 1 / np.sqrt(12))

                        termB = FermionOperator(((ra, 1), (pa, 0), (sb, 1), (qb, 0)), 1 / 2.0)
                        termB += FermionOperator(((rb, 1), (pb, 0), (sa, 1), (qa, 0)), 1 / 2.0)
                        termB += FermionOperator(((ra, 1), (pb, 0), (sb, 1), (qa, 0)), -1 / 2.0)
                        termB += FermionOperator(((rb, 1), (pa, 0), (sa, 1), (qb, 0)), -1 / 2.0)

                        termA -= hermitian_conjugated(termA)
                        termB -= hermitian_conjugated(termB)

                        termA = normal_ordered(termA)
                        termB = normal_ordered(termB)

                        # Normalize
                        coeffA = 0

                        for t in termA.terms:
                            coeff_t = termA.terms[t]
                            coeffA += coeff_t * coeff_t

                        coeffB = 0

                        for t in termB.terms:
                            coeff_t = termB.terms[t]
                            coeffB += coeff_t * coeff_t

                        if termA.many_body_order() > 0:
                            termA = termA / np.sqrt(coeffA)
                            self.fermi_ops.append(termA)

                        if termB.many_body_order() > 0:
                            termB = termB / np.sqrt(coeffB)
                            self.fermi_ops.append(termB)

        self.n_ops = len(self.fermi_ops)
        print(" Number of fermionic operators: ", self.n_ops)

        n = self.n_spin_orb

        pool_vec = np.zeros((4 ** n,))

        for i in self.fermi_ops:
            pauli = openfermion.transforms.jordan_wigner(i)
            for line in pauli.terms:
                line = str(line)
                # print(line)
                Bin = np.zeros((2 * n,), dtype=int)
                X_pat_1 = re.compile("(\d{,2}), 'X'")
                X_1 = X_pat_1.findall(line)
                if X_1:
                    for i in X_1:
                        k = int(i)
                        Bin[n + k] = 1
                Y_pat_1 = re.compile("(\d{,2}), 'Y'")
                Y_1 = Y_pat_1.findall(line)
                if Y_1:
                    for i in Y_1:
                        k = int(i)
                        Bin[n + k] = 1
                        Bin[k] = 1
                Z_pat_1 = re.compile("(\d{,2}), 'Z'")
                Z_1 = Z_pat_1.findall(line)
                if Z_1:
                    for i in Z_1:
                        k = int(i)
                        Bin[k] = 1
                # print(Bin)
                index = int("".join(str(x) for x in Bin), 2)
                # print("index", index)

                pool_vec[index] = int(1)

        nz = np.nonzero(pool_vec)[0]

        print("pauli pool size:", len(pool_vec[nz]))

        self.fermi_ops = []

        m = 2*n

        for i in nz:
            p = int(i)
            bi = bin(p)
            b_string = [int(j) for j in bi[2:].zfill(m)]
            pauli_string = ''
            flip = []
            for k in range(n):
                if b_string[k] == 0:
                    if b_string[k + n] == 1:
                        pauli_string += 'X%d ' % k
                        flip.append(k)
                if b_string[k] == 1:
                    if b_string[k + n] == 1:
                        pauli_string += 'Y%d ' % k
                        flip.append(k)
            flip.sort()
            z_string = list(range(flip[0] + 1,flip[1]))
            if len(flip) == 4:
                for i in range(flip[2] + 1, flip[3]):
                    z_string.append(i)
            print("Z string:", z_string)
            for i in z_string:
                b_string[i] += 1
                b_string[i] = b_string[i] % 2
            for k in range(n):
                if b_string[k] == 1:
                    if b_string[k + n] == 0:
                        pauli_string += 'Z%d ' % k
            A = QubitOperator(pauli_string, 0 + 1j)
            print("Pauli:", pauli_string)
            self.fermi_ops.append(A)

        print("original size", len(self.fermi_ops))

        self.fermi_ops_1 = self.fermi_ops
        self.fermi_ops = []

        np.random.shuffle(self.fermi_ops_1)
        for op in self.fermi_ops_1[:len(self.fermi_ops_1) // 32]:
            self.fermi_ops.append(op)

        for op in self.fermi_ops:
            print(op)

        self.n_ops = len(self.fermi_ops)
        print(" Number of operators: ", self.n_ops)
        return

class GSD_extract_reduce(OperatorPool):
    def generate_SQ_Operators(self):
        """
        n_orb is number of spatial orbitals assuming that spin orbitals are labelled
        0a,0b,1a,1b,2a,2b,3a,3b,....  -> 0,1,2,3,...
        """

        print(" Form singlet GSD operators")

        self.fermi_ops = []
        for p in range(0, self.n_orb):
            pa = 2 * p
            pb = 2 * p + 1

            for q in range(p, self.n_orb):
                qa = 2 * q
                qb = 2 * q + 1

                termA = FermionOperator(((pa, 1), (qa, 0)))
                termA += FermionOperator(((pb, 1), (qb, 0)))

                termA -= hermitian_conjugated(termA)

                termA = normal_ordered(termA)

                # Normalize
                coeffA = 0
                for t in termA.terms:
                    coeff_t = termA.terms[t]
                    coeffA += coeff_t * coeff_t

                if termA.many_body_order() > 0:
                    termA = termA / np.sqrt(coeffA)
                    self.fermi_ops.append(termA)

        pq = -1
        for p in range(0, self.n_orb):
            pa = 2 * p
            pb = 2 * p + 1

            for q in range(p, self.n_orb):
                qa = 2 * q
                qb = 2 * q + 1

                pq += 1

                rs = -1
                for r in range(0, self.n_orb):
                    ra = 2 * r
                    rb = 2 * r + 1

                    for s in range(r, self.n_orb):
                        sa = 2 * s
                        sb = 2 * s + 1

                        rs += 1

                        if (pq > rs):
                            continue

                        termA = FermionOperator(((ra, 1), (pa, 0), (sa, 1), (qa, 0)), 2 / np.sqrt(12))
                        termA += FermionOperator(((rb, 1), (pb, 0), (sb, 1), (qb, 0)), 2 / np.sqrt(12))
                        termA += FermionOperator(((ra, 1), (pa, 0), (sb, 1), (qb, 0)), 1 / np.sqrt(12))
                        termA += FermionOperator(((rb, 1), (pb, 0), (sa, 1), (qa, 0)), 1 / np.sqrt(12))
                        termA += FermionOperator(((ra, 1), (pb, 0), (sb, 1), (qa, 0)), 1 / np.sqrt(12))
                        termA += FermionOperator(((rb, 1), (pa, 0), (sa, 1), (qb, 0)), 1 / np.sqrt(12))

                        termB = FermionOperator(((ra, 1), (pa, 0), (sb, 1), (qb, 0)), 1 / 2.0)
                        termB += FermionOperator(((rb, 1), (pb, 0), (sa, 1), (qa, 0)), 1 / 2.0)
                        termB += FermionOperator(((ra, 1), (pb, 0), (sb, 1), (qa, 0)), -1 / 2.0)
                        termB += FermionOperator(((rb, 1), (pa, 0), (sa, 1), (qb, 0)), -1 / 2.0)

                        termA -= hermitian_conjugated(termA)
                        termB -= hermitian_conjugated(termB)

                        termA = normal_ordered(termA)
                        termB = normal_ordered(termB)

                        # Normalize
                        coeffA = 0

                        for t in termA.terms:
                            coeff_t = termA.terms[t]
                            coeffA += coeff_t * coeff_t

                        coeffB = 0

                        for t in termB.terms:
                            coeff_t = termB.terms[t]
                            coeffB += coeff_t * coeff_t

                        if termA.many_body_order() > 0:
                            termA = termA / np.sqrt(coeffA)
                            self.fermi_ops.append(termA)

                        if termB.many_body_order() > 0:
                            termB = termB / np.sqrt(coeffB)
                            self.fermi_ops.append(termB)

        self.n_ops = len(self.fermi_ops)
        print(" Number of fermionic operators: ", self.n_ops)

        n = self.n_spin_orb

        over_mat = np.zeros(shape=(self.n_ops, self.n_ops))
        vec = np.random.rand(2 ** self.n_spin_orb, 1)
        # print(vec)
        norm = 0

        for i in vec:
            norm += i * i

        vec = np.true_divide(vec, np.sqrt(norm))
        vec = scipy.sparse.csc_matrix(vec)

        self.spmat_ops = []
        print(" Generate Sparse Matrices for operators in pool")
        for op in self.fermi_ops:
            self.spmat_ops.append(transforms.get_sparse_operator(op, n_qubits=self.n_spin_orb))

        for i in range(self.n_ops):
            for j in range(self.n_ops):
                over_mat[i, j] = vec.transpose().conjugate().dot(self.spmat_ops[i].dot(self.spmat_ops[j].dot(vec)))[0, 0]

        rank = np.linalg.matrix_rank(over_mat)

        print("original rank =", rank)

        combined = list(zip(self.spmat_ops, self.fermi_ops))

        for i in range(100):
            random.shuffle(combined)
            self.spmat_ops[:], self.fermi_ops[:] = zip(*combined)
            self.spmat_trial = self.spmat_ops[0:20]
            over_mat = np.zeros(shape=(len(self.spmat_trial), len(self.spmat_trial)))
            vec = np.random.rand(2 ** self.n_spin_orb, 1)
            # print(vec)
            norm = 0
            for i in vec:
                norm += i * i
            vec = np.true_divide(vec, np.sqrt(norm))
            vec = scipy.sparse.csc_matrix(vec)
            for i in range(len(self.spmat_trial)):
                for j in range(len(self.spmat_trial)):
                    over_mat[i, j] = vec.transpose().conjugate().dot(self.spmat_trial[i].dot(self.spmat_trial[j].dot(vec)))[0, 0]
            rank = np.linalg.matrix_rank(over_mat)
            if rank == 20:
                self.spmat_ops = self.spmat_ops[0:20]
                self.fermi_ops = self.fermi_ops[0:20]
                break

        pool_vec = np.zeros((4 ** n,))

        for i in self.fermi_ops:
            pauli = openfermion.transforms.jordan_wigner(i)
            for line in pauli.terms:
                line = str(line)
                # print(line)
                Bin = np.zeros((2 * n,), dtype=int)
                X_pat_1 = re.compile("(\d{,2}), 'X'")
                X_1 = X_pat_1.findall(line)
                if X_1:
                    for i in X_1:
                        k = int(i)
                        Bin[n + k] = 1
                Y_pat_1 = re.compile("(\d{,2}), 'Y'")
                Y_1 = Y_pat_1.findall(line)
                if Y_1:
                    for i in Y_1:
                        k = int(i)
                        Bin[n + k] = 1
                        Bin[k] = 1
                Z_pat_1 = re.compile("(\d{,2}), 'Z'")
                Z_1 = Z_pat_1.findall(line)
                if Z_1:
                    for i in Z_1:
                        k = int(i)
                        Bin[k] = 1
                # print(Bin)
                index = int("".join(str(x) for x in Bin), 2)
                # print("index", index)

                pool_vec[index] = int(1)

        nz = np.nonzero(pool_vec)[0]

        # print("pauli pool size:", len(pool_vec[nz]))

        self.fermi_ops = []

        m = 2*n

        for i in nz:
            p = int(i)
            bi = bin(p)
            b_string = [int(j) for j in bi[2:].zfill(m)]
            pauli_string = ''
            flip = []
            for k in range(n):
                if b_string[k] == 0:
                    if b_string[k + n] == 1:
                        pauli_string += 'X%d ' % k
                        flip.append(k)
                if b_string[k] == 1:
                    if b_string[k + n] == 1:
                        pauli_string += 'Y%d ' % k
                        flip.append(k)
            flip.sort()
            z_string = list(range(flip[0] + 1,flip[1]))
            if len(flip) == 4:
                for i in range(flip[2] + 1, flip[3]):
                    z_string.append(i)
            print("Z string:", z_string)
            for i in z_string:
                b_string[i] += 1
                b_string[i] = b_string[i] % 2
            for k in range(n):
                if b_string[k] == 1:
                    if b_string[k + n] == 0:
                        pauli_string += 'Z%d ' % k
            A = QubitOperator(pauli_string, 0 + 1j)
            print("Pauli:", pauli_string)
            self.fermi_ops.append(A)

        self.n_ops = len(self.fermi_ops)
        print(" Number of pauli operators: ", self.n_ops)

        return

class anti_com(OperatorPool):
    def generate_SQ_Operators(self):

        self.bin_pool = []
        self.fermi_ops = []

        ii = 8

        for i in range(2 ** ii):
            b_string = [int(j) for j in bin(i)[2:].zfill(ii)]
            self.bin_pool.append(b_string)

        self.z = []
        self.x = []

        for i in self.bin_pool:
            for j in self.bin_pool:
                if sum(i[k] * j[k] for k in range(ii)) % 2 == 1:
                    self.z.append(i)
                    self.x.append(j)

        combined = list(zip(self.z, self.x))

        print("total number of antisymmetric ops :"+str(len(self.z)))

        print("Georgi terms")

        self.georgi_z = []
        self.georgi_x = []

        zz = 0

        for i in range(ii):
            zz += 2 ** i
            xx = 2 ** i
            b_string_z = [int(j) for j in bin(zz)[2:].zfill(ii)]
            b_string_x = [int(j) for j in bin(xx)[2:].zfill(ii)]
            self.georgi_z.append(b_string_z)
            self.georgi_x.append(b_string_x)

        for i in range(len(self.georgi_z)):
            print(self.georgi_z[i], self.georgi_x[i])

        self.pool_z = self.georgi_z
        self.pool_x = self.georgi_x

        max_len = 0

        for k in range(100):
            random.shuffle(combined)
            self.z[:], self.x[:] = zip(*combined)
            for i in range(len(self.z)):
                kk = 0
                for j in range(len(self.pool_z)):
                    z1 = [(self.z[i][k] + self.pool_z[j][k]) % 2 for k in range(ii)]
                    x1 = [(self.x[i][k] + self.pool_x[j][k]) % 2 for k in range(ii)]
                    if sum(z1[k] * x1[k] for k in range(ii)) % 2 == 0:
                        kk += 1
                if kk == 0:
                    self.pool_z.append(self.z[i])
                    self.pool_x.append(self.x[i])
            length = len(self.pool_z)
            if length > max_len:
                max_len = length
                self.georgi_z = self.pool_z
                self.georgi_x = self.pool_x

        print('size of set %12i' % (len(self.georgi_z)))

        for i in range(len(self.georgi_z)):
            print(self.georgi_z[i], self.georgi_x[i])

        for i in range(len(self.georgi_z)):
            pauli_string = ''
            for j in range(ii):
                if self.georgi_z[i][j] == 0:
                    if self.georgi_x[i][j] == 1:
                        pauli_string += 'X%d ' % j
                if self.georgi_z[i][j] == 1:
                    if self.georgi_x[i][j] == 0:
                        pauli_string += 'Z%d ' % j
                    else:
                        pauli_string += 'Y%d ' % j
            A = QubitOperator(pauli_string, 0+1j)
            print(self.georgi_z[i], self.georgi_x[i], A)
            self.fermi_ops.append(A)

        self.n_ops = len(self.fermi_ops)

class singlet_SD(OperatorPool):
    def generate_SQ_Operators(self):
        """
        0a,0b,1a,1b,2a,2b,3a,3b,....  -> 0,1,2,3,...
        """

        print(" Form singlet SD operators")
        self.fermi_ops = [] 
       
        n_occ = self.n_occ
        n_vir = self.n_vir
       
        for i in range(0,n_occ):
            ia = 2*i
            ib = 2*i+1

            for a in range(0,n_vir):
                aa = 2*n_occ + 2*a
                ab = 2*n_occ + 2*a+1
                    
                termA =  FermionOperator(((aa,1),(ia,0)), 1/np.sqrt(2))
                termA += FermionOperator(((ab,1),(ib,0)), 1/np.sqrt(2))
                
                termA -= hermitian_conjugated(termA)
                        
                termA = normal_ordered(termA)
               
                #Normalize
                coeffA = 0
                for t in termA.terms:
                    coeff_t = termA.terms[t]
                    coeffA += coeff_t * coeff_t
                
                if termA.many_body_order() > 0:
                    termA = termA/np.sqrt(coeffA)
                    self.fermi_ops.append(termA)
       

        for i in range(0,n_occ):
            ia = 2*i
            ib = 2*i+1

            for j in range(i,n_occ):
                ja = 2*j
                jb = 2*j+1
        
                for a in range(0,n_vir):
                    aa = 2*n_occ + 2*a
                    ab = 2*n_occ + 2*a+1

                    for b in range(a,n_vir):
                        ba = 2*n_occ + 2*b
                        bb = 2*n_occ + 2*b+1

                        termA =  FermionOperator(((aa,1),(ba,1),(ia,0),(ja,0)), 2/np.sqrt(12))
                        termA += FermionOperator(((ab,1),(bb,1),(ib,0),(jb,0)), 2/np.sqrt(12))
                        termA += FermionOperator(((aa,1),(bb,1),(ia,0),(jb,0)), 1/np.sqrt(12))
                        termA += FermionOperator(((ab,1),(ba,1),(ib,0),(ja,0)), 1/np.sqrt(12))
                        termA += FermionOperator(((aa,1),(bb,1),(ib,0),(ja,0)), 1/np.sqrt(12))
                        termA += FermionOperator(((ab,1),(ba,1),(ia,0),(jb,0)), 1/np.sqrt(12))
                                                                      
                        termB  = FermionOperator(((aa,1),(bb,1),(ia,0),(jb,0)), 1/2)
                        termB += FermionOperator(((ab,1),(ba,1),(ib,0),(ja,0)), 1/2)
                        termB += FermionOperator(((aa,1),(bb,1),(ib,0),(ja,0)), -1/2)
                        termB += FermionOperator(((ab,1),(ba,1),(ia,0),(jb,0)), -1/2)
                
                        termA -= hermitian_conjugated(termA)
                        termB -= hermitian_conjugated(termB)
               
                        termA = normal_ordered(termA)
                        termB = normal_ordered(termB)
                        
                        #Normalize
                        coeffA = 0
                        coeffB = 0
                        for t in termA.terms:
                            coeff_t = termA.terms[t]
                            coeffA += coeff_t * coeff_t
                        for t in termB.terms:
                            coeff_t = termB.terms[t]
                            coeffB += coeff_t * coeff_t

                        
                        if termA.many_body_order() > 0:
                            termA = termA/np.sqrt(coeffA)
                            self.fermi_ops.append(termA)
                        
                        if termB.many_body_order() > 0:
                            termB = termB/np.sqrt(coeffB)
                            self.fermi_ops.append(termB)
        
        self.n_ops = len(self.fermi_ops)
        print(" Number of operators: ", self.n_ops)
        return


class qubits(OperatorPool):
    def generate_SQ_Operators(self):

        self.fermi_ops = []

        n_occ = self.n_occ
        n_vir = self.n_vir

        for p in range(0, 2 * self.n_orb):
            X = QubitOperator('X%d' % p, 0 + 1j)
            Y = QubitOperator('Y%d' % p, 0 + 1j)

            # self.fermi_ops.append(Y)
            # self.fermi_ops.append(X)
        for p in range(0, 2 * self.n_orb):
            Z = QubitOperator('Z%d' % p, 0 + 1j)

            # self.fermi_ops.append(Z)

        for p in range(0, 2 * self.n_orb):

            for q in range(p + 1, 2 * self.n_orb):
                ZZ = QubitOperator('Z%d Z%d' % (p, q), 0 + 1j)
                # self.fermi_ops.append(ZZ)

        for p in range(0, 2 * self.n_orb):
            for q in range(p + 1, 2 * self.n_orb):
                XX = QubitOperator('X%d X%d' % (p, q), 0 + 1j)
                XY = QubitOperator('X%d Y%d' % (p, q), 0 + 1j)
                YX = QubitOperator('Y%d X%d' % (p, q), 0 + 1j)
                XZ = QubitOperator('X%d Z%d' % (p, q), 0 + 1j)
                YZ = QubitOperator('Y%d Z%d' % (p, q), 0 + 1j)
                YY = QubitOperator('Y%d Y%d' % (p, q), 0 + 1j)

                # self.fermi_ops.append(XX)
                self.fermi_ops.append(YX)
                # self.fermi_ops.append(XZ)
                # self.fermi_ops.append(YZ)
                # self.fermi_ops.append(YY)
                # self.fermi_ops.append(XY)

        for i in range(0, 2 * self.n_orb):
            for j in range(i + 1, 2 * self.n_orb):

                for k in range(j + 1, 2 * self.n_orb):
                    YXZ = QubitOperator('Y%d X%d Z%d' % (i, j, k), 1j)
                    XYZ = QubitOperator('X%d Y%d Z%d' % (i, j, k), 1j)
                    XZY = QubitOperator('X%d Z%d Y%d' % (i, j, k), 1j)
                    YZX = QubitOperator('Y%d Z%d X%d' % (i, j, k), 1j)
                    ZXY = QubitOperator('Z%d X%d Y%d' % (i, j, k), 1j)
                    ZYX = QubitOperator('Z%d X%d Y%d' % (i, j, k), 1j)
                    # ZZZ = QubitOperator('Z%d Z%d Z%d'% (i, j, k), 1j)

                    self.fermi_ops.append(XYZ)
                    # self.fermi_ops.append(YXZ)
                    self.fermi_ops.append(XZY)
                    # self.fermi_ops.append(YZX)
                    self.fermi_ops.append(ZXY)
                    # self.fermi_ops.append(ZYX)
                    # self.fermi_ops.append(ZZZ)

                    # XXY = QubitOperator('X%d X%d Y%d'% (i, j, k), 1j)
                    # XYY = QubitOperator('X%d Y%d Y%d'% (i, j, k), 1j)

                    # self.fermi_ops.append(XXY)
                    # self.fermi_ops.append(XYY)

        for p in range(0, 2 * self.n_orb):
            for q in range(p + 1, 2 * self.n_orb):
                for r in range(q + 1, 2 * self.n_orb):
                    for s in range(r + 1, 2 * self.n_orb):
                        XYYY = QubitOperator('X%d Y%d Y%d Y%d' % (p, q, r, s), 1j)
                        YXYY = QubitOperator('Y%d X%d Y%d Y%d' % (p, q, r, s), 1j)
                        YYXY = QubitOperator('Y%d Y%d X%d Y%d' % (p, q, r, s), 1j)
                        YYYX = QubitOperator('Y%d Y%d Y%d X%d' % (p, q, r, s), 1j)

                        self.fermi_ops.append(XYYY)
                        self.fermi_ops.append(YXYY)
                        self.fermi_ops.append(YYXY)
                        self.fermi_ops.append(YYYX)

                        XXXY = QubitOperator('X%d X%d X%d Y%d'% (p, q, r, s), 1j)
                        YXXX = QubitOperator('Y%d X%d X%d X%d'% (p, q, r, s), 1j)
                        XYXX = QubitOperator('X%d Y%d X%d X%d'% (p, q, r, s), 1j)
                        XXYX = QubitOperator('X%d X%d Y%d X%d'% (p, q, r, s), 1j)

                        self.fermi_ops.append(XXXY)
                        self.fermi_ops.append(XXYX)
                        self.fermi_ops.append(XYXX)
                        self.fermi_ops.append(YXXX)

                        # ZZXY = QubitOperator('Z%d Z%d X%d Y%d'% (p, q, r, s), 1j)
                        # ZZYX = QubitOperator('Z%d Z%d Y%d X%d'% (p, q, r, s), 1j)
                        # XZZY = QubitOperator('X%d Z%d Z%d Y%d'% (p, q, r, s), 1j)
                        # YZZX = QubitOperator('Y%d Z%d Z%d X%d'% (p, q, r, s), 1j)
                        # ZYXZ = QubitOperator('Z%d Y%d X%d Z%d'% (p, q, r, s), 1j)
                        # ZXYZ = QubitOperator('Z%d X%d Y%d Z%d'% (p, q, r, s), 1j)
                        # XYZZ = QubitOperator('X%d Y%d Z%d Z%d'% (p, q, r, s), 1j)
                        # YXZZ = QubitOperator('Y%d X%d Z%d Z%d'% (p, q, r, s), 1j)
                        # XZYZ = QubitOperator('X%d Z%d Y%d Z%d'% (p, q, r, s), 1j)
                        # YZXZ = QubitOperator('Y%d Z%d X%d Z%d'% (p, q, r, s), 1j)
                        # ZXZY = QubitOperator('Z%d X%d Z%d Y%d'% (p, q, r, s), 1j)
                        # ZYZX = QubitOperator('Z%d Y%d Z%d X%d'% (p, q, r, s), 1j)

                        # self.fermi_ops.append(ZZXY)
                        # self.fermi_ops.append(ZZYX)
                        # self.fermi_ops.append(XZZY)
                        # self.fermi_ops.append(YZZX)
                        # self.fermi_ops.append(ZXYZ)
                        # self.fermi_ops.append(ZYXZ)
                        # self.fermi_ops.append(XYZZ)
                        # self.fermi_ops.append(YXZZ)
                        # self.fermi_ops.append(XZYZ)
                        # self.fermi_ops.append(YZXZ)
                        # self.fermi_ops.append(ZXZY)
                        # self.fermi_ops.append(ZYZX)

                        # XXYY = QubitOperator('X%d X%d Y%d Y%d'% (p, q, r, s), 1j)
                        # XYXY = QubitOperator('X%d Y%d X%d Y%d'% (p, q, r, s), 1j)
                        # XYYX = QubitOperator('X%d Y%d Y%d X%d'% (p, q, r, s), 1j)
                        # YYXX = QubitOperator('Y%d Y%d X%d X%d'% (p, q, r, s), 1j)
                        # YXXY = QubitOperator('Y%d X%d X%d Y%d'% (p, q, r, s), 1j)
                        # YXYX = QubitOperator('Y%d X%d Y%d X%d'% (p, q, r, s), 1j)

                        # self.fermi_ops.append(XXYY)
                        # self.fermi_ops.append(XYXY)
                        # self.fermi_ops.append(XYYX)
                        # self.fermi_ops.append(YYXX)
                        # self.fermi_ops.append(YXXY)
                        # self.fermi_ops.append(YXYX)

                        # XXXX = QubitOperator('X%d X%d X%d X%d'% (p, q, r, s), 1j)
                        # YYYY = QubitOperator('Y%d Y%d Y%d Y%d'% (p, q, r, s), 1j)

                        # self.fermi_ops.append(XXXX)
                        # self.fermi_ops.append(YYYY)

        for op in self.fermi_ops:
            print(op)

        self.n_ops = len(self.fermi_ops)
        print(" Number of operators: ", self.n_ops)
        return

class qubits(OperatorPool):
    def generate_SQ_Operators(self):

        self.fermi_ops = []
        
        n_occ = self.n_occ
        n_vir = self.n_vir
                
        for p in range(0,2*self.n_orb):
            X = QubitOperator('X%d'% p, 0+1j)
            Y = QubitOperator('Y%d'% p, 0+1j)

            # self.fermi_ops.append(Y)
            # self.fermi_ops.append(X)
        for p in range(0,2*self.n_orb):
            Z = QubitOperator('Z%d'% p, 0+1j)

            # self.fermi_ops.append(Z)

        for p in range(0,2*self.n_orb):

            for q in range(p+1,2*self.n_orb):

                ZZ = QubitOperator('Z%d Z%d'% (p, q), 0+1j)
                # self.fermi_ops.append(ZZ)

        for p in range(0,2*self.n_orb):
            for q in range(p+1,2*self.n_orb):

                XX = QubitOperator('X%d X%d'% (p, q), 0+1j)
                XY = QubitOperator('X%d Y%d'% (p, q), 0+1j)
                YX = QubitOperator('Y%d X%d'% (p, q), 0+1j)
                XZ = QubitOperator('X%d Z%d'% (p, q), 0+1j)
                YZ = QubitOperator('Y%d Z%d'% (p, q), 0+1j)
                YY = QubitOperator('Y%d Y%d'% (p, q), 0+1j)

                # self.fermi_ops.append(XX)
                self.fermi_ops.append(YX)
                # self.fermi_ops.append(XZ)
                # self.fermi_ops.append(YZ)
                # self.fermi_ops.append(YY)
                # self.fermi_ops.append(XY) 


        for i in range(0,2*self.n_orb):
            for j in range(i+1, 2*self.n_orb):

                for k in range(j+1, 2*self.n_orb):

                    YXZ = QubitOperator('Y%d X%d Z%d'% (i, j, k), 1j)  
                    XYZ = QubitOperator('X%d Y%d Z%d'% (i, j, k), 1j)
                    XZY = QubitOperator('X%d Z%d Y%d'% (i, j, k), 1j)
                    YZX = QubitOperator('Y%d Z%d X%d'% (i, j, k), 1j)
                    ZXY = QubitOperator('Z%d X%d Y%d'% (i, j, k), 1j)
                    ZYX = QubitOperator('Z%d X%d Y%d'% (i, j, k), 1j)
                    # ZZZ = QubitOperator('Z%d Z%d Z%d'% (i, j, k), 1j)
  
                    self.fermi_ops.append(XYZ)
                    # self.fermi_ops.append(YXZ)
                    self.fermi_ops.append(XZY)
                    # self.fermi_ops.append(YZX)
                    self.fermi_ops.append(ZXY)
                    # self.fermi_ops.append(ZYX)
                    # self.fermi_ops.append(ZZZ)

                    # XXY = QubitOperator('X%d X%d Y%d'% (i, j, k), 1j)
                    # XYY = QubitOperator('X%d Y%d Y%d'% (i, j, k), 1j)

                    # self.fermi_ops.append(XXY)
                    # self.fermi_ops.append(XYY)

        for p in range(0,2*self.n_orb):
            for q in range(p+1,2*self.n_orb):
                for r in range(q+1,2*self.n_orb):
                    for s in range(r+1,2*self.n_orb):
                        XYYY = QubitOperator('X%d Y%d Y%d Y%d'% (p, q, r, s), 1j)
                        YXYY = QubitOperator('Y%d X%d Y%d Y%d'% (p, q, r, s), 1j)
                        YYXY = QubitOperator('Y%d Y%d X%d Y%d'% (p, q, r, s), 1j)
                        YYYX = QubitOperator('Y%d Y%d Y%d X%d'% (p, q, r, s), 1j)

                        self.fermi_ops.append(XYYY)
                        # self.fermi_ops.append(YXYY)
                        # self.fermi_ops.append(YYXY)
                        # self.fermi_ops.append(YYYX)

                        # XXXY = QubitOperator('X%d X%d X%d Y%d'% (p, q, r, s), 1j)
                        # YXXX = QubitOperator('Y%d X%d X%d X%d'% (p, q, r, s), 1j)
                        # XYXX = QubitOperator('X%d Y%d X%d X%d'% (p, q, r, s), 1j)
                        # XXYX = QubitOperator('X%d X%d Y%d X%d'% (p, q, r, s), 1j)

                        # self.fermi_ops.append(XXXY)
                        # self.fermi_ops.append(XXYX)
                        # self.fermi_ops.append(XYXX)
                        # self.fermi_ops.append(YXXX)

                        # ZZXY = QubitOperator('Z%d Z%d X%d Y%d'% (p, q, r, s), 1j)
                        # ZZYX = QubitOperator('Z%d Z%d Y%d X%d'% (p, q, r, s), 1j)
                        # XZZY = QubitOperator('X%d Z%d Z%d Y%d'% (p, q, r, s), 1j)
                        # YZZX = QubitOperator('Y%d Z%d Z%d X%d'% (p, q, r, s), 1j)
                        # ZYXZ = QubitOperator('Z%d Y%d X%d Z%d'% (p, q, r, s), 1j)
                        # ZXYZ = QubitOperator('Z%d X%d Y%d Z%d'% (p, q, r, s), 1j)
                        # XYZZ = QubitOperator('X%d Y%d Z%d Z%d'% (p, q, r, s), 1j)
                        # YXZZ = QubitOperator('Y%d X%d Z%d Z%d'% (p, q, r, s), 1j)
                        # XZYZ = QubitOperator('X%d Z%d Y%d Z%d'% (p, q, r, s), 1j)
                        # YZXZ = QubitOperator('Y%d Z%d X%d Z%d'% (p, q, r, s), 1j)
                        # ZXZY = QubitOperator('Z%d X%d Z%d Y%d'% (p, q, r, s), 1j)
                        # ZYZX = QubitOperator('Z%d Y%d Z%d X%d'% (p, q, r, s), 1j)

                        # self.fermi_ops.append(ZZXY)
                        # self.fermi_ops.append(ZZYX)
                        # self.fermi_ops.append(XZZY)
                        # self.fermi_ops.append(YZZX)
                        # self.fermi_ops.append(ZXYZ)
                        # self.fermi_ops.append(ZYXZ)
                        # self.fermi_ops.append(XYZZ)
                        # self.fermi_ops.append(YXZZ)
                        # self.fermi_ops.append(XZYZ)
                        # self.fermi_ops.append(YZXZ)
                        # self.fermi_ops.append(ZXZY)
                        # self.fermi_ops.append(ZYZX)

                        # XXYY = QubitOperator('X%d X%d Y%d Y%d'% (p, q, r, s), 1j)
                        # XYXY = QubitOperator('X%d Y%d X%d Y%d'% (p, q, r, s), 1j)
                        # XYYX = QubitOperator('X%d Y%d Y%d X%d'% (p, q, r, s), 1j)
                        # YYXX = QubitOperator('Y%d Y%d X%d X%d'% (p, q, r, s), 1j)
                        # YXXY = QubitOperator('Y%d X%d X%d Y%d'% (p, q, r, s), 1j)
                        # YXYX = QubitOperator('Y%d X%d Y%d X%d'% (p, q, r, s), 1j)

                        # self.fermi_ops.append(XXYY)
                        # self.fermi_ops.append(XYXY)
                        # self.fermi_ops.append(XYYX)
                        # self.fermi_ops.append(YYXX)
                        # self.fermi_ops.append(YXXY)
                        # self.fermi_ops.append(YXYX)

                        # XXXX = QubitOperator('X%d X%d X%d X%d'% (p, q, r, s), 1j)
                        # YYYY = QubitOperator('Y%d Y%d Y%d Y%d'% (p, q, r, s), 1j)

                        # self.fermi_ops.append(XXXX)
                        # self.fermi_ops.append(YYYY)                        
 
                
        for op in self.fermi_ops:
            print(op)

        self.n_ops = len(self.fermi_ops)
        print(" Number of operators: ", self.n_ops)        
        return

class rand_half_qubits(OperatorPool):
    def generate_SQ_Operators(self):

        self.fermi_ops_1 = []
        self.fermi_ops_2 = []
        self.fermi_ops_3 = []
        self.fermi_ops_4 = []
        self.fermi_ops = []
        
        n_occ = self.n_occ
        n_vir = self.n_vir
                
        for p in range(0,2*self.n_orb):
            X = QubitOperator('X%d'% p, 0+1j)
            Y = QubitOperator('Y%d'% p, 0+1j)

            self.fermi_ops_1.append(X)
            self.fermi_ops_1.append(Y)
        for p in range(0,2*self.n_orb):
            Z = QubitOperator('Z%d'% p, 0+1j)

            self.fermi_ops_1.append(Z)

        for p in range(0,2*self.n_orb):

            for q in range(p+1,2*self.n_orb):

                ZZ = QubitOperator('Z%d Z%d'% (p, q), 0+1j)
                self.fermi_ops_2.append(ZZ)

        for p in range(0,2*self.n_orb):
            for q in range(p+1,2*self.n_orb):

                XX = QubitOperator('X%d X%d'% (p, q), 0+1j)
                XY = QubitOperator('X%d Y%d'% (p, q), 0+1j)
                YX = QubitOperator('Y%d X%d'% (p, q), 0+1j)
                XZ = QubitOperator('X%d Z%d'% (p, q), 0+1j)
                YZ = QubitOperator('Y%d Z%d'% (p, q), 0+1j)
                YY = QubitOperator('Y%d Y%d'% (p, q), 0+1j)

                self.fermi_ops_2.append(XX)
                self.fermi_ops_2.append(YX)
                self.fermi_ops_2.append(XZ)
                self.fermi_ops_2.append(YZ)
                self.fermi_ops_2.append(YY)
                self.fermi_ops_2.append(XY) 


        for i in range(0,2*self.n_orb):
            for j in range(i+1, 2*self.n_orb):

                for k in range(j+1, 2*self.n_orb):

                    YXZ = QubitOperator('Y%d X%d Z%d'% (i, j, k), 1j)  
                    # XYZ = QubitOperator('X%d Y%d Z%d'% (i, j, k), 1j)
                    XZY = QubitOperator('X%d Z%d Y%d'% (i, j, k), 1j)
                    # YZX = QubitOperator('Y%d Z%d X%d'% (i, j, k), 1j)
                    ZXY = QubitOperator('Z%d X%d Y%d'% (i, j, k), 1j)
                    # ZYX = QubitOperator('Z%d X%d Y%d'% (i, j, k), 1j)
                    # ZZZ = QubitOperator('Z%d Z%d Z%d'% (i, j, k), 1j)
  
                    # self.fermi_ops.append(XYZ)
                    self.fermi_ops_3.append(YXZ)
                    self.fermi_ops_3.append(XZY)
                    # self.fermi_ops.append(YZX)
                    self.fermi_ops_3.append(ZXY)
                    # self.fermi_ops.append(ZYX)
                    # self.fermi_ops.append(ZZZ)

                    # XXY = QubitOperator('X%d X%d Y%d'% (i, j, k), 1j)
                    # XYY = QubitOperator('X%d Y%d Y%d'% (i, j, k), 1j)

                    # self.fermi_ops.append(XXY)
                    # self.fermi_ops.append(XYY)

        for p in range(0,2*self.n_orb):
            for q in range(p+1,2*self.n_orb):
                for r in range(q+1,2*self.n_orb):
                    for s in range(r+1,2*self.n_orb):
                        XYYY = QubitOperator('X%d Y%d Y%d Y%d'% (p, q, r, s), 1j)
                        YXYY = QubitOperator('Y%d X%d Y%d Y%d'% (p, q, r, s), 1j)
                        YYXY = QubitOperator('Y%d Y%d X%d Y%d'% (p, q, r, s), 1j)
                        YYYX = QubitOperator('Y%d Y%d Y%d X%d'% (p, q, r, s), 1j)

                        self.fermi_ops_4.append(XYYY)
                        self.fermi_ops_4.append(YXYY)
                        self.fermi_ops_4.append(YYXY)
                        self.fermi_ops_4.append(YYYX)

                        # XXXY = QubitOperator('X%d X%d X%d Y%d'% (p, q, r, s), 1j)
                        # YXXX = QubitOperator('Y%d X%d X%d X%d'% (p, q, r, s), 1j)
                        # XYXX = QubitOperator('X%d Y%d X%d X%d'% (p, q, r, s), 1j)
                        # XXYX = QubitOperator('X%d X%d Y%d X%d'% (p, q, r, s), 1j)

                        # self.fermi_ops.append(XXXY)
                        # self.fermi_ops.append(XXYX)
                        # self.fermi_ops.append(XYXX)
                        # self.fermi_ops.append(YXXX)

                        # ZZXY = QubitOperator('Z%d Z%d X%d Y%d'% (p, q, r, s), 1j)
                        # ZZYX = QubitOperator('Z%d Z%d Y%d X%d'% (p, q, r, s), 1j)
                        # XZZY = QubitOperator('X%d Z%d Z%d Y%d'% (p, q, r, s), 1j)
                        # YZZX = QubitOperator('Y%d Z%d Z%d X%d'% (p, q, r, s), 1j)
                        # ZYXZ = QubitOperator('Z%d Y%d X%d Z%d'% (p, q, r, s), 1j)
                        # ZXYZ = QubitOperator('Z%d X%d Y%d Z%d'% (p, q, r, s), 1j)
                        # XYZZ = QubitOperator('X%d Y%d Z%d Z%d'% (p, q, r, s), 1j)
                        # YXZZ = QubitOperator('Y%d X%d Z%d Z%d'% (p, q, r, s), 1j)
                        # XZYZ = QubitOperator('X%d Z%d Y%d Z%d'% (p, q, r, s), 1j)
                        # YZXZ = QubitOperator('Y%d Z%d X%d Z%d'% (p, q, r, s), 1j)
                        # ZXZY = QubitOperator('Z%d X%d Z%d Y%d'% (p, q, r, s), 1j)
                        # ZYZX = QubitOperator('Z%d Y%d Z%d X%d'% (p, q, r, s), 1j)

                        # self.fermi_ops.append(ZZXY)
                        # self.fermi_ops.append(ZZYX)
                        # self.fermi_ops.append(XZZY)
                        # self.fermi_ops.append(YZZX)
                        # self.fermi_ops.append(ZXYZ)
                        # self.fermi_ops.append(ZYXZ)
                        # self.fermi_ops.append(XYZZ)
                        # self.fermi_ops.append(YXZZ)
                        # self.fermi_ops.append(XZYZ)
                        # self.fermi_ops.append(YZXZ)
                        # self.fermi_ops.append(ZXZY)
                        # self.fermi_ops.append(ZYZX)

                        # XXYY = QubitOperator('X%d X%d Y%d Y%d'% (p, q, r, s), 1j)
                        # XYXY = QubitOperator('X%d Y%d X%d Y%d'% (p, q, r, s), 1j)
                        # XYYX = QubitOperator('X%d Y%d Y%d X%d'% (p, q, r, s), 1j)
                        # YYXX = QubitOperator('Y%d Y%d X%d X%d'% (p, q, r, s), 1j)
                        # YXXY = QubitOperator('Y%d X%d X%d Y%d'% (p, q, r, s), 1j)
                        # YXYX = QubitOperator('Y%d X%d Y%d X%d'% (p, q, r, s), 1j)

                        # self.fermi_ops.append(XXYY)
                        # self.fermi_ops.append(XYXY)
                        # self.fermi_ops.append(XYYX)
                        # self.fermi_ops.append(YYXX)
                        # self.fermi_ops.append(YXXY)
                        # self.fermi_ops.append(YXYX)

                        # XXXX = QubitOperator('X%d X%d X%d X%d'% (p, q, r, s), 1j)
                        # YYYY = QubitOperator('Y%d Y%d Y%d Y%d'% (p, q, r, s), 1j)

                        # self.fermi_ops.append(XXXX)
                        # self.fermi_ops.append(YYYY)                        
        
        np.random.shuffle(self.fermi_ops_1)
        for op in self.fermi_ops_1[:len(self.fermi_ops_1)//2]:
            self.fermi_ops.append(op)

        np.random.shuffle(self.fermi_ops_2)
        for op in self.fermi_ops_2[:len(self.fermi_ops_2)//2]:
            self.fermi_ops.append(op)

        np.random.shuffle(self.fermi_ops_3)
        for op in self.fermi_ops_3[:len(self.fermi_ops_3)//2]:
            self.fermi_ops.append(op)

        np.random.shuffle(self.fermi_ops_4)
        for op in self.fermi_ops_4[:len(self.fermi_ops_4)//2]:
            self.fermi_ops.append(op)

        for op in self.fermi_ops:
            print(op)

        self.n_ops = len(self.fermi_ops)
        print(" Number of operators: ", self.n_ops)        
        return

class rand_hh_qubits(OperatorPool):
    def generate_SQ_Operators(self):

        self.fermi_ops_1 = []
        self.fermi_ops_2 = []
        self.fermi_ops_3 = []
        self.fermi_ops_4 = []
        self.fermi_ops = []
        
        n_occ = self.n_occ
        n_vir = self.n_vir
                
        for p in range(0,2*self.n_orb):
            X = QubitOperator('X%d'% p, 0+1j)
            Y = QubitOperator('Y%d'% p, 0+1j)

            # self.fermi_ops_1.append(X)
            # self.fermi_ops_1.append(Y)
        for p in range(0,2*self.n_orb):
            Z = QubitOperator('Z%d'% p, 0+1j)

            # self.fermi_ops_1.append(Z)

        for p in range(0,2*self.n_orb):

            for q in range(p+1,2*self.n_orb):

                ZZ = QubitOperator('Z%d Z%d'% (p, q), 0+1j)
                # self.fermi_ops_2.append(ZZ)

        for p in range(0,2*self.n_orb):
            for q in range(p+1,2*self.n_orb):

                XX = QubitOperator('X%d X%d'% (p, q), 0+1j)
                XY = QubitOperator('X%d Y%d'% (p, q), 0+1j)
                YX = QubitOperator('Y%d X%d'% (p, q), 0+1j)
                XZ = QubitOperator('X%d Z%d'% (p, q), 0+1j)
                YZ = QubitOperator('Y%d Z%d'% (p, q), 0+1j)
                YY = QubitOperator('Y%d Y%d'% (p, q), 0+1j)

                # self.fermi_ops_2.append(XX)
                self.fermi_ops_2.append(YX)
                # self.fermi_ops_2.append(XZ)
                # self.fermi_ops_2.append(YZ)
                # self.fermi_ops_2.append(YY)
                self.fermi_ops_2.append(XY) 


        for i in range(0,2*self.n_orb):
            for j in range(i+1, 2*self.n_orb):

                for k in range(j+1, 2*self.n_orb):

                    YXZ = QubitOperator('Y%d X%d Z%d'% (i, j, k), 1j)
                    XYZ = QubitOperator('X%d Y%d Z%d'% (i, j, k), 1j)
                    XZY = QubitOperator('X%d Z%d Y%d'% (i, j, k), 1j)
                    YZX = QubitOperator('Y%d Z%d X%d'% (i, j, k), 1j)
                    ZXY = QubitOperator('Z%d X%d Y%d'% (i, j, k), 1j)
                    ZYX = QubitOperator('Z%d X%d Y%d'% (i, j, k), 1j)
                    # ZZZ = QubitOperator('Z%d Z%d Z%d'% (i, j, k), 1j)
  
                    self.fermi_ops_3.append(XYZ)
                    self.fermi_ops_3.append(YXZ)
                    self.fermi_ops_3.append(XZY)
                    self.fermi_ops_3.append(YZX)
                    self.fermi_ops_3.append(ZXY)
                    self.fermi_ops_3.append(ZYX)
                    # self.fermi_ops.append(ZZZ)

                    # XXY = QubitOperator('X%d X%d Y%d'% (i, j, k), 1j)
                    # XYY = QubitOperator('X%d Y%d Y%d'% (i, j, k), 1j)

                    # self.fermi_ops.append(XXY)
                    # self.fermi_ops.append(XYY)

        for p in range(0,2*self.n_orb):
            for q in range(p+1,2*self.n_orb):
                for r in range(q+1,2*self.n_orb):
                    for s in range(r+1,2*self.n_orb):
                        XYYY = QubitOperator('X%d Y%d Y%d Y%d'% (p, q, r, s), 1j)
                        YXYY = QubitOperator('Y%d X%d Y%d Y%d'% (p, q, r, s), 1j)
                        YYXY = QubitOperator('Y%d Y%d X%d Y%d'% (p, q, r, s), 1j)
                        YYYX = QubitOperator('Y%d Y%d Y%d X%d'% (p, q, r, s), 1j)

                        self.fermi_ops_4.append(XYYY)
                        self.fermi_ops_4.append(YXYY)
                        self.fermi_ops_4.append(YYXY)
                        self.fermi_ops_4.append(YYYX)

                        # XXXY = QubitOperator('X%d X%d X%d Y%d'% (p, q, r, s), 1j)
                        # YXXX = QubitOperator('Y%d X%d X%d X%d'% (p, q, r, s), 1j)
                        # XYXX = QubitOperator('X%d Y%d X%d X%d'% (p, q, r, s), 1j)
                        # XXYX = QubitOperator('X%d X%d Y%d X%d'% (p, q, r, s), 1j)

                        # self.fermi_ops.append(XXXY)
                        # self.fermi_ops.append(XXYX)
                        # self.fermi_ops.append(XYXX)
                        # self.fermi_ops.append(YXXX)

                        # ZZXY = QubitOperator('Z%d Z%d X%d Y%d'% (p, q, r, s), 1j)
                        # ZZYX = QubitOperator('Z%d Z%d Y%d X%d'% (p, q, r, s), 1j)
                        # XZZY = QubitOperator('X%d Z%d Z%d Y%d'% (p, q, r, s), 1j)
                        # YZZX = QubitOperator('Y%d Z%d Z%d X%d'% (p, q, r, s), 1j)
                        # ZYXZ = QubitOperator('Z%d Y%d X%d Z%d'% (p, q, r, s), 1j)
                        # ZXYZ = QubitOperator('Z%d X%d Y%d Z%d'% (p, q, r, s), 1j)
                        # XYZZ = QubitOperator('X%d Y%d Z%d Z%d'% (p, q, r, s), 1j)
                        # YXZZ = QubitOperator('Y%d X%d Z%d Z%d'% (p, q, r, s), 1j)
                        # XZYZ = QubitOperator('X%d Z%d Y%d Z%d'% (p, q, r, s), 1j)
                        # YZXZ = QubitOperator('Y%d Z%d X%d Z%d'% (p, q, r, s), 1j)
                        # ZXZY = QubitOperator('Z%d X%d Z%d Y%d'% (p, q, r, s), 1j)
                        # ZYZX = QubitOperator('Z%d Y%d Z%d X%d'% (p, q, r, s), 1j)

                        # self.fermi_ops.append(ZZXY)
                        # self.fermi_ops.append(ZZYX)
                        # self.fermi_ops.append(XZZY)
                        # self.fermi_ops.append(YZZX)
                        # self.fermi_ops.append(ZXYZ)
                        # self.fermi_ops.append(ZYXZ)
                        # self.fermi_ops.append(XYZZ)
                        # self.fermi_ops.append(YXZZ)
                        # self.fermi_ops.append(XZYZ)
                        # self.fermi_ops.append(YZXZ)
                        # self.fermi_ops.append(ZXZY)
                        # self.fermi_ops.append(ZYZX)

                        # XXYY = QubitOperator('X%d X%d Y%d Y%d'% (p, q, r, s), 1j)
                        # XYXY = QubitOperator('X%d Y%d X%d Y%d'% (p, q, r, s), 1j)
                        # XYYX = QubitOperator('X%d Y%d Y%d X%d'% (p, q, r, s), 1j)
                        # YYXX = QubitOperator('Y%d Y%d X%d X%d'% (p, q, r, s), 1j)
                        # YXXY = QubitOperator('Y%d X%d X%d Y%d'% (p, q, r, s), 1j)
                        # YXYX = QubitOperator('Y%d X%d Y%d X%d'% (p, q, r, s), 1j)

                        # self.fermi_ops.append(XXYY)
                        # self.fermi_ops.append(XYXY)
                        # self.fermi_ops.append(XYYX)
                        # self.fermi_ops.append(YYXX)
                        # self.fermi_ops.append(YXXY)
                        # self.fermi_ops.append(YXYX)

                        # XXXX = QubitOperator('X%d X%d X%d X%d'% (p, q, r, s), 1j)
                        # YYYY = QubitOperator('Y%d Y%d Y%d Y%d'% (p, q, r, s), 1j)

                        # self.fermi_ops.append(XXXX)
                        # self.fermi_ops.append(YYYY)                        
        
        # np.random.shuffle(self.fermi_ops_1)
        # for op in self.fermi_ops_1[:len(self.fermi_ops_1)//4]:
        #     self.fermi_ops.append(op)

        print("original size", len(self.fermi_ops_2)+len(self.fermi_ops_3)+len(self.fermi_ops_4))

        np.random.shuffle(self.fermi_ops_2)
        for op in self.fermi_ops_2[:len(self.fermi_ops_2)//4]:
            self.fermi_ops.append(op)

        np.random.shuffle(self.fermi_ops_3)
        for op in self.fermi_ops_3[:len(self.fermi_ops_3)//4]:
            self.fermi_ops.append(op)

        np.random.shuffle(self.fermi_ops_4)
        for op in self.fermi_ops_4[:len(self.fermi_ops_4)//4]:
            self.fermi_ops.append(op)

        for op in self.fermi_ops:
            print(op)

        self.n_ops = len(self.fermi_ops)
        print(" Number of operators: ", self.n_ops)        
        return


class rand_hhh_qubits(OperatorPool):
    def generate_SQ_Operators(self):

        self.fermi_ops_1 = []
        self.fermi_ops_2 = []
        self.fermi_ops_3 = []
        self.fermi_ops_4 = []
        self.fermi_ops = []

        n_occ = self.n_occ
        n_vir = self.n_vir

        for p in range(0, 2 * self.n_orb):
            X = QubitOperator('X%d' % p, 0 + 1j)
            Y = QubitOperator('Y%d' % p, 0 + 1j)

            # self.fermi_ops_1.append(X)
            # self.fermi_ops_1.append(Y)
        for p in range(0, 2 * self.n_orb):
            Z = QubitOperator('Z%d' % p, 0 + 1j)

            # self.fermi_ops_1.append(Z)

        for p in range(0, 2 * self.n_orb):

            for q in range(p + 1, 2 * self.n_orb):
                ZZ = QubitOperator('Z%d Z%d' % (p, q), 0 + 1j)
                # self.fermi_ops_2.append(ZZ)

        for p in range(0, 2 * self.n_orb):
            for q in range(p + 1, 2 * self.n_orb):
                XX = QubitOperator('X%d X%d' % (p, q), 0 + 1j)
                XY = QubitOperator('X%d Y%d' % (p, q), 0 + 1j)
                YX = QubitOperator('Y%d X%d' % (p, q), 0 + 1j)
                XZ = QubitOperator('X%d Z%d' % (p, q), 0 + 1j)
                YZ = QubitOperator('Y%d Z%d' % (p, q), 0 + 1j)
                YY = QubitOperator('Y%d Y%d' % (p, q), 0 + 1j)

                # self.fermi_ops_2.append(XX)
                self.fermi_ops_2.append(YX)
                # self.fermi_ops_2.append(XZ)
                # self.fermi_ops_2.append(YZ)
                # self.fermi_ops_2.append(YY)
                self.fermi_ops_2.append(XY)

        for i in range(0, 2 * self.n_orb):
            for j in range(i + 1, 2 * self.n_orb):

                for k in range(j + 1, 2 * self.n_orb):
                    YXZ = QubitOperator('Y%d X%d Z%d' % (i, j, k), 1j)
                    XYZ = QubitOperator('X%d Y%d Z%d' % (i, j, k), 1j)
                    XZY = QubitOperator('X%d Z%d Y%d' % (i, j, k), 1j)
                    YZX = QubitOperator('Y%d Z%d X%d' % (i, j, k), 1j)
                    ZXY = QubitOperator('Z%d X%d Y%d' % (i, j, k), 1j)
                    ZYX = QubitOperator('Z%d X%d Y%d' % (i, j, k), 1j)
                    # ZZZ = QubitOperator('Z%d Z%d Z%d'% (i, j, k), 1j)

                    self.fermi_ops_3.append(XYZ)
                    self.fermi_ops_3.append(YXZ)
                    self.fermi_ops_3.append(XZY)
                    self.fermi_ops_3.append(YZX)
                    self.fermi_ops_3.append(ZXY)
                    self.fermi_ops_3.append(ZYX)
                    # self.fermi_ops.append(ZZZ)

                    # XXY = QubitOperator('X%d X%d Y%d'% (i, j, k), 1j)
                    # XYY = QubitOperator('X%d Y%d Y%d'% (i, j, k), 1j)

                    # self.fermi_ops.append(XXY)
                    # self.fermi_ops.append(XYY)

        for p in range(0, 2 * self.n_orb):
            for q in range(p + 1, 2 * self.n_orb):
                for r in range(q + 1, 2 * self.n_orb):
                    for s in range(r + 1, 2 * self.n_orb):
                        XYYY = QubitOperator('X%d Y%d Y%d Y%d' % (p, q, r, s), 1j)
                        YXYY = QubitOperator('Y%d X%d Y%d Y%d' % (p, q, r, s), 1j)
                        YYXY = QubitOperator('Y%d Y%d X%d Y%d' % (p, q, r, s), 1j)
                        YYYX = QubitOperator('Y%d Y%d Y%d X%d' % (p, q, r, s), 1j)

                        self.fermi_ops_4.append(XYYY)
                        self.fermi_ops_4.append(YXYY)
                        self.fermi_ops_4.append(YYXY)
                        self.fermi_ops_4.append(YYYX)

                        # XXXY = QubitOperator('X%d X%d X%d Y%d'% (p, q, r, s), 1j)
                        # YXXX = QubitOperator('Y%d X%d X%d X%d'% (p, q, r, s), 1j)
                        # XYXX = QubitOperator('X%d Y%d X%d X%d'% (p, q, r, s), 1j)
                        # XXYX = QubitOperator('X%d X%d Y%d X%d'% (p, q, r, s), 1j)

                        # self.fermi_ops.append(XXXY)
                        # self.fermi_ops.append(XXYX)
                        # self.fermi_ops.append(XYXX)
                        # self.fermi_ops.append(YXXX)

                        # ZZXY = QubitOperator('Z%d Z%d X%d Y%d'% (p, q, r, s), 1j)
                        # ZZYX = QubitOperator('Z%d Z%d Y%d X%d'% (p, q, r, s), 1j)
                        # XZZY = QubitOperator('X%d Z%d Z%d Y%d'% (p, q, r, s), 1j)
                        # YZZX = QubitOperator('Y%d Z%d Z%d X%d'% (p, q, r, s), 1j)
                        # ZYXZ = QubitOperator('Z%d Y%d X%d Z%d'% (p, q, r, s), 1j)
                        # ZXYZ = QubitOperator('Z%d X%d Y%d Z%d'% (p, q, r, s), 1j)
                        # XYZZ = QubitOperator('X%d Y%d Z%d Z%d'% (p, q, r, s), 1j)
                        # YXZZ = QubitOperator('Y%d X%d Z%d Z%d'% (p, q, r, s), 1j)
                        # XZYZ = QubitOperator('X%d Z%d Y%d Z%d'% (p, q, r, s), 1j)
                        # YZXZ = QubitOperator('Y%d Z%d X%d Z%d'% (p, q, r, s), 1j)
                        # ZXZY = QubitOperator('Z%d X%d Z%d Y%d'% (p, q, r, s), 1j)
                        # ZYZX = QubitOperator('Z%d Y%d Z%d X%d'% (p, q, r, s), 1j)

                        # self.fermi_ops.append(ZZXY)
                        # self.fermi_ops.append(ZZYX)
                        # self.fermi_ops.append(XZZY)
                        # self.fermi_ops.append(YZZX)
                        # self.fermi_ops.append(ZXYZ)
                        # self.fermi_ops.append(ZYXZ)
                        # self.fermi_ops.append(XYZZ)
                        # self.fermi_ops.append(YXZZ)
                        # self.fermi_ops.append(XZYZ)
                        # self.fermi_ops.append(YZXZ)
                        # self.fermi_ops.append(ZXZY)
                        # self.fermi_ops.append(ZYZX)

                        # XXYY = QubitOperator('X%d X%d Y%d Y%d'% (p, q, r, s), 1j)
                        # XYXY = QubitOperator('X%d Y%d X%d Y%d'% (p, q, r, s), 1j)
                        # XYYX = QubitOperator('X%d Y%d Y%d X%d'% (p, q, r, s), 1j)
                        # YYXX = QubitOperator('Y%d Y%d X%d X%d'% (p, q, r, s), 1j)
                        # YXXY = QubitOperator('Y%d X%d X%d Y%d'% (p, q, r, s), 1j)
                        # YXYX = QubitOperator('Y%d X%d Y%d X%d'% (p, q, r, s), 1j)

                        # self.fermi_ops.append(XXYY)
                        # self.fermi_ops.append(XYXY)
                        # self.fermi_ops.append(XYYX)
                        # self.fermi_ops.append(YYXX)
                        # self.fermi_ops.append(YXXY)
                        # self.fermi_ops.append(YXYX)

                        # XXXX = QubitOperator('X%d X%d X%d X%d'% (p, q, r, s), 1j)
                        # YYYY = QubitOperator('Y%d Y%d Y%d Y%d'% (p, q, r, s), 1j)

                        # self.fermi_ops.append(XXXX)
                        # self.fermi_ops.append(YYYY)

        # np.random.shuffle(self.fermi_ops_1)
        # for op in self.fermi_ops_1[:len(self.fermi_ops_1)//4]:
        #     self.fermi_ops.append(op)

        print("original size", len(self.fermi_ops_2) + len(self.fermi_ops_3) + len(self.fermi_ops_4))

        np.random.shuffle(self.fermi_ops_2)
        for op in self.fermi_ops_2[:len(self.fermi_ops_2) // 8]:
            self.fermi_ops.append(op)

        np.random.shuffle(self.fermi_ops_3)
        for op in self.fermi_ops_3[:len(self.fermi_ops_3) // 8]:
            self.fermi_ops.append(op)

        np.random.shuffle(self.fermi_ops_4)
        for op in self.fermi_ops_4[:len(self.fermi_ops_4) // 8]:
            self.fermi_ops.append(op)

        for op in self.fermi_ops:
            print(op)

        self.n_ops = len(self.fermi_ops)
        print(" Number of operators: ", self.n_ops)
        return

class rand_hhhh_qubits(OperatorPool):
    def generate_SQ_Operators(self):

        self.fermi_ops_1 = []
        self.fermi_ops_2 = []
        self.fermi_ops_3 = []
        self.fermi_ops_4 = []
        self.fermi_ops = []

        n_occ = self.n_occ
        n_vir = self.n_vir

        for p in range(0, 2 * self.n_orb):
            X = QubitOperator('X%d' % p, 0 + 1j)
            Y = QubitOperator('Y%d' % p, 0 + 1j)

            # self.fermi_ops_1.append(X)
            # self.fermi_ops_1.append(Y)
        for p in range(0, 2 * self.n_orb):
            Z = QubitOperator('Z%d' % p, 0 + 1j)

            # self.fermi_ops_1.append(Z)

        for p in range(0, 2 * self.n_orb):

            for q in range(p + 1, 2 * self.n_orb):
                ZZ = QubitOperator('Z%d Z%d' % (p, q), 0 + 1j)
                # self.fermi_ops_2.append(ZZ)

        for p in range(0, 2 * self.n_orb):
            for q in range(p + 1, 2 * self.n_orb):
                XX = QubitOperator('X%d X%d' % (p, q), 0 + 1j)
                XY = QubitOperator('X%d Y%d' % (p, q), 0 + 1j)
                YX = QubitOperator('Y%d X%d' % (p, q), 0 + 1j)
                XZ = QubitOperator('X%d Z%d' % (p, q), 0 + 1j)
                YZ = QubitOperator('Y%d Z%d' % (p, q), 0 + 1j)
                YY = QubitOperator('Y%d Y%d' % (p, q), 0 + 1j)

                # self.fermi_ops_2.append(XX)
                self.fermi_ops_2.append(YX)
                # self.fermi_ops_2.append(XZ)
                # self.fermi_ops_2.append(YZ)
                # self.fermi_ops_2.append(YY)
                self.fermi_ops_2.append(XY)

        for i in range(0, 2 * self.n_orb):
            for j in range(i + 1, 2 * self.n_orb):

                for k in range(j + 1, 2 * self.n_orb):
                    YXZ = QubitOperator('Y%d X%d Z%d' % (i, j, k), 1j)
                    XYZ = QubitOperator('X%d Y%d Z%d' % (i, j, k), 1j)
                    XZY = QubitOperator('X%d Z%d Y%d' % (i, j, k), 1j)
                    YZX = QubitOperator('Y%d Z%d X%d' % (i, j, k), 1j)
                    ZXY = QubitOperator('Z%d X%d Y%d' % (i, j, k), 1j)
                    ZYX = QubitOperator('Z%d X%d Y%d' % (i, j, k), 1j)
                    # ZZZ = QubitOperator('Z%d Z%d Z%d'% (i, j, k), 1j)

                    self.fermi_ops_3.append(XYZ)
                    self.fermi_ops_3.append(YXZ)
                    self.fermi_ops_3.append(XZY)
                    self.fermi_ops_3.append(YZX)
                    self.fermi_ops_3.append(ZXY)
                    self.fermi_ops_3.append(ZYX)
                    # self.fermi_ops.append(ZZZ)

                    # XXY = QubitOperator('X%d X%d Y%d'% (i, j, k), 1j)
                    # XYY = QubitOperator('X%d Y%d Y%d'% (i, j, k), 1j)

                    # self.fermi_ops.append(XXY)
                    # self.fermi_ops.append(XYY)

        for p in range(0, 2 * self.n_orb):
            for q in range(p + 1, 2 * self.n_orb):
                for r in range(q + 1, 2 * self.n_orb):
                    for s in range(r + 1, 2 * self.n_orb):
                        XYYY = QubitOperator('X%d Y%d Y%d Y%d' % (p, q, r, s), 1j)
                        YXYY = QubitOperator('Y%d X%d Y%d Y%d' % (p, q, r, s), 1j)
                        YYXY = QubitOperator('Y%d Y%d X%d Y%d' % (p, q, r, s), 1j)
                        YYYX = QubitOperator('Y%d Y%d Y%d X%d' % (p, q, r, s), 1j)

                        self.fermi_ops_4.append(XYYY)
                        self.fermi_ops_4.append(YXYY)
                        self.fermi_ops_4.append(YYXY)
                        self.fermi_ops_4.append(YYYX)

                        # XXXY = QubitOperator('X%d X%d X%d Y%d'% (p, q, r, s), 1j)
                        # YXXX = QubitOperator('Y%d X%d X%d X%d'% (p, q, r, s), 1j)
                        # XYXX = QubitOperator('X%d Y%d X%d X%d'% (p, q, r, s), 1j)
                        # XXYX = QubitOperator('X%d X%d Y%d X%d'% (p, q, r, s), 1j)

                        # self.fermi_ops.append(XXXY)
                        # self.fermi_ops.append(XXYX)
                        # self.fermi_ops.append(XYXX)
                        # self.fermi_ops.append(YXXX)

                        # ZZXY = QubitOperator('Z%d Z%d X%d Y%d'% (p, q, r, s), 1j)
                        # ZZYX = QubitOperator('Z%d Z%d Y%d X%d'% (p, q, r, s), 1j)
                        # XZZY = QubitOperator('X%d Z%d Z%d Y%d'% (p, q, r, s), 1j)
                        # YZZX = QubitOperator('Y%d Z%d Z%d X%d'% (p, q, r, s), 1j)
                        # ZYXZ = QubitOperator('Z%d Y%d X%d Z%d'% (p, q, r, s), 1j)
                        # ZXYZ = QubitOperator('Z%d X%d Y%d Z%d'% (p, q, r, s), 1j)
                        # XYZZ = QubitOperator('X%d Y%d Z%d Z%d'% (p, q, r, s), 1j)
                        # YXZZ = QubitOperator('Y%d X%d Z%d Z%d'% (p, q, r, s), 1j)
                        # XZYZ = QubitOperator('X%d Z%d Y%d Z%d'% (p, q, r, s), 1j)
                        # YZXZ = QubitOperator('Y%d Z%d X%d Z%d'% (p, q, r, s), 1j)
                        # ZXZY = QubitOperator('Z%d X%d Z%d Y%d'% (p, q, r, s), 1j)
                        # ZYZX = QubitOperator('Z%d Y%d Z%d X%d'% (p, q, r, s), 1j)

                        # self.fermi_ops.append(ZZXY)
                        # self.fermi_ops.append(ZZYX)
                        # self.fermi_ops.append(XZZY)
                        # self.fermi_ops.append(YZZX)
                        # self.fermi_ops.append(ZXYZ)
                        # self.fermi_ops.append(ZYXZ)
                        # self.fermi_ops.append(XYZZ)
                        # self.fermi_ops.append(YXZZ)
                        # self.fermi_ops.append(XZYZ)
                        # self.fermi_ops.append(YZXZ)
                        # self.fermi_ops.append(ZXZY)
                        # self.fermi_ops.append(ZYZX)

                        # XXYY = QubitOperator('X%d X%d Y%d Y%d'% (p, q, r, s), 1j)
                        # XYXY = QubitOperator('X%d Y%d X%d Y%d'% (p, q, r, s), 1j)
                        # XYYX = QubitOperator('X%d Y%d Y%d X%d'% (p, q, r, s), 1j)
                        # YYXX = QubitOperator('Y%d Y%d X%d X%d'% (p, q, r, s), 1j)
                        # YXXY = QubitOperator('Y%d X%d X%d Y%d'% (p, q, r, s), 1j)
                        # YXYX = QubitOperator('Y%d X%d Y%d X%d'% (p, q, r, s), 1j)

                        # self.fermi_ops.append(XXYY)
                        # self.fermi_ops.append(XYXY)
                        # self.fermi_ops.append(XYYX)
                        # self.fermi_ops.append(YYXX)
                        # self.fermi_ops.append(YXXY)
                        # self.fermi_ops.append(YXYX)

                        # XXXX = QubitOperator('X%d X%d X%d X%d'% (p, q, r, s), 1j)
                        # YYYY = QubitOperator('Y%d Y%d Y%d Y%d'% (p, q, r, s), 1j)

                        # self.fermi_ops.append(XXXX)
                        # self.fermi_ops.append(YYYY)

        # np.random.shuffle(self.fermi_ops_1)
        # for op in self.fermi_ops_1[:len(self.fermi_ops_1)//4]:
        #     self.fermi_ops.append(op)

        print("original size", len(self.fermi_ops_2) + len(self.fermi_ops_3) + len(self.fermi_ops_4))

        np.random.shuffle(self.fermi_ops_2)
        for op in self.fermi_ops_2[:len(self.fermi_ops_2) // 16]:
            self.fermi_ops.append(op)

        np.random.shuffle(self.fermi_ops_3)
        for op in self.fermi_ops_3[:len(self.fermi_ops_3) // 16]:
            self.fermi_ops.append(op)

        np.random.shuffle(self.fermi_ops_4)
        for op in self.fermi_ops_4[:len(self.fermi_ops_4) // 16]:
            self.fermi_ops.append(op)

        for op in self.fermi_ops:
            print(op)

        self.n_ops = len(self.fermi_ops)
        print(" Number of operators: ", self.n_ops)
        return

class qubits_A_new(OperatorPool):
    def generate_SQ_Operators(self):

        self.fermi_ops = []
        
        n_occ = self.n_occ
        n_vir = self.n_vir

                
        for p in range(0,2*self.n_orb):
            X = QubitOperator('X%d'% p, 0+1j)
            Y = QubitOperator('Y%d'% p, 0+1j)

            # self.fermi_ops.append(X)
        for p in range(0,2*self.n_orb):
            Z = QubitOperator('Z%d'% p, 0+1j)

            # self.fermi_ops.append(Z)

        for p in range(0,2*self.n_orb):

            for q in range(p+1,2*self.n_orb):

                ZZ = QubitOperator('Z%d Z%d'% (p, q), 0+1j)
                # self.fermi_ops.append(ZZ)

        for p in range(0,2*self.n_orb):
            for q in range(p+1,2*self.n_orb):

                XX = QubitOperator('X%d X%d'% (p, q), 0+1j)
                XY = QubitOperator('X%d Y%d'% (p, q), 0+1j)
                YX = QubitOperator('Y%d X%d'% (p, q), 0+1j)
                XZ = QubitOperator('X%d Z%d'% (p, q), 0+1j)
                YZ = QubitOperator('Y%d Z%d'% (p, q), 0+1j)
                YY = QubitOperator('Y%d Y%d'% (p, q), 0+1j)

                self.fermi_ops.append(XX)
                self.fermi_ops.append(YX)
                self.fermi_ops.append(XZ)
                self.fermi_ops.append(YZ)
                self.fermi_ops.append(YY)
                self.fermi_ops.append(XY)

                A =  QubitOperator('X%d Y%d'% (p, q), 0+1j)
                A -= QubitOperator('Y%d X%d'% (p, q), 0+1j)

                self.fermi_ops.append(A)
 

        for i in range(0,2*self.n_orb):
            for j in range(i+1, 2*self.n_orb):

                for k in range(j+1, 2*self.n_orb):

                    YXZ = QubitOperator('Y%d X%d Z%d'% (i, j, k), 1j)  
                    # XYZ = QubitOperator('X%d Y%d Z%d'% (i, j, k), 1j)
                    XZY = QubitOperator('X%d Z%d Y%d'% (i, j, k), 1j)
                    # YZX = QubitOperator('Y%d Z%d X%d'% (i, j, k), 1j)
                    ZXY = QubitOperator('Z%d X%d Y%d'% (i, j, k), 1j)
                    # ZYX = QubitOperator('Z%d X%d Y%d'% (i, j, k), 1j)
                    # ZZZ = QubitOperator('Z%d Z%d Z%d'% (i, j, k), 1j)
  
                    # # self.fermi_ops.append(XYZ)
                    # self.fermi_ops.append(YXZ)
                    # self.fermi_ops.append(XZY)
                    # # self.fermi_ops.append(YZX)  
                    # self.fermi_ops.append(ZXY)
                    # # self.fermi_ops.append(ZYX)
                    # # self.fermi_ops.append(ZZZ) 

                    XXY = QubitOperator('X%d X%d Y%d'% (i, j, k), 1j)
                    XYY = QubitOperator('X%d Y%d Y%d'% (i, j, k), 1j)

                    # self.fermi_ops.append(XXY)
                    # self.fermi_ops.append(XYY)

        for p in range(0,2*self.n_orb):
            for q in range(p+1,2*self.n_orb):
                for r in range(q+1,2*self.n_orb):
                    for s in range(r+1,2*self.n_orb):
                        XYYY = QubitOperator('X%d Y%d Y%d Y%d'% (p, q, r, s), 1j)
                        YXYY = QubitOperator('Y%d X%d Y%d Y%d'% (p, q, r, s), 1j)
                        YYXY = QubitOperator('Y%d Y%d X%d Y%d'% (p, q, r, s), 1j)
                        YYYX = QubitOperator('Y%d Y%d Y%d X%d'% (p, q, r, s), 1j)

                        # self.fermi_ops.append(XYYY)
                        # self.fermi_ops.append(YXYY)
                        # self.fermi_ops.append(YYXY)
                        # self.fermi_ops.append(YYYX)

                        # XXXY = QubitOperator('X%d X%d X%d Y%d'% (p, q, r, s), 1j)
                        # YXXX = QubitOperator('Y%d X%d X%d X%d'% (p, q, r, s), 1j)
                        # XYXX = QubitOperator('X%d Y%d X%d X%d'% (p, q, r, s), 1j)
                        # XXYX = QubitOperator('X%d X%d Y%d X%d'% (p, q, r, s), 1j)

                        # self.fermi_ops.append(XXXY)
                        # self.fermi_ops.append(XXYX)
                        # self.fermi_ops.append(XYXX)
                        # self.fermi_ops.append(YXXX)

                        # ZZXY = QubitOperator('Z%d Z%d X%d Y%d'% (p, q, r, s), 1j)
                        # ZZYX = QubitOperator('Z%d Z%d Y%d X%d'% (p, q, r, s), 1j)
                        # XZZY = QubitOperator('X%d Z%d Z%d Y%d'% (p, q, r, s), 1j)
                        # YZZX = QubitOperator('Y%d Z%d Z%d X%d'% (p, q, r, s), 1j)
                        # ZYXZ = QubitOperator('Z%d Y%d X%d Z%d'% (p, q, r, s), 1j)
                        # ZXYZ = QubitOperator('Z%d X%d Y%d Z%d'% (p, q, r, s), 1j)
                        # XYZZ = QubitOperator('X%d Y%d Z%d Z%d'% (p, q, r, s), 1j)
                        # YXZZ = QubitOperator('Y%d X%d Z%d Z%d'% (p, q, r, s), 1j)
                        # XZYZ = QubitOperator('X%d Z%d Y%d Z%d'% (p, q, r, s), 1j)
                        # YZXZ = QubitOperator('Y%d Z%d X%d Z%d'% (p, q, r, s), 1j)
                        # ZXZY = QubitOperator('Z%d X%d Z%d Y%d'% (p, q, r, s), 1j)
                        # ZYZX = QubitOperator('Z%d Y%d Z%d X%d'% (p, q, r, s), 1j)

                        # self.fermi_ops.append(ZZXY)
                        # self.fermi_ops.append(ZZYX)
                        # self.fermi_ops.append(XZZY)
                        # self.fermi_ops.append(YZZX)
                        # self.fermi_ops.append(ZXYZ)
                        # self.fermi_ops.append(ZYXZ)
                        # self.fermi_ops.append(XYZZ)
                        # self.fermi_ops.append(YXZZ)
                        # self.fermi_ops.append(XZYZ)
                        # self.fermi_ops.append(YZXZ)
                        # self.fermi_ops.append(ZXZY)
                        # self.fermi_ops.append(ZYZX)

                        # XXYY = QubitOperator('X%d X%d Y%d Y%d'% (p, q, r, s), 1j)
                        # XYXY = QubitOperator('X%d Y%d X%d Y%d'% (p, q, r, s), 1j)
                        # XYYX = QubitOperator('X%d Y%d Y%d X%d'% (p, q, r, s), 1j)
                        # YYXX = QubitOperator('Y%d Y%d X%d X%d'% (p, q, r, s), 1j)
                        # YXXY = QubitOperator('Y%d X%d X%d Y%d'% (p, q, r, s), 1j)
                        # YXYX = QubitOperator('Y%d X%d Y%d X%d'% (p, q, r, s), 1j)

                        # self.fermi_ops.append(XXYY)
                        # self.fermi_ops.append(XYXY)
                        # self.fermi_ops.append(XYYX)
                        # self.fermi_ops.append(YYXX)
                        # self.fermi_ops.append(YXXY)
                        # self.fermi_ops.append(YXYX)

                        # XXXX = QubitOperator('X%d X%d X%d X%d'% (p, q, r, s), 1j)
                        # YYYY = QubitOperator('Y%d Y%d Y%d Y%d'% (p, q, r, s), 1j)

                        # self.fermi_ops.append(XXXX)
                        # self.fermi_ops.append(YYYY)                        
 
                
        for op in self.fermi_ops:
            print(op)

        self.n_ops = len(self.fermi_ops)
        print(" Number of operators: ", self.n_ops)        
        return

class qubits2(OperatorPool):
    def generate_SQ_Operators(self):
        """
        n_orb is number of spatial orbitals assuming that spin orbitals are labelled
        0a,0b,1a,1b,2a,2b,3a,3b,....  -> 0,1,2,3,...
        """
        
        print(" Form singlet GSD operators")
        
        self.fermi_ops = []
        for p in range(0,self.n_orb):
            pa = 2*p
            pb = 2*p+1
 
            for q in range(p,self.n_orb):
                qa = 2*q
                qb = 2*q+1


                termA = QubitOperator('X%d Y%d'% (pa,qa), 1j)
                termA -= hermitian_conjugated(termA)
                self.fermi_ops.append(termA)

                termA = QubitOperator('Y%d X%d'% (pa,qa), 1j)
                termA -= hermitian_conjugated(termA)
                self.fermi_ops.append(termA)

                termA = QubitOperator('X%d Y%d'% (pb,qb), 1j)
                termA -= hermitian_conjugated(termA)
                self.fermi_ops.append(termA)

                termA = QubitOperator('Y%d X%d'% (pb,qb), 1j)
                termA -= hermitian_conjugated(termA)
                self.fermi_ops.append(termA)  

        pq = -1 
        for p in range(0,self.n_orb):
            pa = 2*p
            pb = 2*p+1
 
            for q in range(p,self.n_orb):
                qa = 2*q
                qb = 2*q+1
        
                pq += 1
        
                rs = -1 
                for r in range(0,self.n_orb):
                    ra = 2*r
                    rb = 2*r+1
                    
                    for s in range(r,self.n_orb):
                        sa = 2*s
                        sb = 2*s+1
                    
                        rs += 1
                    
                        if(pq > rs):
                            continue

                        OA = QubitOperator('X1', 0)
                        OB = QubitOperator('X1', 0)

                        # termA =  FermionOperator(((ra,1),(pa,0),(sa,1),(qa,0)), 2/np.sqrt(12))

                        OA =  QubitOperator('Y%d X%d X%d X%d'% (ra,sa,pa,qa), 1j)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)  

                        OA =  QubitOperator('X%d Y%d X%d X%d'% (ra,sa,pa,qa), 1j)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)  

                        OA =  QubitOperator('X%d X%d Y%d X%d'% (ra,sa,pa,qa), 1j)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)  

                        OA =  QubitOperator('X%d X%d X%d Y%d'% (ra,sa,pa,qa), 1j)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)  

                        OA =  QubitOperator('X%d Y%d Y%d Y%d'% (ra,sa,pa,qa), 1j)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)  

                        OA =  QubitOperator('Y%d X%d Y%d Y%d'% (ra,sa,pa,qa), 1j)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)  

                        OA =  QubitOperator('Y%d Y%d X%d Y%d'% (ra,sa,pa,qa), 1j)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)  
                        
                        OA =  QubitOperator('Y%d Y%d Y%d X%d'% (ra,sa,pa,qa), 1)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)  
                        

                        OA =  QubitOperator('X%d X%d X%d X%d'% (ra,sa,pa,qa), 1)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)  
                        
                        OA =  QubitOperator('Y%d Y%d Y%d Y%d'% (ra,sa,pa,qa), 1)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)  
                        
                        OA =  QubitOperator('X%d X%d Y%d Y%d'% (ra,sa,pa,qa), 1)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)  
                        
                        OA =  QubitOperator('X%d Y%d X%d Y%d'% (ra,sa,pa,qa), 1)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)  
                        
                        OA =  QubitOperator('Y%d X%d X%d Y%d'% (ra,sa,pa,qa), 1)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)  
                        
                        OA =  QubitOperator('X%d Y%d Y%d X%d'% (ra,sa,pa,qa), 1)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)  
                        
                        OA =  QubitOperator('Y%d Y%d X%d X%d'% (ra,sa,pa,qa), 1)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)  
                        
                        OA =  QubitOperator('Y%d X%d Y%d X%d'% (ra,sa,pa,qa), 1)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)  
                        
 
                        # termA += FermionOperator(((rb,1),(pb,0),(sb,1),(qb,0)), 2/np.sqrt(12))

                        OA =  QubitOperator('Y%d X%d X%d X%d'% (rb,sb,pb,qb), 1j)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)

                        OA =  QubitOperator('X%d Y%d X%d X%d'% (rb,sb,pb,qb), 1j)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)
                        
                        OA =  QubitOperator('X%d X%d Y%d X%d'% (rb,sb,pb,qb), 1j)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)
                        
                        OA =  QubitOperator('X%d X%d X%d Y%d'% (rb,sb,pb,qb), 1j)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)
                        
                        OA =  QubitOperator('X%d Y%d Y%d Y%d'% (rb,sb,pb,qb), 1j)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)
                        
                        OA =  QubitOperator('Y%d X%d Y%d Y%d'% (rb,sb,pb,qb), 1j)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)
                        
                        OA =  QubitOperator('Y%d Y%d X%d Y%d'% (rb,sb,pb,qb), 1j)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)
                        
                        OA =  QubitOperator('Y%d Y%d Y%d X%d'% (rb,sb,pb,qb), 1j)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)
                        

                        OA =  QubitOperator('X%d X%d X%d X%d'% (rb,sb,pb,qb), 1)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)
                        
                        OA =  QubitOperator('Y%d Y%d Y%d Y%d'% (rb,sb,pb,qb), 1)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)
                        
                        OA =  QubitOperator('X%d X%d Y%d Y%d'% (rb,sb,pb,qb), 1)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)
                        
                        OA =  QubitOperator('X%d Y%d X%d Y%d'% (rb,sb,pb,qb), 1)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)
                        
                        OA =  QubitOperator('Y%d X%d X%d Y%d'% (rb,sb,pb,qb), 1)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)
                        
                        OA =  QubitOperator('X%d Y%d Y%d X%d'% (rb,sb,pb,qb), 1)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)
                        
                        OA =  QubitOperator('Y%d Y%d X%d X%d'% (rb,sb,pb,qb), 1)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)
                        
                        OA =  QubitOperator('Y%d X%d Y%d X%d'% (rb,sb,pb,qb), 1)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)
                        

                        # termA += FermionOperator(((ra,1),(pa,0),(sb,1),(qb,0)), 1/np.sqrt(12))

                        OA  = QubitOperator('Y%d X%d X%d X%d'% (ra,sb,pa,qb), 1j)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)
                        
                        OA  = QubitOperator('X%d Y%d X%d X%d'% (ra,sb,pa,qb), 1j)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)
                        
                        OA  = QubitOperator('X%d X%d Y%d X%d'% (ra,sb,pa,qb), 1j)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)
                        
                        OA  = QubitOperator('X%d X%d X%d Y%d'% (ra,sb,pa,qb), 1j)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)
                        
                        OA  = QubitOperator('X%d Y%d Y%d Y%d'% (ra,sb,pa,qb), 1j)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)
                        
                        OA  = QubitOperator('Y%d X%d Y%d Y%d'% (ra,sb,pa,qb), 1j)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)
                        
                        OA  = QubitOperator('Y%d Y%d X%d Y%d'% (ra,sb,pa,qb), 1j)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)
                        
                        OA  = QubitOperator('Y%d Y%d Y%d X%d'% (ra,sb,pa,qb), 1j)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)
                        

                        OA  = QubitOperator('X%d X%d X%d X%d'% (ra,sb,pa,qb), 1)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)
                        
                        OA  = QubitOperator('Y%d Y%d Y%d Y%d'% (ra,sb,pa,qb), 1)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)
                        
                        OA  = QubitOperator('X%d X%d Y%d Y%d'% (ra,sb,pa,qb), 1)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)
                        
                        OA  = QubitOperator('X%d Y%d X%d Y%d'% (ra,sb,pa,qb), 1)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)
                        
                        OA  = QubitOperator('Y%d X%d X%d Y%d'% (ra,sb,pa,qb), 1)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)
                        
                        OA  = QubitOperator('X%d Y%d Y%d X%d'% (ra,sb,pa,qb), 1)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)
                        
                        OA  = QubitOperator('Y%d Y%d X%d X%d'% (ra,sb,pa,qb), 1)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)
                        
                        OA  = QubitOperator('Y%d X%d Y%d X%d'% (ra,sb,pa,qb), 1)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)
                        

                        # termA += FermionOperator(((rb,1),(pb,0),(sa,1),(qa,0)), 1/np.sqrt(12))

                        OA  = QubitOperator('Y%d X%d X%d X%d'% (rb,sa,pb,qa), 1j)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)
                        
                        OA  = QubitOperator('X%d Y%d X%d X%d'% (rb,sa,pb,qa), 1j)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)
                        
                        OA  = QubitOperator('X%d X%d Y%d X%d'% (rb,sa,pb,qa), 1j)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)
                        
                        OA  = QubitOperator('X%d X%d X%d Y%d'% (rb,sa,pb,qa), 1j)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)
                        
                        OA  = QubitOperator('X%d Y%d Y%d Y%d'% (rb,sa,pb,qa), 1j)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)
                        
                        OA  = QubitOperator('Y%d X%d Y%d Y%d'% (rb,sa,pb,qa), 1j)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)
                        
                        OA  = QubitOperator('Y%d Y%d X%d Y%d'% (rb,sa,pb,qa), 1j)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)
                        
                        OA  = QubitOperator('Y%d Y%d Y%d X%d'% (rb,sa,pb,qa), 1j)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)
                        
                        OA  = QubitOperator('X%d X%d X%d X%d'% (rb,sa,pb,qa), 1)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)
                        
                        OA  = QubitOperator('Y%d Y%d Y%d Y%d'% (rb,sa,pb,qa), 1)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)
                        
                        OA  = QubitOperator('X%d X%d Y%d Y%d'% (rb,sa,pb,qa), 1)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)
                        
                        OA  = QubitOperator('X%d Y%d X%d Y%d'% (rb,sa,pb,qa), 1)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)
                        
                        OA  = QubitOperator('Y%d X%d X%d Y%d'% (rb,sa,pb,qa), 1)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)
                        
                        OA  = QubitOperator('X%d Y%d Y%d X%d'% (rb,sa,pb,qa), 1)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)
                        
                        OA  = QubitOperator('Y%d Y%d X%d X%d'% (rb,sa,pb,qa), 1)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)
                        
                        OA  = QubitOperator('Y%d X%d Y%d X%d'% (rb,sa,pb,qa), 1)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)
                        

                        # termA += FermionOperator(((ra,1),(pb,0),(sb,1),(qa,0)), 1/np.sqrt(12))
                        
                        OA  = QubitOperator('Y%d X%d X%d X%d'% (ra,sb,pb,qa), 1j)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)
                        
                        OA  = QubitOperator('X%d Y%d X%d X%d'% (ra,sb,pb,qa), 1j)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)
                        
                        OA  = QubitOperator('X%d X%d Y%d X%d'% (ra,sb,pb,qa), 1j)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)
                        
                        OA  = QubitOperator('X%d X%d X%d Y%d'% (ra,sb,pb,qa), 1j)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)
                        
                        OA  = QubitOperator('X%d Y%d Y%d Y%d'% (ra,sb,pb,qa), 1j)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)
                        
                        OA  = QubitOperator('Y%d X%d Y%d Y%d'% (ra,sb,pb,qa), 1j)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)
                        
                        OA  = QubitOperator('Y%d Y%d X%d Y%d'% (ra,sb,pb,qa), 1j)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)
                        
                        OA  = QubitOperator('Y%d Y%d Y%d X%d'% (ra,sb,pb,qa), 1j)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)
                        

                        OA  = QubitOperator('X%d X%d X%d X%d'% (ra,sb,pb,qa), 1)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)
                        
                        OA  = QubitOperator('Y%d Y%d Y%d Y%d'% (ra,sb,pb,qa), 1)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)
                        
                        OA  = QubitOperator('X%d X%d Y%d Y%d'% (ra,sb,pb,qa), 1)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)
                        
                        OA  = QubitOperator('X%d Y%d X%d Y%d'% (ra,sb,pb,qa), 1)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)
                        
                        OA  = QubitOperator('Y%d X%d X%d Y%d'% (ra,sb,pb,qa), 1)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)
                        
                        OA  = QubitOperator('X%d Y%d Y%d X%d'% (ra,sb,pb,qa), 1)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)
                        
                        OA  = QubitOperator('Y%d Y%d X%d X%d'% (ra,sb,pb,qa), 1)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)
                        
                        OA  = QubitOperator('Y%d X%d Y%d X%d'% (ra,sb,pb,qa), 1)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)
                        

                        # termA += FermionOperator(((rb,1),(pa,0),(sa,1),(qb,0)), 1/np.sqrt(12))
                                                                
                        OA  = QubitOperator('Y%d X%d X%d X%d'% (rb,sa,pa,qb), 1j)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)
                        
                        OA  = QubitOperator('X%d Y%d X%d X%d'% (rb,sa,pa,qb), 1j)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)
                        
                        OA  = QubitOperator('X%d X%d Y%d X%d'% (rb,sa,pa,qb), 1j)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)
                        
                        OA  = QubitOperator('X%d X%d X%d Y%d'% (rb,sa,pa,qb), 1j)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)
                        
                        OA  = QubitOperator('X%d Y%d Y%d Y%d'% (rb,sa,pa,qb), 1j)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)
                        
                        OA  = QubitOperator('Y%d X%d Y%d Y%d'% (rb,sa,pa,qb), 1j)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)
                        
                        OA  = QubitOperator('Y%d Y%d X%d Y%d'% (rb,sa,pa,qb), 1j)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)
                        
                        OA  = QubitOperator('Y%d Y%d Y%d X%d'% (rb,sa,pa,qb), 1j)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)
                        
                        OA  = QubitOperator('X%d X%d X%d X%d'% (rb,sa,pa,qb), 1)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)
                        
                        OA  = QubitOperator('Y%d Y%d Y%d Y%d'% (rb,sa,pa,qb), 1)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)
                        
                        OA  = QubitOperator('X%d X%d Y%d Y%d'% (rb,sa,pa,qb), 1)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)
                        
                        OA  = QubitOperator('X%d Y%d X%d Y%d'% (rb,sa,pa,qb), 1)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)
                        
                        OA  = QubitOperator('Y%d X%d X%d Y%d'% (rb,sa,pa,qb), 1)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)
                        
                        OA  = QubitOperator('X%d Y%d Y%d X%d'% (rb,sa,pa,qb), 1)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)
                        
                        OA  = QubitOperator('Y%d Y%d X%d X%d'% (rb,sa,pa,qb), 1)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)
                        
                        OA  = QubitOperator('Y%d X%d Y%d X%d'% (rb,sa,pa,qb), 1)
                        OA -= hermitian_conjugated(OA)
                        self.fermi_ops.append(OA)
                        
        for op in self.fermi_ops:
            print(op)

        self.n_ops = len(self.fermi_ops)
        print(" Number of operators: ", self.n_ops)
        return 

class qubitsZ(OperatorPool):
    def generate_SQ_Operators(self):
        """
        0a,0b,1a,1b,2a,2b,3a,3b,....  -> 0,1,2,3,...
        """
        n_occ = self.n_occ
        n_vir = self.n_vir

        print(" Form singlet SD operators")
        self.fermi_ops = [] 

        # for p in range(0,2*self.n_orb):
        #     X = QubitOperator('X%d'% p, 0+1j)
        #     Y = QubitOperator('Y%d'% p, 0+1j)

        #     self.fermi_ops.append(X)
        # for p in range(0,2*self.n_orb):
        #     Z = QubitOperator('Z%d'% p, 0+1j)

        #     self.fermi_ops.append(Z)

        # for p in range(0,2*self.n_orb):

        #     for q in range(p+1,2*self.n_orb):

        #         ZZ = QubitOperator('Z%d Z%d'% (p, q), 0+1j)
        #         self.fermi_ops.append(ZZ)

        # for p in range(0,2*self.n_orb):
        #     for q in range(p+1,2*self.n_orb):

        #         XX = QubitOperator('X%d X%d'% (p, q), 0+1j)
        #         XY = QubitOperator('X%d Y%d'% (p, q), 0+1j)
        #         YX = QubitOperator('Y%d X%d'% (p, q), 0+1j)
        #         XZ = QubitOperator('X%d Z%d'% (p, q), 0+1j)
        #         YZ = QubitOperator('Y%d Z%d'% (p, q), 0+1j)
        #         YY = QubitOperator('Y%d Y%d'% (p, q), 0+1j)

        #         self.fermi_ops.append(XX)
        #         self.fermi_ops.append(YX)
        #         self.fermi_ops.append(XZ)
        #         self.fermi_ops.append(YZ)
        #         self.fermi_ops.append(YY)
        #         self.fermi_ops.append(XY)


        # for i in range(0,2*self.n_orb):
        #     for j in range(i+1, 2*self.n_orb):

        #         for k in range(j+1, 2*self.n_orb):

        #             YXZ = QubitOperator('Y%d X%d Z%d'% (i, j, k), 1j)  
        #             XYZ = QubitOperator('X%d Y%d Z%d'% (i, j, k), 1j)
        #             XZY = QubitOperator('X%d Z%d Y%d'% (i, j, k), 1j)
        #             YZX = QubitOperator('Y%d Z%d X%d'% (i, j, k), 1j)
        #             ZXY = QubitOperator('Z%d X%d Y%d'% (i, j, k), 1j)
        #             ZYX = QubitOperator('Z%d X%d Y%d'% (i, j, k), 1j)
  
        #             self.fermi_ops.append(XYZ)
        #             self.fermi_ops.append(YXZ)
        #             self.fermi_ops.append(XZY)
        #             self.fermi_ops.append(YZX)  
        #             self.fermi_ops.append(ZXY)
        #             self.fermi_ops.append(ZYX) 

        #             XXY = QubitOperator('X%d X%d Y%d'% (i, j, k), 1j)
        #             XYY = QubitOperator('X%d Y%d Y%d'% (i, j, k), 1j)

        #             self.fermi_ops.append(XXY)
        #             self.fermi_ops.append(XYY)

        # for p in range(0,2*self.n_orb):
        #     for q in range(p+1,2*self.n_orb):
        #         for r in range(q+1,2*self.n_orb):
        #             for s in range(r+1,2*self.n_orb):
        #                 XYYY = QubitOperator('X%d Y%d Y%d Y%d'% (p, q, r, s), 1j)
        #                 YXYY = QubitOperator('Y%d X%d Y%d Y%d'% (p, q, r, s), 1j)
        #                 YYXY = QubitOperator('Y%d Y%d X%d Y%d'% (p, q, r, s), 1j)
        #                 YYYX = QubitOperator('Y%d Y%d Y%d X%d'% (p, q, r, s), 1j)

        #                 self.fermi_ops.append(XYYY)
        #                 self.fermi_ops.append(YXYY)
        #                 self.fermi_ops.append(YYXY)
        #                 self.fermi_ops.append(YYYX)

        #                 XXXY = QubitOperator('X%d X%d X%d Y%d'% (p, q, r, s), 1j)
        #                 YXXX = QubitOperator('Y%d X%d X%d X%d'% (p, q, r, s), 1j)
        #                 XYXX = QubitOperator('X%d Y%d X%d X%d'% (p, q, r, s), 1j)
        #                 XXYX = QubitOperator('X%d X%d Y%d X%d'% (p, q, r, s), 1j)

        #                 self.fermi_ops.append(XXXY)
        #                 self.fermi_ops.append(XXYX)
        #                 self.fermi_ops.append(XYXX)
        #                 self.fermi_ops.append(YXXX)

        #                 ZZXY = QubitOperator('Z%d Z%d X%d Y%d'% (p, q, r, s), 1j)
        #                 ZZYX = QubitOperator('Z%d Z%d Y%d X%d'% (p, q, r, s), 1j)
        #                 XZZY = QubitOperator('X%d Z%d Z%d Y%d'% (p, q, r, s), 1j)
        #                 YZZX = QubitOperator('Y%d Z%d Z%d X%d'% (p, q, r, s), 1j)
        #                 ZYXZ = QubitOperator('Z%d Y%d X%d Z%d'% (p, q, r, s), 1j)
        #                 ZXYZ = QubitOperator('Z%d X%d Y%d Z%d'% (p, q, r, s), 1j)
        #                 XYZZ = QubitOperator('X%d Y%d Z%d Z%d'% (p, q, r, s), 1j)
        #                 YXZZ = QubitOperator('Y%d X%d Z%d Z%d'% (p, q, r, s), 1j)
        #                 XZYZ = QubitOperator('X%d Z%d Y%d Z%d'% (p, q, r, s), 1j)
        #                 YZXZ = QubitOperator('Y%d Z%d X%d Z%d'% (p, q, r, s), 1j)
        #                 ZXZY = QubitOperator('Z%d X%d Z%d Y%d'% (p, q, r, s), 1j)
        #                 ZYZX = QubitOperator('Z%d Y%d Z%d X%d'% (p, q, r, s), 1j)

        #                 self.fermi_ops.append(ZZXY)
        #                 self.fermi_ops.append(ZZYX)
        #                 self.fermi_ops.append(XZZY)
        #                 self.fermi_ops.append(YZZX)
        #                 self.fermi_ops.append(ZXYZ)
        #                 self.fermi_ops.append(ZYXZ)
        #                 self.fermi_ops.append(XYZZ)
        #                 self.fermi_ops.append(YXZZ)
        #                 self.fermi_ops.append(XZYZ)
        #                 self.fermi_ops.append(YZXZ)
        #                 self.fermi_ops.append(ZXZY)
        #                 self.fermi_ops.append(ZYZX)
        
        for p in range(0,2*self.n_orb):
            for q in range(0,2*self.n_orb):
                    
                termA =  FermionOperator(((p,1),(q,0)), 1)
                
                termAQ = openfermion.transforms.jordan_wigner(termA)
                print(termAQ)
               
                #Normalize
                coeffA = 0
                for t in termAQ.terms:
                    self.fermi_ops.append(QubitOperator(t,1j))
                    print(t)
       
        pq = -1 
        for p in range(0,self.n_orb):
            pa = 2*p
            pb = 2*p+1
 
            for q in range(p,self.n_orb):
                qa = 2*q
                qb = 2*q+1
        
                pq += 1
        
                rs = -1 
                for r in range(0,self.n_orb):
                    ra = 2*r
                    rb = 2*r+1
                    
                    for s in range(r,self.n_orb):
                        sa = 2*s
                        sb = 2*s+1
                    
                        rs += 1
                    
                        if(pq > rs):
                            continue

                        termA =  FermionOperator(((ra,1),(pa,0),(sa,1),(qa,0)), 1)

                        termAQ = openfermion.transforms.jordan_wigner(termA)
                        print(termAQ)

                        for t in termAQ.terms:
                            self.fermi_ops.append(QubitOperator(t,1j))
                            print(t)

                        termA =  FermionOperator(((rb,1),(pb,0),(sb,1),(qb,0)), 1)

                        termAQ = openfermion.transforms.jordan_wigner(termA)
                        print(termAQ)

                        for t in termAQ.terms:
                            self.fermi_ops.append(QubitOperator(t,1j))
                            print(t)

                        termA =  FermionOperator(((ra,1),(pa,0),(sb,1),(qb,0)), 1)

                        termAQ = openfermion.transforms.jordan_wigner(termA)
                        print(termAQ)

                        for t in termAQ.terms:
                            self.fermi_ops.append(QubitOperator(t,1j))
                            print(t)

                        termA =  FermionOperator(((rb,1),(pb,0),(sa,1),(qa,0)), 1)

                        termAQ = openfermion.transforms.jordan_wigner(termA)
                        print(termAQ)

                        for t in termAQ.terms:
                            self.fermi_ops.append(QubitOperator(t,1j))
                            print(t)

                        termA =  FermionOperator(((ra,1),(pb,0),(sb,1),(qa,0)), 1)

                        termAQ = openfermion.transforms.jordan_wigner(termA)
                        print(termAQ)

                        for t in termAQ.terms:
                            self.fermi_ops.append(QubitOperator(t,1j))
                            print(t)

                        termA =  FermionOperator(((rb,1),(pa,0),(sa,1),(qb,0)), 1)

                        termAQ = openfermion.transforms.jordan_wigner(termA)
                        print(termAQ)

                        for t in termAQ.terms:
                            self.fermi_ops.append(QubitOperator(t,1j))
                            print(t)


        for op in self.fermi_ops:
            print(op)
        
        self.n_ops = len(self.fermi_ops)
        print(" Number of operators: ", self.n_ops)
        return 

class qubits_spin(OperatorPool):
    def generate_SQ_Operators(self):
        """
        n_orb is number of spatial orbitals assuming that spin orbitals are labelled
        0a,0b,1a,1b,2a,2b,3a,3b,....  -> 0,1,2,3,...
        """
        
        print(" Form singlet GSD operators")
        
        self.fermi_ops = []
        for p in range(0,self.n_orb):
            pa = 2*p
            pb = 2*p+1
 
            for q in range(p,self.n_orb):
                qa = 2*q
                qb = 2*q+1


                termA =  QubitOperator('X%d Y%d'% (pa,qa), 1j)
                termA += QubitOperator('X%d Y%d'% (pb,qb), 1j)
                termA -= hermitian_conjugated(termA)
                coeffA = 0

                for t in termA.terms:
                    coeff_t = termA.terms[t]
                    coeffA += coeff_t * coeff_t

                if np.abs(coeffA) > 1e-6:
                  self.fermi_ops.append(termA)

                termA = QubitOperator('Y%d X%d'% (pa,qa), 1j)
                termA += QubitOperator('Y%d X%d'% (pb,qb), 1j)
                termA -= hermitian_conjugated(termA)
                coeffA = 0

                for t in termA.terms:
                    coeff_t = termA.terms[t]
                    coeffA += coeff_t * coeff_t

                if np.abs(coeffA) > 1e-6:
                  self.fermi_ops.append(termA)

        pq = -1 
        for p in range(0,self.n_orb):
            pa = 2*p
            pb = 2*p+1
 
            for q in range(p,self.n_orb):
                qa = 2*q
                qb = 2*q+1
        
                pq += 1
        
                rs = -1 
                for r in range(0,self.n_orb):
                    ra = 2*r
                    rb = 2*r+1
                    
                    for s in range(r,self.n_orb):
                        sa = 2*s
                        sb = 2*s+1
                    
                        rs += 1
                    
                        if(pq > rs):
                            continue

                        OA = QubitOperator('X1', 0)
                        OB = QubitOperator('X1', 0)

                        # YXXX

                        OA += QubitOperator('Y%d X%d X%d X%d'% (ra,sa,pa,qa), 1j*2/np.sqrt(12))
                        OA += QubitOperator('Y%d X%d X%d X%d'% (rb,sb,pb,qb), 1j*2/np.sqrt(12))
                        OA += QubitOperator('Y%d X%d X%d X%d'% (ra,sb,pa,qb), 1j/np.sqrt(12))
                        OA += QubitOperator('Y%d X%d X%d X%d'% (rb,sa,pb,qa), 1j/np.sqrt(12))
                        OA += QubitOperator('Y%d X%d X%d X%d'% (ra,sb,pb,qa), 1j/np.sqrt(12))
                        OA += QubitOperator('Y%d X%d X%d X%d'% (rb,sa,pa,qb), 1j/np.sqrt(12))

                        OB += QubitOperator('Y%d X%d X%d X%d'% (ra,sb,pa,qb), 1j/2)
                        OB += QubitOperator('Y%d X%d X%d X%d'% (rb,sa,pb,qa), 1j/2)
                        OB -= QubitOperator('Y%d X%d X%d X%d'% (ra,sb,pb,qa), 1j/2)
                        OB -= QubitOperator('Y%d X%d X%d X%d'% (rb,sa,pa,qb), 1j/2)

                        OA -= hermitian_conjugated(OA)

                        OB -= hermitian_conjugated(OB)

                        coeffA = 0
                        coeffB = 0

                        for t in OA.terms:
                            coeff_t = OA.terms[t]
                            coeffA += coeff_t * coeff_t

                        if np.abs(coeffA) > 1e-6:
                          self.fermi_ops.append(OA)

                        for t in OB.terms:
                            coeff_t = OB.terms[t]
                            coeffB += coeff_t * coeff_t
                            
                        if np.abs(coeffB) > 1e-6:
                          self.fermi_ops.append(OB)

                        # XYXX

                        OA = QubitOperator('X1', 0)
                        OB = QubitOperator('X1', 0)

                        OA += QubitOperator('X%d Y%d X%d X%d'% (ra,sa,pa,qa), 1j*2/np.sqrt(12))
                        OA += QubitOperator('X%d Y%d X%d X%d'% (rb,sb,pb,qb), 1j*2/np.sqrt(12))
                        OA += QubitOperator('X%d Y%d X%d X%d'% (ra,sb,pa,qb), 1j/np.sqrt(12))
                        OA += QubitOperator('X%d Y%d X%d X%d'% (rb,sa,pb,qa), 1j/np.sqrt(12))
                        OA += QubitOperator('X%d Y%d X%d X%d'% (ra,sb,pb,qa), 1j/np.sqrt(12))
                        OA += QubitOperator('X%d Y%d X%d X%d'% (rb,sa,pa,qb), 1j/np.sqrt(12))

                        OB += QubitOperator('X%d Y%d X%d X%d'% (ra,sb,pa,qb), 1j/2)
                        OB += QubitOperator('X%d Y%d X%d X%d'% (rb,sa,pb,qa), 1j/2)
                        OB -= QubitOperator('X%d Y%d X%d X%d'% (ra,sb,pb,qa), 1j/2)
                        OB -= QubitOperator('X%d Y%d X%d X%d'% (rb,sa,pa,qb), 1j/2)

                        OA -= hermitian_conjugated(OA)

                        OB -= hermitian_conjugated(OB)

                        coeffA = 0
                        coeffB = 0

                        for t in OA.terms:
                            coeff_t = OA.terms[t]
                            coeffA += coeff_t * coeff_t

                        if np.abs(coeffA) > 1e-6:
                          self.fermi_ops.append(OA)

                        for t in OB.terms:
                            coeff_t = OB.terms[t]
                            coeffB += coeff_t * coeff_t
                            
                        if np.abs(coeffB) > 1e-6:
                          self.fermi_ops.append(OB)

                        # XXYX

                        OA = QubitOperator('X1', 0)
                        OB = QubitOperator('X1', 0)

                        OA += QubitOperator('X%d X%d Y%d X%d'% (ra,sa,pa,qa), -1j*2/np.sqrt(12))
                        OA += QubitOperator('X%d X%d Y%d X%d'% (rb,sb,pb,qb), -1j*2/np.sqrt(12))
                        OA += QubitOperator('X%d X%d Y%d X%d'% (ra,sb,pa,qb), -1j/np.sqrt(12))
                        OA += QubitOperator('X%d X%d Y%d X%d'% (rb,sa,pb,qa), -1j/np.sqrt(12))
                        OA += QubitOperator('X%d X%d Y%d X%d'% (ra,sb,pb,qa), -1j/np.sqrt(12))
                        OA += QubitOperator('X%d X%d Y%d X%d'% (rb,sa,pa,qb), -1j/np.sqrt(12))

                        OB += QubitOperator('X%d X%d Y%d X%d'% (ra,sb,pa,qb), -1j/2)
                        OB += QubitOperator('X%d X%d Y%d X%d'% (rb,sa,pb,qa), -1j/2)
                        OB -= QubitOperator('X%d X%d Y%d X%d'% (ra,sb,pb,qa), -1j/2)
                        OB -= QubitOperator('X%d X%d Y%d X%d'% (rb,sa,pa,qb), -1j/2)

                        OA -= hermitian_conjugated(OA)

                        OB -= hermitian_conjugated(OB)

                        coeffA = 0
                        coeffB = 0

                        for t in OA.terms:
                            coeff_t = OA.terms[t]
                            coeffA += coeff_t * coeff_t

                        if np.abs(coeffA) > 1e-6:
                          self.fermi_ops.append(OA)

                        for t in OB.terms:
                            coeff_t = OB.terms[t]
                            coeffB += coeff_t * coeff_t
                            
                        if np.abs(coeffB) > 1e-6:
                          self.fermi_ops.append(OB)

                        # XXXY

                        OA = QubitOperator('X1', 0)
                        OB = QubitOperator('X1', 0)

                        OA += QubitOperator('X%d X%d X%d Y%d'% (ra,sa,pa,qa), -1j*2/np.sqrt(12))
                        OA += QubitOperator('X%d X%d X%d Y%d'% (rb,sb,pb,qb), -1j*2/np.sqrt(12))
                        OA += QubitOperator('X%d X%d X%d Y%d'% (ra,sb,pa,qb), -1j/np.sqrt(12))
                        OA += QubitOperator('X%d X%d X%d Y%d'% (rb,sa,pb,qa), -1j/np.sqrt(12))
                        OA += QubitOperator('X%d X%d X%d Y%d'% (ra,sb,pb,qa), -1j/np.sqrt(12))
                        OA += QubitOperator('X%d X%d X%d Y%d'% (rb,sa,pa,qb), -1j/np.sqrt(12))

                        OB += QubitOperator('X%d X%d X%d Y%d'% (ra,sb,pa,qb), -1j/2)
                        OB += QubitOperator('X%d X%d X%d Y%d'% (rb,sa,pb,qa), -1j/2)
                        OB -= QubitOperator('X%d X%d X%d Y%d'% (ra,sb,pb,qa), -1j/2)
                        OB -= QubitOperator('X%d X%d X%d Y%d'% (rb,sa,pa,qb), -1j/2)

                        OA -= hermitian_conjugated(OA)

                        OB -= hermitian_conjugated(OB)

                        coeffA = 0
                        coeffB = 0

                        for t in OA.terms:
                            coeff_t = OA.terms[t]
                            coeffA += coeff_t * coeff_t

                        if np.abs(coeffA) > 1e-6:
                          self.fermi_ops.append(OA)

                        for t in OB.terms:
                            coeff_t = OB.terms[t]
                            coeffB += coeff_t * coeff_t
                            
                        if np.abs(coeffB) > 1e-6:
                          self.fermi_ops.append(OB)

                        # XYYY

                        OA = QubitOperator('X1', 0)
                        OB = QubitOperator('X1', 0)

                        OA += QubitOperator('X%d Y%d Y%d Y%d'% (ra,sa,pa,qa), -1j*2/np.sqrt(12))
                        OA += QubitOperator('X%d Y%d Y%d Y%d'% (rb,sb,pb,qb), -1j*2/np.sqrt(12))
                        OA += QubitOperator('X%d Y%d Y%d Y%d'% (ra,sb,pa,qb), -1j/np.sqrt(12))
                        OA += QubitOperator('X%d Y%d Y%d Y%d'% (rb,sa,pb,qa), -1j/np.sqrt(12))
                        OA += QubitOperator('X%d Y%d Y%d Y%d'% (ra,sb,pb,qa), -1j/np.sqrt(12))
                        OA += QubitOperator('X%d Y%d Y%d Y%d'% (rb,sa,pa,qb), -1j/np.sqrt(12))

                        OB += QubitOperator('X%d Y%d Y%d Y%d'% (ra,sb,pa,qb), -1j/2)
                        OB += QubitOperator('X%d Y%d Y%d Y%d'% (rb,sa,pb,qa), -1j/2)
                        OB -= QubitOperator('X%d Y%d Y%d Y%d'% (ra,sb,pb,qa), -1j/2)
                        OB -= QubitOperator('X%d Y%d Y%d Y%d'% (rb,sa,pa,qb), -1j/2)

                        OA -= hermitian_conjugated(OA)

                        OB -= hermitian_conjugated(OB)

                        coeffA = 0
                        coeffB = 0

                        for t in OA.terms:
                            coeff_t = OA.terms[t]
                            coeffA += coeff_t * coeff_t

                        if np.abs(coeffA) > 1e-6:
                          self.fermi_ops.append(OA)

                        for t in OB.terms:
                            coeff_t = OB.terms[t]
                            coeffB += coeff_t * coeff_t
                            
                        if np.abs(coeffB) > 1e-6:
                          self.fermi_ops.append(OB)

                        # YXYY

                        OA = QubitOperator('X1', 0)
                        OB = QubitOperator('X1', 0)

                        OA += QubitOperator('Y%d X%d Y%d Y%d'% (ra,sa,pa,qa), -1j*2/np.sqrt(12))
                        OA += QubitOperator('Y%d X%d Y%d Y%d'% (rb,sb,pb,qb), -1j*2/np.sqrt(12))
                        OA += QubitOperator('Y%d X%d Y%d Y%d'% (ra,sb,pa,qb), -1j/np.sqrt(12))
                        OA += QubitOperator('Y%d X%d Y%d Y%d'% (rb,sa,pb,qa), -1j/np.sqrt(12))
                        OA += QubitOperator('Y%d X%d Y%d Y%d'% (ra,sb,pb,qa), -1j/np.sqrt(12))
                        OA += QubitOperator('Y%d X%d Y%d Y%d'% (rb,sa,pa,qb), -1j/np.sqrt(12))

                        OB += QubitOperator('Y%d X%d Y%d Y%d'% (ra,sb,pa,qb), -1j/2)
                        OB += QubitOperator('Y%d X%d Y%d Y%d'% (rb,sa,pb,qa), -1j/2)
                        OB -= QubitOperator('Y%d X%d Y%d Y%d'% (ra,sb,pb,qa), -1j/2)
                        OB -= QubitOperator('Y%d X%d Y%d Y%d'% (rb,sa,pa,qb), -1j/2)

                        OA -= hermitian_conjugated(OA)

                        OB -= hermitian_conjugated(OB)

                        coeffA = 0
                        coeffB = 0

                        for t in OA.terms:
                            coeff_t = OA.terms[t]
                            coeffA += coeff_t * coeff_t

                        if np.abs(coeffA) > 1e-6:
                          self.fermi_ops.append(OA)

                        for t in OB.terms:
                            coeff_t = OB.terms[t]
                            coeffB += coeff_t * coeff_t
                            
                        if np.abs(coeffB) > 1e-6:
                          self.fermi_ops.append(OB)

                        # YYXY

                        OA = QubitOperator('X1', 0)
                        OB = QubitOperator('X1', 0)

                        OA += QubitOperator('Y%d Y%d X%d Y%d'% (ra,sa,pa,qa), 1j*2/np.sqrt(12))
                        OA += QubitOperator('Y%d Y%d X%d Y%d'% (rb,sb,pb,qb), 1j*2/np.sqrt(12))
                        OA += QubitOperator('Y%d Y%d X%d Y%d'% (ra,sb,pa,qb), 1j/np.sqrt(12))
                        OA += QubitOperator('Y%d Y%d X%d Y%d'% (rb,sa,pb,qa), 1j/np.sqrt(12))
                        OA += QubitOperator('Y%d Y%d X%d Y%d'% (ra,sb,pb,qa), 1j/np.sqrt(12))
                        OA += QubitOperator('Y%d Y%d X%d Y%d'% (rb,sa,pa,qb), 1j/np.sqrt(12))

                        OB += QubitOperator('Y%d Y%d X%d Y%d'% (ra,sb,pa,qb), 1j/2)
                        OB += QubitOperator('Y%d Y%d X%d Y%d'% (rb,sa,pb,qa), 1j/2)
                        OB -= QubitOperator('Y%d Y%d X%d Y%d'% (ra,sb,pb,qa), 1j/2)
                        OB -= QubitOperator('Y%d Y%d X%d Y%d'% (rb,sa,pa,qb), 1j/2)

                        OA -= hermitian_conjugated(OA)

                        OB -= hermitian_conjugated(OB)

                        coeffA = 0
                        coeffB = 0

                        for t in OA.terms:
                            coeff_t = OA.terms[t]
                            coeffA += coeff_t * coeff_t

                        if np.abs(coeffA) > 1e-6:
                          self.fermi_ops.append(OA)

                        for t in OB.terms:
                            coeff_t = OB.terms[t]
                            coeffB += coeff_t * coeff_t
                            
                        if np.abs(coeffB) > 1e-6:
                          self.fermi_ops.append(OB)

                        # YYYX

                        OA = QubitOperator('X1', 0)
                        OB = QubitOperator('X1', 0)

                        OA += QubitOperator('Y%d Y%d Y%d X%d'% (ra,sa,pa,qa), 1j*2/np.sqrt(12))
                        OA += QubitOperator('Y%d Y%d Y%d X%d'% (rb,sb,pb,qb), 1j*2/np.sqrt(12))
                        OA += QubitOperator('Y%d Y%d Y%d X%d'% (ra,sb,pa,qb), 1j/np.sqrt(12))
                        OA += QubitOperator('Y%d Y%d Y%d X%d'% (rb,sa,pb,qa), 1j/np.sqrt(12))
                        OA += QubitOperator('Y%d Y%d Y%d X%d'% (ra,sb,pb,qa), 1j/np.sqrt(12))
                        OA += QubitOperator('Y%d Y%d Y%d X%d'% (rb,sa,pa,qb), 1j/np.sqrt(12))

                        OB += QubitOperator('Y%d Y%d Y%d X%d'% (ra,sb,pa,qb), 1j/2)
                        OB += QubitOperator('Y%d Y%d Y%d X%d'% (rb,sa,pb,qa), 1j/2)
                        OB -= QubitOperator('Y%d Y%d Y%d X%d'% (ra,sb,pb,qa), 1j/2)
                        OB -= QubitOperator('Y%d Y%d Y%d X%d'% (rb,sa,pa,qb), 1j/2)

                        OA -= hermitian_conjugated(OA)

                        OB -= hermitian_conjugated(OB)

                        coeffA = 0
                        coeffB = 0

                        for t in OA.terms:
                            coeff_t = OA.terms[t]
                            coeffA += coeff_t * coeff_t

                        if np.abs(coeffA) > 1e-6:
                          self.fermi_ops.append(OA)

                        for t in OB.terms:
                            coeff_t = OB.terms[t]
                            coeffB += coeff_t * coeff_t
                            
                        if np.abs(coeffB) > 1e-6:
                          self.fermi_ops.append(OB)

                        # XXXX

                        OA = QubitOperator('X1', 0)
                        OB = QubitOperator('X1', 0)

                        OA += QubitOperator('X%d X%d X%d X%d'% (ra,sa,pa,qa), 2/np.sqrt(12))
                        OA += QubitOperator('X%d X%d X%d X%d'% (rb,sb,pb,qb), 2/np.sqrt(12))
                        OA += QubitOperator('X%d X%d X%d X%d'% (ra,sb,pa,qb), 1/np.sqrt(12))
                        OA += QubitOperator('X%d X%d X%d X%d'% (rb,sa,pb,qa), 1/np.sqrt(12))
                        OA += QubitOperator('X%d X%d X%d X%d'% (ra,sb,pb,qa), 1/np.sqrt(12))
                        OA += QubitOperator('X%d X%d X%d X%d'% (rb,sa,pa,qb), 1/np.sqrt(12))

                        OB += QubitOperator('X%d X%d X%d X%d'% (ra,sb,pa,qb), 1/2)
                        OB += QubitOperator('X%d X%d X%d X%d'% (rb,sa,pb,qa), 1/2)
                        OB -= QubitOperator('X%d X%d X%d X%d'% (ra,sb,pb,qa), 1/2)
                        OB -= QubitOperator('X%d X%d X%d X%d'% (rb,sa,pa,qb), 1/2)

                        OA -= hermitian_conjugated(OA)

                        OB -= hermitian_conjugated(OB)

                        coeffA = 0
                        coeffB = 0

                        for t in OA.terms:
                            coeff_t = OA.terms[t]
                            coeffA += coeff_t * coeff_t

                        if np.abs(coeffA) > 1e-6:
                          self.fermi_ops.append(OA)

                        for t in OB.terms:
                            coeff_t = OB.terms[t]
                            coeffB += coeff_t * coeff_t
                            
                        if np.abs(coeffB) > 1e-6:
                          self.fermi_ops.append(OB)

                        # YYYY

                        OA = QubitOperator('X1', 0)
                        OB = QubitOperator('X1', 0)

                        OA += QubitOperator('Y%d Y%d Y%d Y%d'% (ra,sa,pa,qa), 2/np.sqrt(12))
                        OA += QubitOperator('Y%d Y%d Y%d Y%d'% (rb,sb,pb,qb), 2/np.sqrt(12))
                        OA += QubitOperator('Y%d Y%d Y%d Y%d'% (ra,sb,pa,qb), 1/np.sqrt(12))
                        OA += QubitOperator('Y%d Y%d Y%d Y%d'% (rb,sa,pb,qa), 1/np.sqrt(12))
                        OA += QubitOperator('Y%d Y%d Y%d Y%d'% (ra,sb,pb,qa), 1/np.sqrt(12))
                        OA += QubitOperator('Y%d Y%d Y%d Y%d'% (rb,sa,pa,qb), 1/np.sqrt(12))

                        OB += QubitOperator('Y%d Y%d Y%d Y%d'% (ra,sb,pa,qb), 1/2)
                        OB += QubitOperator('Y%d Y%d Y%d Y%d'% (rb,sa,pb,qa), 1/2)
                        OB -= QubitOperator('Y%d Y%d Y%d Y%d'% (ra,sb,pb,qa), 1/2)
                        OB -= QubitOperator('Y%d Y%d Y%d Y%d'% (rb,sa,pa,qb), 1/2)

                        OA -= hermitian_conjugated(OA)

                        OB -= hermitian_conjugated(OB)

                        coeffA = 0
                        coeffB = 0

                        for t in OA.terms:
                            coeff_t = OA.terms[t]
                            coeffA += coeff_t * coeff_t

                        if np.abs(coeffA) > 1e-6:
                          self.fermi_ops.append(OA)

                        for t in OB.terms:
                            coeff_t = OB.terms[t]
                            coeffB += coeff_t * coeff_t
                            
                        if np.abs(coeffB) > 1e-6:
                          self.fermi_ops.append(OB)

                        # XXYY

                        OA = QubitOperator('X1', 0)
                        OB = QubitOperator('X1', 0)

                        OA += QubitOperator('X%d X%d Y%d Y%d'% (ra,sa,pa,qa), -2/np.sqrt(12))
                        OA += QubitOperator('X%d X%d Y%d Y%d'% (rb,sb,pb,qb), -2/np.sqrt(12))
                        OA += QubitOperator('X%d X%d Y%d Y%d'% (ra,sb,pa,qb), -1/np.sqrt(12))
                        OA += QubitOperator('X%d X%d Y%d Y%d'% (rb,sa,pb,qa), -1/np.sqrt(12))
                        OA += QubitOperator('X%d X%d Y%d Y%d'% (ra,sb,pb,qa), -1/np.sqrt(12))
                        OA += QubitOperator('X%d X%d Y%d Y%d'% (rb,sa,pa,qb), -1/np.sqrt(12))

                        OB += QubitOperator('X%d X%d Y%d Y%d'% (ra,sb,pa,qb), -1/2)
                        OB += QubitOperator('X%d X%d Y%d Y%d'% (rb,sa,pb,qa), -1/2)
                        OB -= QubitOperator('X%d X%d Y%d Y%d'% (ra,sb,pb,qa), -1/2)
                        OB -= QubitOperator('X%d X%d Y%d Y%d'% (rb,sa,pa,qb), -1/2)

                        OA -= hermitian_conjugated(OA)

                        OB -= hermitian_conjugated(OB)

                        coeffA = 0
                        coeffB = 0

                        for t in OA.terms:
                            coeff_t = OA.terms[t]
                            coeffA += coeff_t * coeff_t

                        if np.abs(coeffA) > 1e-6:
                          self.fermi_ops.append(OA)

                        for t in OB.terms:
                            coeff_t = OB.terms[t]
                            coeffB += coeff_t * coeff_t
                            
                        if np.abs(coeffB) > 1e-6:
                          self.fermi_ops.append(OB)

                        # XYXY

                        OA = QubitOperator('X1', 0)
                        OB = QubitOperator('X1', 0)

                        OA += QubitOperator('X%d Y%d X%d Y%d'% (ra,sa,pa,qa), 2/np.sqrt(12))
                        OA += QubitOperator('X%d Y%d X%d Y%d'% (rb,sb,pb,qb), 2/np.sqrt(12))
                        OA += QubitOperator('X%d Y%d X%d Y%d'% (ra,sb,pa,qb), 1/np.sqrt(12))
                        OA += QubitOperator('X%d Y%d X%d Y%d'% (rb,sa,pb,qa), 1/np.sqrt(12))
                        OA += QubitOperator('X%d Y%d X%d Y%d'% (ra,sb,pb,qa), 1/np.sqrt(12))
                        OA += QubitOperator('X%d Y%d X%d Y%d'% (rb,sa,pa,qb), 1/np.sqrt(12))

                        OB += QubitOperator('X%d Y%d X%d Y%d'% (ra,sb,pa,qb), 1/2)
                        OB += QubitOperator('X%d Y%d X%d Y%d'% (rb,sa,pb,qa), 1/2)
                        OB -= QubitOperator('X%d Y%d X%d Y%d'% (ra,sb,pb,qa), 1/2)
                        OB -= QubitOperator('X%d Y%d X%d Y%d'% (rb,sa,pa,qb), 1/2)

                        OA -= hermitian_conjugated(OA)

                        OB -= hermitian_conjugated(OB)

                        coeffA = 0
                        coeffB = 0

                        for t in OA.terms:
                            coeff_t = OA.terms[t]
                            coeffA += coeff_t * coeff_t

                        if np.abs(coeffA) > 1e-6:
                          self.fermi_ops.append(OA)

                        for t in OB.terms:
                            coeff_t = OB.terms[t]
                            coeffB += coeff_t * coeff_t
                            
                        if np.abs(coeffB) > 1e-6:
                          self.fermi_ops.append(OB)

                        # YXXY

                        OA = QubitOperator('X1', 0)
                        OB = QubitOperator('X1', 0)

                        OA += QubitOperator('Y%d X%d X%d Y%d'% (ra,sa,pa,qa), 2/np.sqrt(12))
                        OA += QubitOperator('Y%d X%d X%d Y%d'% (rb,sb,pb,qb), 2/np.sqrt(12))
                        OA += QubitOperator('Y%d X%d X%d Y%d'% (ra,sb,pa,qb), 1/np.sqrt(12))
                        OA += QubitOperator('Y%d X%d X%d Y%d'% (rb,sa,pb,qa), 1/np.sqrt(12))
                        OA += QubitOperator('Y%d X%d X%d Y%d'% (ra,sb,pb,qa), 1/np.sqrt(12))
                        OA += QubitOperator('Y%d X%d X%d Y%d'% (rb,sa,pa,qb), 1/np.sqrt(12))

                        OB += QubitOperator('Y%d X%d X%d Y%d'% (ra,sb,pa,qb), 1/2)
                        OB += QubitOperator('Y%d X%d X%d Y%d'% (rb,sa,pb,qa), 1/2)
                        OB -= QubitOperator('Y%d X%d X%d Y%d'% (ra,sb,pb,qa), 1/2)
                        OB -= QubitOperator('Y%d X%d X%d Y%d'% (rb,sa,pa,qb), 1/2)

                        OA -= hermitian_conjugated(OA)

                        OB -= hermitian_conjugated(OB)

                        coeffA = 0
                        coeffB = 0

                        for t in OA.terms:
                            coeff_t = OA.terms[t]
                            coeffA += coeff_t * coeff_t

                        if np.abs(coeffA) > 1e-6:
                          self.fermi_ops.append(OA)

                        for t in OB.terms:
                            coeff_t = OB.terms[t]
                            coeffB += coeff_t * coeff_t
                            
                        if np.abs(coeffB) > 1e-6:
                          self.fermi_ops.append(OB)

                        # XYYX

                        OA = QubitOperator('X1', 0)
                        OB = QubitOperator('X1', 0)

                        OA += QubitOperator('X%d Y%d Y%d X%d'% (ra,sa,pa,qa), 2/np.sqrt(12))
                        OA += QubitOperator('X%d Y%d Y%d X%d'% (rb,sb,pb,qb), 2/np.sqrt(12))
                        OA += QubitOperator('X%d Y%d Y%d X%d'% (ra,sb,pa,qb), 1/np.sqrt(12))
                        OA += QubitOperator('X%d Y%d Y%d X%d'% (rb,sa,pb,qa), 1/np.sqrt(12))
                        OA += QubitOperator('X%d Y%d Y%d X%d'% (ra,sb,pb,qa), 1/np.sqrt(12))
                        OA += QubitOperator('X%d Y%d Y%d X%d'% (rb,sa,pa,qb), 1/np.sqrt(12))

                        OB += QubitOperator('X%d Y%d Y%d X%d'% (ra,sb,pa,qb), 1/2)
                        OB += QubitOperator('X%d Y%d Y%d X%d'% (rb,sa,pb,qa), 1/2)
                        OB -= QubitOperator('X%d Y%d Y%d X%d'% (ra,sb,pb,qa), 1/2)
                        OB -= QubitOperator('X%d Y%d Y%d X%d'% (rb,sa,pa,qb), 1/2)

                        OA -= hermitian_conjugated(OA)

                        OB -= hermitian_conjugated(OB)

                        coeffA = 0
                        coeffB = 0

                        for t in OA.terms:
                            coeff_t = OA.terms[t]
                            coeffA += coeff_t * coeff_t

                        if np.abs(coeffA) > 1e-6:
                          self.fermi_ops.append(OA)

                        for t in OB.terms:
                            coeff_t = OB.terms[t]
                            coeffB += coeff_t * coeff_t
                            
                        if np.abs(coeffB) > 1e-6:
                          self.fermi_ops.append(OB)

                        # YYXX

                        OA = QubitOperator('X1', 0)
                        OB = QubitOperator('X1', 0)

                        OA += QubitOperator('Y%d Y%d X%d X%d'% (ra,sa,pa,qa), -2/np.sqrt(12))
                        OA += QubitOperator('Y%d Y%d X%d X%d'% (rb,sb,pb,qb), -2/np.sqrt(12))
                        OA += QubitOperator('Y%d Y%d X%d X%d'% (ra,sb,pa,qb), -1/np.sqrt(12))
                        OA += QubitOperator('Y%d Y%d X%d X%d'% (rb,sa,pb,qa), -1/np.sqrt(12))
                        OA += QubitOperator('Y%d Y%d X%d X%d'% (ra,sb,pb,qa), -1/np.sqrt(12))
                        OA += QubitOperator('Y%d Y%d X%d X%d'% (rb,sa,pa,qb), -1/np.sqrt(12))

                        OB += QubitOperator('Y%d Y%d X%d X%d'% (ra,sb,pa,qb), -1/2)
                        OB += QubitOperator('Y%d Y%d X%d X%d'% (rb,sa,pb,qa), -1/2)
                        OB -= QubitOperator('Y%d Y%d X%d X%d'% (ra,sb,pb,qa), -1/2)
                        OB -= QubitOperator('Y%d Y%d X%d X%d'% (rb,sa,pa,qb), -1/2)

                        OA -= hermitian_conjugated(OA)

                        OB -= hermitian_conjugated(OB)

                        coeffA = 0
                        coeffB = 0

                        for t in OA.terms:
                            coeff_t = OA.terms[t]
                            coeffA += coeff_t * coeff_t

                        if np.abs(coeffA) > 1e-6:
                          self.fermi_ops.append(OA)

                        for t in OB.terms:
                            coeff_t = OB.terms[t]
                            coeffB += coeff_t * coeff_t
                            
                        if np.abs(coeffB) > 1e-6:
                          self.fermi_ops.append(OB)

                        # YXYX

                        OA = QubitOperator('X1', 0)
                        OB = QubitOperator('X1', 0)

                        OA += QubitOperator('Y%d X%d Y%d X%d'% (ra,sa,pa,qa), 2/np.sqrt(12))
                        OA += QubitOperator('Y%d X%d Y%d X%d'% (rb,sb,pb,qb), 2/np.sqrt(12))
                        OA += QubitOperator('Y%d X%d Y%d X%d'% (ra,sb,pa,qb), 1/np.sqrt(12))
                        OA += QubitOperator('Y%d X%d Y%d X%d'% (rb,sa,pb,qa), 1/np.sqrt(12))
                        OA += QubitOperator('Y%d X%d Y%d X%d'% (ra,sb,pb,qa), 1/np.sqrt(12))
                        OA += QubitOperator('Y%d X%d Y%d X%d'% (rb,sa,pa,qb), 1/np.sqrt(12))

                        OB += QubitOperator('Y%d X%d Y%d X%d'% (ra,sb,pa,qb), 1/2)
                        OB += QubitOperator('Y%d X%d Y%d X%d'% (rb,sa,pb,qa), 1/2)
                        OB -= QubitOperator('Y%d X%d Y%d X%d'% (ra,sb,pb,qa), 1/2)
                        OB -= QubitOperator('Y%d X%d Y%d X%d'% (rb,sa,pa,qb), 1/2)

                        OA -= hermitian_conjugated(OA)

                        OB -= hermitian_conjugated(OB)

                        coeffA = 0
                        coeffB = 0

                        for t in OA.terms:
                            coeff_t = OA.terms[t]
                            coeffA += coeff_t * coeff_t

                        if np.abs(coeffA) > 1e-6:
                          self.fermi_ops.append(OA)

                        for t in OB.terms:
                            coeff_t = OB.terms[t]
                            coeffB += coeff_t * coeff_t
                            
                        if np.abs(coeffB) > 1e-6:
                          self.fermi_ops.append(OB)

        for op in self.fermi_ops:
            print(op)

        self.n_ops = len(self.fermi_ops)
        print(" Number of operators: ", self.n_ops)
        return 

class qubits_GP2(OperatorPool): # factorize group
    def generate_SQ_Operators(self):
        """
        n_orb is number of spatial orbitals assuming that spin orbitals are labelled
        0a,0b,1a,1b,2a,2b,3a,3b,....  -> 0,1,2,3,...
        """
        
        print(" Form singlet GSD operators")
        
        self.fermi_ops = []
        for p in range(0,self.n_orb):
            pa = 2*p
            pb = 2*p+1
 
            for q in range(p,self.n_orb):
                qa = 2*q
                qb = 2*q+1


                termA =  QubitOperator('X%d Y%d'% (pa,qa), 1j)
                termA -= QubitOperator('Y%d X%d'% (pa,qa), 1j)
                termA -= hermitian_conjugated(termA)
                coeffA = 0

                for t in termA.terms:
                    coeff_t = termA.terms[t]
                    coeffA += coeff_t * coeff_t

                if np.abs(coeffA) > 1e-6:
                  self.fermi_ops.append(termA)

                termA =  QubitOperator('X%d Y%d'% (pb,qb), 1j)
                termA -= QubitOperator('Y%d X%d'% (pb,qb), 1j)
                termA -= hermitian_conjugated(termA)
                coeffA = 0

                for t in termA.terms:
                    coeff_t = termA.terms[t]
                    coeffA += coeff_t * coeff_t

                if np.abs(coeffA) > 1e-6:
                  self.fermi_ops.append(termA)

        pq = -1 
        for p in range(0,self.n_orb):
            pa = 2*p
            pb = 2*p+1
 
            for q in range(p,self.n_orb):
                qa = 2*q
                qb = 2*q+1
        
                pq += 1
        
                rs = -1 
                for r in range(0,self.n_orb):
                    ra = 2*r
                    rb = 2*r+1
                    
                    for s in range(r,self.n_orb):
                        sa = 2*s
                        sb = 2*s+1
                    
                        rs += 1
                    
                        if(pq > rs):
                            continue

                        OA = QubitOperator('X1', 0)

                        OA += QubitOperator('Y%d X%d X%d X%d'% (ra,sa,pa,qa), 1j)
                        OA += QubitOperator('X%d Y%d X%d X%d'% (ra,sa,pa,qa), 1j)
                        OA += QubitOperator('X%d X%d Y%d X%d'% (ra,sa,pa,qa),-1j)
                        OA += QubitOperator('X%d X%d X%d Y%d'% (ra,sa,pa,qa),-1j)
                        OA += QubitOperator('X%d Y%d Y%d Y%d'% (ra,sa,pa,qa),-1j)
                        OA += QubitOperator('Y%d X%d Y%d Y%d'% (ra,sa,pa,qa),-1j)
                        OA += QubitOperator('Y%d Y%d X%d Y%d'% (ra,sa,pa,qa), 1j)
                        OA += QubitOperator('Y%d Y%d Y%d X%d'% (ra,sa,pa,qa), 1j)

                        OA += QubitOperator('X%d X%d X%d X%d'% (ra,sa,pa,qa), 1)
                        OA += QubitOperator('Y%d Y%d Y%d Y%d'% (ra,sa,pa,qa), 1)
                        OA += QubitOperator('X%d X%d Y%d Y%d'% (ra,sa,pa,qa),-1)
                        OA += QubitOperator('X%d Y%d X%d Y%d'% (ra,sa,pa,qa), 1)
                        OA += QubitOperator('Y%d X%d X%d Y%d'% (ra,sa,pa,qa), 1)
                        OA += QubitOperator('X%d Y%d Y%d X%d'% (ra,sa,pa,qa), 1)
                        OA += QubitOperator('Y%d Y%d X%d X%d'% (ra,sa,pa,qa),-1)
                        OA += QubitOperator('Y%d X%d Y%d X%d'% (ra,sa,pa,qa), 1)

                        OA -= hermitian_conjugated(OA)
                        coeffA = 0
        
                        for t in OA.terms:
                            coeff_t = OA.terms[t]
                            coeffA += coeff_t * coeff_t
        
                        if np.abs(coeffA) > 1e-6:
                          self.fermi_ops.append(OA)

                        # termA += FermionOperator(((rb,1),(pb,0),(sb,1),(qb,0)), 2/np.sqrt(12))

                        OA = QubitOperator('X1', 0)

                        OA += QubitOperator('Y%d X%d X%d X%d'% (rb,sb,pb,qb), 1j)
                        OA += QubitOperator('X%d Y%d X%d X%d'% (rb,sb,pb,qb), 1j)
                        OA += QubitOperator('X%d X%d Y%d X%d'% (rb,sb,pb,qb),-1j)
                        OA += QubitOperator('X%d X%d X%d Y%d'% (rb,sb,pb,qb),-1j)
                        OA += QubitOperator('X%d Y%d Y%d Y%d'% (rb,sb,pb,qb),-1j)
                        OA += QubitOperator('Y%d X%d Y%d Y%d'% (rb,sb,pb,qb),-1j)
                        OA += QubitOperator('Y%d Y%d X%d Y%d'% (rb,sb,pb,qb), 1j)
                        OA += QubitOperator('Y%d Y%d Y%d X%d'% (rb,sb,pb,qb), 1j)

                        OA += QubitOperator('X%d X%d X%d X%d'% (rb,sb,pb,qb), 1)
                        OA += QubitOperator('Y%d Y%d Y%d Y%d'% (rb,sb,pb,qb), 1)
                        OA += QubitOperator('X%d X%d Y%d Y%d'% (rb,sb,pb,qb),-1)
                        OA += QubitOperator('X%d Y%d X%d Y%d'% (rb,sb,pb,qb), 1)
                        OA += QubitOperator('Y%d X%d X%d Y%d'% (rb,sb,pb,qb), 1)
                        OA += QubitOperator('X%d Y%d Y%d X%d'% (rb,sb,pb,qb), 1)
                        OA += QubitOperator('Y%d Y%d X%d X%d'% (rb,sb,pb,qb),-1)
                        OA += QubitOperator('Y%d X%d Y%d X%d'% (rb,sb,pb,qb), 1)

                        OA -= hermitian_conjugated(OA)
                        coeffA = 0
        
                        for t in OA.terms:
                            coeff_t = OA.terms[t]
                            coeffA += coeff_t * coeff_t
        
                        if np.abs(coeffA) > 1e-6:
                          self.fermi_ops.append(OA)

                        # termA += FermionOperator(((ra,1),(pa,0),(sb,1),(qb,0)), 1/np.sqrt(12))

                        OA = QubitOperator('X1', 0)

                        OA += QubitOperator('Y%d X%d X%d X%d'% (ra,sb,pa,qb), 1j)
                        OA += QubitOperator('X%d Y%d X%d X%d'% (ra,sb,pa,qb), 1j)
                        OA += QubitOperator('X%d X%d Y%d X%d'% (ra,sb,pa,qb),-1j)
                        OA += QubitOperator('X%d X%d X%d Y%d'% (ra,sb,pa,qb),-1j)
                        OA += QubitOperator('X%d Y%d Y%d Y%d'% (ra,sb,pa,qb),-1j)
                        OA += QubitOperator('Y%d X%d Y%d Y%d'% (ra,sb,pa,qb),-1j)
                        OA += QubitOperator('Y%d Y%d X%d Y%d'% (ra,sb,pa,qb), 1j)
                        OA += QubitOperator('Y%d Y%d Y%d X%d'% (ra,sb,pa,qb), 1j)

                        OA += QubitOperator('X%d X%d X%d X%d'% (ra,sb,pa,qb), 1)
                        OA += QubitOperator('Y%d Y%d Y%d Y%d'% (ra,sb,pa,qb), 1)
                        OA += QubitOperator('X%d X%d Y%d Y%d'% (ra,sb,pa,qb),-1)
                        OA += QubitOperator('X%d Y%d X%d Y%d'% (ra,sb,pa,qb), 1)
                        OA += QubitOperator('Y%d X%d X%d Y%d'% (ra,sb,pa,qb), 1)
                        OA += QubitOperator('X%d Y%d Y%d X%d'% (ra,sb,pa,qb), 1)
                        OA += QubitOperator('Y%d Y%d X%d X%d'% (ra,sb,pa,qb),-1)
                        OA += QubitOperator('Y%d X%d Y%d X%d'% (ra,sb,pa,qb), 1)

                        OA -= hermitian_conjugated(OA)
                        coeffA = 0
        
                        for t in OA.terms:
                            coeff_t = OA.terms[t]
                            coeffA += coeff_t * coeff_t
        
                        if np.abs(coeffA) > 1e-6:
                          self.fermi_ops.append(OA)

                        # termA += FermionOperator(((rb,1),(pb,0),(sa,1),(qa,0)), 1/np.sqrt(12))

                        OA = QubitOperator('X1', 0)

                        OA += QubitOperator('Y%d X%d X%d X%d'% (rb,sa,pb,qa), 1j)
                        OA += QubitOperator('X%d Y%d X%d X%d'% (rb,sa,pb,qa), 1j)
                        OA += QubitOperator('X%d X%d Y%d X%d'% (rb,sa,pb,qa),-1j)
                        OA += QubitOperator('X%d X%d X%d Y%d'% (rb,sa,pb,qa),-1j)
                        OA += QubitOperator('X%d Y%d Y%d Y%d'% (rb,sa,pb,qa),-1j)
                        OA += QubitOperator('Y%d X%d Y%d Y%d'% (rb,sa,pb,qa),-1j)
                        OA += QubitOperator('Y%d Y%d X%d Y%d'% (rb,sa,pb,qa), 1j)
                        OA += QubitOperator('Y%d Y%d Y%d X%d'% (rb,sa,pb,qa), 1j)

                        OA += QubitOperator('X%d X%d X%d X%d'% (rb,sa,pb,qa), 1)
                        OA += QubitOperator('Y%d Y%d Y%d Y%d'% (rb,sa,pb,qa), 1)
                        OA += QubitOperator('X%d X%d Y%d Y%d'% (rb,sa,pb,qa),-1)
                        OA += QubitOperator('X%d Y%d X%d Y%d'% (rb,sa,pb,qa), 1)
                        OA += QubitOperator('Y%d X%d X%d Y%d'% (rb,sa,pb,qa), 1)
                        OA += QubitOperator('X%d Y%d Y%d X%d'% (rb,sa,pb,qa), 1)
                        OA += QubitOperator('Y%d Y%d X%d X%d'% (rb,sa,pb,qa),-1)
                        OA += QubitOperator('Y%d X%d Y%d X%d'% (rb,sa,pb,qa), 1)

                        OA -= hermitian_conjugated(OA)
                        coeffA = 0
        
                        for t in OA.terms:
                            coeff_t = OA.terms[t]
                            coeffA += coeff_t * coeff_t
        
                        if np.abs(coeffA) > 1e-6:
                          self.fermi_ops.append(OA)

                        # termA += FermionOperator(((ra,1),(pb,0),(sb,1),(qa,0)), 1/np.sqrt(12))

                        OA = QubitOperator('X1', 0)
                        
                        OA += QubitOperator('Y%d X%d X%d X%d'% (ra,sb,pb,qa), 1j)
                        OA += QubitOperator('X%d Y%d X%d X%d'% (ra,sb,pb,qa), 1j)
                        OA += QubitOperator('X%d X%d Y%d X%d'% (ra,sb,pb,qa),-1j)
                        OA += QubitOperator('X%d X%d X%d Y%d'% (ra,sb,pb,qa),-1j)
                        OA += QubitOperator('X%d Y%d Y%d Y%d'% (ra,sb,pb,qa),-1j)
                        OA += QubitOperator('Y%d X%d Y%d Y%d'% (ra,sb,pb,qa),-1j)
                        OA += QubitOperator('Y%d Y%d X%d Y%d'% (ra,sb,pb,qa), 1j)
                        OA += QubitOperator('Y%d Y%d Y%d X%d'% (ra,sb,pb,qa), 1j)

                        OA += QubitOperator('X%d X%d X%d X%d'% (ra,sb,pb,qa), 1)
                        OA += QubitOperator('Y%d Y%d Y%d Y%d'% (ra,sb,pb,qa), 1)
                        OA += QubitOperator('X%d X%d Y%d Y%d'% (ra,sb,pb,qa),-1)
                        OA += QubitOperator('X%d Y%d X%d Y%d'% (ra,sb,pb,qa), 1)
                        OA += QubitOperator('Y%d X%d X%d Y%d'% (ra,sb,pb,qa), 1)
                        OA += QubitOperator('X%d Y%d Y%d X%d'% (ra,sb,pb,qa), 1)
                        OA += QubitOperator('Y%d Y%d X%d X%d'% (ra,sb,pb,qa),-1)
                        OA += QubitOperator('Y%d X%d Y%d X%d'% (ra,sb,pb,qa), 1)

                        OA -= hermitian_conjugated(OA)
                        coeffA = 0
        
                        for t in OA.terms:
                            coeff_t = OA.terms[t]
                            coeffA += coeff_t * coeff_t
        
                        if np.abs(coeffA) > 1e-6:
                          self.fermi_ops.append(OA)

                        # termA += FermionOperator(((rb,1),(pa,0),(sa,1),(qb,0)), 1/np.sqrt(12))

                        OA = QubitOperator('X1', 0)
                                                                
                        OA += QubitOperator('Y%d X%d X%d X%d'% (rb,sa,pa,qb), 1j)
                        OA += QubitOperator('X%d Y%d X%d X%d'% (rb,sa,pa,qb), 1j)
                        OA += QubitOperator('X%d X%d Y%d X%d'% (rb,sa,pa,qb),-1j)
                        OA += QubitOperator('X%d X%d X%d Y%d'% (rb,sa,pa,qb),-1j)
                        OA += QubitOperator('X%d Y%d Y%d Y%d'% (rb,sa,pa,qb),-1j)
                        OA += QubitOperator('Y%d X%d Y%d Y%d'% (rb,sa,pa,qb),-1j)
                        OA += QubitOperator('Y%d Y%d X%d Y%d'% (rb,sa,pa,qb), 1j)
                        OA += QubitOperator('Y%d Y%d Y%d X%d'% (rb,sa,pa,qb), 1j)

                        OA += QubitOperator('X%d X%d X%d X%d'% (rb,sa,pa,qb), 1)
                        OA += QubitOperator('Y%d Y%d Y%d Y%d'% (rb,sa,pa,qb), 1)
                        OA += QubitOperator('X%d X%d Y%d Y%d'% (rb,sa,pa,qb),-1)
                        OA += QubitOperator('X%d Y%d X%d Y%d'% (rb,sa,pa,qb), 1)
                        OA += QubitOperator('Y%d X%d X%d Y%d'% (rb,sa,pa,qb), 1)
                        OA += QubitOperator('X%d Y%d Y%d X%d'% (rb,sa,pa,qb), 1)
                        OA += QubitOperator('Y%d Y%d X%d X%d'% (rb,sa,pa,qb),-1)
                        OA += QubitOperator('Y%d X%d Y%d X%d'% (rb,sa,pa,qb), 1)

                        OA -= hermitian_conjugated(OA)
                        coeffA = 0
        
                        for t in OA.terms:
                            coeff_t = OA.terms[t]
                            coeffA += coeff_t * coeff_t
        
                        if np.abs(coeffA) > 1e-6:
                          self.fermi_ops.append(OA)

        for op in self.fermi_ops:
            print(op)

        self.n_ops = len(self.fermi_ops)
        print(" Number of operators: ", self.n_ops)
        return 

class qubits_sym_check(OperatorPool): # factorize group all sign +
    def generate_SQ_Operators(self):
        """
        n_orb is number of spatial orbitals assuming that spin orbitals are labelled
        0a,0b,1a,1b,2a,2b,3a,3b,....  -> 0,1,2,3,...
        """
        
        print(" Form singlet GSD operators")
        
        self.fermi_ops = []
        for p in range(0,self.n_orb):
            pa = 2*p
            pb = 2*p+1
 
            for q in range(p,self.n_orb):
                qa = 2*q
                qb = 2*q+1


                termA =  QubitOperator('X%d Y%d'% (pa,qa), 1j)
                termA += QubitOperator('Y%d X%d'% (pa,qa), 1j)
                termA -= hermitian_conjugated(termA)
                coeffA = 0

                for t in termA.terms:
                    coeff_t = termA.terms[t]
                    coeffA += coeff_t * coeff_t

                if np.abs(coeffA) > 1e-6:
                  self.fermi_ops.append(termA)

                termA =  QubitOperator('X%d Y%d'% (pb,qb), 1j)
                termA += QubitOperator('Y%d X%d'% (pb,qb), 1j)
                termA -= hermitian_conjugated(termA)
                coeffA = 0

                for t in termA.terms:
                    coeff_t = termA.terms[t]
                    coeffA += coeff_t * coeff_t

                if np.abs(coeffA) > 1e-6:
                  self.fermi_ops.append(termA)

        pq = -1 
        for p in range(0,self.n_orb):
            pa = 2*p
            pb = 2*p+1
 
            for q in range(p,self.n_orb):
                qa = 2*q
                qb = 2*q+1
        
                pq += 1
        
                rs = -1 
                for r in range(0,self.n_orb):
                    ra = 2*r
                    rb = 2*r+1
                    
                    for s in range(r,self.n_orb):
                        sa = 2*s
                        sb = 2*s+1
                    
                        rs += 1
                    
                        if(pq > rs):
                            continue

                        OA = QubitOperator('X1', 0)

                        OA += QubitOperator('Y%d X%d X%d X%d'% (ra,sa,pa,qa), 1j)
                        OA += QubitOperator('X%d Y%d X%d X%d'% (ra,sa,pa,qa), 1j)
                        OA += QubitOperator('X%d X%d Y%d X%d'% (ra,sa,pa,qa), 1j)
                        OA += QubitOperator('X%d X%d X%d Y%d'% (ra,sa,pa,qa), 1j)
                        OA += QubitOperator('X%d Y%d Y%d Y%d'% (ra,sa,pa,qa), 1j)
                        OA += QubitOperator('Y%d X%d Y%d Y%d'% (ra,sa,pa,qa), 1j)
                        OA += QubitOperator('Y%d Y%d X%d Y%d'% (ra,sa,pa,qa), 1j)
                        OA += QubitOperator('Y%d Y%d Y%d X%d'% (ra,sa,pa,qa), 1j)

                        OA += QubitOperator('X%d X%d X%d X%d'% (ra,sa,pa,qa), 1)
                        OA += QubitOperator('Y%d Y%d Y%d Y%d'% (ra,sa,pa,qa), 1)
                        OA += QubitOperator('X%d X%d Y%d Y%d'% (ra,sa,pa,qa), 1)
                        OA += QubitOperator('X%d Y%d X%d Y%d'% (ra,sa,pa,qa), 1)
                        OA += QubitOperator('Y%d X%d X%d Y%d'% (ra,sa,pa,qa), 1)
                        OA += QubitOperator('X%d Y%d Y%d X%d'% (ra,sa,pa,qa), 1)
                        OA += QubitOperator('Y%d Y%d X%d X%d'% (ra,sa,pa,qa), 1)
                        OA += QubitOperator('Y%d X%d Y%d X%d'% (ra,sa,pa,qa), 1)

                        OA -= hermitian_conjugated(OA)
                        coeffA = 0
        
                        for t in OA.terms:
                            coeff_t = OA.terms[t]
                            coeffA += coeff_t * coeff_t
        
                        if np.abs(coeffA) > 1e-6:
                          self.fermi_ops.append(OA)

                        # termA += FermionOperator(((rb,1),(pb,0),(sb,1),(qb,0)), 2/np.sqrt(12))

                        OA = QubitOperator('X1', 0)

                        OA += QubitOperator('Y%d X%d X%d X%d'% (rb,sb,pb,qb), 1j)
                        OA += QubitOperator('X%d Y%d X%d X%d'% (rb,sb,pb,qb), 1j)
                        OA += QubitOperator('X%d X%d Y%d X%d'% (rb,sb,pb,qb), 1j)
                        OA += QubitOperator('X%d X%d X%d Y%d'% (rb,sb,pb,qb), 1j)
                        OA += QubitOperator('X%d Y%d Y%d Y%d'% (rb,sb,pb,qb), 1j)
                        OA += QubitOperator('Y%d X%d Y%d Y%d'% (rb,sb,pb,qb), 1j)
                        OA += QubitOperator('Y%d Y%d X%d Y%d'% (rb,sb,pb,qb), 1j)
                        OA += QubitOperator('Y%d Y%d Y%d X%d'% (rb,sb,pb,qb), 1j)

                        OA += QubitOperator('X%d X%d X%d X%d'% (rb,sb,pb,qb), 1)
                        OA += QubitOperator('Y%d Y%d Y%d Y%d'% (rb,sb,pb,qb), 1)
                        OA += QubitOperator('X%d X%d Y%d Y%d'% (rb,sb,pb,qb), 1)
                        OA += QubitOperator('X%d Y%d X%d Y%d'% (rb,sb,pb,qb), 1)
                        OA += QubitOperator('Y%d X%d X%d Y%d'% (rb,sb,pb,qb), 1)
                        OA += QubitOperator('X%d Y%d Y%d X%d'% (rb,sb,pb,qb), 1)
                        OA += QubitOperator('Y%d Y%d X%d X%d'% (rb,sb,pb,qb), 1)
                        OA += QubitOperator('Y%d X%d Y%d X%d'% (rb,sb,pb,qb), 1)

                        OA -= hermitian_conjugated(OA)
                        coeffA = 0
        
                        for t in OA.terms:
                            coeff_t = OA.terms[t]
                            coeffA += coeff_t * coeff_t
        
                        if np.abs(coeffA) > 1e-6:
                          self.fermi_ops.append(OA)

                        # termA += FermionOperator(((ra,1),(pa,0),(sb,1),(qb,0)), 1/np.sqrt(12))

                        OA = QubitOperator('X1', 0)

                        OA += QubitOperator('Y%d X%d X%d X%d'% (ra,sb,pa,qb), 1j)
                        OA += QubitOperator('X%d Y%d X%d X%d'% (ra,sb,pa,qb), 1j)
                        OA += QubitOperator('X%d X%d Y%d X%d'% (ra,sb,pa,qb), 1j)
                        OA += QubitOperator('X%d X%d X%d Y%d'% (ra,sb,pa,qb), 1j)
                        OA += QubitOperator('X%d Y%d Y%d Y%d'% (ra,sb,pa,qb), 1j)
                        OA += QubitOperator('Y%d X%d Y%d Y%d'% (ra,sb,pa,qb), 1j)
                        OA += QubitOperator('Y%d Y%d X%d Y%d'% (ra,sb,pa,qb), 1j)
                        OA += QubitOperator('Y%d Y%d Y%d X%d'% (ra,sb,pa,qb), 1j)

                        OA += QubitOperator('X%d X%d X%d X%d'% (ra,sb,pa,qb), 1)
                        OA += QubitOperator('Y%d Y%d Y%d Y%d'% (ra,sb,pa,qb), 1)
                        OA += QubitOperator('X%d X%d Y%d Y%d'% (ra,sb,pa,qb), 1)
                        OA += QubitOperator('X%d Y%d X%d Y%d'% (ra,sb,pa,qb), 1)
                        OA += QubitOperator('Y%d X%d X%d Y%d'% (ra,sb,pa,qb), 1)
                        OA += QubitOperator('X%d Y%d Y%d X%d'% (ra,sb,pa,qb), 1)
                        OA += QubitOperator('Y%d Y%d X%d X%d'% (ra,sb,pa,qb), 1)
                        OA += QubitOperator('Y%d X%d Y%d X%d'% (ra,sb,pa,qb), 1)

                        OA -= hermitian_conjugated(OA)
                        coeffA = 0
        
                        for t in OA.terms:
                            coeff_t = OA.terms[t]
                            coeffA += coeff_t * coeff_t
        
                        if np.abs(coeffA) > 1e-6:
                          self.fermi_ops.append(OA)

                        # termA += FermionOperator(((rb,1),(pb,0),(sa,1),(qa,0)), 1/np.sqrt(12))

                        OA = QubitOperator('X1', 0)

                        OA += QubitOperator('Y%d X%d X%d X%d'% (rb,sa,pb,qa), 1j)
                        OA += QubitOperator('X%d Y%d X%d X%d'% (rb,sa,pb,qa), 1j)
                        OA += QubitOperator('X%d X%d Y%d X%d'% (rb,sa,pb,qa), 1j)
                        OA += QubitOperator('X%d X%d X%d Y%d'% (rb,sa,pb,qa), 1j)
                        OA += QubitOperator('X%d Y%d Y%d Y%d'% (rb,sa,pb,qa), 1j)
                        OA += QubitOperator('Y%d X%d Y%d Y%d'% (rb,sa,pb,qa), 1j)
                        OA += QubitOperator('Y%d Y%d X%d Y%d'% (rb,sa,pb,qa), 1j)
                        OA += QubitOperator('Y%d Y%d Y%d X%d'% (rb,sa,pb,qa), 1j)

                        OA += QubitOperator('X%d X%d X%d X%d'% (rb,sa,pb,qa), 1)
                        OA += QubitOperator('Y%d Y%d Y%d Y%d'% (rb,sa,pb,qa), 1)
                        OA += QubitOperator('X%d X%d Y%d Y%d'% (rb,sa,pb,qa), 1)
                        OA += QubitOperator('X%d Y%d X%d Y%d'% (rb,sa,pb,qa), 1)
                        OA += QubitOperator('Y%d X%d X%d Y%d'% (rb,sa,pb,qa), 1)
                        OA += QubitOperator('X%d Y%d Y%d X%d'% (rb,sa,pb,qa), 1)
                        OA += QubitOperator('Y%d Y%d X%d X%d'% (rb,sa,pb,qa), 1)
                        OA += QubitOperator('Y%d X%d Y%d X%d'% (rb,sa,pb,qa), 1)

                        OA -= hermitian_conjugated(OA)
                        coeffA = 0
        
                        for t in OA.terms:
                            coeff_t = OA.terms[t]
                            coeffA += coeff_t * coeff_t
        
                        if np.abs(coeffA) > 1e-6:
                          self.fermi_ops.append(OA)

                        # termA += FermionOperator(((ra,1),(pb,0),(sb,1),(qa,0)), 1/np.sqrt(12))

                        OA = QubitOperator('X1', 0)
                        
                        OA += QubitOperator('Y%d X%d X%d X%d'% (ra,sb,pb,qa), 1j)
                        OA += QubitOperator('X%d Y%d X%d X%d'% (ra,sb,pb,qa), 1j)
                        OA += QubitOperator('X%d X%d Y%d X%d'% (ra,sb,pb,qa), 1j)
                        OA += QubitOperator('X%d X%d X%d Y%d'% (ra,sb,pb,qa), 1j)
                        OA += QubitOperator('X%d Y%d Y%d Y%d'% (ra,sb,pb,qa), 1j)
                        OA += QubitOperator('Y%d X%d Y%d Y%d'% (ra,sb,pb,qa), 1j)
                        OA += QubitOperator('Y%d Y%d X%d Y%d'% (ra,sb,pb,qa), 1j)
                        OA += QubitOperator('Y%d Y%d Y%d X%d'% (ra,sb,pb,qa), 1j)

                        OA += QubitOperator('X%d X%d X%d X%d'% (ra,sb,pb,qa), 1)
                        OA += QubitOperator('Y%d Y%d Y%d Y%d'% (ra,sb,pb,qa), 1)
                        OA += QubitOperator('X%d X%d Y%d Y%d'% (ra,sb,pb,qa), 1)
                        OA += QubitOperator('X%d Y%d X%d Y%d'% (ra,sb,pb,qa), 1)
                        OA += QubitOperator('Y%d X%d X%d Y%d'% (ra,sb,pb,qa), 1)
                        OA += QubitOperator('X%d Y%d Y%d X%d'% (ra,sb,pb,qa), 1)
                        OA += QubitOperator('Y%d Y%d X%d X%d'% (ra,sb,pb,qa), 1)
                        OA += QubitOperator('Y%d X%d Y%d X%d'% (ra,sb,pb,qa), 1)

                        OA -= hermitian_conjugated(OA)
                        coeffA = 0
        
                        for t in OA.terms:
                            coeff_t = OA.terms[t]
                            coeffA += coeff_t * coeff_t
        
                        if np.abs(coeffA) > 1e-6:
                          self.fermi_ops.append(OA)

                        # termA += FermionOperator(((rb,1),(pa,0),(sa,1),(qb,0)), 1/np.sqrt(12))

                        OA = QubitOperator('X1', 0)
                                                                
                        OA += QubitOperator('Y%d X%d X%d X%d'% (rb,sa,pa,qb), 1j)
                        OA += QubitOperator('X%d Y%d X%d X%d'% (rb,sa,pa,qb), 1j)
                        OA += QubitOperator('X%d X%d Y%d X%d'% (rb,sa,pa,qb), 1j)
                        OA += QubitOperator('X%d X%d X%d Y%d'% (rb,sa,pa,qb), 1j)
                        OA += QubitOperator('X%d Y%d Y%d Y%d'% (rb,sa,pa,qb), 1j)
                        OA += QubitOperator('Y%d X%d Y%d Y%d'% (rb,sa,pa,qb), 1j)
                        OA += QubitOperator('Y%d Y%d X%d Y%d'% (rb,sa,pa,qb), 1j)
                        OA += QubitOperator('Y%d Y%d Y%d X%d'% (rb,sa,pa,qb), 1j)

                        OA += QubitOperator('X%d X%d X%d X%d'% (rb,sa,pa,qb), 1)
                        OA += QubitOperator('Y%d Y%d Y%d Y%d'% (rb,sa,pa,qb), 1)
                        OA += QubitOperator('X%d X%d Y%d Y%d'% (rb,sa,pa,qb), 1)
                        OA += QubitOperator('X%d Y%d X%d Y%d'% (rb,sa,pa,qb), 1)
                        OA += QubitOperator('Y%d X%d X%d Y%d'% (rb,sa,pa,qb), 1)
                        OA += QubitOperator('X%d Y%d Y%d X%d'% (rb,sa,pa,qb), 1)
                        OA += QubitOperator('Y%d Y%d X%d X%d'% (rb,sa,pa,qb), 1)
                        OA += QubitOperator('Y%d X%d Y%d X%d'% (rb,sa,pa,qb), 1)

                        OA -= hermitian_conjugated(OA)
                        coeffA = 0
        
                        for t in OA.terms:
                            coeff_t = OA.terms[t]
                            coeffA += coeff_t * coeff_t
        
                        if np.abs(coeffA) > 1e-6:
                          self.fermi_ops.append(OA)

        for op in self.fermi_ops:
            print(op)

        self.n_ops = len(self.fermi_ops)
        print(" Number of operators: ", self.n_ops)
        return 

class singlet_GSD_nZs(OperatorPool):
    def generate_SQ_Operators(self):
        """
        n_orb is number of spatial orbitals assuming that spin orbitals are labelled
        0a,0b,1a,1b,2a,2b,3a,3b,....  -> 0,1,2,3,...
        """
        
        print(" Form singlet GSD operators")
        
        self.fermi_ops = []
        for p in range(0,self.n_orb):
            pa = 2*p
            pb = 2*p+1
 
            for q in range(p,self.n_orb):
                qa = 2*q
                qb = 2*q+1


                termA =  QubitOperator('X%d Y%d'% (pa,qa), 1j)
                termA -= QubitOperator('Y%d X%d'% (pa,qa), 1j)
                termA += QubitOperator('X%d Y%d'% (pb,qb), 1j)
                termA -= QubitOperator('Y%d X%d'% (pb,qb), 1j)

                termA -= hermitian_conjugated(termA)

                coeffA = 0

                for t in termA.terms:
                    coeff_t = termA.terms[t]
                    coeffA += coeff_t * coeff_t

                if np.abs(coeffA) > 1e-6:
                  self.fermi_ops.append(termA)

        pq = -1 
        for p in range(0,self.n_orb):
            pa = 2*p
            pb = 2*p+1
 
            for q in range(p,self.n_orb):
                qa = 2*q
                qb = 2*q+1
        
                pq += 1
        
                rs = -1 
                for r in range(0,self.n_orb):
                    ra = 2*r
                    rb = 2*r+1
                    
                    for s in range(r,self.n_orb):
                        sa = 2*s
                        sb = 2*s+1
                    
                        rs += 1
                    
                        if(pq > rs):
                            continue

                        OA = QubitOperator('X1', 0)
                        OB = QubitOperator('X1', 0)

                        # termA =  FermionOperator(((ra,1),(pa,0),(sa,1),(qa,0)), 2/np.sqrt(12))

                        OA += QubitOperator('Y%d X%d X%d X%d'% (ra,sa,pa,qa), 1j*2/np.sqrt(12))
                        OA += QubitOperator('X%d Y%d X%d X%d'% (ra,sa,pa,qa), 1j*2/np.sqrt(12))
                        OA += QubitOperator('X%d X%d Y%d X%d'% (ra,sa,pa,qa), -1j*2/np.sqrt(12))
                        OA += QubitOperator('X%d X%d X%d Y%d'% (ra,sa,pa,qa), -1j*2/np.sqrt(12))
                        OA += QubitOperator('X%d Y%d Y%d Y%d'% (ra,sa,pa,qa), -1j*2/np.sqrt(12))
                        OA += QubitOperator('Y%d X%d Y%d Y%d'% (ra,sa,pa,qa), -1j*2/np.sqrt(12))
                        OA += QubitOperator('Y%d Y%d X%d Y%d'% (ra,sa,pa,qa), 1j*2/np.sqrt(12))
                        OA += QubitOperator('Y%d Y%d Y%d X%d'% (ra,sa,pa,qa), 1j*2/np.sqrt(12))

                        OA += QubitOperator('X%d X%d X%d X%d'% (ra,sa,pa,qa), 2/np.sqrt(12))
                        OA += QubitOperator('Y%d Y%d Y%d Y%d'% (ra,sa,pa,qa), 2/np.sqrt(12))
                        OA += QubitOperator('X%d X%d Y%d Y%d'% (ra,sa,pa,qa), -2/np.sqrt(12))
                        OA += QubitOperator('X%d Y%d X%d Y%d'% (ra,sa,pa,qa), 2/np.sqrt(12))
                        OA += QubitOperator('Y%d X%d X%d Y%d'% (ra,sa,pa,qa), 2/np.sqrt(12))
                        OA += QubitOperator('X%d Y%d Y%d X%d'% (ra,sa,pa,qa), 2/np.sqrt(12))
                        OA += QubitOperator('Y%d Y%d X%d X%d'% (ra,sa,pa,qa), -2/np.sqrt(12))
                        OA += QubitOperator('Y%d X%d Y%d X%d'% (ra,sa,pa,qa), 2/np.sqrt(12))

                        # termA += FermionOperator(((rb,1),(pb,0),(sb,1),(qb,0)), 2/np.sqrt(12))

                        OA += QubitOperator('Y%d X%d X%d X%d'% (rb,sb,pb,qb), 1j*2/np.sqrt(12))
                        OA += QubitOperator('X%d Y%d X%d X%d'% (rb,sb,pb,qb), 1j*2/np.sqrt(12))
                        OA += QubitOperator('X%d X%d Y%d X%d'% (rb,sb,pb,qb), -1j*2/np.sqrt(12))
                        OA += QubitOperator('X%d X%d X%d Y%d'% (rb,sb,pb,qb), -1j*2/np.sqrt(12))
                        OA += QubitOperator('X%d Y%d Y%d Y%d'% (rb,sb,pb,qb), -1j*2/np.sqrt(12))
                        OA += QubitOperator('Y%d X%d Y%d Y%d'% (rb,sb,pb,qb), -1j*2/np.sqrt(12))
                        OA += QubitOperator('Y%d Y%d X%d Y%d'% (rb,sb,pb,qb), 1j*2/np.sqrt(12))
                        OA += QubitOperator('Y%d Y%d Y%d X%d'% (rb,sb,pb,qb), 1j*2/np.sqrt(12))

                        OA += QubitOperator('X%d X%d X%d X%d'% (rb,sb,pb,qb), 2/np.sqrt(12))
                        OA += QubitOperator('Y%d Y%d Y%d Y%d'% (rb,sb,pb,qb), 2/np.sqrt(12))
                        OA += QubitOperator('X%d X%d Y%d Y%d'% (rb,sb,pb,qb), -2/np.sqrt(12))
                        OA += QubitOperator('X%d Y%d X%d Y%d'% (rb,sb,pb,qb), 2/np.sqrt(12))
                        OA += QubitOperator('Y%d X%d X%d Y%d'% (rb,sb,pb,qb), 2/np.sqrt(12))
                        OA += QubitOperator('X%d Y%d Y%d X%d'% (rb,sb,pb,qb), 2/np.sqrt(12))
                        OA += QubitOperator('Y%d Y%d X%d X%d'% (rb,sb,pb,qb), -2/np.sqrt(12))
                        OA += QubitOperator('Y%d X%d Y%d X%d'% (rb,sb,pb,qb), 2/np.sqrt(12))

                        # termA += FermionOperator(((ra,1),(pa,0),(sb,1),(qb,0)), 1/np.sqrt(12))

                        OA += QubitOperator('Y%d X%d X%d X%d'% (ra,sb,pa,qb), 1j/np.sqrt(12))
                        OA += QubitOperator('X%d Y%d X%d X%d'% (ra,sb,pa,qb), 1j/np.sqrt(12))
                        OA += QubitOperator('X%d X%d Y%d X%d'% (ra,sb,pa,qb), -1j/np.sqrt(12))
                        OA += QubitOperator('X%d X%d X%d Y%d'% (ra,sb,pa,qb), -1j/np.sqrt(12))
                        OA += QubitOperator('X%d Y%d Y%d Y%d'% (ra,sb,pa,qb), -1j/np.sqrt(12))
                        OA += QubitOperator('Y%d X%d Y%d Y%d'% (ra,sb,pa,qb), -1j/np.sqrt(12))
                        OA += QubitOperator('Y%d Y%d X%d Y%d'% (ra,sb,pa,qb), 1j/np.sqrt(12))
                        OA += QubitOperator('Y%d Y%d Y%d X%d'% (ra,sb,pa,qb), 1j/np.sqrt(12))

                        OA += QubitOperator('X%d X%d X%d X%d'% (ra,sb,pa,qb), 1/np.sqrt(12))
                        OA += QubitOperator('Y%d Y%d Y%d Y%d'% (ra,sb,pa,qb), 1/np.sqrt(12))
                        OA += QubitOperator('X%d X%d Y%d Y%d'% (ra,sb,pa,qb), -1/np.sqrt(12))
                        OA += QubitOperator('X%d Y%d X%d Y%d'% (ra,sb,pa,qb), 1/np.sqrt(12))
                        OA += QubitOperator('Y%d X%d X%d Y%d'% (ra,sb,pa,qb), 1/np.sqrt(12))
                        OA += QubitOperator('X%d Y%d Y%d X%d'% (ra,sb,pa,qb), 1/np.sqrt(12))
                        OA += QubitOperator('Y%d Y%d X%d X%d'% (ra,sb,pa,qb), -1/np.sqrt(12))
                        OA += QubitOperator('Y%d X%d Y%d X%d'% (ra,sb,pa,qb), 1/np.sqrt(12))

                        # termA += FermionOperator(((rb,1),(pb,0),(sa,1),(qa,0)), 1/np.sqrt(12))

                        OA += QubitOperator('Y%d X%d X%d X%d'% (rb,sa,pb,qa), 1j/np.sqrt(12))
                        OA += QubitOperator('X%d Y%d X%d X%d'% (rb,sa,pb,qa), 1j/np.sqrt(12))
                        OA += QubitOperator('X%d X%d Y%d X%d'% (rb,sa,pb,qa), -1j/np.sqrt(12))
                        OA += QubitOperator('X%d X%d X%d Y%d'% (rb,sa,pb,qa), -1j/np.sqrt(12))
                        OA += QubitOperator('X%d Y%d Y%d Y%d'% (rb,sa,pb,qa), -1j/np.sqrt(12))
                        OA += QubitOperator('Y%d X%d Y%d Y%d'% (rb,sa,pb,qa), -1j/np.sqrt(12))
                        OA += QubitOperator('Y%d Y%d X%d Y%d'% (rb,sa,pb,qa), 1j/np.sqrt(12))
                        OA += QubitOperator('Y%d Y%d Y%d X%d'% (rb,sa,pb,qa), 1j/np.sqrt(12))

                        OA += QubitOperator('X%d X%d X%d X%d'% (rb,sa,pb,qa), 1/np.sqrt(12))
                        OA += QubitOperator('Y%d Y%d Y%d Y%d'% (rb,sa,pb,qa), 1/np.sqrt(12))
                        OA += QubitOperator('X%d X%d Y%d Y%d'% (rb,sa,pb,qa), -1/np.sqrt(12))
                        OA += QubitOperator('X%d Y%d X%d Y%d'% (rb,sa,pb,qa), 1/np.sqrt(12))
                        OA += QubitOperator('Y%d X%d X%d Y%d'% (rb,sa,pb,qa), 1/np.sqrt(12))
                        OA += QubitOperator('X%d Y%d Y%d X%d'% (rb,sa,pb,qa), 1/np.sqrt(12))
                        OA += QubitOperator('Y%d Y%d X%d X%d'% (rb,sa,pb,qa), -1/np.sqrt(12))
                        OA += QubitOperator('Y%d X%d Y%d X%d'% (rb,sa,pb,qa), 1/np.sqrt(12))

                        # termA += FermionOperator(((ra,1),(pb,0),(sb,1),(qa,0)), 1/np.sqrt(12))
                        
                        OA += QubitOperator('Y%d X%d X%d X%d'% (ra,sb,pb,qa), 1j/np.sqrt(12))
                        OA += QubitOperator('X%d Y%d X%d X%d'% (ra,sb,pb,qa), 1j/np.sqrt(12))
                        OA += QubitOperator('X%d X%d Y%d X%d'% (ra,sb,pb,qa), -1j/np.sqrt(12))
                        OA += QubitOperator('X%d X%d X%d Y%d'% (ra,sb,pb,qa), -1j/np.sqrt(12))
                        OA += QubitOperator('X%d Y%d Y%d Y%d'% (ra,sb,pb,qa), -1j/np.sqrt(12))
                        OA += QubitOperator('Y%d X%d Y%d Y%d'% (ra,sb,pb,qa), -1j/np.sqrt(12))
                        OA += QubitOperator('Y%d Y%d X%d Y%d'% (ra,sb,pb,qa), 1j/np.sqrt(12))
                        OA += QubitOperator('Y%d Y%d Y%d X%d'% (ra,sb,pb,qa), 1j/np.sqrt(12))

                        OA += QubitOperator('X%d X%d X%d X%d'% (ra,sb,pb,qa), 1/np.sqrt(12))
                        OA += QubitOperator('Y%d Y%d Y%d Y%d'% (ra,sb,pb,qa), 1/np.sqrt(12))
                        OA += QubitOperator('X%d X%d Y%d Y%d'% (ra,sb,pb,qa), -1/np.sqrt(12))
                        OA += QubitOperator('X%d Y%d X%d Y%d'% (ra,sb,pb,qa), 1/np.sqrt(12))
                        OA += QubitOperator('Y%d X%d X%d Y%d'% (ra,sb,pb,qa), 1/np.sqrt(12))
                        OA += QubitOperator('X%d Y%d Y%d X%d'% (ra,sb,pb,qa), 1/np.sqrt(12))
                        OA += QubitOperator('Y%d Y%d X%d X%d'% (ra,sb,pb,qa), -1/np.sqrt(12))
                        OA += QubitOperator('Y%d X%d Y%d X%d'% (ra,sb,pb,qa), 1/np.sqrt(12))

                        # termA += FermionOperator(((rb,1),(pa,0),(sa,1),(qb,0)), 1/np.sqrt(12))
                                                                
                        OA += QubitOperator('Y%d X%d X%d X%d'% (rb,sa,pa,qb), 1j/np.sqrt(12))
                        OA += QubitOperator('X%d Y%d X%d X%d'% (rb,sa,pa,qb), 1j/np.sqrt(12))
                        OA += QubitOperator('X%d X%d Y%d X%d'% (rb,sa,pa,qb), -1j/np.sqrt(12))
                        OA += QubitOperator('X%d X%d X%d Y%d'% (rb,sa,pa,qb), -1j/np.sqrt(12))
                        OA += QubitOperator('X%d Y%d Y%d Y%d'% (rb,sa,pa,qb), -1j/np.sqrt(12))
                        OA += QubitOperator('Y%d X%d Y%d Y%d'% (rb,sa,pa,qb), -1j/np.sqrt(12))
                        OA += QubitOperator('Y%d Y%d X%d Y%d'% (rb,sa,pa,qb), 1j/np.sqrt(12))
                        OA += QubitOperator('Y%d Y%d Y%d X%d'% (rb,sa,pa,qb), 1j/np.sqrt(12))

                        OA += QubitOperator('X%d X%d X%d X%d'% (rb,sa,pa,qb), 1/np.sqrt(12))
                        OA += QubitOperator('Y%d Y%d Y%d Y%d'% (rb,sa,pa,qb), 1/np.sqrt(12))
                        OA += QubitOperator('X%d X%d Y%d Y%d'% (rb,sa,pa,qb), -1/np.sqrt(12))
                        OA += QubitOperator('X%d Y%d X%d Y%d'% (rb,sa,pa,qb), 1/np.sqrt(12))
                        OA += QubitOperator('Y%d X%d X%d Y%d'% (rb,sa,pa,qb), 1/np.sqrt(12))
                        OA += QubitOperator('X%d Y%d Y%d X%d'% (rb,sa,pa,qb), 1/np.sqrt(12))
                        OA += QubitOperator('Y%d Y%d X%d X%d'% (rb,sa,pa,qb), -1/np.sqrt(12))
                        OA += QubitOperator('Y%d X%d Y%d X%d'% (rb,sa,pa,qb), 1/np.sqrt(12))

                        # termB =  FermionOperator(((ra,1),(pa,0),(sb,1),(qb,0)),  1/2.0)

                        OB += QubitOperator('Y%d X%d X%d X%d'% (ra,sb,pa,qb), 1j/2)
                        OB += QubitOperator('X%d Y%d X%d X%d'% (ra,sb,pa,qb), 1j/2)
                        OB += QubitOperator('X%d X%d Y%d X%d'% (ra,sb,pa,qb), -1j/2)
                        OB += QubitOperator('X%d X%d X%d Y%d'% (ra,sb,pa,qb), -1j/2)
                        OB += QubitOperator('X%d Y%d Y%d Y%d'% (ra,sb,pa,qb), -1j/2)
                        OB += QubitOperator('Y%d X%d Y%d Y%d'% (ra,sb,pa,qb), -1j/2)
                        OB += QubitOperator('Y%d Y%d X%d Y%d'% (ra,sb,pa,qb), 1j/2)
                        OB += QubitOperator('Y%d Y%d Y%d X%d'% (ra,sb,pa,qb), 1j/2)

                        OB += QubitOperator('X%d X%d X%d X%d'% (ra,sb,pa,qb), 1/2)
                        OB += QubitOperator('Y%d Y%d Y%d Y%d'% (ra,sb,pa,qb), 1/2)
                        OB += QubitOperator('X%d X%d Y%d Y%d'% (ra,sb,pa,qb), -1/2)
                        OB += QubitOperator('X%d Y%d X%d Y%d'% (ra,sb,pa,qb), 1/2)
                        OB += QubitOperator('Y%d X%d X%d Y%d'% (ra,sb,pa,qb), 1/2)
                        OB += QubitOperator('X%d Y%d Y%d X%d'% (ra,sb,pa,qb), 1/2)
                        OB += QubitOperator('Y%d Y%d X%d X%d'% (ra,sb,pa,qb), -1/2)
                        OB += QubitOperator('Y%d X%d Y%d X%d'% (ra,sb,pa,qb), 1/2)

                        #BtermB += FermionOperator(((rb,1),(pb,0),(sa,1),(qa,0)),  1/2.0)

                        OB += QubitOperator('Y%d X%d X%d X%d'% (rb,sa,pb,qa), 1j/2)
                        OB += QubitOperator('X%d Y%d X%d X%d'% (rb,sa,pb,qa), 1j/2)
                        OB += QubitOperator('X%d X%d Y%d X%d'% (rb,sa,pb,qa), -1j/2)
                        OB += QubitOperator('X%d X%d X%d Y%d'% (rb,sa,pb,qa), -1j/2)
                        OB += QubitOperator('X%d Y%d Y%d Y%d'% (rb,sa,pb,qa), -1j/2)
                        OB += QubitOperator('Y%d X%d Y%d Y%d'% (rb,sa,pb,qa), -1j/2)
                        OB += QubitOperator('Y%d Y%d X%d Y%d'% (rb,sa,pb,qa), 1j/2)
                        OB += QubitOperator('Y%d Y%d Y%d X%d'% (rb,sa,pb,qa), 1j/2)

                        OB += QubitOperator('X%d X%d X%d X%d'% (rb,sa,pb,qa), 1/2)
                        OB += QubitOperator('Y%d Y%d Y%d Y%d'% (rb,sa,pb,qa), 1/2)
                        OB += QubitOperator('X%d X%d Y%d Y%d'% (rb,sa,pb,qa), -1/2)
                        OB += QubitOperator('X%d Y%d X%d Y%d'% (rb,sa,pb,qa), 1/2)
                        OB += QubitOperator('Y%d X%d X%d Y%d'% (rb,sa,pb,qa), 1/2)
                        OB += QubitOperator('X%d Y%d Y%d X%d'% (rb,sa,pb,qa), 1/2)
                        OB += QubitOperator('Y%d Y%d X%d X%d'% (rb,sa,pb,qa), -1/2)
                        OB += QubitOperator('Y%d X%d Y%d X%d'% (rb,sa,pb,qb), 1/2)

                        #BtermB += FermionOperator(((ra,1),(pb,0),(sb,1),(qa,0)), -1/2.0)

                        OB -= QubitOperator('Y%d X%d X%d X%d'% (ra,sb,pb,qa), 1j/2)
                        OB -= QubitOperator('X%d Y%d X%d X%d'% (ra,sb,pb,qa), 1j/2)
                        OB -= QubitOperator('X%d X%d Y%d X%d'% (ra,sb,pb,qa), -1j/2)
                        OB -= QubitOperator('X%d X%d X%d Y%d'% (ra,sb,pb,qa), -1j/2)
                        OB -= QubitOperator('X%d Y%d Y%d Y%d'% (ra,sb,pb,qa), -1j/2)
                        OB -= QubitOperator('Y%d X%d Y%d Y%d'% (ra,sb,pb,qa), -1j/2)
                        OB -= QubitOperator('Y%d Y%d X%d Y%d'% (ra,sb,pb,qa), 1j/2)
                        OB -= QubitOperator('Y%d Y%d Y%d X%d'% (ra,sb,pb,qa), 1j/2)

                        OB -= QubitOperator('X%d X%d X%d X%d'% (ra,sb,pb,qa), 1/2)
                        OB -= QubitOperator('Y%d Y%d Y%d Y%d'% (ra,sb,pb,qa), 1/2)
                        OB -= QubitOperator('X%d X%d Y%d Y%d'% (ra,sb,pb,qa), -1/2)
                        OB -= QubitOperator('X%d Y%d X%d Y%d'% (ra,sb,pb,qa), 1/2)
                        OB -= QubitOperator('Y%d X%d X%d Y%d'% (ra,sb,pb,qa), 1/2)
                        OB -= QubitOperator('X%d Y%d Y%d X%d'% (ra,sb,pb,qa), 1/2)
                        OB -= QubitOperator('Y%d Y%d X%d X%d'% (ra,sb,pb,qa), -1/2)
                        OB -= QubitOperator('Y%d X%d Y%d X%d'% (ra,sb,pb,qa), 1/2)

                        #BtermB += FermionOperator(((rb,1),(pa,0),(sa,1),(qb,0)), -1/2.0)

                        OB -= QubitOperator('Y%d X%d X%d X%d'% (rb,sa,pa,qb), 1j/2)
                        OB -= QubitOperator('X%d Y%d X%d X%d'% (rb,sa,pa,qb), 1j/2)
                        OB -= QubitOperator('X%d X%d Y%d X%d'% (rb,sa,pa,qb), -1j/2)
                        OB -= QubitOperator('X%d X%d X%d Y%d'% (rb,sa,pa,qb), -1j/2)
                        OB -= QubitOperator('X%d Y%d Y%d Y%d'% (rb,sa,pa,qb), -1j/2)
                        OB -= QubitOperator('Y%d X%d Y%d Y%d'% (rb,sa,pa,qb), -1j/2)
                        OB -= QubitOperator('Y%d Y%d X%d Y%d'% (rb,sa,pa,qb), 1j/2)
                        OB -= QubitOperator('Y%d Y%d Y%d X%d'% (rb,sa,pa,qb), 1j/2)

                        OB -= QubitOperator('X%d X%d X%d X%d'% (rb,sa,pa,qb), 1/2)
                        OB -= QubitOperator('Y%d Y%d Y%d Y%d'% (rb,sa,pa,qb), 1/2)
                        OB -= QubitOperator('X%d X%d Y%d Y%d'% (rb,sa,pa,qb), -1/2)
                        OB -= QubitOperator('X%d Y%d X%d Y%d'% (rb,sa,pa,qb), 1/2)
                        OB -= QubitOperator('Y%d X%d X%d Y%d'% (rb,sa,pa,qb), 1/2)
                        OB -= QubitOperator('X%d Y%d Y%d X%d'% (rb,sa,pa,qb), 1/2)
                        OB -= QubitOperator('Y%d Y%d X%d X%d'% (rb,sa,pa,qb), -1/2)
                        OB -= QubitOperator('Y%d X%d Y%d X%d'% (rb,sa,pa,qb), 1/2)
 

                        # for t in OA.terms:
                        #     if(OA.terms[t] - np.conj(OA.terms[t]) < 1e-6):
                        #       OA.terms[t] = 0
                        # for t in OB.terms:
                        #     if(OB.terms[t] - np.conj(OB.terms[t]) < 1e-6):
                        #       OB.terms[t] = 0
                        OA -= hermitian_conjugated(OA)

                        OB -= hermitian_conjugated(OB)

                        coeffA = 0
                        coeffB = 0

                        for t in OA.terms:
                            coeff_t = OA.terms[t]
                            coeffA += coeff_t * coeff_t

                        if np.abs(coeffA) > 1e-6:
                          self.fermi_ops.append(OA)

                        for t in OB.terms:
                            coeff_t = OB.terms[t]
                            coeffB += coeff_t * coeff_t
                            
                        if np.abs(coeffB) > 1e-6:
                          self.fermi_ops.append(OB)

                        # termA -= hermitian_conjugated(termA)
                        # termB -= hermitian_conjugated(termB)
                      
               
                        # termA = normal_ordered(termA)
                        # termB = normal_ordered(termB)
                        
                        # #Normalize
                        # for t in termA.terms:
                        #     coeff_t = termA.terms[t]
                        #     coeffA += coeff_t * coeff_t
                        # for t in termB.terms:
                        #     coeff_t = termB.terms[t]
                        #     coeffB += coeff_t * coeff_t

                        
                        # if termA.many_body_order() > 0:
                        #     termA = termA/np.sqrt(coeffA)
                        #     self.fermi_ops.append(termA)
                        
                        # if termB.many_body_order() > 0:
                        #     termB = termB/np.sqrt(coeffB)
                            # self.fermi_ops.append(termB)
        for op in self.fermi_ops:
            print(op)

        self.n_ops = len(self.fermi_ops)
        print(" Number of operators: ", self.n_ops)
        return 

class singlet_GSD_nospin(OperatorPool):
    def generate_SQ_Operators(self):
        """
        n_orb is number of spatial orbitals assuming that spin orbitals are labelled
        0a,0b,1a,1b,2a,2b,3a,3b,....  -> 0,1,2,3,...
        """
        
        print(" Form singlet GSD operators")
        
        self.fermi_ops = []
        for p in range(0,self.n_orb):
            pa = 2*p
            pb = 2*p+1
 
            for q in range(p,self.n_orb):
                qa = 2*q
                qb = 2*q+1
        
                termA  = FermionOperator(((pa,1),(qa,0)))
                termA -= hermitian_conjugated(termA)
                self.fermi_ops.append(termA)

                termA  = FermionOperator(((pb,1),(qb,0)))
                termA -= hermitian_conjugated(termA)
                self.fermi_ops.append(termA)
               
                termA = normal_ordered(termA)
                      
      
        pq = -1 
        for p in range(0,self.n_orb):
            pa = 2*p
            pb = 2*p+1
 
            for q in range(p,self.n_orb):
                qa = 2*q
                qb = 2*q+1
        
                pq += 1
        
                rs = -1 
                for r in range(0,self.n_orb):
                    ra = 2*r
                    rb = 2*r+1
                    
                    for s in range(r,self.n_orb):
                        sa = 2*s
                        sb = 2*s+1
                    
                        rs += 1
                    
                        if(pq > rs):
                            continue

                        termA  = FermionOperator(((ra,1),(pa,0),(sa,1),(qa,0)))
                        termA -= hermitian_conjugated(termA)
                        self.fermi_ops.append(termA)

                        termA  = FermionOperator(((rb,1),(pb,0),(sb,1),(qb,0)))
                        termA -= hermitian_conjugated(termA)
                        self.fermi_ops.append(termA)

                        termA  = FermionOperator(((ra,1),(pa,0),(sb,1),(qb,0)))
                        termA -= hermitian_conjugated(termA)
                        self.fermi_ops.append(termA)

                        termA  = FermionOperator(((rb,1),(pb,0),(sa,1),(qa,0)))
                        termA -= hermitian_conjugated(termA)
                        self.fermi_ops.append(termA)

                        termA  = FermionOperator(((ra,1),(pb,0),(sb,1),(qa,0)))
                        termA -= hermitian_conjugated(termA)
                        self.fermi_ops.append(termA)

                        termA  = FermionOperator(((rb,1),(pa,0),(sa,1),(qb,0)))
                        termA -= hermitian_conjugated(termA)
                        self.fermi_ops.append(termA)

        self.n_ops = len(self.fermi_ops)
        print(" Number of operators: ", self.n_ops)
        return 

class A(OperatorPool):
    def generate_SQ_Operators(self):

        self.fermi_ops = []

        self.sup_index = []
        self.left_ops = []
        self.right_ops = []
        self.scalar = []

        sup_ind = 1
        
        n_occ = self.n_occ
        n_vir = self.n_vir

                
        for p in range(0,2*self.n_orb):
            X = QubitOperator('X%d'% p, 0+1j)
            Y = QubitOperator('Y%d'% p, 0+1j)

            self.fermi_ops.append(X)
            self.sup_index.append(0)

        for p in range(0,2*self.n_orb):
            Z = QubitOperator('Z%d'% p, 0+1j)

            self.fermi_ops.append(Z)
            self.sup_index.append(0)

        for p in range(0,2*self.n_orb):

            for q in range(p+1,2*self.n_orb):

                ZZ = QubitOperator('Z%d Z%d'% (p, q), 0+1j)
                self.fermi_ops.append(ZZ)
                self.sup_index.append(0)

        for p in range(0,2*self.n_orb):
            for q in range(p+1,2*self.n_orb):

                XX = QubitOperator('X%d X%d'% (p, q), 0+1j)
                XY = QubitOperator('X%d Y%d'% (p, q), 0+1j)
                YX = QubitOperator('Y%d X%d'% (p, q), 0+1j)
                XZ = QubitOperator('X%d Z%d'% (p, q), 0+1j)
                YZ = QubitOperator('Y%d Z%d'% (p, q), 0+1j)
                YY = QubitOperator('Y%d Y%d'% (p, q), 0+1j)

                self.fermi_ops.append(XX)
                self.fermi_ops.append(YX)
                self.fermi_ops.append(XZ)
                self.fermi_ops.append(YZ)
                self.fermi_ops.append(YY)
                self.fermi_ops.append(XY)

                self.sup_index.append(0)
                self.sup_index.append(0)
                self.sup_index.append(0)
                self.sup_index.append(0)
                self.sup_index.append(0)
                self.sup_index.append(0)

                A =  QubitOperator('X%d X%d' % (p,q),1j/2)
                A += QubitOperator('Y%d Y%d' % (p,q),1j/2)
                self.sup_index.append(sup_ind)

                left = QubitOperator('Z%d' %q , 1j*np.pi/4)
                self.left_ops.append(left)

                right = QubitOperator('Z%d Z%d' %(p,q), -1j*np.pi/4)
                right += QubitOperator('Z%d' %p, -1j*np.pi/4)
                self.right_ops.append(right)

                scalar = np.exp(1j*np.pi/4)
                self.scalar.append(scalar)

                sup_ind += 1

                self.fermi_ops.append(A)


        for i in range(0,2*self.n_orb):
            for j in range(i+1, 2*self.n_orb):

                for k in range(j+1, 2*self.n_orb):

                    YXZ = QubitOperator('Y%d X%d Z%d'% (i, j, k), 1j)  
                    # XYZ = QubitOperator('X%d Y%d Z%d'% (i, j, k), 1j)
                    XZY = QubitOperator('X%d Z%d Y%d'% (i, j, k), 1j)
                    # YZX = QubitOperator('Y%d Z%d X%d'% (i, j, k), 1j)
                    ZXY = QubitOperator('Z%d X%d Y%d'% (i, j, k), 1j)
                    # ZYX = QubitOperator('Z%d X%d Y%d'% (i, j, k), 1j)
                    # ZZZ = QubitOperator('Z%d Z%d Z%d'% (i, j, k), 1j)
  
                    # self.fermi_ops.append(XYZ)
                    self.fermi_ops.append(YXZ)
                    self.sup_index.append(0)
                    self.fermi_ops.append(XZY)
                    self.sup_index.append(0)
                    # self.fermi_ops.append(YZX)  
                    self.fermi_ops.append(ZXY)
                    self.sup_index.append(0)
                    # self.fermi_ops.append(ZYX)
                    # self.fermi_ops.append(ZZZ) 

                    # XXY = QubitOperator('X%d X%d Y%d'% (i, j, k), 1j)
                    # XYY = QubitOperator('X%d Y%d Y%d'% (i, j, k), 1j)

                    # self.fermi_ops.append(XXY)
                    # self.fermi_ops.append(XYY)

        for p in range(0,2*self.n_orb):
            for q in range(p+1,2*self.n_orb):
                for r in range(q+1,2*self.n_orb):
                    for s in range(r+1,2*self.n_orb):
                        XYYY = QubitOperator('X%d Y%d Y%d Y%d'% (p, q, r, s), 1j)
                        YXYY = QubitOperator('Y%d X%d Y%d Y%d'% (p, q, r, s), 1j)
                        YYXY = QubitOperator('Y%d Y%d X%d Y%d'% (p, q, r, s), 1j)
                        YYYX = QubitOperator('Y%d Y%d Y%d X%d'% (p, q, r, s), 1j)

                        self.fermi_ops.append(XYYY)
                        self.fermi_ops.append(YXYY)
                        self.fermi_ops.append(YYXY)
                        self.fermi_ops.append(YYYX)

                        self.sup_index.append(0)
                        self.sup_index.append(0)
                        self.sup_index.append(0)
                        self.sup_index.append(0)

                        # XXXY = QubitOperator('X%d X%d X%d Y%d'% (p, q, r, s), 1j)
                        # YXXX = QubitOperator('Y%d X%d X%d X%d'% (p, q, r, s), 1j)
                        # XYXX = QubitOperator('X%d Y%d X%d X%d'% (p, q, r, s), 1j)
                        # XXYX = QubitOperator('X%d X%d Y%d X%d'% (p, q, r, s), 1j)

                        # self.fermi_ops.append(XXXY)
                        # self.fermi_ops.append(XXYX)
                        # self.fermi_ops.append(XYXX)
                        # self.fermi_ops.append(YXXX)

                        # ZZXY = QubitOperator('Z%d Z%d X%d Y%d'% (p, q, r, s), 1j)
                        # ZZYX = QubitOperator('Z%d Z%d Y%d X%d'% (p, q, r, s), 1j)
                        # XZZY = QubitOperator('X%d Z%d Z%d Y%d'% (p, q, r, s), 1j)
                        # YZZX = QubitOperator('Y%d Z%d Z%d X%d'% (p, q, r, s), 1j)
                        # ZYXZ = QubitOperator('Z%d Y%d X%d Z%d'% (p, q, r, s), 1j)
                        # ZXYZ = QubitOperator('Z%d X%d Y%d Z%d'% (p, q, r, s), 1j)
                        # XYZZ = QubitOperator('X%d Y%d Z%d Z%d'% (p, q, r, s), 1j)
                        # YXZZ = QubitOperator('Y%d X%d Z%d Z%d'% (p, q, r, s), 1j)
                        # XZYZ = QubitOperator('X%d Z%d Y%d Z%d'% (p, q, r, s), 1j)
                        # YZXZ = QubitOperator('Y%d Z%d X%d Z%d'% (p, q, r, s), 1j)
                        # ZXZY = QubitOperator('Z%d X%d Z%d Y%d'% (p, q, r, s), 1j)
                        # ZYZX = QubitOperator('Z%d Y%d Z%d X%d'% (p, q, r, s), 1j)

                        # self.fermi_ops.append(ZZXY)
                        # self.fermi_ops.append(ZZYX)
                        # self.fermi_ops.append(XZZY)
                        # self.fermi_ops.append(YZZX)
                        # self.fermi_ops.append(ZXYZ)
                        # self.fermi_ops.append(ZYXZ)
                        # self.fermi_ops.append(XYZZ)
                        # self.fermi_ops.append(YXZZ)
                        # self.fermi_ops.append(XZYZ)
                        # self.fermi_ops.append(YZXZ)
                        # self.fermi_ops.append(ZXZY)
                        # self.fermi_ops.append(ZYZX)

                        # XXYY = QubitOperator('X%d X%d Y%d Y%d'% (p, q, r, s), 1j)
                        # XYXY = QubitOperator('X%d Y%d X%d Y%d'% (p, q, r, s), 1j)
                        # XYYX = QubitOperator('X%d Y%d Y%d X%d'% (p, q, r, s), 1j)
                        # YYXX = QubitOperator('Y%d Y%d X%d X%d'% (p, q, r, s), 1j)
                        # YXXY = QubitOperator('Y%d X%d X%d Y%d'% (p, q, r, s), 1j)
                        # YXYX = QubitOperator('Y%d X%d Y%d X%d'% (p, q, r, s), 1j)

                        # self.fermi_ops.append(XXYY)
                        # self.fermi_ops.append(XYXY)
                        # self.fermi_ops.append(XYYX)
                        # self.fermi_ops.append(YYXX)
                        # self.fermi_ops.append(YXXY)
                        # self.fermi_ops.append(YXYX)

                        # XXXX = QubitOperator('X%d X%d X%d X%d'% (p, q, r, s), 1j)
                        # YYYY = QubitOperator('Y%d Y%d Y%d Y%d'% (p, q, r, s), 1j)

                        # self.fermi_ops.append(XXXX)
                        # self.fermi_ops.append(YYYY)                        
 
                
        # for op in self.fermi_ops:
        #     print(op)

        self.n_ops = len(self.fermi_ops)
        print(" Number of operators: ", self.n_ops)        
        return
