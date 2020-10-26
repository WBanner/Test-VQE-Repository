"""
Collection of operator pools and ways to construct, combine, and manipulate them.
"""
import itertools
import logging
from itertools import product
from typing import List, Tuple, Union

import numpy as np
from openfermion import QubitOperator
from qiskit.aqua.operators import WeightedPauliOperator, BaseOperator
from qiskit.quantum_info import Pauli, pauli_group
from qisresearch.utils.ofer import openfermion_to_qiskit

#from .fermionic_mapper import fermionic_mapping_to_qubit_ops

logger = logging.getLogger(__name__)


def chunks(l, n):
    """Iterable to return parts of a list.
    Parameters
    ----------
    l : list
        List to get chunks from.
    n : type
        Number of chunks to get.
    """
    for i in range(0, len(l), n):
        yield l[i:i+n]


def pauli_word_len(my_string):
    """Get the number of non-trivial terms in Pauli string.
    Parameters
    ----------
    my_string : str
        Pauli string to get count from.
    Returns
    -------
    int
        Number of non-trivial terms in Pauli string.
    """
    word = my_string.upper()
    if not all(p in 'IXYZ' for p in word):
        raise ValueError('Encountered invalid Pauli word')
    return len(word) - word.count('I')


def tuples_to_string(tuples, num_qubits):
    """Create Pauli string from indices of non-trivial terms.
    Parameters
    ----------
    tuples : List[Tuple[int, str]]
        List of indices and axes for Pauli string, e.g.:
        `[(1, 'X'), (3, 'Y'), (10, 'Z'), (2, 'Z')]`
    num_qubits : int
        Number of qubits in resulting string.
    Returns
    -------
    str
        Final Pauli string.
    """
    identity = ['I'] * num_qubits
    res = identity
    for tup in tuples:
        res[tup[0]] = tup[1]
    return ''.join(res)


def paul(n_qubits: int, index: int, axis: Union[int, str]):
    """Return a Pauli string with a single non-trivial element.
    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    index : int
        Index of non-trivial term.
    axis : Union[int, str]
        Axis for Pauli string. Either a string, or an int, e.g. `'X'` or `'1'`.
    Returns
    -------
    WeightedPauliOperator($0)
        Resulting Pauli operator.
    """
    label = ['I']*n_qubits
    pauli_dict = {0: 'I', 1: 'X', 2: 'Y', 3: 'Z'}
    if isinstance(axis, str):
        if not axis in list(pauli_dict.values()):
            raise ValueError('Invalid axis provided')
        label[index] = axis
    elif isinstance(axis, int):
        label[index] = pauli_dict[axis]
    return WeightedPauliOperator(paulis=[[1.0, Pauli.from_label(label)]])


def pauli_string_specs(
        num_qubits: int,
        axes: List[str] = ('I', 'X', 'Y', 'Z'),
        max_operators: int = None,
        min_operators: int = None
):
    """Construct Iterable for list of Pauli strings.
    Parameters
    ----------
    num_qubits : int
        Number of qubits.
    axes : List[str]
        Axes to use in strings.
    max_operators : int
        Maximum number of non-trivial terms in Pauli strings.
    min_operators : int
        Minimum number of non-trivial terms in Pauli strings.
    Returns
    -------
    Iterable
            Iterable to loop through all Pauli strings of specification.
    """
    if max_operators is None:
        max_operators = num_qubits
    if min_operators is None:
        min_operators = 0

    if min_operators < 0:
        raise ValueError('Min operators must be non-negative integer')
    if max_operators > num_qubits:
        raise ValueError('Max operators should be less than or equal to number of qubits')
    if not set(axes) <= {'I', 'X', 'Y', 'Z'}:
        raise ValueError('Axes should be a subset of I, X, Y, Z')

    def str_counts(pauli_str):
        id_counts = pauli_str.count('I')
        pa_counts = pauli_str.count('X') + pauli_str.count('Y') + pauli_str.count('Z')
        return id_counts, pa_counts

    combinations = itertools.product(axes, repeat=num_qubits)
    combinations = map(lambda pauli_list: ''.join(pauli_list), combinations)
    combinations = filter(
        lambda pauli_str: min_operators <= str_counts(pauli_str)[1] <= max_operators,
        combinations
    )
    return combinations

"""
Complete pool on two qubits, derived by Ed Barnes.
"""
_TWO_QUBIT_COMPLETE_POOL = [
    'XY',
    'ZY',
    'YI',
    'YX',
    'YZ',
    'IY'
]


"""
Complete pool on two qubits, derived by Vlad Shkolnikov.
"""
_THREE_QUBIT_COMPLETE_POOL = ['ZZY', 'ZYI', 'YII', 'IYI']


def normalize_operator(operator: WeightedPauliOperator, power: float = 0.5) -> WeightedPauliOperator:
    """Return an operator scaled according to the number of terms.
    Parameters
    ----------
    operator : WeightedPauliOperator
        Operator to scale.
    power : float
        Exponent in denominator of scaling.
    Returns
    -------
    WeightedPauliOperator
        Scaled operator.
    """
    n_terms = len(operator.paulis)
    return operator * (1/(n_terms**power))


class OperatorPool:
    """Base class for operator pool.
    Parameters
    ----------
    name : type
        Name of pool.
    Attributes
    ----------
    pool : type
        List of the elements in the pool.
    num_qubits : type
        Number of qubits on which the pool acts.
    name
    """
    def __init__(self, name=''):
        self._pool = []
        self._num_qubits = None
        self.name = name

    @property
    def pool(self) -> List[WeightedPauliOperator]:
        """List of elements in the pool.
        Returns
        -------
        List[WeightedPauliOperator]
            List of elements in the pool.
        """
        return self._pool

    @property
    def num_qubits(self):
        """Number of qubits.
        Returns
        -------
        int
            Number of qubits on which the pool acts.
        """
        return self._num_qubits

    def reverse(self):
        """Reverses the order of the elements in the pool.
        """
        self._pool.reverse()

    def __len__(self):
        """Returns the number of elements in the pool.
        Returns
        -------
        int
            Number of elements in pool.
        """
        return len(self.pool)

    def __add__(self, rhs):
        """Add combine pool with `rhs`. For example, `pool = pool_1 + pool_2`.
        """
        if self.num_qubits is not rhs.num_qubits:
            raise ValueError('Pools must have same number of qubits')
        self.add_ops(rhs.pool)
        self.name = self.name + '_+_' + rhs.name
        return self

    def add_ops(self, op_list):
        """Add operators to pool.
        Parameters
        ----------
        op_list : List[Operator]
            Operators to add to pool.
        """
        qubits_list = {op.num_qubits for op in op_list}
        if len(qubits_list) > 1:
            raise ValueError('All operators should have same number of qubits')
        if self._num_qubits is None:
            self._num_qubits = op_list[0].num_qubits
            self._pool = self._pool + op_list
        elif self._num_qubits is not op_list[0].num_qubits:
            raise ValueError(
                'New operators do not match number of existing qubits')
        else:
            self._pool = self._pool + op_list

    def thanos_pool(self, frac=1/2):
        """Returns random selection of elements in pool.
        Parameters
        ----------
        frac : float
            Fraction of operators to return.
        Returns
        -------
        OperatorPool
            Randomly selected operators.
        """
        
        random_ops = list(np.random.choice(self.pool, int(frac * len(self.pool))))
        
        res = self.from_operator_list(random_ops)
        
        return res

    @classmethod
    def from_operator_list(cls, op_list):
        """Create OperatorPool from lsit of operators. For example,
        ```
        my_ops = [op_1, op_2]
        pool = OperatorPool.from_operator_list(my_ops)
        ```
        """
        result = cls()
        if len(op_list) == 0:
            return result
        result.add_ops(op_list)
        return result

    @staticmethod
    def drop_zero_operators(operator_list: List[WeightedPauliOperator]) -> List[WeightedPauliOperator]:
        """Remove empty operators from list of operators.
        Parameters
        ----------
        operator_list : List[WeightedPauliOperator]
            Operators to filter.
        Returns
        -------
        List[WeightedPauliOperator]
            Filtered list of operators.
        """
        return list(filter(
            lambda op: len(op.paulis) > 0,
            operator_list
        ))

    def drop_duplicates(self, inplace: bool = True):
        """Remove duplicate operators from pool.
        Parameters
        ----------
        inplace : bool
            If `True`, original object will be mutated. If `False`, returns a
            copy of the mutated object, while leaving the original untouched.
        Returns
        -------
        OperatorPool
            Returned operator pool if `inplace=False`.
        """
        temp_list = []
        for op in self.pool:
            if op not in temp_list:
                temp_list.append(op)
        if inplace:
            self._pool = temp_list
        else:
            return temp_list


class PauliMixerPool(OperatorPool):
    """Operator pool with elements analogous to the QAOA mixer.
    """

    def __init__(self, name=''):
        super().__init__(name=name)

    @classmethod
    def from_pauli_terms_list(cls, terms: List[Tuple[str, int]]):
        """Create pool from list of single qubit terms. E.g.,
        ```
        PauliPool.from_pauli_terms_list([('X', 1), ('Y', 3)])
        ```
        """
        result = cls()
        qubit_indices = [tup[1] for tup in terms]
        num_qubits = max(qubit_indices)+1
        operator_list = [
            paul(num_qubits, index, axis)
            for axis, index in terms
        ]
        result.add_ops(operator_list)
        result.name = 'From pauli terms list {}'.format(terms)
        return result

    @classmethod
    def from_exact_term_number(cls, num_qubits: int, num_terms: int, axes: List[str]):
        if num_terms > num_qubits:
            raise ValueError('Cannot have more terms than qubits.')
        if not set(axes).issubset({'I', 'X', 'Y', 'Z'}):
            raise ValueError('Invalid set of axes provided.')

        qubit_indices = list(itertools.combinations(range(num_qubits), r=num_terms))
        strin_indices = list(itertools.permutations(axes, r=num_terms))

        pairings = list(itertools.product(qubit_indices, strin_indices))

        pauli_list = []
        for indices, letters in pairings:
            current_pauli = paul(num_qubits, 0, 'I')
            current_pauli = 0.0 * current_pauli
            for ind, let in zip(indices, letters):
                current_pauli += paul(num_qubits, ind, let)
            current_pauli.simplify()
            pauli_list.append(current_pauli)

        if len(pauli_list) == 0:
            raise ValueError('Empty list of operators found')

        result = cls.from_operator_list(pauli_list)
        result.name = 'From exact number'
        return result

    @classmethod
    def from_max_term_number(cls, num_qubits: int, max_num_terms: int, axes: List[str]):
        if max_num_terms > num_qubits:
            raise ValueError('Cannot have more terms than qubits.')
        if not set(axes).issubset({'I', 'X', 'Y', 'Z'}):
            raise ValueError('Invalid set of axes provided.')

        pools = [cls.from_exact_term_number(num_qubits, i, axes) for i in range(1, max_num_terms+1)]

        result = cls()
        result._num_qubits = num_qubits
        for pool in pools:
            result += pool
        result.name = 'From max number'
        return result


class PauliPool(OperatorPool):
    """Pool consisting of only Pauli strings. Subject of qubit-ADAPT paper.
    """
    def __init__(self, name=''):
        super().__init__(name=name)

    @staticmethod
    def _drop_even_y_paulis(operator: WeightedPauliOperator) -> WeightedPauliOperator:
        new_paulis = []
        if len(operator.paulis) > 1:
            raise ValueError(
                'Removal of Pauli strings from operator with more than one term is not supported\n'
                'Tried to remove even Y strings from {}'.format(operator.paulis)
            )
        for weight, pauli in operator.paulis:
            if pauli.to_label().count('Y') % 2 == 0:
                new_weight = 0.0 + 0.j
            else:
                new_weight = weight
            new_paulis.append([new_weight, pauli])
        return WeightedPauliOperator(paulis=new_paulis).chop()

    def cast_out_even_y_paulis(self, inplace: bool = True) -> List[WeightedPauliOperator]:
        """Remove terms that have an even number of Pauli Y operators.
        Parameters
        ----------
        inplace : bool
            If `True`, original object will be mutated. If `False`, returns a
            copy of the mutated object, while leaving the original untouched.
        Returns
        -------
        List[WeightedPauliOperator]
            List of elements in the pool that have odd number of Y's.
        """
        logger.info('Casting out terms from pool with even number of Ys, originally have {} operators'.format(len(self)))
        new_pool = list(map(self._drop_even_y_paulis, self.pool))
        if inplace:
            self._pool = self.drop_zero_operators(new_pool)
            logger.info(
                'Pool now has {} operators'.format(
                    len(self)))
            return self.pool
        else:
            logger.info(
                'Pool now has {} operators'.format(
                    len(new_pool)))
            return self.drop_zero_operators(new_pool)

    @staticmethod
    def _drop_particle_number_violating_paulis(operator: WeightedPauliOperator) -> WeightedPauliOperator:
        new_paulis = []
        if len(operator.paulis) > 1:
            raise ValueError(
                'Removal of Pauli strings from operator with more than one term is not supported\n'
                'Tried to remove particle number violating strings from {}'.format(operator.paulis)
            )
        for weight, pauli in operator.paulis:
            x_terms = pauli.to_label().count('X')
            y_terms = pauli.to_label().count('Y')
            if (x_terms + y_terms) % 2 == 1:
                new_weight = 0.0 + 0.j
            else:
                new_weight = weight
            new_paulis.append([new_weight, pauli])
        return WeightedPauliOperator(paulis=new_paulis).chop()

    def cast_out_particle_number_violating_strings(self, inplace: bool = True) -> List[WeightedPauliOperator]:
        """Remove terms that have an odd number of `X` or `Y`, which do not preserve
        the number of particles for the state on which they act.
        Parameters
        ----------
        inplace : bool
            If `True`, original object will be mutated. If `False`, returns a
            copy of the mutated object, while leaving the original untouched.
        Returns
        -------
        List[WeightedPauliOperator]
            List of elements in the pool that preserve particle number.
        """
        logger.info('Casting out terms from pool that violate particle number conservation, originally have {} operators'.format(len(self)))
        new_pool = list(map(self._drop_particle_number_violating_paulis, self.pool))
        if inplace:
            self._pool = self.drop_zero_operators(new_pool)
            logger.info(
                'Pool now has {} operators'.format(
                    len(self)))
            return self.pool
        else:
            logger.info(
                'Pool now has {} operators'.format(
                    len(new_pool)))
            return self.drop_zero_operators(new_pool)

    @staticmethod
    def _drop_higher_order_z_strings(operator: WeightedPauliOperator) -> WeightedPauliOperator:
        new_paulis = []
        if len(operator.paulis) > 1:
            raise ValueError(
                'Removal of Pauli strings from operator with more than one term is not supported\n'
                'Tried to remove particle number violating strings from {}'.format(operator.paulis)
            )
        for weight, pauli in operator.paulis:
            nontrivial_terms = len(pauli.to_label()) - pauli.to_label().count('I')
            if pauli.to_label().count('Z') >= 2 and nontrivial_terms > 2:
                new_weight = 0.0 + 0.j
            else:
                new_weight = weight
            new_paulis.append([new_weight, pauli])
        return WeightedPauliOperator(paulis=new_paulis).chop()

    def cast_out_higher_order_z_strings(self, inplace: bool = True) -> List[WeightedPauliOperator]:
        """Remove terms that have more than one `Z`s.
        Parameters
        ----------
        inplace : bool
            If `True`, original object will be mutated. If `False`, returns a
            copy of the mutated object, while leaving the original untouched.
        Returns
        -------
        List[WeightedPauliOperator]
            List of elements in the pool that have at most one `Z`.
        """
        logger.info('Casting out terms from pool that have 2 or more Z terms, originally have {} operators'.format(len(self)))
        new_pool = list(map(self._drop_higher_order_z_strings, self.pool))
        if inplace:
            self._pool = self.drop_zero_operators(new_pool)
            logger.info(
                'Pool now has {} operators'.format(
                    len(self)))
            return self.pool
        else:
            logger.info(
                'Pool now has {} operators'.format(
                    len(new_pool)))
            return self.drop_zero_operators(new_pool)

    @classmethod
    def from_max_word_length(cls, num_qubits, max_word_length):
        """Create pool from maximum number of non-identity terms. E.g.,
        ```
        pool = PauliPool.from_max_word_length(4, 2)
        ```
        This pool has elements like `IIXZ`, `IIIZ`, `YIXI`, etc.
        """
        result = cls()
        result.name = 'max_word_length_num_qubits_{}_max_word_length_{}'.format(num_qubits, max_word_length)
        words = filter(lambda word: 0 < pauli_word_len(word) <= max_word_length,
                       [''.join(word) for word in product('IXYZ', repeat=num_qubits)]
                       )
        res = [Pauli.from_label(word) for word in words] + \
              [Pauli.from_label(''.join(['I'] * num_qubits))]
        operator_pool = list(map(lambda p: WeightedPauliOperator(paulis=[[1.0, p]]), res))
        result.add_ops(operator_pool)
        return result

    @classmethod
    def from_all_pauli_strings(cls, num_qubits):
        """Creates pool from all possible Pauli strings for given number of qubits.
        ```
        pool = PauliPool.from_all_pauli_strings(4)
        ```
        """
        result = cls().from_max_word_length(num_qubits, num_qubits)
        return result

    @classmethod
    def from_exact_word_length(cls, num_qubits, word_length):
        """Creates pool from strings that have a specific number of non-identity
        Paulis. E.g.,
        ```
        pool = PauliPool.from_exact_word_length(3, 2)
        ```
        This pool has elements like `XXI`, `ZIY`, etc.
        """
        result = cls()
        result.name = 'exact_word_length_num_qubits_{}_word_length_{}'.format(num_qubits, word_length)
        words = list(filter(lambda word: pauli_word_len(word) == word_length,
                            [''.join(word) for word in product('IXYZ', repeat=num_qubits)]
                            ))
        operator_pool = list(map(lambda p: WeightedPauliOperator(paulis=[[1.0, Pauli.from_label(p)]]), words))
        result.add_ops(operator_pool)
        return result

    @classmethod
    def from_pauli_strings(cls, string_list):
        """Creates pool from list of given Pauli strings.
        ```
        pool = PauliPool.from_pauli_strings(['IZZ', 'XXY'])
        ```
        """
        result = cls()
        result.name = 'pauli_string_list_{}'.format(string_list)
        lengths = list(map(len, string_list))
        if not all(x == lengths[0] for x in lengths):
            raise ValueError('All strings should be of equal length')
        res = [Pauli.from_label(word.upper()) for word in string_list]
        res = list(map(lambda p: WeightedPauliOperator(paulis=[[1.0, p]]), res))
        result.add_ops(res)
        return result

    @classmethod
    def from_coupling_list_pairs(cls, coupling_list, pauli_pairs, include_reverses=False):
        """Creates pool with 2-local Paulis where the qubits that interact are
        given by a coupling list.
        Parameters
        ----------
        coupling_list : List[Tuple[Int, Int]]
            List of which qubits are connected. E.g., `[[0, 1], [1, 2]]`.
        pauli_pairs : str
            Pauli pairs to place at every edge of the connectivity graph. E.g.,
            `['ZZ', 'ZX']`.
        include_reverses : bool
            If `True`, then also include `'XZ'` in the above example.
        """
        labels = list(set(np.array(coupling_list).flatten()))
        if labels != list(range(0, len(labels))):
            raise ValueError('Invalid coupling list provided')
        num_qubits = len(labels)

        if include_reverses:
            reverses = list(map(lambda x: x[::-1], pauli_pairs))
            pauli_pairs += reverses

        for pair in pauli_pairs:
            if len(pair) != 2 or not isinstance(pair, str):
                raise ValueError('All Pauli pairs must be strings of length 2')
            if not set(pair).issubset({'X', 'Y', 'Z', 'I'}):
                raise ValueError('All Pauli pairs must be strings with chars X, Y, Z, or I')

        final_strings = []
        for connected_pair in coupling_list:
            for pauli_pair in pauli_pairs:
                id = list('I' * num_qubits)
                q1, q2 = connected_pair
                p1, p2 = pauli_pair[0], pauli_pair[1]
                id[q1] = p1
                id[q2] = p2
                id = ''.join(id)
                final_strings.append(id)
        final_strings = list(set(final_strings))
        res = cls.from_pauli_strings(final_strings)
        res.name = 'from coupling list pairs'
        return res


    @classmethod
    def from_openfermion_qubit_operator_list(cls, of_op_list: List[QubitOperator]):
        for op in of_op_list:
            if not isinstance(op, QubitOperator):
                raise ValueError('Found element in list that is not a QubitOperator')

        num_qubits = np.max(np.max(
            [[pair[0] for pair in tups] for tups in list(QubitOperator.accumulate(of_op_list).terms.keys())])) + 1
        op_list_res = []
        for op in of_op_list:
            pauli_list = []
            for tup, coeff in op.terms.items():
                if not np.isreal(coeff):
                    raise ValueError('All coefficients must be real so that resulting operator is Hermitian')
                pauli_list.append([coeff, Pauli.from_label(tuples_to_string(tup, num_qubits))])
            op_res = WeightedPauliOperator(paulis=pauli_list)
            op_list_res.append(op_res)
        result = cls.from_operator_list(op_list_res)
        result.name = 'openfermion_qubit_operator_list'.format(of_op_list)
        return result

    @classmethod
    def from_complete_two_qubit_list(cls):
        """Uses Ed Barnes' two qubit complete pool to construct a 2 qubit pool."""
        res = cls.from_pauli_strings(_TWO_QUBIT_COMPLETE_POOL)
        res.name = 'Complete two qubit pool'
        return res


def x_strings(n: int, axis: str = 'X'):
    """Returns a list of strings with a single X and all others I."""
    def x_string(i: int):
        l = ['I']*n
        l[i] = axis
        return ''.join(l)
    return [x_string(i) for i in range(n)]


def x_mixer(n: int, axis: str = 'X'):
    """Construct the original QAOA mixer.
    Parameters
    ----------
    n : int
        Number of qubits.
    Returns
    -------
    WeightedPauliOperator
        QAOA Mixer, `X_1+X_2+...`.
    """
    op = WeightedPauliOperator([])
    for o in PauliPool.from_pauli_strings(x_strings(n, axis=axis)).pool:
        op += o
    return op


def overcomplete_pool(n: int, non_local: str = 'ZY', local: List[str] = ['Z', 'Y']) -> PauliPool:
    """Construct a pool based on interacting chain of qubits.
    Parameters
    ----------
    n : int
        Number of qubits.
    non_local : str
        Interaction term for neighboring qubits.
    local : List[str]
        Single qubit terms.
    Returns
    -------
    PauliPool
        Resulting pool.
    """
    couplings = [[i, i+1] for i in range(n-1)]
    pool = PauliPool.from_coupling_list_pairs(couplings, [non_local], include_reverses=False)
    local_ops = []
    for a in local:
        for i in range(n):
            local_ops.append(paul(n_qubits=n, index=i, axis=a))
    pool += OperatorPool.from_operator_list(local_ops)
    return pool


def make_complete_pool(n_qubits: int ,local_string: str = 'ZY' ):
    """Construct a complete pool on `n` qubits with `2n-2` elements.
    Parameters
    ----------
    n_qubits : int
        Number of qubits.
    Returns
    -------
    List[str]
        Pauli strings for elements in pool.
    """
    if local_string is 'ZY':
        if n_qubits == 3:
            return _THREE_QUBIT_COMPLETE_POOL
        elif n_qubits > 3:
            _lower_pool = make_complete_pool(n_qubits-1)
            result = ['Z'+pauli for pauli in _lower_pool]

            Yn = 'Y' + 'I'*(n_qubits-1)
            Ynm1 = 'IY' + 'I'*(n_qubits-2)

            result.extend([Yn, Ynm1])
            return result
        else:
            raise ValueError('Invalid number of qubits: {}'.format())
    elif local_string is 'YX':
        combinations = []
        combinations3 = []
        for i in range(n_qubits-1):
            combinations.append('I'*(i)+'YX'+'I'*(n_qubits-i-2))
        for i in range(n_qubits-2):
            combinations3.append('I'*(i)+'YIX'+'I'*(n_qubits-i-3))
        combinations3=combinations3[1:len(combinations3)]
        Yn = 'I'*(n_qubits-1)+'Y' 
        Ynm1 = 'I'*(n_qubits-2)+'YI' 
        result = combinations+combinations3
        result.extend([Yn, Ynm1])
        return result
    else:
        raise ValueError('Only "ZY" and "YX" are acceptable local strings')


class CompletePauliPool(PauliPool):
    """Complete pool as constructed by Vlad Shkolnikov."""

    @classmethod
    def from_num_qubits(cls, num_qubits: int, local_string: str = 'ZY'):
        """Construct complete pool from number of qubits.
        ```
        pool = CompletePauliPool.from_num_qubits(5)
        ```
        """
        result = cls()

        if type(num_qubits) != int:
            raise ValueError('num_qubits must be an int, got {}'.format(num_qubits))
        if num_qubits < 1:
            raise ValueError('num_qubits must be non-zero, got {}'.format(num_qubits))
        
        if local_string is 'ZY':
            if num_qubits == 1:
                op_list = PauliPool.from_pauli_strings(['Y']).pool
            elif num_qubits == 2:
                op_list = PauliPool.from_pauli_strings(_TWO_QUBIT_COMPLETE_POOL).pool
            elif num_qubits == 3:
                op_list = PauliPool.from_pauli_strings(_THREE_QUBIT_COMPLETE_POOL).pool
            elif num_qubits > 3:
                pauli_strings = make_complete_pool(num_qubits)
                op_list = PauliPool.from_pauli_strings(pauli_strings).pool
        elif local_string is 'YX':
            if num_qubits == 1:
                op_list = PauliPool.from_pauli_strings(['Y']).pool
            elif num_qubits > 1:
                pauli_strings = make_complete_pool(num_qubits,local_string)
                op_list = PauliPool.from_pauli_strings(pauli_strings).pool
        else:
            raise ValueError('Local string only supports "ZY" or "YX" options')
        result.add_ops(op_list)
        return result


class ADAPTQAOAPool(OperatorPool):

    @classmethod
    def from_pool_type(cls, num_qubits: int, pool_type: str):
        res = cls()
        res.add_ops([x_mixer(num_qubits)])
        if pool_type != 'qaoa':
            res.add_ops(PauliPool.from_exact_word_length(num_qubits, 1).pool)
            res.add_ops([x_mixer(num_qubits, axis='Y')])
        if pool_type == 'multi':
            res.add_ops(PauliPool.from_exact_word_length(num_qubits, 2).pool)
        res.name = 'adapt_qaoa_{}'.format(pool_type)

        return res

    def drop_z_2_symmetry_ops(self, inplace: bool = True):
        f_op = WeightedPauliOperator.from_list(paulis=[Pauli.from_label('X'*self.num_qubits)], weights=[1.0])
        new_ops = list(filter(
            lambda op: f_op.commute_with(op),
            self.pool
        ))
        if inplace:
            self._pool = new_ops
        else:
            res = ADAPTQAOAPool.from_operator_list(new_ops)
            return res


def plus_op(k: int, minus: bool = False) -> QubitOperator:
    sign = -1 if minus else 1
    res = QubitOperator().identity() + sign * QubitOperator('X{}'.format(k))
    res /= 2
    return res


def plus_product_mixer(n: int, of: bool = False):
    res = QubitOperator().identity()
    for i in range(n):
        res *= plus_op(i)
    if of:
        return res
    else:
        return openfermion_to_qiskit(res, num_qubits=n)