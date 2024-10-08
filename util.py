import numpy as np

def adjust_global_phase(A, B):
    """
    Adjust the global phase of B relative to A so that they are aligned.
    """
    # Compute the global phase difference using the trace of A†B
    phase_difference = np.angle(np.trace(np.dot(A.conj().T, B)))

    # Remove the global phase from B by multiplying by e^(-i * phase_difference)
    B_adjusted = np.exp(-1j * phase_difference) * B

    return B_adjusted


def diamond_norm(op1, op2):
    """
    Compute the diamond norm between two superoperators.
    """
    from qutip import Qobj
    from qutip.core.metrics import dnorm
    
    op2 = adjust_global_phase(op1, op2)
    d = dnorm(Qobj(op1 - op2))

    # Ensure that the diamond norm is between 0 and 2
    assert d >= 0 and d <= 2
    return d