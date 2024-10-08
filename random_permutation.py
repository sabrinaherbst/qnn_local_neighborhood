import os
import numpy as np
import pennylane as qml
import math
from multiprocessing import Pool
import pandas as pd
import os

from util import diamond_norm

#### Utility functions ####

def adjoint(matrix):
    "Returns the adjont of a matrix"
    return matrix.conjugate().transpose()

#### Circuit functions ####

def circuit(params, rx, ry, rz, cnot, cz):
    """
    Implements the circuit in pennylane and returns the quantum state
    """
    
    for layer in range(params.shape[0]):
        for qubit in range(params.shape[1]):
            i = 0
            if rx:
                qml.RX(params[layer][qubit][i], qubit)
                i += 1
            if ry:
                qml.RY(params[layer][qubit][i], qubit)
                i += 1
            if rz:
                qml.RZ(params[layer][qubit][i], qubit)
                i += 1
        
        if cnot:
            for qubit in range(params.shape[1] - 1):
                qml.CNOT([qubit, qubit+1])
            qml.CNOT([params.shape[1]-1, 0])

        if cz:
            for qubit in range(params.shape[1] - 1):
                qml.CZ([qubit, qubit+1])
            qml.CZ([params.shape[1]-1, 0])

    return qml.state()

def channel(params, rx, ry, rz, cnot, cz, qnode):
    return qml.matrix(qnode)(params, rx, ry, rz, cnot, cz)

def difference_upon_perturbation_mulitple_params(config):
    layers, qubits, params_per_layer, perturbation_threshold, param_threshold, args = config

    path = f'perturbation/{layers}_{qubits}_{params_per_layer}_{"_".join([str(x) for x in args])}_{perturbation_threshold}_{param_threshold}.txt'

    if os.path.exists(path):
        print("Already exists", path)
        return
    
    N = 500 * (layers * qubits * params_per_layer)
    res = []
    layer_str = []

    if args[0]:
        layer_str.append('RX')
    if args[1]:
        layer_str.append('RY')
    if args[2]:
        layer_str.append('RZ')

    dev = qml.device("default.qubit", wires=qubits)
    qnode = qml.QNode(circuit, dev)

    for trial in range(N):
        print("Trial", trial)
        # randomly sample (layers * qubits * params_per_layer) * param_threshold parameters
        perturbed_params = np.random.choice(np.arange(0, layers*qubits*params_per_layer), size=int(layers*qubits*params_per_layer*param_threshold), replace=False)

        p = np.random.uniform(0, 2*np.pi, (layers, qubits, params_per_layer))
        U_orig = channel(p, qnode=qnode, *args)

        rg = perturbation_threshold * (np.pi*2)

        for param in perturbed_params:
            l = int(param / (qubits * params_per_layer))
            rest = int(param % (qubits * params_per_layer))
            qb = int(rest / params_per_layer)
            param = int(rest % params_per_layer)

            # perturb either up or down
            p[l][qb][param] += rg * np.random.choice([-1, 1])
            # make sure between 0 and 2pi
            p[l][qb][param] = p[l][qb][param] % (2*np.pi)

        op = U_orig - channel(p, qnode=qnode, *args)

        delta = np.repeat(rg, int(layers * qubits * params_per_layer * param_threshold))
        delta = np.linalg.norm(delta)

        twonorm = np.linalg.norm(op, ord=2)
        tracenorm = np.trace(np.absolute(op))
        d = diamond_norm(U_orig, channel(p, qnode=qnode, *args))
        print(d)
        res.append([l, qb, trial, twonorm, tracenorm, d, delta])

    # save
    df = pd.DataFrame(res, columns=["Layer", "Qubit", "Run", "2-Norm", "Trace Norm", "Diamond Norm", "Bound"])
    df.to_csv(path)
    print("Done", layers, qubits, params_per_layer, args)


if not os.path.exists("perturbation"):
    os.mkdir("perturbation")

configs = []

for qubits in range(1, 5):
    for layers in range(1, 6):
        for rx in [True, False]:
            for ry in [True, False]:
                for rz in [True, False]:
                    for cnot in [True, False]:
                        for cz in [True, False]:
                            for pert_threshold in [0.001, 0.005, 0.01]:
                                for param_threshold in [0.5, 0.75,  0.95]:
                                    if (qubits * layers * (rx + ry + rz)) * param_threshold < 1:
                                        continue
                                    if qubits == 1 and (cz + cnot) > 0:
                                        continue
                                    if rx + ry + rz == 0:
                                        continue
                                    configs.append((layers, qubits, rx+ry+rz, pert_threshold, param_threshold, [rx, ry, rz, cnot, cz]))

with Pool(16) as p:
    p.map(difference_upon_perturbation_mulitple_params, configs)