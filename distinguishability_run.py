import pennylane as qml
from pennylane import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.datasets import load_breast_cancer, load_wine

import pandas as pd

from util import diamond_norm

from multiprocessing.pool import Pool

# number of repetitions for training
REPS = 50

def statepreparation(x, qubits):
    """
    Encode input x into the quantum state of the qubits.
    """
    # pad to 2^qubits with 0
    if x.shape[1] < 2**qubits:
        x = np.pad(x, ((0, 0), (0, 2**qubits - x.shape[1])), mode='constant')

    # norm to 1
    norm = np.linalg.norm(x, axis=1)
    x = x / norm[:, None]

    # encode into the amplitudes
    qml.StatePrep(x, wires=range(qubits))

def layer(parameters, x, y, z, cnot, cz, qubits):
    """
    Variational circuit representing a layer of an ansaz.
    """
    ind = 0
    for i in range(qubits):
        if x:
            qml.RX(parameters[ind], wires=i)
            ind += 1
        if y:
            qml.RY(parameters[ind], wires=i)
            ind += 1
        if z:
            qml.RZ(parameters[ind], wires=i)
            ind += 1

    if cnot:
        for j in range(qubits-1):
            qml.CNOT(wires=[j, j+1])
        qml.CNOT(wires=[qubits-1, 0])
    if cz:
        for j in range(qubits-1):
            qml.CZ(wires=[j, j+1])
        qml.CZ(wires=[qubits-1, 0])

def get_circuit(data, parameters, reps, x, y, z, cnot, cz, qubits):
    """
    Quantum circuit for the variational classifier.
    """
    statepreparation(data, qubits)
    for i in range(reps):
        layer(parameters[i*qubits*(x+y+z):(i+1)*qubits*(x+y+z)], x, y, z, cnot, cz, qubits)

    # measure first qubit in z-basis
    return qml.expval(qml.PauliZ(0))

def get_channel(parameters, reps, x, y, z, cnot, cz, qubits):
    """"
    Function to obtain the unitary applied to the encoded state.
    """
    for i in range(reps):
        layer(parameters[i*qubits*(x+y+z):(i+1)*qubits*(x+y+z)], x, y, z, cnot, cz, qubits)
    return qml.state()

def square_loss(labels, predictions):
    """
    Square loss function.
    """
    return np.mean((labels - qml.math.stack(predictions)) ** 2)

def accuracy(labels, predictions):
    """
    Accuracy function.
    """
    acc = sum(abs(l - p) < 1e-5 for l, p in zip(labels, predictions))
    acc = acc / len(labels)
    return acc

def cost(parameters, X, Y, circuit, configuration, reps, qubits):
    """
    Runs predictions and returns the square loss.
    """
    predictions = circuit(X, parameters, reps=reps, x=configuration[0], y=configuration[1], z=configuration[2], cnot=configuration[3], cz=configuration[4], qubits=qubits)
    return square_loss(Y, predictions)

def relative_change(prev, new):
    """
    Compute the relative change between two sets of parameters.
    """
    diff = prev - new
    adjust_circular = (diff + np.pi) % (2*np.pi) - np.pi
    return np.abs(adjust_circular)

def train_model(parameters, rx, ry, rz, cnot, cz, depth, qubits, circuit):
    """
    Train the model using the given parameters.
    """
    opt = qml.optimize.AdamOptimizer()

    X_train = pd.read_csv("data/X_train.csv").drop(columns=["Unnamed: 0"])
    X_test = pd.read_csv("data/X_test.csv").drop(columns=["Unnamed: 0"])
    y_train = pd.read_csv("data/y_train.csv").drop(columns=["Unnamed: 0"])
    y_test = pd.read_csv("data/y_test.csv").drop(columns=["Unnamed: 0"])

    if X_train.shape[1] > 2**qubits:
        pca = PCA(n_components=2**qubits)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

    X_train = np.array(X_train, requires_grad=False)
    X_test = np.array(X_test, requires_grad=False)

    y_train = np.array(y_train, requires_grad=False)
    y_test = np.array(y_test, requires_grad=False)

    # define device for obtaining the channel
    dev2 = qml.device("default.qubit", wires=qubits)
    get_channel_qnode = qml.QNode(get_channel, dev2)
    
    conv = []
    param_updates = []
    for i in range(150):
        prev_parameters = parameters
        ind = 0

        # update parameters
        parameters, loss = opt.step_and_cost(cost, parameters, X_train, y_train, circuit, [rx, ry, rz, cnot, cz], reps=depth, qubits=qubits)
        parameters = parameters[0]
        conv.append([i, loss])

        # get mean percentual change
        change = ((np.abs(prev_parameters - parameters) % np.pi) / 2*np.pi)
        mean_change = np.mean(change)
        std_change = np.std(change)

        # Compute predictions on train and validation set
        predictions_train = np.sign(circuit(X_train, parameters, depth, rx, ry, rz, cnot, cz, qubits))
        predictions_val = np.sign(circuit(X_test, parameters, depth, rx, ry, rz, cnot, cz, qubits))

        # Compute accuracy on train and validation set
        acc_train = accuracy(y_train, predictions_train)
        acc_val = accuracy(y_test, predictions_val)

        # calc diamond norm
        dnorm = diamond_norm(qml.matrix(get_channel_qnode)(parameters, depth, rx, ry, rz, cnot, cz, qubits), qml.matrix(get_channel_qnode)(prev_parameters, depth, rx, ry, rz, cnot, cz, qubits))
        print(f"Step {i}, loss {loss}, acc_train {acc_train}, acc_val {acc_val}")

        # get absolute change in parameters
        chg = ((np.abs(prev_parameters - parameters) % np.pi) / 2*np.pi).reshape((depth, qubits, rx+ry+rz))  
        # get absolute change in parameters
        rel_chg = relative_change(prev_parameters, parameters)
        rel_chg = np.sum(rel_chg)/2

        # get mean change per layer, per qubit, per gate
        per_layer = np.sum(np.sum(chg, axis=2), axis=1) / (qubits*(rx+ry+rz))
        per_qubit = np.sum(np.sum(chg, axis=2), axis=0) / (depth*(rx+ry+rz))
        per_gate = np.sum(np.sum(chg, axis=1), axis=0) / (depth*qubits)
        
        # get changes in rx, ry, rz
        ind = 0
        rx_changes = np.inf
        ry_changes = np.inf
        rz_changes = np.inf
        
        if rx:
            rx_changes = per_gate[ind]
            ind += 1
        if ry:
            ry_changes = per_gate[ind]
            ind += 1
        if rz:
            rz_changes = per_gate[ind]
            ind += 1

        # number of parameters that changed
        changed = np.sum(np.abs(prev_parameters - parameters) > 0.0)/len(parameters)

        param_updates.append([i, mean_change, std_change, per_layer, per_qubit, rx_changes, ry_changes, rz_changes, changed, dnorm, rel_chg])
        print(f"Step {i}, mean change {mean_change}, std change {std_change}, changed {changed}, dnorm {dnorm}, rel_chg {rel_chg}")

    # get final results         
    pred_train = circuit(X_train, parameters, depth, rx, ry, rz, cnot, cz, qubits)
    pred_test = circuit(X_test, parameters, depth, rx, ry, rz, cnot, cz, qubits) 
    loss_train = square_loss(y_train, pred_train)
    loss_val = square_loss(y_test, pred_test)
    acc_train = accuracy(y_train, np.sign(pred_train))
    acc_val = accuracy(y_test, np.sign(pred_test))
    
    # save results
    res = pd.DataFrame(conv, columns=["step", "loss"])
    param_updates = pd.DataFrame(param_updates, columns=["step", "mean_change", "std_change", "per_layer", "per_qubit", "rx_changes", "ry_changes", "rz_changes",  "changed", "dnorm", "rel_chg"])
    final = [loss_train, acc_train, loss_val, acc_val]
    return res, param_updates, final
    

def run_simulation(configuration):
    """
    Run the simulation for the given configuration.
    """
    layers, qubits, rx, ry, rz, cnot, cz = configuration
    print(f"Running simulation for {layers} layers, {qubits} qubits, RX={rx}, RY={ry}, RZ={rz}, CNOT={cnot}, CZ={cz}")

    # check if exists
    if os.path.exists(f"convergence/res_{ds}_{rx}_{ry}_{rz}_{cnot}_{cz}_{layers}_{qubits}.csv"):
        print("Already exists")
        return
    
    shape = (layers*qubits*(rx+ry+rz))
    dev = qml.device("default.qubit", wires=qubits)
    circuit = qml.QNode(get_circuit, dev)

    res = []
    param_updates = []
    final = []
    for i in range(REPS):
        print(f"Running simulation for {layers} layers, {qubits} qubits, RX={rx}, RY={ry}, RZ={rz}, CNOT={cnot}, CZ={cz}, rep={i}")

        parameters = np.random.uniform(size=shape, low=0, high=2*np.pi, requires_grad=True)
        r, p, f = train_model(parameters, rx, ry, rz, cnot, cz, layers, qubits, circuit)

        r['rep'] = i
        p['rep'] = i
        f.insert(0, i)

        r['opt'] = 'adam'
        p['opt'] = 'adam'
        f.insert(1, 'adam')

        res.append(r)
        param_updates.append(p)
        final.append(f)
    
    res = pd.concat(res)
    param_updates = pd.concat(param_updates)
    final = pd.DataFrame(final, columns=["rep", "opt", "train_loss", "train_acc", "val_loss", "val_acc"])

    # to csv
    res.to_csv(f"convergence/res_{ds}_{rx}_{ry}_{rz}_{cnot}_{cz}_{layers}_{qubits}.csv")
    param_updates.to_csv(f"param_updates/param_updates_{ds}_{rx}_{ry}_{rz}_{cnot}_{cz}_{layers}_{qubits}.csv")
    final.to_csv(f"final_results/final_{ds}_{rx}_{ry}_{rz}_{cnot}_{cz}_{layers}_{qubits}.csv")    

for d in ["breast", "wine"]:
    ds = d

    if ds == 'breast':
        X, y = load_breast_cancer(return_X_y=True)

        train_samples = 150
        test_samples = 50
    elif ds == 'wine':
        X, y = load_wine(return_X_y=True)
        
        # keep only classes 0 & 1
        X = X[y < 2]
        y = y[y < 2]

        # sample size is 130
        train_samples = 100
        test_samples = 30

    # sample training and test set
    np.random.seed(0)
    train_idx = np.random.choice(range(len(X)), train_samples, replace=False)
    test_idx = np.array([i for i in range(len(X)) if i not in train_idx])

    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    # save
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    y_train = pd.DataFrame(y_train)
    y_test = pd.DataFrame(y_test)

    import os

    pth = "data/"
    if not os.path.exists(pth):
        os.makedirs(pth)

    if not os.path.exists("convergence"):
        os.makedirs("convergence")

    if not os.path.exists("param_updates"):
        os.makedirs("param_updates")

    if not os.path.exists("final_results"):
        os.makedirs("final_results")

    X_train.to_csv("data/X_train.csv")
    X_test.to_csv("data/X_test.csv")
    y_train.to_csv("data/y_train.csv")
    y_test.to_csv("data/y_test.csv")

    # get all combinations of x, y, z
    combinations = []
    for qubits in range(1, 5):
        for layers in range(1, 6):
            for x in [True, False]:
                for y in [True, False]:
                    for z in [True, False]:
                        for cnot in [True, False]:
                            for cz in [True, False]:
                                if not x and not y and not z: continue
                                if qubits == 1 and (cnot or cz): continue
                                combinations.append((layers, qubits, x, y, z, cnot, cz))


    with Pool(16) as p:
        p.map(run_simulation, combinations)

