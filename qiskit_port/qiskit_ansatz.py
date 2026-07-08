"""Qiskit ansatz builders."""


from qiskit import QuantumCircuit


def all_symmetry_ansatzae_qiskit(
    theta,
    n_qubits,
    n_layers,
    initial_occupations_indices,
    connected_graphs,
    compilation="generic",
):
    if compilation != "generic":
        raise ValueError("Only generic compilation is implemented for now.")

    qc = QuantumCircuit(n_qubits)

    graph_up = connected_graphs["graph_up"]
    graph_down = connected_graphs["graph_down"]
    graph_stitch = connected_graphs["graph_stitch"]

    for idx in initial_occupations_indices:
        qc.x(idx)

    k = 0

    for _ in range(n_layers):
        for edge in graph_up.edges():
            qubit_a, qubit_b = edge

            qc.s(qubit_a)
            qc.s(qubit_b)
            qc.h(qubit_a)
            qc.cx(qubit_a, qubit_b)
            qc.ry(-theta[k], qubit_a)
            qc.ry(-theta[k], qubit_b)
            k += 1
            qc.cx(qubit_a, qubit_b)
            qc.h(qubit_a)
            qc.sdg(qubit_a)
            qc.sdg(qubit_b)

        for edge in graph_down.edges():
            qubit_a, qubit_b = edge

            qc.s(qubit_a)
            qc.s(qubit_b)
            qc.h(qubit_a)
            qc.cx(qubit_a, qubit_b)
            qc.ry(-theta[k], qubit_a)
            qc.ry(-theta[k], qubit_b)
            k += 1
            qc.cx(qubit_a, qubit_b)
            qc.h(qubit_a)
            qc.sdg(qubit_a)
            qc.sdg(qubit_b)

        for edge in graph_stitch.edges():
            qubit_a, qubit_b = edge

            qc.cx(qubit_a, qubit_b)
            qc.rz(-theta[k], qubit_b)
            qc.cx(qubit_a, qubit_b)
            k += 1

        for q in range(n_qubits):
            qc.rz(-theta[k], q)
            k += 1

    return qc