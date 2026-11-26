import numpy as np
import os
from qiskit.quantum_info import Z2Symmetries
from qiskit_nature.second_q.operators import FermionicOp
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.circuit.library import UCCSD
from qiskit_nature.second_q.circuit.library import HartreeFock
from qiskit_nature.second_q.mappers import (
    JordanWignerMapper,
    BravyiKitaevMapper,
    ParityMapper,
)

from torchquantum.plugin.qiskit.qiskit_processor import QiskitProcessor
from torchquantum.plugin import qiskit2tq

from torchpack.utils.config import configs

processor = QiskitProcessor(
    use_real_qc=False,
    backend_name=None,
    noise_model_name=None,
    # coupling_map_name=None,
    basis_gates=None,
    n_shots=8192,
    initial_layout=None,
    seed_transpiler=42,
    seed_simulator=42,
    optimization_level=None,
    max_jobs=5,
    remove_ops=False,
    remove_ops_thres=1e-4,
)

# import pdb
# pdb.set_trace()

def get_mapper(transform):
    if transform == 'BK':
        return BravyiKitaevMapper()

    elif transform == 'JW':
        return JordanWignerMapper()

    else:
        return ParityMapper()


def load_qubitop_for_molecule(molecule_data):

    atom_list = [
        a[0] + ' ' + " ".join([str(elem) for elem in a[1]])
        for a in molecule_data['geometry']
    ]

    atom = "; ".join(atom_list)

    basis = molecule_data['basis']
    transform = molecule_data['transform']

    driver = PySCFDriver(
        atom=atom,
        unit=DistanceUnit.ANGSTROM,
        basis=basis,
        charge=0,
        spin=0
    )

    problem = driver.run()

    molecule = problem

    mapper = get_mapper(transform)

    # ----------------------------------------
    # particle information
    # ----------------------------------------

    num_particles_tuple = (
        problem.num_alpha,
        problem.num_beta
    )

    num_particles = (
        problem.num_alpha +
        problem.num_beta
    )

    num_spatial_orbitals = problem.num_spatial_orbitals

    num_spin_orbitals = 2 * num_spatial_orbitals

    # ----------------------------------------
    # second quantized Hamiltonian
    # ----------------------------------------

    second_q_op = problem.hamiltonian.second_q_op()

    # ----------------------------------------
    # qubit mapping
    # ----------------------------------------

    qubitOp = mapper.map(second_q_op)

    return (
        molecule,
        qubitOp,
        mapper,
        num_particles_tuple,
        num_spatial_orbitals
    )


def generate_uccsd(molecule_data):

    (
        molecule,
        qubitOp,
        mapper,
        num_particles,
        num_spatial_orbitals
    ) = load_qubitop_for_molecule(molecule_data)

    print(
        "# of electrons: {}".format(
            sum(num_particles)
        )
    )

    print(
        "# of spatial orbitals: {}".format(
            num_spatial_orbitals
        )
    )

    # ----------------------------------------
    # Hartree-Fock initial state
    # ----------------------------------------

    HF_state = HartreeFock(
        num_spatial_orbitals=num_spatial_orbitals,
        num_particles=num_particles,
        qubit_mapper=mapper,
    )

    # ----------------------------------------
    # UCCSD ansatz
    # ----------------------------------------

    uccsd_ansatz = UCCSD(
        reps=1,
        num_spatial_orbitals=num_spatial_orbitals,
        num_particles=num_particles,
        qubit_mapper=mapper,
        initial_state=HF_state,
    )

    # ----------------------------------------
    # assign parameters
    # ----------------------------------------

    circ = uccsd_ansatz.assign_parameters(
        [0.4242] * uccsd_ansatz.num_parameters
    )

    circ.measure_all()

    circ_transpiled = processor.transpile(circ)

    q_layer = qiskit2tq(circ_transpiled)

    for name, param in q_layer.named_parameters():

        if not (
            param % (np.pi / 2)
        ).detach().cpu().numpy().any():

            param.requires_grad = False

    if getattr(
        configs.model.arch,
        'n_truncate_ops',
        None
    ) is not None:

        q_layer.ops = q_layer.ops[
            :configs.model.arch.n_truncate_ops
        ]

    return q_layer

def molecule_data2str(md):
    return md['name'] + ' ' + md['basis'] + ' ' + md['transform']+ ' ' + str(md['active_orbitals'])

def write_ansatz(molecule_data):
    #filename = ...
    ansatz = generate_uccsd(molecule_data)
    randlist = np.random.rand(uccsd_ansatz.num_parameters) # ansatz parameters
    uccsd_ansatz_circuit = uccsd_ansatz.assign_parameters(randlist)
    print(uccsd_ansatz_circuit)

def write_observable(molecule_data, root):
    #filename = ...
    _, qubitOp, _, _ , _ = load_qubitop_for_molecule(molecule_data)
    molecule_str = molecule_data2str(molecule_data)
    numq = qubitOp.num_qubits
    molecule_str += ' q' + str(numq) + '\n'
    op_str = qubitOp.print_details()
    filename = f"{molecule_data['name'].lower()}_" \
               f"{molecule_data['transform'].lower()}"
    with open(os.path.join(root, filename, f"{filename}.txt"), 'w') as wfid:
        wfid.write(f"{molecule_data['name'].lower()} "
                   f"{molecule_data['transform'].lower()} {numq}\n")
        for line in op_str.splitlines():
            molecule_str = ''
            #print(ord(line[6])) #ZXXIII (6.505213034913027e-19+0j)
            linedata = line.split(chr(9))
            if not complex(linedata[1]).imag == 0:
                print(f"WARNING: imaginary is not zero!!")

            molecule_str += str(complex(linedata[1]).real) + ' '
            for (i, c) in enumerate(linedata[0]):
                molecule_str += c+str(i)+' '

            wfid.write(f"{molecule_str}\n")

        # molecule_str
    # print(molecule_str)

        

    

# Molecule parameters for H2
h2_molecule = {
'name' : 'H2',
'basis' : 'sto-3g',
'transform' : 'BK',
'electrons' : 2,
'geometry' : [('H', (0., 0., 0.)), ('H', (0., 0., 0.72))],
'active_orbitals' : 2
}

# Molecule parameters for H2O
h2o_molecule = {
'name' : 'H2O',
'basis' : 'sto-3g',
'transform' : 'BK',
'electrons' : 8,
'geometry' : [('O', (0.,0.,0.)), ('H', (0.757,0.586,0.)), ('H', (-0.757,0.586,0.))],
'active_orbitals' : 4
}

# Molecule parameters for LiH
lih_molecule = {
'name' : 'LiH',
'basis' : 'sto-3g',
'transform' : 'BK',
'electrons' : 4,
'geometry' : [('Li', (0., 0., 0.)), ('H', (0., 0., 1.45))],
'active_orbitals' : 4
}

# Molecule parameters for CH4
ch4_molecule = {
'name' : 'CH4',
'basis' : 'sto-3g',
'transform' : 'BK',
'electrons' : 10,
'geometry' : [('C', (0, 0, 0)), ('H', (0.5541, 0.7996, 0.4965)), 
            ('H', (0.6833, -0.8134, -0.2536)), ('H', (-0.7782, -0.3735, 0.6692)), 
            ('H', (-0.4593, 0.3874, -0.9121))],
'active_orbitals' : 4
}

beh2_molecule = {
    'name' : 'BeH2',
    'basis' : 'sto-3g',
    'transform' : 'BK',
    'electrons' : 8,
    'geometry' : [('Be', (0., 0., 0.)), ('H', (0., 0., -1.7)), ('H', (0., 0., 1.7))],
    'active_orbitals' : 7
}

ch4new_molecule = {
    'name' : 'CH4NEW',
    'basis' : 'sto-3g',
    'transform' : 'BK',
    'electrons' : 10,
    'geometry' : [('C', (0, 0, 0)), ('H', (0.5541, 0.7996, 0.4965)),
                  ('H', (0.6833, -0.8134, -0.2536)), ('H', (-0.7782, -0.3735, 0.6692)),
                  ('H', (-0.4593, 0.3874, -0.9121))],
    'active_orbitals' : 6
}

#generate_uccsd(h2_molecule)
#generate_uccsd(h2o_molecule)
#generate_uccsd(lih_molecule)
#generate_uccsd(ch4_molecule)

molecule_name_dict = {
    'h2': h2_molecule,
    'h2o': h2o_molecule,
    'lih': lih_molecule,
    'ch4': ch4_molecule,
    'beh2': beh2_molecule,
    'ch4new': ch4new_molecule,
}

if __name__ == '__main__':
    import pdb
    pdb.set_trace()
    # generate_uccsd(molecule_name_dict['ch4'])

# for transform in ['BK', 'JW']:
#     for name, info in molecule_name_dict.items():
#         root = './examples/data/vqe/'
#         info['transform'] = transform
#         os.makedirs(os.path.join(root, f"{name}_{transform.lower()}"),
#                     exist_ok=True)
#
#         write_observable(info, root)
