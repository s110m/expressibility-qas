from torchquantum.plugin import tq2qiskit

import torchquantum as tq

class MyQauntum(tq.QuantumModule):

    def __init__(self):
        super().__init__()
        self.num_wires = 4
        self.q_device = tq.QuantumDevice(4)
        self.first = tq.RX(has_params=True, trainable=True)

    
    def forward(self, q_dev):
        # q_dev = tq.QuantumDevice(self.num_wires)

        self.first(q_dev, wires=0)

myquantum = MyQauntum()

circ = tq2qiskit(myquantum.q_device, myquantum)

print(list(circ.parameters))

print(circ.assign_parameters)
