import numpy as np

def generate_traffic():
    packet_size = np.random.randint(20, 1500)
    destination_port = np.random.choice([80, 443, 22, 23, 25, 8080])
    protocol = np.random.choice([0, 1])

    if destination_port in [23, 25]:
        label = 1  # Attack
    else:
        label = 0  # Normal

    return [packet_size, destination_port, protocol], label
