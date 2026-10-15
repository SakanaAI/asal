import numpy as np

def generate_random_states():
    state_tensor = np.random.uniform(low=-1, high=1, size=(7, 10, 10))
    for i in range(100):
        state_tensor[0] = np.random.uniform(low=-1, high=1, size=(10, 10)) # activations

        state_tensor[1] += np.random.uniform(low=0, high=1, size=(10, 10)) # cytoplasm
        state_tensor[1] = state_tensor[1] / np.linalg.norm(state_tensor[1])

        state_tensor[2:7] += np.random.uniform(low=-1, high=1, size=(5, 10, 10)) # muscle radii
        state_tensor[2:7] = state_tensor[2:7] / np.linalg.norm(state_tensor[2:7])

        np.save(f'./fake_data/state_tensor{i}.npy', state_tensor)
    
generate_random_states()