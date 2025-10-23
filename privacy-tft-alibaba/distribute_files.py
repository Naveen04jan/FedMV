# import os
# import json
# import numpy as np
# from config import DATA_DIR, NUM_CLIENTS

# def distribute_files():
#     all_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
#     np.random.shuffle(all_files)
#     client_files = np.array_split(all_files, NUM_CLIENTS)
#     allocation = {f"client_{i}": list(files) for i, files in enumerate(client_files)}

#     with open('file_allocation.json', 'w') as f:
#         json.dump(allocation, f, indent=4)

#     for i, files in enumerate(client_files):
#         print(f"Client {i} has {len(files)} files.")

# if __name__ == "__main__":
#     distribute_files()

#Dirichlet distribution of files to clients
import os
import json
import numpy as np
from numpy.random import dirichlet
from config import DATA_DIR, NUM_CLIENTS, ALPHA

def distribute_files():
    all_files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith('.csv')]
    num_files = len(all_files)
    
    # Get the proportions using the Dirichlet distribution
    proportions = dirichlet(np.ones(NUM_CLIENTS) * ALPHA, size=1).flatten()
    
    # Calculate the number of files for each client
    split_indices = (proportions * num_files).astype(int)
    current_sum = split_indices.sum()
    diff = num_files - current_sum
    
    # Adjust the split_indices to ensure all files are distributed
    if diff > 0:
        for i in range(diff):
            split_indices[np.argmin(split_indices)] += 1
    elif diff < 0:
        for i in range(-diff):
            split_indices[np.argmax(split_indices)] -= 1

    client_files = {}
    start_index = 0
    
    # Distribute files to clients
    for client_id in range(NUM_CLIENTS):
        end_index = start_index + split_indices[client_id]
        client_files[f"client_{client_id}"] = all_files[start_index:end_index]
        start_index = end_index
        print(f"Client {client_id} has {len(client_files[f'client_{client_id}'])} files.")

    # Save the file allocation to a JSON file
    with open('file_allocation.json', 'w') as f:
        json.dump(client_files, f, indent=4)

if __name__ == "__main__":
    distribute_files()
