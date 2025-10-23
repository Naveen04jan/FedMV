import subprocess
import os
from config import NUM_CLIENTS

def start_clients():
    processes = []
    for client_id in range(NUM_CLIENTS):
        env = os.environ.copy()
        env["CLIENT_ID"] = str(client_id)
        print(f"Starting client {client_id} with CLIENT_ID={client_id}")
        p = subprocess.Popen(["python", "client.py"], env=env)
        processes.append(p)

    for p in processes:
        p.wait()

if __name__ == "__main__":
    start_clients()
