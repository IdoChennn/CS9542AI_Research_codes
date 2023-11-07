# Server Code
import socket
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import threading
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


# Initialize a CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, 1)  # Reduced channel size
        self.conv2 = nn.Conv2d(8, 16, 3, 1)  # Reduced channel size
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(2304, 32)  # Adjusted for reduced channel size
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# data checksum and reformatting
def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf:
            raise ConnectionError("Incomplete data received")
        buf += newbuf
        count -= len(newbuf)
    return buf


# Sending global model as well as receiving model updates
def handle_client(conn, addr):
    print("Connected by", addr)
    data = pickle.dumps(model.state_dict())

    # Send the current model to the client
    conn.sendall(len(data).to_bytes(4, byteorder='big'))
    conn.sendall(data)

    # Receive client model state after training
    data_size = int.from_bytes(conn.recv(4), byteorder='big')
    try:
        client_state_data = recvall(conn, data_size)
        if client_state_data is None:
            print("Received incomplete data from client.")
            return
        client_state = pickle.loads(client_state_data)
    except ConnectionError as e:
        print(f"Error: {e}")
        return

    clients_state_dicts.append(client_state)
    print(f"Client updates received from {addr}")


# aggregate model updates after each round of training
def aggregate_updates():
    # Ensure that the length of clients_state_dicts matches the expected number of clients
    assert len(
        clients_state_dicts) == NUM_CLIENTS, f"Expected {NUM_CLIENTS} client updates, but received {len(clients_state_dicts)}."

    aggregated_state = {}
    for key in model.state_dict().keys():
        tensors_to_aggregate = []
        for client_state in clients_state_dicts:  # This dynamically gets the number of client states
            client_tensor = client_state[key]
            tensors_to_aggregate.append(client_tensor)

        # Calculate the mean of the tensors from all clients
        stacked_tensors = torch.stack(tensors_to_aggregate, dim=0)
        mean_tensor = torch.mean(stacked_tensors, dim=0)

        aggregated_state[key] = mean_tensor

    # Update the global model's weights with the aggregated weights
    model.load_state_dict(aggregated_state)


def evaluate(model, loader):
    correct = 0
    total = 0
    model.eval()  # Put the model in evaluation mode
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(myGPU), labels.to(myGPU)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Show some sample results
            fig, axes = plt.subplots(1, 5, figsize=(12, 3))
            for ax, image, label, pred in zip(axes, images, labels, predicted):
                ax.imshow(image.squeeze().cpu().numpy(), cmap='gray')
                ax.set_title(f"True: {label}, Pred: {pred}")
                ax.axis('off')
            plt.show()
            break  # Only show one batch of results for simplicity

    print(f"Accuracy on test set: {(100 * correct / total):.2f}%")


def main():
    # driver code for server
    for _ in range(epoch):  # 3 epochs

        client_connected = 0

        print(f"Epoch {_ + 1}: Waiting for connection...")
        while True:

            if client_connected < NUM_CLIENTS:

                conn, addr = s.accept()
                connectedClientAddr.append(conn)
                client_thread = threading.Thread(target=handle_client, args=(conn, addr))

                client_thread.start()
                client_connected += 1

            else:
                break

        while True:
            if len(clients_state_dicts) != client_connected:
                pass
            else:
                break

        # Now aggregate updates and evaluate

        aggregate_updates()
        for connection in connectedClientAddr:
            connection.sendall("AGGREGATION_DONE".encode())
            connection.close()

        # Clear the clients_state_dicts and connected clients for the next epoch
        clients_state_dicts.clear()
        connectedClientAddr.clear()
        print("\n")

    s.close()
    evaluate(model, test_loader)


if __name__ == "__main__":
    HOST = '127.0.0.1'
    PORT = 65433
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((HOST, PORT))
    s.listen()

    connected_clients = 0
    clients_state_dicts = []
    connectedClientAddr = []

    # Load the MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=True)

    NUM_CLIENTS = int(input("Enter number of Federated Learning Clients: "))
    epoch = int(input("Enter number of communication round you would like: "))
    myGPU = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(myGPU)

    main()
