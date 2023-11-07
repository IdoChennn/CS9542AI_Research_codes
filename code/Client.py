# Client Code
import socket
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# Define the CNN model
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


def receive_data(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf:
            raise ConnectionError("Incomplete data received")
        buf += newbuf
        count -= len(newbuf)
    return buf


def downloading_global_model(epoch):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, PORT))
    print(f"Connection established for epoch {epoch + 1}")

    # Receive the global model from the server
    data_size = int.from_bytes(s.recv(4), byteorder='big')
    data = receive_data(s, data_size)
    return data, s


def local_model_training(model):

    num_samples = 1000
    indices = torch.randperm(len(dataset))[:num_samples]
    subset_dataset = Subset(dataset, indices)

    train_loader = DataLoader(subset_dataset, batch_size=64, shuffle=True)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(NUM_EPOCHS):

        global_model, conn = downloading_global_model(epoch)
        model.load_state_dict(pickle.loads(global_model))
        model.to(myGPU)

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(myGPU), target.to(myGPU)  # Move data and target to GPU
            optimizer.zero_grad()
            output = model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            loss.backward()
            optimizer.step()

        updates = pickle.dumps(model.state_dict())
        sending_local_updates(updates, conn, epoch)

        accuracy = test_model_accuracy(model, myGPU)
        print(f'Accuracy of the local model on the test dataset: {accuracy:.2f}%')
        print("\n")


def sending_local_updates(updates, conn, epoch):
    conn.sendall(len(updates).to_bytes(4, byteorder='big'))
    conn.sendall(updates)
    print("Updates status: sent")

    while True:
        confirmation = conn.recv(16).decode()
        if confirmation == "AGGREGATION_DONE":
            print(f"Server has finished aggregation for epoch {epoch + 1}")
            break
        else:
            continue

    conn.close()


def test_model_accuracy(model, device):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        transform=transforms.ToTensor(),
        download=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=64,
        shuffle=True
    )
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    return accuracy


if __name__ == "__main__":

    myGPU = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    local_model = CNN().to(myGPU)
    # Load MNIST Training data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)

    # Use only a subset (e.g., 1000 samples)

    # num_samples = 10000
    # indices = torch.randperm(len(dataset))[:num_samples]
    # subset_dataset = Subset(dataset, indices)

    HOST = '127.0.0.1'
    PORT = 65433
    NUM_EPOCHS = 3

    local_model_training(local_model)
