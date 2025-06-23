from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST, ImageFolder
import os
from torch.utils.data import DataLoader
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import zipfile


#---Hiperparameters---
ROOT = Path("../data")
BATCH_SIZE = 64
NUM_WORKERS = 4

#configure transforms mnist
def mnist_loaders():
    tf_train = T.Compose([T.RandomRotation(10),
                                   T.ToTensor(),
                                   T.Normalize((0.1307,), (0.3081,))])
    tf_test  = T.Compose([T.ToTensor(),
                                   T.Normalize((0.1307,), (0.3081,))])
    tr = MNIST(
        root=ROOT,
        train=True,
        transform=tf_train,
        download=True
     )
    te = MNIST(
        root=ROOT,
        train=False,
        transform=tf_test,
        download=True
     )
    return (DataLoader(tr, BATCH_SIZE, True,  num_workers=NUM_WORKERS),
            DataLoader(te, BATCH_SIZE*4, False, num_workers=NUM_WORKERS))


#configure transforms cifar10
def cifar10_loaders():
    tf_train = T.Compose([T.Grayscale(),
                                   T.RandomHorizontalFlip(),
                                   T.RandomRotation(10),
                                   T.Resize((28,28)),
                                   T.ToTensor(),
                                   T.Normalize((0.5,), (0.5,))])
    tf_test  = T.Compose([T.Grayscale(),
                                   T.Resize((28,28)),
                                   T.ToTensor(),
                                   T.Normalize((0.5,), (0.5,))])
    tr = CIFAR10(ROOT, True,  True, tf_train)
    te = CIFAR10(ROOT, False, True, tf_test)
    return (DataLoader(tr, BATCH_SIZE, True,  num_workers=NUM_WORKERS),
            DataLoader(te, BATCH_SIZE*4, False, num_workers=NUM_WORKERS))

def extrair_zip(arquivo_zip, pasta_destino):
    with zipfile.ZipFile(arquivo_zip, 'r') as zip_ref:
        zip_ref.extractall(pasta_destino)

    
def medical_loaders():
    arq_zip = '../medical_mnist_dataset.zip'  # caminho do arquivo zip
    if not os.path.exists(ROOT/"medical_mnist"):
        if not os.path.exists(arq_zip):
            raise FileNotFoundError(f"Arquivo {arq_zip} não encontrado. Baixe o arquivo do Kaggle.")
        extrair_zip(arq_zip, ROOT/"medical_mnist")
    
    med_root = ROOT/"medical_mnist"             # baixe via Kaggle antes!
    tf_train = T.Compose([T.Grayscale(),
                                   T.RandomRotation(10),
                                   T.RandomHorizontalFlip(),
                                   T.Resize((64,64)),
                                   T.ToTensor(),
                                   T.Normalize((0.5,), (0.5,))])
    tf_test  = T.Compose([T.Grayscale(),
                                   T.Resize((64,64)),
                                   T.ToTensor(),
                                   T.Normalize((0.5,), (0.5,))])
    full = ImageFolder(med_root, transform=tf_train)
    tr_len, te_len = 47163, len(full)-47163
    tr, te = random_split(full, [tr_len, te_len],
                          generator=torch.Generator().manual_seed(42))
    te.dataset.transform = tf_test
    return (DataLoader(tr, BATCH_SIZE, True,  num_workers=NUM_WORKERS),
            DataLoader(te, BATCH_SIZE*4, False, num_workers=NUM_WORKERS))

# ---------- checagem rápida ----------
print("MNIST  :", len(mnist_train),  len(mnist_test))
print("CIFAR10:", len(cifar_train), len(cifar_test))
print("MedMNIST:", len(med_train_ds), len(med_test_ds))

def quanv_pqc():
    import pennylane as qml
    dev = qml.device("lightning.qubit", wires=4)

    @qml.qnode(dev, interface="torch", diff_method="adjoint")
    def circuit(inputs, weights):            # inputs: (batch, 4)
        # 1) Y-encoding batelado
        qml.AngleEmbedding(
            features=inputs, wires=range(4), rotation="Y")

        qml.RX(weights[0], wires=0)
        qml.RX(weights[1], wires=1)
        # 3) Triângulo de CNOTs (4→3, 3→1, 4→1)
        qml.CNOT(wires=[3, 2])
        qml.CNOT(wires=[2, 0])
        qml.CNOT(wires=[3, 0])

        # 4) Y-gates apenas nos fios 0 e 3
        qml.RY(weights[2], wires=0)
        qml.RY(weights[3], wires=3)

        # expval de Z em cada qubit  →   (batch, 4)
        return [qml.expval(qml.PauliZ(k)) for k in range(4)]

    return qml.qnn.TorchLayer(circuit,
                              weight_shapes={"weights": (4,)})



class QuanvLayer(nn.Module):
    def __init__(self, patch=2):
        super().__init__()
        self.patch = patch
        self.pqc = quanv_pqc()

    def forward(self, x):                     # B×1×H×W
        B, _, H, W = x.shape
        px = (x.unfold(2, self.patch, self.patch)
                .unfold(3, self.patch, self.patch)          # B×1×H'×W'×2×2
                .contiguous()
                .view(-1, self.patch**2))                  # (B·H'·W')×4

        # normalização *por patch*
        px_min, px_max = px.min(dim=1, keepdim=True).values, px.max(dim=1, keepdim=True).values
        px = (px - px_min) / (px_max - px_min + 1e-8) * torch.pi

        z = self.pqc(px)                                   # (B·H'·W')×4
        z = z.view(B, H//self.patch, W//self.patch, 4).permute(0,3,1,2)
        return z.contiguous()                              # B×4×H'×W'


class HQuanvNet10(nn.Module):                       # MNIST & CIFAR-10
    def __init__(self):
        super().__init__()
        self.pre = nn.Sequential(                   # cinza + 28×28
            T.Grayscale(num_output_channels=1),
            T.Resize((28, 28))
        )
        self.quanv = QuanvLayer()
        self.fc = nn.Linear(4*14*14, 10)

    def forward(self, x):
        with torch.no_grad():           # transforms funciona em tensor?
            x = torch.stack([self.pre(img) for img in x])  # minibatch loop
        x = self.quanv(x)
        return self.fc(x.flatten(1))
    
class HQuanvNet6(nn.Module):
    def __init__(self):
        super().__init__()
        self.resize = T.Resize((28, 28))
        self.gray   = T.Grayscale()
        self.quanv  = QuanvLayer()                 # seu bloco corrigido
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(4 * 14 * 14, 6)        # 6 categorias

    def forward(self, x):
        # x B×3×64×64  (Medical MNIST)
        x = self.gray(x)                           # → B×1×64×64
        x = self.resize(x)                         # → B×1×28×28
        x = self.quanv(x)                          # → B×4×14×14
        x = self.flatten(x)                        # → B×784
        return self.fc(x)                          # → B×6


class HQuanvNet6(nn.Module):                        # Medical MNIST
    def __init__(self):
        super().__init__()
        self.pre = nn.Sequential(
            T.Grayscale(1),
            T.Resize((28, 28))
        )
        self.quanv = QuanvLayer()
        self.fc = nn.Linear(4*14*14, 6)

    def forward(self, x):
        with torch.no_grad():
            x = torch.stack([self.pre(img) for img in x])
        x = self.quanv(x)
        return self.fc(x.flatten(1))


MODELS = {"mnist": HQuanvNet10, "cifar10": HQuanvNet10, "medical": HQuanvNet6}
LOADERS = {"mnist": mnist_loaders, "cifar10": cifar10_loaders, "medical": medical_loaders}


def train_one(dataset_name, epochs=10, lr=1e-3, device="cuda"):
    device = device if torch.cuda.is_available() else "cpu"
    train_dl, test_dl = LOADERS[dataset_name]()
    model = MODELS[dataset_name]().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for ep in range(epochs):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        print(f"[{dataset_name}] época {ep+1}/{epochs} concluída")

    # avaliação
    model.eval(); correct = total = 0
    with torch.no_grad():
        for xb, yb in test_dl:
            pred = model(xb.to(device)).argmax(1).cpu()
            correct += (pred == yb).sum().item(); total += yb.size(0)
    print(f"[{dataset_name}] Acurácia teste: {correct/total:.4f}")



if __name__ == "__main__":
    for ds in ("mnist", "cifar10", "medical"):
        train_one(ds, epochs=10)