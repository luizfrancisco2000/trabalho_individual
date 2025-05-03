import os, random, math, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import gymnasium as gym
from gymnasium import spaces
import pennylane as qml
from pennylane import numpy as pnp
from stable_baselines3 import PPO
from medmnist import ChestMNIST

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED   = 42
torch.manual_seed(SEED)  
np.random.seed(SEED)
random.seed(SEED)

BATCH_SZ = 64
IMG_SIZE  = 32
NUM_CLASSES = 3


transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

train = ChestMNIST(split="train", download=True, as_rgb=True, transform=transform)
test  = ChestMNIST(split="test", download=True, as_rgb=True, transform=transform)

#Apenas 3 classes classificadas nesse momento
train_idx = [i for i,(x,y) in enumerate(train) if y[0] < NUM_CLASSES]
test_idx  = [i for i,(x,y) in enumerate(test)  if y[0] < NUM_CLASSES]

train_loader = DataLoader(Subset(train, train_idx), batch_size=BATCH_SZ, shuffle=True)
test_loader  = DataLoader(Subset(test,  test_idx),  batch_size=BATCH_SZ, shuffle=False)

# ---------- 3. Hiper‑parâmetros de RL / PQC ----------
N_QUBITS   = 4
MAX_DEPTH  = 8          # nº máximo de gates que o agente pode adicionar
ACTION_SET = (['rx','ry','rz'] + ['cnot'])   # gates disponíveis
N_ACTIONS  = len(ACTION_SET) * N_QUBITS

dev = qml.device("default.qubit", wires=N_QUBITS)

import torch
import torch.nn as nn
import pennylane as qml

class Quantumnet(nn.Module):
    def __init__(self, circuit, n_qubits=4, input_dim=3072, n_classes=3):
        super(Quantumnet, self).__init__()
        self.n_qubits = n_qubits
        self.circuit = circuit
        self.q_params = nn.Parameter(0.01 * torch.randn(len(circuit)))
        self.input_dim = input_dim

        self.pre_net = nn.Linear(input_dim, n_qubits)
        self.post_net = nn.Linear(n_qubits, n_classes)

        # Define QNode
        self.dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self.dev, interface="torch")
        def qnode(inputs, weights):
            for i in range(n_qubits):
                qml.RY(inputs[i], wires=i)
            for i, (gate, wire) in enumerate(circuit):
                angle = weights[i]
                if gate == 'rx':
                    qml.RX(angle, wires=wire)
                elif gate == 'ry':
                    qml.RY(angle, wires=wire)
                elif gate == 'rz':
                    qml.RZ(angle, wires=wire)
                elif gate == 'cnot':
                    qml.CNOT(wires=[wire, (wire + 1) % n_qubits])
            return qml.math.stack([qml.expval(qml.PauliZ(i)) for i in range(n_qubits)])

        self.qnode = qnode
    def forward(self, x):
        # x shape: [batch_size, input_dim]
        x = self.pre_net(x)                        # [batch, n_qubits]
        x = torch.tanh(x) * (torch.pi / 3.0)

        q_out = []
        for xi in x:
            q_result = self.qnode(xi, self.q_params)
            q_out.append(torch.tensor(q_result, dtype=torch.float32, device=DEVICE))  # <- converte aqui

        q_out = torch.stack(q_out)
        return self.post_net(q_out)

class CircuitBuilderEnv(gym.Env):
    def __init__(self):
        super(CircuitBuilderEnv, self).__init__()
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(MAX_DEPTH * 2,), dtype=np.float32)
        self.train_iter = iter(train_loader)
        #elf.reset()
    
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
        
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        self.circuit = []
        self.steps = 0
        self.state = np.zeros(MAX_DEPTH * 2, dtype=np.float32)
        return self.state,{}



    def step(self, action):
        gate_idx = action // N_QUBITS
        wire = action % N_QUBITS
        gate = ACTION_SET[gate_idx]

        if self.steps < MAX_DEPTH:
            self.circuit.append((gate, wire))
            self.steps += 1
            self._update_state(gate_idx, wire)

        reward = self._evaluate_circuit()
        terminated = self.steps == MAX_DEPTH
        truncated = False  # Você pode definir lógica de truncamento se quiser
        info = {}

        return self.state, reward, terminated, truncated, info

    def _update_state(self, gate_idx, wire):
        self.state[2 * (self.steps - 1)]     = gate_idx / (len(ACTION_SET) - 1)
        self.state[2 * (self.steps - 1) + 1] = wire / (N_QUBITS - 1)

    def _evaluate_circuit(self):
        if len(self.circuit) == 0:
            return 0.0  # ou um pequeno valor de recompensa neutra

        try:
            x, y = next(self.train_iter)
        except StopIteration:
            self.train_iter = iter(train_loader)
            x, y = next(self.train_iter)

        x = x.to(DEVICE)
        y = y.argmax(dim=1)
        x = x.view(x.size(0), -1)

        model = Quantumnet(self.circuit, n_qubits=N_QUBITS).to(DEVICE)
        model.eval()
        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=1)
            acc = (pred == y).float().mean().item()
        return acc


#-- Train RL agent --
def train_agent():
    env = CircuitBuilderEnv()

    model_rl = PPO("MlpPolicy", env, 
                    learning_rate=3e-4,n_steps=512,
                    batch_size=64,gamma=0.95,
                    verbose=1,seed=SEED)
    
    print("Training RL agent...")
    model_rl.learn(total_timesteps=10000)
    model_rl.save("rl_agent")
    print("RL agent trained!")
    return model_rl

#-- Test RL agent --
def train_final_agent():
    best_env = train_agent()
    best_circuit = best_env.circuit

    final_model = best_env._build_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(final_model.parameters(), lr=1e-3)

    EPOCHS_SUP = 3 #upgrade to production
    for epoch in range(EPOCHS_SUP):
        final_model.train()
        for image, labels in train_loader:
            image = image.to(DEVICE)
            labels = labels.squeeze().long().to(DEVICE)

            optimizer.zero_grad()
            logits = final_model(image)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        #--Sample evaluation--
        final_model.eval()
        correct = total = 0
        with torch.no_grad():
            for image, labels in test_loader:
                image = image.to(DEVICE)
                labels = labels.squeeze().long().to(DEVICE)
                logits = final_model(image)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = correct / total
        print(f"Epoch {epoch+1}/{EPOCHS_SUP}, Test Accuracy: {acc:.4f}")
    print("Final model trained!")


if __name__ == "__main__":
    #-- Train RL agent --
    train_agent()

    #-- Train final model --
    train_final_agent()
