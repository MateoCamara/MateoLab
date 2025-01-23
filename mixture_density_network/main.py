import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


# Crear un dataset sintético multimodal
def create_dataset(num_samples=1000):
    x = np.random.uniform(-10, 10, size=num_samples)
    y = []
    for i in x:
        if np.random.rand() > 0.5:
            y.append(np.sin(i) + np.random.normal(0, 0.1))
        else:
            y.append(np.cos(i) + np.random.normal(0, 0.1))
    return x, np.array(y)


# Dataset
x_train, y_train = create_dataset()
x_train = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)


# Modelo Mixture Density Network
class MDN(nn.Module):
    def __init__(self, input_dim, output_dim, num_mixtures):
        super(MDN, self).__init__()
        self.num_mixtures = num_mixtures

        # Capa densa para predecir los parámetros de la mezcla
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_mixtures * 3 * output_dim)  # Parámetros: pi, mu, sigma
        )

    def forward(self, x):
        params = self.fc(x)
        pi, mu, sigma = torch.chunk(params, 3, dim=-1)  # Dividir los parámetros

        # Activaciones específicas
        pi = nn.Softmax(dim=-1)(pi)  # Probabilidades de mezcla
        sigma = torch.exp(sigma)  # Las desviaciones estándar deben ser positivas
        return pi, mu, sigma


# Pérdida de log-verosimilitud negativa
def mdn_loss(pi, mu, sigma, y):
    # Expandir dimensiones para permitir operaciones entre mezclas y datos
    y = y.expand(-1, mu.size(-1))

    # Calcular la densidad gaussiana
    normal_dist = torch.exp(-0.5 * ((y - mu) / sigma) ** 2) / (sigma * torch.sqrt(torch.tensor(2 * np.pi)))

    # Ponderar las gaussianas por las probabilidades pi
    weighted_gaussians = pi * normal_dist  # Producto sobre las dimensiones de salida

    # Sumar sobre las componentes de mezcla y calcular el log
    loss = -torch.log(weighted_gaussians.sum(dim=-1) + 1e-6)

    return loss.mean()


# Entrenamiento
input_dim = 1
output_dim = 1
num_mixtures = 5

model = MDN(input_dim, output_dim, num_mixtures)
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 500

losses = []
for epoch in range(epochs):
    optimizer.zero_grad()
    pi, mu, sigma = model(x_train)
    loss = mdn_loss(pi, mu, sigma, y_train)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

# Visualización de resultados
x_test = torch.linspace(-10, 10, 1000).unsqueeze(1)
pi, mu, sigma = model(x_test.detach())

# Muestrear de la mezcla para visualizar
samples = []
for i in range(1000):
    component = np.random.choice(num_mixtures, p=pi[i].detach().numpy())
    sample = np.random.normal(mu[i, component].item(), sigma[i, component].item())
    samples.append(sample)

plt.figure(figsize=(10, 6))
plt.scatter(x_train.numpy(), y_train.numpy(), s=5, alpha=0.5, label="Datos de entrenamiento")
plt.scatter(x_test.numpy(), samples, s=5, alpha=0.5, label="Muestras de MDN")
plt.legend()
plt.title("Mixture Density Network")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
