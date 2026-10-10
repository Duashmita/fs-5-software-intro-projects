import os
import time
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from pid_template import make_car
from pid_template import update
from pid_template import calculate_desired_acceleration
from pid_template import acceleration_to_throttle_percentage

K_P = 0.2
K_I = 0.1
K_D = 0.1

DATA_EPISODES = 20 # each episode starts a fresh car, like the comparison runs
EPISODE_STEPS = 550
CHANGE_EVERY = 200 # steps between desired velocity changes in the training data
MAX_V = 40.0 # highest desired velocity in the training data

EPOCHS = 1000
LEARNING_RATE = 0.01

COMPARE_STEPS = 550
DESIRED_VS = [10.0, 20.0, 30.0]

DEVICE = "mps" # Apple GPU
assert torch.backends.mps.is_available(), "MPS not available, refusing to silently train on CPU"

torch.manual_seed(0)
np.random.seed(0)

#folder for saved graphs
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "project-outputs", "MLP-graphs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def pid_throttle(car, k_i=0.0, k_d=0.0):
    desired_acceleration, _ = calculate_desired_acceleration(car, K_P, k_i, k_d)
    return acceleration_to_throttle_percentage(desired_acceleration)

def features(car):
    return [car["v"], car["desired_v"], car["net_integral"]]

def mlp_throttle(feature_rows):
    #feature_rows: list of [v, desired_v, net_integral], all go through the model together as one batch
    with torch.inference_mode():
        return controller(torch.tensor(feature_rows, dtype=torch.float32).to(DEVICE)).cpu().squeeze(1).numpy()

def mlp_drive_throttles(cars):
    for car in cars:
        car["net_integral"] += (car["desired_v"] - car["v"]) * car["dt"] # same integral the PID keeps
    return mlp_throttle([features(car) for car in cars])

def generate_data():
    #runs the PID on fresh cars, with a desired velocity that changes every CHANGE_EVERY steps
    inputs = []
    throttles = []
    for episode in range(DATA_EPISODES):
        car = make_car(dt=0.1)
        for step in range(EPISODE_STEPS):
            if step % CHANGE_EVERY == 0:
                car["desired_v"] = np.random.uniform(0, MAX_V)
            throttle_percentage = pid_throttle(car, K_I, K_D) # updates net_integral first
            inputs.append(features(car))
            throttles.append(throttle_percentage)
            update(car, throttle_percentage)

    X = torch.tensor(inputs, dtype=torch.float32)
    y = torch.tensor(throttles, dtype=torch.float32).unsqueeze(1)
    return X, y

def drive(controller):
    #drives one car per desired velocity, returns velocities and times for each
    cars = [make_car(desired_v=desired_v, dt=0.1) for desired_v in DESIRED_VS]
    velocities = [[] for car in cars]
    times = []
    for step in range(COMPARE_STEPS):
        throttles = controller(cars)
        times.append(cars[0]["t"])
        for i, car in enumerate(cars):
            velocities[i].append(car["v"])
            update(car, throttles[i])
    return velocities, times

def finish_plot(figure_num, xlabel, ylabel, title, filename):
    plt.figure(figure_num)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, filename))

#data: shuffle then split 80% train, 20% test
X, y = generate_data()
order = torch.randperm(len(X))
split = int(0.8 * len(X))

#standardize each input with the training data's mean and std
X_mean = X[order[:split]].mean(0)
X_std = X[order[:split]].std(0)
X = (X - X_mean) / X_std

X_train, y_train = X[order[:split]].to(DEVICE), y[order[:split]].to(DEVICE)
X_test, y_test = X[order[split:]].to(DEVICE), y[order[split:]].to(DEVICE)

#model: [v, desired_v, net_integral] (standardized) -> throttle percentage
model = nn.Sequential(
    nn.Linear(3, 16),
    nn.ReLU(),
    nn.Linear(16, 16),
    nn.ReLU(),
    nn.Linear(16, 1),
).to(DEVICE)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

#training loop
train_losses = []
test_losses = []
start = time.time()
for epoch in range(EPOCHS):
    model.train()
    loss = loss_fn(model(X_train), y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.inference_mode():
        test_loss = loss_fn(model(X_test), y_test)
    train_losses.append(loss.item())
    test_losses.append(test_loss.item())

    if epoch % 100 == 0 or epoch == EPOCHS - 1:
        print(f"epoch {epoch}: train loss {loss.item():.6f}, test loss {test_loss.item():.6f}")

print(f"device {DEVICE}, {len(X_train)} training rows, trained in {time.time() - start:.1f}s")

#controller: raw [v, desired_v, net_integral] in, throttle percentage (-1 to 1) out
class Controller(nn.Module):
    def __init__(self, model, mean, std):
        super().__init__()
        self.model = model
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, feature_rows):
        return torch.clamp(self.model((feature_rows - self.mean) / self.std), -1, 1) # same -100% to 100% range as the PID

controller = Controller(model, X_mean, X_std).to(DEVICE).eval()

#save it ready to use
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(MODEL_DIR, exist_ok=True)
torch.jit.trace(controller.cpu(), torch.tensor([[0.0, 20.0, 0.0]])).save(os.path.join(MODEL_DIR, "mlp_controller.pt"))
controller.to(DEVICE)

#comparison: PID and MLP each drive the car at every desired velocity
results = {
    "PID": drive(lambda cars: [pid_throttle(car, K_I, K_D) for car in cars]),
    "MLP": drive(mlp_drive_throttles),
}

#summary table
print(f"\n{'controller':<11}{'desired_v':>10}{'final v':>10}{'final error':>13}{'overshoot':>11}")
for name, (velocities, times) in results.items():
    for desired_v, v in zip(DESIRED_VS, velocities):
        overshoot = (max(v) - desired_v) / desired_v * 100
        print(f"{name:<11}{desired_v:>10.1f}{v[-1]:>10.3f}{desired_v - v[-1]:>13.3f}{overshoot:>10.1f}%")

# Figure 1: loss with epochs
plt.figure(1)
plt.plot(train_losses, label="Train loss")
plt.plot(test_losses, label="Test loss")
plt.yscale("log")
finish_plot(1, "Epoch", "MSE loss", "Loss over Epochs", "loss.png")

# Figure 2: true vs predicted throttle for a sweep of velocities at desired_v = 20, net_integral = 0 (D term left out, it needs the previous error)
sweep_v = np.linspace(0, MAX_V, 200)
plt.figure(2)
plt.plot(sweep_v, [acceleration_to_throttle_percentage(K_P * (20.0 - v)) for v in sweep_v], label="PID throttle")
plt.plot(sweep_v, mlp_throttle([[v, 20.0, 0.0] for v in sweep_v]), "--", label="MLP prediction")
finish_plot(2, "Velocity", "Throttle percentage", "Throttle vs Velocity (desired_v = 20, net_integral = 0)", "throttle_prediction.png")

# Figure 3: velocity with time, thick faded PID with thin dashed MLP on top (they overlap closely)
plt.figure(3)
for i, desired_v in enumerate(DESIRED_VS):
    pid_line, = plt.plot(results["PID"][1], results["PID"][0][i], linewidth=5, alpha=0.35, label=f"PID v_des={desired_v}")
    plt.plot(results["MLP"][1], results["MLP"][0][i], "--", linewidth=1.5, color=pid_line.get_color(), label=f"MLP v_des={desired_v}")
finish_plot(3, "Time", "Velocity", "PID vs MLP Controller", "pid_vs_mlp_velocity.png")

# Figure 4: difference between the two controllers' velocities
plt.figure(4)
for i, desired_v in enumerate(DESIRED_VS):
    difference = np.array(results["MLP"][0][i]) - np.array(results["PID"][0][i])
    plt.plot(results["PID"][1], difference, label=f"v_des={desired_v}")
finish_plot(4, "Time", "MLP velocity - PID velocity", "MLP vs PID Velocity Difference", "pid_vs_mlp_difference.png")

plt.show()
