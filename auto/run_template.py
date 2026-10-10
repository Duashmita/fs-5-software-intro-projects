import os
import matplotlib.pyplot as plt
from pid_template import make_car
from pid_template import update
from pid_template import calculate_desired_acceleration
from pid_template import acceleration_to_throttle_percentage

K_P = 0.2
K_I = 0.1
K_D = 0.1

STEPS = 550

GRAPH_FOLDER = "Integral-graphs" # use "PID-graphs" for the step 7 runs

#WRITE CODE HERE

#folder for saved graphs
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "project-outputs", GRAPH_FOLDER)
os.makedirs(OUTPUT_DIR, exist_ok=True)

GAINS = f"KP{K_P}_KI{K_I}_KD{K_D}"

def run(desired_v):
    car = make_car(desired_v=desired_v, dt=0.1)

    #data storage for plotting
    velocities = []
    errors = []
    times = []

    #simulation loop
    for step in range(STEPS):
        desired_acceleration, error = calculate_desired_acceleration(car, K_P, K_I, K_D)
        throttle_percentage = acceleration_to_throttle_percentage(desired_acceleration)
        velocities.append(car["v"])
        errors.append(error)
        times.append(car["t"])
        update(car, throttle_percentage)

    return velocities, errors, times

def finish_plot(figure_num, ylabel):
    plt.figure(figure_num)
    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} over Time")
    plt.legend()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{ylabel.lower()}_{GAINS}.png")) #save before show, show clears the figure

#single run
velocities, errors, times = run(20.0)
plt.figure(1)
plt.plot(times, velocities, label="Velocity")
plt.figure(2)
plt.plot(times, errors, label="Error")

# #step 7: run the same gains at several desired velocities and compare them on shared plots
# #to use: comment out the single run above and set GRAPH_FOLDER to "PID-graphs"
# for desired_v in [10.0, 20.0, 30.0]:
#     velocities, errors, times = run(desired_v)
#     overshoot = (max(velocities) - desired_v) / desired_v * 100
#     print(f"desired_v={desired_v}: final v={velocities[-1]:.3f}, overshoot={overshoot:.1f}%")
#     plt.figure(1)
#     plt.plot(times, velocities, label=f"v_des={desired_v}")
#     plt.figure(2)
#     plt.plot(times, errors, label=f"v_des={desired_v}")

finish_plot(1, "Velocity")
finish_plot(2, "Error")
plt.show()
