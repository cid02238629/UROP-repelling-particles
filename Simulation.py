import numpy as np
import matplotlib.pyplot as plt

# Define simulation box size
Box_Length = 30

# Length of simulation
Time_Step = 1 / 50000
Last_Frame = 22000

# Initial positions for 2 particles (manually set)
xPosition = np.array([4.5, 6.0])  # X-coordinates of particles
yPosition = np.array([5.0, 5.0])  # Y-coordinates of particles

# Initial speeds
xSpeed = np.array([0.0, 0.0])  # X-components of velocity
ySpeed = np.array([5000.0, 5000.0])  # Y-components of velocity

# Simulation Parameters and Constants
n = 2
Mass = 4e-27
Radius = 0.2
Bond_Length = 4.5 * Radius
Bond_Length_inverse = 1 / Bond_Length
Bond_Length_Sixth_Power = Bond_Length**6
Bond_Strength = 400e-24  # epsilon
Lennard_Jones_Constant = 12 * (Bond_Length**6) * Bond_Strength
Radius_Of_Effect = 2 * Bond_Length
Radius_Of_Effect_Squared = Radius_Of_Effect**2


def Wall_Repulsion_Force_Function(Particle_spacing):
    Beta = 5.99
    return (Bond_Strength / (1 - 6 / Beta)) * (
        6
        * Bond_Length_inverse
        * np.exp(Beta - Beta * Particle_spacing * Bond_Length_inverse)
    )


# Initialize total kinetic energy array
Total_Kinetic_Energy = np.zeros(Last_Frame)
Boltzmann_Constant = 1.3810e-23

# Pre-allocate force arrays
xForce = np.zeros(n)
yForce = np.zeros(n)

# Set up the plot
plt.ion()
fig, ax = plt.subplots()
scat = ax.scatter([], [], s=100)
ax.set_xlim(0, Box_Length)
ax.set_ylim(0, Box_Length)
title_text = ax.set_title("")

# Animation loop
for t in range(Last_Frame):
    # Half-step position update for Velocity Verlet scheme
    xPosition += 0.5 * xSpeed * Time_Step
    yPosition += 0.5 * ySpeed * Time_Step

    # Reset forces
    xForce[:] = 0
    yForce[:] = 0

    # Compute interaction between the two particles
    dx = xPosition[1] - xPosition[0]
    dy = yPosition[1] - yPosition[0]
    Distance_squared = dx**2 + dy**2
    Distance_Sixth_Power = Distance_squared**3

    if Distance_squared > 0:
        force_magnitude = (
            Lennard_Jones_Constant
            * (-Bond_Length_Sixth_Power)
            / (Distance_squared * Distance_Sixth_Power**2)
        )

        # Compute force components
        fx = force_magnitude * dx
        fy = force_magnitude * dy

        # Update forces on both particles (Newton's third law)
        xForce[0] += fx
        yForce[0] += fy
        xForce[1] -= fx
        yForce[1] -= fy

    # Wall boundary interactions
    xForce += Wall_Repulsion_Force_Function(
        2 * (Box_Length - xPosition)
    ) - Wall_Repulsion_Force_Function(2 * xPosition)
    yForce += Wall_Repulsion_Force_Function(
        2 * (Box_Length - yPosition)
    ) - Wall_Repulsion_Force_Function(2 * yPosition)

    # Full-step velocity update
    xSpeed += (xForce / Mass) * Time_Step
    ySpeed += (yForce / Mass) * Time_Step

    # Complete position update
    xPosition += 0.5 * xSpeed * Time_Step
    yPosition += 0.5 * ySpeed * Time_Step

    # Calculate and store total kinetic energy
    Speed_Absolut = np.sqrt(xSpeed**2 + ySpeed**2)
    Current_Kinetic_Energy = np.sum(0.5 * Mass * Speed_Absolut**2)
    Total_Kinetic_Energy[t] = Current_Kinetic_Energy

    # Plot particle positions (every 20 frames)
    if t % 20 == 0:
        scat.set_offsets(np.c_[xPosition, yPosition])
        title_text.set_text(f"Frame: {t} | Total KE: {Current_Kinetic_Energy:.2e} J")
        plt.pause(0.00001)

plt.ioff()
plt.show()
