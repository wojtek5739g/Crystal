import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import re
import random
import math
import argparse
import time

steps = 10000
tau = 0.001
time_total = steps*tau

def generate_positions():
    positions = np.zeros((N, dim))

    for m in range(0, n):
            for k in range (0, n):
                for l in range (0, n):
                    i = m+k*n+l*n**2
                    positions[i] = (m - (n-1)/2)*b[0] + (k-(n-1)/2)*b[1] + (l-(n-1)/2)*b[2]

    return positions
    
def generate_momentums():
    energies_random = np.zeros((N, dim))
    for i in range(0, N):
        energies_random[i] = np.array([(-1/2)*kB*T*(np.log((random.uniform(0, 1)+1)/2)), (-1/2)*kB*T*(np.log((random.uniform(0, 1)+1)/2)), (-1/2)*kB*T*(np.log((random.uniform(0, 1)+1)/2))])

    momentums = np.zeros((N, dim))
    for i in range(0, N):
        momentums[i] = np.array([(random.randint(0, 1)*2-1)*np.sqrt(2*m*energies_random[i, 0]), (random.randint(0, 1)*2-1)*np.sqrt(2*m*energies_random[i, 1]), (random.randint(0, 1)*2-1)*np.sqrt(2*m*energies_random[i, 2])])

    P = np.zeros(dim)
    for i in range(N):
        P+=momentums[i]

    for i in range(N):
        momentums[i] = momentums[i]-(1/N)*P

    return momentums
    
def write_initial_positions(positions):
    file = open('positions.xyz', 'w')

    file.write('{}\n'.format(N))
    file.write('Positions\n')
    for i in range(0, N):
        file.write('Ar '+ '{} {} {}\n'.format(positions[i, 0], positions[i, 1], positions[i, 2]))
        
    file.close()

@jit(nopython=True)
def calculate_total_potential(positions):
    potentialsP = np.zeros((N))

    for i in range(N):
        for j in range(i):
            if (i!=j):
                dist = np.linalg.norm(positions[i]-positions[j])
                potentialsP[i] += epsilon*((R/dist)**12-2*(R/dist)**6)

    potentialsS = np.zeros((N))
    for i in range(0, N):
        norm = np.linalg.norm(positions[i])
        if not norm<L:
            potentialsS[i] = 1/2*f*(norm-L)**2

    return (np.sum(potentialsP)+np.sum(potentialsS))

@jit(nopython=True)
def calculate_kinetic_energy(momentums):
    kin_energy = 0
    for i in range(0, N):
        kin_energy+=(np.linalg.norm(momentums[i])**2)/(2*m)

    return kin_energy

@jit(nopython=True)
def calculate_wall_force(position, nor):
    return f*(L-nor)*position/nor

@jit(nopython=True)
def calculate_forces(positions):
    forces = np.zeros((N, dim))

    for i in range(N):
        nor = np.linalg.norm(positions[i])
        for j in range(N):
            if (i!=j):
                dist = np.linalg.norm(positions[i]-positions[j])
                forces[i]+= 12*epsilon*((R/dist)**12-(R/dist)**6)*((positions[i]-positions[j])/(dist*dist))
        if (nor>=L):
            forces[i]+=calculate_wall_force(positions[i], nor)

    return forces

@jit(nopython=True)
def calculate_pressure(positions):
    wall_force_sum = 0
    for i in range(N):
        nor = np.linalg.norm(positions[i])
        if (nor>=L):
            wall_force_sum+=np.linalg.norm(calculate_wall_force(positions[i], nor))

    pressure = 1/(4*np.pi*L**2)*wall_force_sum
    return pressure

@jit(nopython=True)
def calculate_temperature(E_kin_total):
    return 2/(3*N*kB)*E_kin_total


def calculate_motion(V_total, E_kins, positions_motion, momentums_motion, forces, pressures, temperatures):
    '''
    Main loop performing time evaluation of the system
    '''
    for s in range (1, steps+1):
        sup = momentums_motion[s-1] + 1/2*forces*tau
        positions_motion[s] = positions_motion[s-1]+(1/m)*sup*tau
        forces = calculate_forces(positions_motion[s])
        momentums_motion[s] = sup + 1/2*forces*tau
        V_total[s] = calculate_total_potential(positions_motion[s])
        E_kins[s] = calculate_kinetic_energy(momentums_motion[s])
        pressures[s] = calculate_pressure(positions_motion[s])
        temperatures[s] = calculate_temperature(E_kins[s])
        
    return positions_motion, momentums_motion

def use_regex(input_text):
    pattern = re.compile(r"[A-Za-z]+ = ([+-]?(?=\.\d|\d)(?:\d+)?(?:\.?\d*))(?:[Ee]([+-]?\d+))?", re.IGNORECASE)
    return pattern.match(input_text)

def import_parameters(filename : str):
    with open(filename, 'r') as file:
        global N, dim, epsilon, f, R, a, m, T, kB, n, L
        N = int(use_regex(file.readline()).group(1))
        dim = int(use_regex(file.readline()).group(1))
        epsilon = float(use_regex(file.readline()).group(1))
        reg = use_regex(file.readline())
        f = float(reg.group(1)+'e'+reg.group(2))
        R = float(use_regex(file.readline()).group(1))
        a = float(use_regex(file.readline()).group(1))
        m = float(use_regex(file.readline()).group(1))
        T = float(use_regex(file.readline()).group(1))
        reg = use_regex(file.readline())
        kB = float(reg.group(1)+'e'+reg.group(2))
        n = round(N**(1/3))
        L = 1.22*a*(n-1) + 0.1

def save_trajectory(positions_motion, num):
    file = open(f'trajectory{num}.xyz', 'w')

    for s in range(steps):
        if (s%100 == 0):
            file.write('{}\n'.format(N))
            file.write('Positions\n')
            for i in range(0, N):
                file.write('Ar '+ '{} {} {}\n'.format(positions_motion[s, i, 0], positions_motion[s, i, 1], positions_motion[s, i, 2]))
    
    file.close()

def save_energies(V_total, E_kins, num):
    with open(f'erergies{num}.txt', 'w') as file:
        file.write('Timestep    Total   Potential   Kinetic\n')
        for i in range(0, steps+1):
            file.write('{}: {} {} {}\n'.format(i, V_total[i]+E_kins[i], V_total[i], E_kins[i]))

def show_momentums_histogram(momentums):
    plt.hist(momentums[:, 2], 30, color='green')

    # plt.show()

def plot_energies(V_total, E_kins, num):
    plt.plot(V_total+E_kins, label='Total energy')
    plt.plot(V_total, label='Potential energy')
    plt.plot(E_kins, label='Kinetic energy')
    plt.title('Energies')
    plt.xlabel('timestep')
    plt.ylabel('E [kJ]', rotation=0, labelpad=20)
    plt.legend()

    plt.savefig(f'energies{num}.png')
    plt.clf()

def plot_pressures(pressures, num):
    plt.plot(pressures)
    plt.xlabel('timestep')
    plt.title('Pressure')
    plt.ylabel('p [Pa]', rotation=0, labelpad=20)

    plt.savefig(f'pressures{num}.png')
    plt.clf()

def plot_temperature(temperatures, num):
    plt.plot(temperatures)
    plt.xlabel('timestep')
    plt.title('Temperature')
    plt.ylabel('T [K]', rotation=0, labelpad=20)

    plt.savefig(f"temperatures{num}.png")
    plt.clf()

def main():
    parser = argparse.ArgumentParser(
                    prog='crystal',
                    description='Program that performs simple molecular dynamics task regarding phase transition from a solid crystal to gas')
    
    parser.add_argument('filename')
    args = parser.parse_args()
    filename = str(args.filename)

    import_parameters(filename)
    print(N)

    '''
    Initializing simulation
    '''
    global b
    b = np.array([[a, 0, 0], [a/2, a*np.sqrt(3)/2, 0], 
            [a/2, a*np.sqrt(3)/6, a*np.sqrt(2/3)]])

    init_positions = generate_positions()
    init_momentums = generate_momentums()

    print(init_positions)
    print(init_momentums)

    init_V = calculate_total_potential(init_positions)
    print(init_V)

    show_momentums_histogram(init_momentums)
    init_forces = calculate_forces(init_positions)

    print(init_forces)
    init_pressure = calculate_pressure(init_positions)
    print(init_pressure)

    '''
    Defining arrays for simulation'
    '''
    pressures = np.zeros(1+steps)
    pressures[0] = init_pressure

    temperatures = np.zeros(1+steps)
    temperatures[0] = T

    kinetic_energies = np.zeros((1+steps))
    kinetic_energies[0] = calculate_kinetic_energy(init_momentums)

    print(kinetic_energies)

    total_potential_energies = np.zeros(1+steps)
    total_potential_energies[0] = init_V
    print(total_potential_energies)

    positions_motion = np.zeros((1+steps, N, dim))
    momentums_motion = np.zeros((1+steps, N, dim))
    positions_motion[0] = init_positions
    momentums_motion[0] = init_momentums

    '''
    Running the simulation
    '''

    start_time = time.time()
    positions_motion, momentums_motion = calculate_motion(total_potential_energies, kinetic_energies, positions_motion, momentums_motion, init_forces, pressures, temperatures)
    end_time = time.time()
    with open('times.txt', 'a') as file:
        file.write(f'{end_time-start_time}\n')

    num = int(re.findall(r'\d+', filename)[0])
    save_trajectory(positions_motion, num)
    save_energies(total_potential_energies, kinetic_energies, num)
    plot_energies(total_potential_energies, kinetic_energies, num)
    plot_pressures(pressures, num)
    plot_temperature(temperatures, num)

if __name__ == "__main__":
    main()