# CENG482 Project - Evolutionary Computation
# Project Source Code - Knapsack Algorithm for Efficient Resource Allocation
# Halil Can Uyanık & Yaşar Mehmet Bağdatlı

import random
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Task Class
class Task:
    def __init__(self, task_id, compute_demand, priority):
        self.task_id = task_id                # Unique identifier for the task
        self.compute_demand = compute_demand  # Computational requirement
        self.priority = priority              # Priority level of the task

# Server Class
class Server:
    def __init__(self, server_id, compute_capacity, server_type, cost_per_hour, latency):
        self.server_id = server_id                # Unique identifier for the server
        self.compute_capacity = compute_capacity  # Maximum computational capacity
        self.server_type = server_type            # 'Edge' or 'Cloud'
        self.cost_per_hour = cost_per_hour        # Operational cost per hour
        self.latency = latency                    # Latency of the server

# Generate tasks
def generate_tasks(num_tasks):
    tasks = []
    for i in range(1, num_tasks + 1):
        task_id = f"T{i}"
        priority = random.randint(0, 2)
        compute_demand = {
            0: random.randint(1, 4),  # Low priority
            1: random.randint(3, 7),  # Medium priority
            2: random.randint(6, 10)  # High priority
        }[priority]
        tasks.append(Task(task_id, compute_demand, priority))
    return tasks

# Generate servers
def generate_servers(num_servers):
    servers = []
    for i in range(1, num_servers + 1):
        server_id = f"S{i}"
        if random.random() > 0.5:
            server_type = "Edge"
            compute_capacity = random.randint(8, 12)
            cost_per_hour = round(random.uniform(0.2, 0.5), 3)  # Edge servers are costlier
            latency = round(random.randint(1, 3), 3)           # Low latency
        else:
            server_type = "Cloud"
            compute_capacity = random.randint(16, 32)
            cost_per_hour = round(random.uniform(0.1, 1.0), 3)  # Cost per hour with 3 digits
            latency = round(random.randint(5, 10), 3)  # Latency with 3 digits
        servers.append(Server(server_id, compute_capacity, server_type, cost_per_hour, latency))
    return servers

# Initialize population
def initialize_population(num_servers, num_tasks, population_size, p_zero=0.7):
    # Initialize a population of task assignments with a probabilistic approach
    # where the majority of values tend to be zero.
    population = []
    for _ in range(population_size):
        individual = []
        for _ in range(num_tasks):
            # With probability p_zero, choose 0; otherwise, choose a value between 1 and num_servers
            if random.random() < p_zero:
                individual.append(0)
            else:
                individual.append(random.randint(1, num_servers))
        population.append(individual)
    return population

# Fitness function
def fitness_function(chromosome, tasks, servers, alpha, beta):
    server_demand = {server.server_id: 0 for server in servers}
    total_cost = 0
    total_latency_penalty = 0
    total_utilization = 0

    for task_idx, server_idx in enumerate(chromosome):
        if server_idx != 0:  # Task is assigned to a server
            server = servers[server_idx - 1]
            task = tasks[task_idx]

            # Update server demand
            server_demand[server.server_id] += task.compute_demand

            # Check if this server exceeds its capacity
            if server_demand[server.server_id] > server.compute_capacity:
                return 0  # Invalid fitness

            # Add utilization
            total_utilization += task.compute_demand

            # Calculate cost penalty
            total_cost += server.cost_per_hour * task.compute_demand

            # Calculate latency penalty (higher priority tasks are more sensitive to latency)
            if task.priority == 2:  # High priority
                total_latency_penalty += server.latency * 2
            elif task.priority == 1:  # Medium priority
                total_latency_penalty += server.latency

    # Calculate fitness with penalties
    fitness = total_utilization - (alpha * total_cost) - (beta * total_latency_penalty)
    return max(fitness, 0)  # Ensure fitness is non-negative

# Mutation
def mutate(chromosome, num_servers, mutation_rate):
    mutated = chromosome[:]
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            # Mutate this gene by assigning it a random server or removing assignment
            mutated[i] = random.randint(0, num_servers)
    return mutated

# Crossover (one-point)
def crossover(parent1, parent2, crossover_rate):
    # Check if a random value is less than the crossover rate
    if random.random() < crossover_rate:
        point = random.randint(1, len(parent1) - 1) # Random crossover point
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2
    else:
        # No crossover, return parents as children
        return parent1, parent2

# Parent selection
def select_parents(population, fitness_values, num_parents):
    # Selects parents for crossover using Roulette Wheel Selection.
    # Calculate total fitness for normalization
    total_fitness = sum(fitness_values)
    if total_fitness == 0:
        # If all fitness values are zero, select random individuals
        return random.sample(population, num_parents)

    # Normalize fitness values to probabilities
    probabilities = [f / total_fitness for f in fitness_values]

    # Use random.choices to select parents based on probabilities
    selected_parents = random.choices(population, weights=probabilities, k=num_parents)

    return selected_parents


# Knapsack algorithm
def knapsack_algorithm(num_tasks, num_servers, population_size, generations, crossover_rate, mutation_rate, alpha,
                       beta):
    # Initialize the population with random chromosome assignments of tasks to servers
    population = initialize_population(num_servers, num_tasks, population_size)

    # Generate tasks and servers with their respective properties
    tasks = generate_tasks(num_tasks)
    servers = generate_servers(num_servers)

    # DEBUGGING: Display a subset of generated tasks and servers for verification
    print("Tasks:")
    for task in tasks[:num_tasks]:
        print(vars(task))  # Print task attributes as a dictionary

    print("\nServers:")
    for server in servers[:num_servers]:
        print(vars(server))  # Print server attributes as a dictionary
    print()

    # DEBUGGING: Display the initial population and corresponding fitness values
    print("INITIAL POPULATION")
    for idx, ind in enumerate(population):
        print(f"ind {idx + 1} - {ind} - fitness -> {fitness_function(ind, tasks, servers, alpha, beta)}")
    print()

    # Variables to track the best solution and fitness across generations
    best_solution = None
    best_fitness = 0
    fitness_over_generations = []  # List to store the best fitness value of each generation

    # Main loop to run the genetic algorithm for a fixed number of generations
    for generation in range(generations):
        # Calculate the fitness value of each individual in the population
        fitness_values = [fitness_function(chromosome, tasks, servers, alpha, beta) for chromosome in population]

        # Update the best solution and fitness if a new optimum is found
        for i, fitness in enumerate(fitness_values):
            if fitness > best_fitness:
                best_fitness = fitness
                best_solution = population[i]

        # Record the best fitness value of the current generation
        fitness_over_generations.append(best_fitness)
        print(f"Generation {generation + 1}: Best Solution {best_solution} - Best Fitness = {best_fitness}")

        # Select parent chromosomes based on their fitness values
        # Retain only half the population size as parents for crossover
        parents = select_parents(population, fitness_values, population_size // 2)

        # Perform crossover and mutation to produce offspring
        offspring = []
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents):  # Ensure pairs for crossover
                child1, child2 = crossover(parents[i], parents[i + 1], crossover_rate)
                offspring.extend([child1, child2])  # Add children to the offspring list

        # Mutate the offspring population with a given mutation rate
        offspring = [mutate(child, num_servers, mutation_rate) for child in offspring]

        # Combine parents and offspring, truncating to maintain population size
        population = parents + offspring
        population = population[:population_size]  # Trim population to the specified size

    # Calculate the theoretical maximum fitness possible based on task demands
    possible_max_fitness = sum(task.compute_demand for task in tasks)

    # Plot the progression of the best fitness values across generations
    plt.figure(figsize=(10, 6))
    plt.plot(fitness_over_generations, label="Best Fitness per Generation", color="blue")
    plt.axhline(y=possible_max_fitness, color="green", linestyle="--", label="Maximum Possible Fitness")
    plt.axhline(y=best_fitness, color="red", linestyle="-.", label="Final Best Fitness")
    plt.title("Fitness Progression Over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.grid()
    plt.show()

    # Return the best solution found, its fitness, and the maximum possible fitness
    return best_solution, best_fitness, possible_max_fitness

# Parameters
num_tasks = 100
num_servers = 40
population_size = 100
generations = 600
p_x = 0.6
p_m = 0.05
alpha = 0.01
beta = 0.005

bs, bf, pmf = knapsack_algorithm(num_tasks, num_servers, population_size, generations, p_x, p_m, alpha, beta)
print(f"Best Solution: {bs}- Best Fitness: {bf}")
print(f"\nBest Fitness: {round(bf, 3)}")
print(f"Possible Maximum Fitness: {pmf}")
print("--------------------------------")
print(f"Normalized Fitness: {round(bf / pmf, 3)}")
