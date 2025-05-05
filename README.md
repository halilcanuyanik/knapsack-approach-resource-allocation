# Knapsack Problem with Genetic Algorithm

This project implements a **Genetic Algorithm (GA)** to solve a variant of the **Knapsack Problem**, where tasks with computational demands and priorities are assigned to servers with limited capacities. The aim is to maximize resource utilization without exceeding capacity constraints.

## 🚀 Features

- **Task Generation:** Randomly generates tasks with varying compute demands and priorities.
- **Server Configuration:** Simulates edge and cloud servers with different compute capacities.
- **Genetic Algorithm Workflow:**
  - Population Initialization
  - Fitness Evaluation
  - Parent Selection (Roulette Wheel Selection)
  - Crossover (One-point Crossover)
  - Mutation
- **Flexible Parameters:** Easily configurable population size, mutation rate, crossover rate, alpha parameter and number of generations.

## 🔧 Getting Started

### Prerequisites

- Python 3.7+
- `random` module (default Python library)

### Running the Code

1. Clone the repository:
   ```bash
   git clone https://github.com/halilcanuyanik/knapsack-approach-resource-allocation.git
   cd knapsack-approach-resource-allocation
   ```
2. Run the script:
   ```bash
   python source_code.py
   ```

## 📁 Project Structure

```plaintext
├── source_code.py          # Main script with the genetic algorithm
├── README.md               # Project documentation
```

## 🧠 Key Components

**Tasks** represent items in the knapsack. Each task has:

- **Compute Demand:** The computational requirement of the task.
- **Priority:** A value indicating the task's importance.

**Servers** represent bins in the knapsack. Each server has:

- **Compute Capacity:** The maximum load it can handle.
- **Type:** "Edge" or "Cloud."

**Genetic Algorithm Functions**

- **Fitness Function:** Calculates how well a solution (chromosome) satisfies server capacity constraints.
- **Selection:** Chooses parents based on their fitness values.
- **Crossover:** Combines genes from two parents to produce offspring.
- **Mutation:** Randomly alters genes in offspring to maintain diversity.

## ⚙️ Configurable Parameters

You can customize the following parameters in the main.py script:

- **Number of Tasks (num_tasks):** Total tasks to be assigned.
- **Number of Servers (num_servers):** Total servers available.
- **Population Size (population_size):** Number of chromosomes in each generation.
- **Number of Generations (generations):** Iterations of the genetic algorithm.
- **Mutation Rate (mutation_rate):** Probability of mutating a gene.
- **Crossover Rate (crossover_rate):** Probability of performing crossover.
- **Alpha Parameter:** Parameter (probability) that determines the zero distribution of the initial chromosomes (also has to be tuned)

## 🤝 Contributing

Contributions are welcome! Here's how you can get started:

- Fork this repository.
- Create a new branch (git checkout -b feature-branch).
- Make your changes and commit them (git commit -m "Description of changes").
- Push to your branch (git push origin feature-branch).
- Open a Pull Request.

## ❤️ Contributors

- Halil Can Uyanık - Creator & Maintainer **@halilcanuyanik**
- Yaşar Mehmet Bağdatlı - Creator & Maintainer **@yasarmb**

## 📜 License

This project is licensed under the General Public License (GPL)

## 🛠️ Future Enhancements

- Add visualization for fitness trends across generations.
- Implement parallel execution for faster performance.
- Explore other selection and mutation strategies.
