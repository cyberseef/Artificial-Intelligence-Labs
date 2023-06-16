import random


def generate_individual():
    # Generate a random individual
    return [random.choice(gene_pool) for _ in range(solution_length)]


def calculate_fitness(individual):
    # Calculate the fitness of an individual
    # Modify this function according to your problem
    # The higher the fitness value, the better the solution
    return abs(sum(individual) - target_solution)


def crossover(parent1, parent2):
    # Perform crossover between two parents to produce offspring
    # Modify this function according to your problem
    # This example performs single-point crossover
    crossover_point = random.randint(1, solution_length - 1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2


def mutate(individual):
    # Perform mutation on an individual
    # Modify this function according to your problem
    # This example randomly selects a gene and replaces it with a random value from the gene pool
    mutated_individual = individual[:]
    for i in range(solution_length):
        if random.random() < mutation_rate:
            mutated_individual[i] = random.choice(gene_pool)
    return mutated_individual


def select_parents(population):
    # Select two parents from the population using tournament selection
    # Modify this function according to your problem
    tournament_size = 5
    tournament = random.sample(population, tournament_size)
    return min(tournament, key=lambda individual: calculate_fitness(individual))


def main():
    # Initialize the population
    population = [generate_individual() for _ in range(population_size)]

    # Main loop
    for generation in range(max_generations):
        # Calculate fitness for each individual in the population
        fitness_scores = [calculate_fitness(individual) for individual in population]

        # Find the best individual and its fitness score
        best_index = fitness_scores.index(min(fitness_scores))
        best_individual = population[best_index]
        best_fitness = fitness_scores[best_index]

        print(f"Generation {generation + 1}: Best Fitness = {best_fitness}, Best Individual = {best_individual}")

        # Check termination condition
        if best_fitness == 0:
            break

        # Create the next generation
        next_generation = []

        # Elitism: Preserve the best individual
        next_generation.append(best_individual)

        # Generate offspring through selection, crossover, and mutation
        while len(next_generation) < population_size:
            parent1 = select_parents(population)
            parent2 = select_parents(population)

            if random.random() < crossover_rate:
                child1, child2 = crossover(parent1, parent2)
            else:
                child1, child2 = parent1, parent2

            child1 = mutate(child1)
            child2 = mutate(child2)

            next_generation.extend([child1, child2])

        # Update the population with the new generation
        population = next_generation


# Genetic Algorithm parameters
population_size = 100
mutation_rate = 0.01
crossover_rate = 0.8
max_generations = 100

# Problem-specific parameters
# Modify these according to your problem
target_solution = 42
solution_length = 8
gene_pool = [i for i in range(10)]  # Example gene pool for numbers 0-9

if __name__ == "__main__":
    main()
