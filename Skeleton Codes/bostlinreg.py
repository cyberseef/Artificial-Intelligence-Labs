import random
import numpy as np
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# Genetic Algorithm parameters
population_size = 100
mutation_rate = 0.01
crossover_rate = 0.8
max_generations = 100

# Problem-specific parameters
random_seed = 42
data = load_boston()
target = data.target
features = data.data
feature_names = data.feature_names
num_features = features.shape[1]


def generate_individual():
    # Generate a random individual as a binary string indicating feature selection
    return [random.randint(0, 1) for _ in range(num_features)]


def calculate_fitness(individual):
    # Calculate the fitness of an individual by training a Linear Regression model
    selected_features = [feature for feature, select in zip(range(num_features), individual) if select == 1]
    if not selected_features:
        return float('-inf')  # Penalize empty feature subset

    X_train, X_test, y_train, y_test = train_test_split(features[:, selected_features], target, test_size=0.2, random_state=random_seed)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)  # Higher R^2 score is better


def crossover(parent1, parent2):
    # Perform crossover between two parents to produce offspring
    # This example performs uniform crossover
    child1 = parent1[:]
    child2 = parent2[:]
    for i in range(num_features):
        if random.random() < crossover_rate:
            child1[i], child2[i] = child2[i], child1[i]
    return child1, child2


def mutate(individual):
    # Perform mutation on an individual
    # This example flips the bit with a certain probability for each feature
    mutated_individual = individual[:]
    for i in range(num_features):
        if random.random() < mutation_rate:
            mutated_individual[i] = 1 - mutated_individual[i]
    return mutated_individual


def select_parents(population):
    # Select two parents from the population using tournament selection
    tournament_size = 5
    tournament = random.sample(population, tournament_size)
    return max(tournament, key=lambda individual: calculate_fitness(individual))


def main():
    # Initialize the population
    population = [generate_individual() for _ in range(population_size)]

    # Main loop
    for generation in range(max_generations):
        # Calculate fitness for each individual in the population
        fitness_scores = [calculate_fitness(individual) for individual in population]

        # Find the best individual and its fitness score
        best_index = fitness_scores.index(max(fitness_scores))
        best_individual = population[best_index]
        best_fitness = fitness_scores[best_index]

        print(f"Generation {generation + 1}: Best Fitness = {best_fitness}")

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


if __name__ == "__main__":
    main()
