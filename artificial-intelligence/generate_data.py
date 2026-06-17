import os
from environment import GRID_SIZE_8, GRID_SIZE_15, generate_solvable_state
from search import a_star, manhattan_distance

if __name__ == "__main__":
    print("Generating puzzle configurations and optimal solutions")

    # number of samples we want to generate
    NUMBER_OF_SAMPLES = 1000
    # list to store the dataset entries
    dataset_entries = []
    
    for i in range(NUMBER_OF_SAMPLES):
        # generate a random solvable initial configuration
        initial_configuration = generate_solvable_state(GRID_SIZE_8)
        # compute the optimal solution using A*
        path, nodes_expanded, optimal_cost = a_star(initial_configuration, manhattan_distance)
        # store the initial configuration and the optimal cost
        dataset_entries.append((initial_configuration, optimal_cost))

    current_directory = os.path.dirname(os.path.abspath(__file__))
    training_dataset_file = os.path.join(current_directory, "training_dataset.txt")

    # save the dataset in a file
    with open(training_dataset_file, "w") as f:
        for entry in dataset_entries:
            f.write(f"{entry[0]} {entry[1]}\n")

    print("Dataset generated successfully")