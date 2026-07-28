import json
from davi_utils import GOAL_STATE, generate_true_scramble

def generate_dataset(output_file="test_dataset_1000.json", samples_per_depth=200):
    depths = [20, 40, 60, 80, 100]
    dataset = {}

    print("Generating evaluation dataset...")
    for d in depths:
        dataset[str(d)] = [generate_true_scramble(GOAL_STATE, d) for _ in range(samples_per_depth)]
        print(f"Generated {samples_per_depth} test instances for depth {d}")

    with open(output_file, "w") as f:
        json.dump(dataset, f)

    print(f"Dataset saved to {output_file}")

if __name__ == "__main__":
    generate_dataset()