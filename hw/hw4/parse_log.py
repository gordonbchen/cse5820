import re
import sys
import os
import matplotlib.pyplot as plt

def parse_and_plot(file_path):
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    # Read the file content
    with open(file_path, 'r') as f:
        data = f.read()

    # Regex to extract episode number and reward
    # Matches "0: reward=-20.0" etc.
    pattern = r"(\d+): reward=([-+]?\d*\.\d+|\d+)"
    matches = re.findall(pattern, data)
    
    if not matches:
        print("No valid reward data found in the file.")
        return

    episodes = [int(m[0]) for m in matches]
    rewards = [float(m[1]) for m in matches]

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, rewards, marker='o', markersize=4, linestyle='-', alpha=0.7)
    
    plt.title(f"Training Progress: {os.path.basename(file_path)}")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True, linestyle='--', alpha=0.6)

    # Construct the output path: same directory, same name + .png
    output_path = f"{file_path}.png"
    
    plt.savefig(output_path)
    plt.close() # Close plot to free up memory
    print(f"Success! Plot saved to: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <path_to_log_file>")
    else:
        parse_and_plot(sys.argv[1])
