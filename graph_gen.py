import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV files.
# Make sure these paths are correct.
hdqn_csv_path = "/home/srinivasan/Downloads/hdqn_monopoly.csv"
ppo_csv_path = "/home/srinivasan/Downloads/ppo_monopoly_PPO_3.csv"

# Load the CSV files into DataFrames
hdqn_df = pd.read_csv(hdqn_csv_path)
ppo_df = pd.read_csv(ppo_csv_path)

# Print first few rows for inspection.
# Inspect the data (optional)
print("HDQN Data Head:")
print(hdqn_df.head())
print("\nPPO Data Head:")
print(ppo_df.head())

# Create separate graph for HDQN (raw reward values).
plt.figure(figsize=(10, 6))
plt.plot(hdqn_df["Step"], hdqn_df["Value"], label="HDQN", marker="o", linestyle="-")
plt.xlabel("Step")
plt.ylabel("Reward")
plt.title("HDQN Rewards")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("hdqn_rewards.png")
plt.show()

# Create separate graph for PPO (raw reward values).
plt.figure(figsize=(10, 6))
plt.plot(ppo_df["Step"], ppo_df["Value"], label="PPO", marker="x", linestyle="--")
plt.xlabel("Step")
plt.ylabel("Reward")
plt.title("PPO Rewards")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("ppo_rewards.png")
plt.show()