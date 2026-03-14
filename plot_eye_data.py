import pandas as pd
import matplotlib.pyplot as plt

# Load data
data = pd.read_excel("blink_gaze_data.xlsx")

# -----------------------------
# Reaction Time Graph
# -----------------------------
plt.figure()
plt.plot(data["Trial"], data["Reaction_Time"], marker="o")
plt.title("Reaction Time vs Trial")
plt.xlabel("Trial")
plt.ylabel("Reaction Time (sec)")
plt.grid(True)
plt.show()

# -----------------------------
# Saccade Speed Graph
# -----------------------------
plt.figure()
plt.plot(data["Trial"], data["Saccade_Speed"], marker="o")
plt.title("Saccade Speed vs Trial")
plt.xlabel("Trial")
plt.ylabel("Saccade Speed (px/sec)")
plt.grid(True)
plt.show()

# -----------------------------
# Accuracy Graph
# -----------------------------
correct = data["Correct"].value_counts()

plt.figure()
plt.bar(correct.index.astype(str), correct.values)
plt.title("Accuracy")
plt.xlabel("Correct / Incorrect")
plt.ylabel("Count")
plt.show()