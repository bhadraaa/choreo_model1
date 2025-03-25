import matplotlib.pyplot as plt

tasks = ["Research", "UI Design", "ML Model", "Testing", "Finalization"]
start_weeks = [0, 1, 2, 3, 4]
durations = [1, 1, 1, 1, 1]

fig, ax = plt.subplots(figsize=(8, 4))
for i, (task, start, duration) in enumerate(zip(tasks, start_weeks, durations)):
    ax.barh(task, duration, left=start, color="skyblue", edgecolor="black")

ax.set_xlabel("Weeks")
ax.set_title("Gantt Chart for Project")
ax.set_xticks(range(6))
ax.set_xticklabels(["Week 1", "Week 2", "Week 3", "Week 4", "Week 5", "Week 6"])
ax.invert_yaxis()

plt.show()
