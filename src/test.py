from llm4bbo.benchmark import make_task
import numpy as np
from tqdm import tqdm


task = make_task("ant", num_designs=500)

x_offline = task._task.x
y_offline = task._task.y
y_offline_correct = task.y_offline

task._task.dataset.subsample()
x_all = task._task.x
y_all = task._task.y

print(x_offline.shape)
print(y_offline.shape)
print(x_all.shape)
print(y_all.shape)

# Predict with batch size 32, task.predict returns 2D array of shape (n, 1)
y_offline_pred = np.zeros((len(x_offline), 1))
y_offline_corrected = np.zeros((len(x_offline), 1))
for i in tqdm(range(0, len(x_offline), 1024)):
    y_offline_pred[i:i+5, :] = task.predict(x_offline[i:i+5])
    y_offline_corrected[i:i+5, :] = y_offline_correct[i:i+5, :]


print(all(y_offline_pred == y_offline))
print(all(y_offline_corrected == y_offline_pred))

# Print these arrays
print("y_offline_pred")
print(y_offline_pred)
print("y_offline_corrected")
print(y_offline_corrected)
print("y_offline")
print(y_offline)
