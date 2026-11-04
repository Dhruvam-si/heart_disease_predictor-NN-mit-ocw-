import matplotlib
print("Backend:", matplotlib.get_backend())

import matplotlib.pyplot as plt

plt.plot([1, 2, 3], [1, 4, 9])
plt.title("Rendering Test")
plt.savefig("test_plot.png")
plt.show()