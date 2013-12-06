
import os
import scipy as sp
from scipy.stats import gamma
import matplotlib.pyplot as plt

sp.random.seed(3)  # to reproduce the data later on

x = sp.arange(1, 31 * 24)
y = sp.array(200 * (sp.sin(2 * sp.pi * x / (7 * 24))), dtype=int)
y += gamma.rvs(15, loc=0, scale=100, size=len(x))
y += 2 * sp.exp(x / 100.0)
y = sp.ma.array(y, mask=[y < 0])
print(sum(y), sum(y < 0))

plt.scatter(x, y)
plt.title("Web traffic over the last month")
plt.xlabel("Time")
plt.ylabel("Hits/hour")
plt.xticks([w * 7 * 24 for w in [0, 1, 2, 3, 4]], ['week %i' % (w + 1) for w in [
           0, 1, 2, 3, 4]])

plt.autoscale(tight=True)
plt.grid()

data_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "data")
	
plt.savefig(os.path.join(data_dir, "web_traffic.png"))
sp.savetxt(os.path.join(
    data_dir, "web_traffic.tsv"), list(zip(x, y)), delimiter="\t", fmt="%s")


# sp.savetxt(os.path.join("..", "web_traffic.tsv"),
# zip(x[~y.mask],y[~y.mask]), delimiter="\t", fmt="%i")
#sp.savetxt(os.path.join(
#    data_dir, "web_traffic.tsv"), list(zip(x, y)), delimiter="\t", fmt="%s")
