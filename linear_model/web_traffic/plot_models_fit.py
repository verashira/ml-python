import scipy as sp
import matplotlib.pyplot as plt

colors = ['g', 'k', 'b', 'm', 'r']
linestyles = ['-', '-.', '--', ':', '-']

def plot_models(x, y, models, fname, mx=None, 
        ymax=None, ymin=None, xmax=None, xmin=None):
    plt.clf()
    plt.scatter(x, y, s=10)
    plt.title("Web traffic over hours");
    plt.xlabel("Time")
    plt.ylabel("Hits/hour")
    plt.xticks(
        [w * 7 * 24 for w in range(10)],
        ["week %i" % w for w in range(10)]);
    
    if models:
        if mx is None:
            mx = sp.linspace(0, x[-1], 1000)
            
        for model, color, style in zip(models, colors, linestyles):
            plt.plot(mx, model(mx), linestyle=style, linewidth=2, c=color)
        plt.legend(["d=%i" % m.order for m in models])
        
    plt.autoscale(tight=True)
    plt.xlim(xmin = 0)
    plt.ylim(ymin = 0)
    if xmin: plt.xlim(xmin=xmin)
    if xmax: plt.xlim(xmax=xmax)
    if ymin: plt.ylim(ymin=ymin)
    if ymax: plt.ylim(ymax=ymax)
    plt.grid(True, linestyle="-", color="0.75")
    plt.savefig(fname)