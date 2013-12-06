import os
import scipy as sp
import matplotlib.pyplot as plt

from plot_models_fit import plot_models

# Initialize the directories
file_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "data");
data = sp.genfromtxt(os.path.join(data_dir, "web_traffic.tsv"), delimiter='\t')

# Fetch the dataset
x = data[:, 0]
y = data[:, 1]
x = x[~sp.isnan(y)]
y = y[~sp.isnan(y)]

# Plot the original data, 'data_origin_png'
plot_models(x, y, None, os.path.join(file_dir, "result", "data_origin.png"))

# Plot the model with digree of 1,2,3,4,10,100
model_d1 = sp.poly1d(sp.polyfit(x, y, 1))
model_d2 = sp.poly1d(sp.polyfit(x, y, 2))
model_d3 = sp.poly1d(sp.polyfit(x, y, 3))
model_d4 = sp.poly1d(sp.polyfit(x, y, 4))
model_d10 = sp.poly1d(sp.polyfit(x, y, 10))
model_d100 = sp.poly1d(sp.polyfit(x, y, 100))
plot_models(x, y,
    [model_d1, model_d2, model_d3, model_d4, model_d10, model_d100],
    os.path.join(file_dir, "result", "polyfits.png"))
    
    
# Plot the model by analysing data divided into two parts
inflection_point = 3.5 * 7 * 24
xa = x[:inflection_point]
ya = y[:inflection_point]
xb = x[inflection_point:]
yb = y[inflection_point:]
model_ap = sp.poly1d(sp.polyfit(xa, ya, 1))
model_bp = sp.poly1d(sp.polyfit(xb, yb, 1))
plot_models(x, y,
    [model_ap, model_bp],
    os.path.join(file_dir, "result", "polyfits_2_parts.png"))
    
   
# Compare errors with those models
def error(f, x, y):
    return sp.sum((f(x) - y) ** 2)

print("Errors for the complete data set:")
for m in [model_d1, model_d2, model_d3, model_d4, model_d10, model_d100]:
    print("Error for d=%i: %f" % (m.order, error(m, x, y)))
print("Errors for the last part of the data set:")
for m in [model_d1, model_d2, model_d3, model_d4, model_d10, model_d100]:
    print("Error for d=%i: %f" % (m.order, error(m, xb, yb)))
print("Error with inflection: %f" % 
    (error(model_ap, xa, ya) + error(model_bp, xb, yb)))
    
    
# Plot the model with possible future data
plot_models(x, y,
    [model_d1, model_d2, model_d3, model_d4, model_d10, model_d100],    
    os.path.join(file_dir, "result", "polyfits_fits_future.png"),
    mx=sp.linspace(0*7*24, 6*7*24, 100),
    ymax=10000)
    

# Predict the day that traffic will achieve 100000
from scipy.optimize import fsolve
print(model_d2)
print(model_d2-100000)
zero_point = fsolve(model_d2-100000, 800) / (7*24)
print("100,000 hits/hour expected at week %f" % zero_point)
