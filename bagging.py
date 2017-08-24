from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.plotting import ColumnDataSource, Figure
from bokeh.models.widgets import Slider, Div
import numpy as np
from numpy import random
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

def get_y(X, power, coeff):
    ''' b0 + b1 * x + b2 * x^2 +... '''
    return reduce(lambda x, y: x+y, [coeff[i]*X**i for i in range(power)])

def func(x):
    ''' The 'true' function we use to generate data'''
    return x+5*np.sin(x)

x_points = 200
x = np.linspace(0,20,x_points)
err = np.random.normal(size=x_points)

p = Figure(title="bagging demo", plot_height=400, plot_width=800, y_range=(-5,30))

slider_degrees = Slider(start=1, end=10, step=1, value=5, title="Degrees")
slider_lines = Slider(start=1, end=50, step=1, value=10, title="Lines")
slider_points = Slider(start=1, end=100, step=1, value=20, title="Points")

# The datapoints
source_points = ColumnDataSource(data=dict(x=x, y=func(x)+err))
p.scatter(x='x', y='y', source=source_points, color="blue", line_width=3)

# The function where the datapoints come from
source_function = ColumnDataSource(data=dict(x=x, y=func(x)))
p.line(x='x', y='y', source=source_function, color="blue", line_width=1)

# The bootstrap lines
source_lines = ColumnDataSource(data=dict(xs=[ [], [] ], ys=[ [], [] ]))
p.multi_line(xs='xs', ys='ys', source=source_lines, color="pink", line_width=0)

# Their average
source_avg = ColumnDataSource(data=dict(x=[], y=[]))
p.line(x='x', y='y', source=source_avg, color="red", line_width=2)

# Basic instructions
div_instr = Div(text="<font color=black>\
<br> <font color=blue>The blue line </font>is the “true” curve we are trying to approximate.\
<br> <font color=blue><b>The blue dots </b></font>are points drawn from the blue curve with an added random error.\
<br> <font color=pink>Each pink line</font> is the polynomial regression fit over a randomly drawn (with replacement) subset of points.\
<br> <font color=red><b>The thick red line </b></font>is the average of the pink lines.</font>", width=800, height=100)

def update(attrname, old, new):
    D=slider_degrees.value  # number of degrees for the polynomial
    L=slider_lines.value    # number of bootstrap lines
    N=slider_points.value   # number of points used for the bootstrap lines

    list_xy=[]
    for i in range(L):
        # randomly pick points (with replacement) from the original dataset
        filt = np.random.choice(len(x), size=N, replace=True)
        list_xy.append((x[filt], (func(x)+err)[filt]))

    #update the data points, NB: showing only the points used for the last bootsrap line as a guide
    source_points.data = dict(x=x[filt], y=(func(x) + err)[filt])

    # polynomial regression model for the bootstrap lines
    model = Pipeline([('poly', PolynomialFeatures(degree=D)), ('linear', LinearRegression(fit_intercept=False))])

    coeff_list=[]
    for xy in list_xy:
        model = model.fit(xy[0][:, np.newaxis], xy[1])
        coeff_list.append(model.named_steps['linear'].coef_)

    vtot=np.zeros(len(x))
    xs=[]
    ys=[]
    for i in range(L):
        v=np.array(get_y(x, len(coeff_list[i]),coeff_list[i]))
        xs.append(x)
        ys.append(v)
        vtot+=v

    #update the bootstrap lines and their average
    source_lines.data = dict(xs=xs, ys=ys)
    source_avg.data = dict(x=x, y=vtot/float(L))

for w in [slider_degrees, slider_lines, slider_points]:
    w.on_change('value', update)

layout = column(div_instr, p, slider_degrees, slider_lines, slider_points)
curdoc().add_root(layout)
