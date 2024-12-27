---
jupyter:
  jupytext:
    formats: ipynb,src//md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.6
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- #region id="e808cacf-697a-4481-acb0-2867b1a763c5" -->
<a href="https://colab.research.google.com/github/project-ida/classical-analogs/blob/master/01-harmonic-oscillators.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<a href="https://nbviewer.org/github/project-ida/classical-analogs/blob/master/01-harmonic-oscillators.ipynb" target="_parent"><img src="https://nbviewer.org/static/img/nav_logo.svg" alt="Open In nbviewer" width="100"/></a>
<!-- #endregion -->

<!-- #region id="ST2ywugL0wNs" -->
# Harmonic oscillators and coupled harmonic oscillators
<!-- #endregion -->

<!-- #region id="ITC4l6CP0wNs" -->
In this tutorial we will talk about various kinds of harmonic oscillators.
<!-- #endregion -->

<!-- #region id="xFbrqI3b0wNs" -->
## 1. Python helper functions
<!-- #endregion -->

```python id="PFOAJj3u0wNs"
# loading some common python libraries

import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
from matplotlib import animation, rc
from IPython.display import Image
from IPython.core.display import HTML
import sympy as sp
#from google.colab.output._publish import javascript
mathjaxurl = "https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.3/latest.js?config=default"
sp.init_printing(use_latex='mathjax')
import matplotlib.patches as patches
import pandas as pd
```

```python id="PVySuLvu0wNt"
def create_fig1(): # create 1 empty subplots with 3 empty lines in each
    global lines1A

    fig1 = plt.figure() #figsize=(8,2)

    ax1A = fig1.add_subplot(111, xlim=(-5, 5), ylim=(-3, 3))
    ax1A.grid(color='lightgrey',alpha=1)
    ax1A.set_xticks(np.arange(-5,5,1))
    ax1A.set_aspect('equal', 'box')

    lines1A = []
    for i in np.arange(0,5,1):
        lobj = ax1A.plot([], [], 'o-')[0] # , lw=0, color='green'
        lines1A.append(lobj)

    plt.close()
    return fig1
```

```python id="d0IqicA00wNt"
def create_fig2(ymax=50): # create 2 empty subplots with 3 empty lines in each
    global lines2A,lines2B

    fig2 = plt.figure(figsize=(12,8))

    ax2A = fig2.add_subplot(211, xlim=(-5, 5), ylim=(-3, 3))
    ax2A.grid(color='lightgrey',alpha=1)
    #ax2A.axis('equal')
    #ax2A.tick_params(which = 'both', direction = 'out')
    ax2A.set_xticks(np.arange(-5,5,1))
    ax2A.set_aspect('equal', 'box')

    lines2A = []
    for i in np.arange(0,3,1):
        lobj = ax2A.plot([], [], 'o-')[0]
        lines2A.append(lobj)

    ax2B = fig2.add_subplot(212, xlim=(0, ymax), ylim=(-1.2,1.2))

    lines2B = []
    for i in np.arange(0,3,1):
        lobj = ax2B.plot([], [], '-')[0]
        lines2B.append(lobj)

    plt.close()
    return fig2
```

```python id="Jirvxs_H0wNt"
def animate1(i,data_graph1): # animation function that is called once for every frame (for fig 1)
    global lines1A

    for lineno,_ in enumerate(data_graph1): # strep through the list and the data for each line and assign data for frame i

        lines1A[lineno].set_xdata(data_graph1[lineno][i][0])
        lines1A[lineno].set_ydata(data_graph1[lineno][i][1])
```

```python id="BVBOBohH0wNt"
def animate2(i,data_graph1,data_graph2): # animation function that is called once for every frame (for fig 2)

    for lineno,value in enumerate(data_graph1):
        lines2A[lineno].set_xdata(data_graph1[lineno][i][0])
        lines2A[lineno].set_ydata(data_graph1[lineno][i][1])

    for lineno,value in enumerate(data_graph2):
        lines2B[lineno].set_xdata(data_graph2[lineno][i][0])
        lines2B[lineno].set_ydata(data_graph2[lineno][i][1])
```

```python id="-ldnEZpP0wNt"
def add_massspring_patches(thisfigure):

    thisaxis = thisfigure.axes[0]

    thisaxis.add_patch(patches.Rectangle(xy=(-5,-3),width=1,height=6,facecolor="gainsboro"))
    thisaxis.add_patch(patches.Rectangle(xy=(4,-3),width=1,height=6,facecolor="gainsboro"))
    thisaxis.add_patch(patches.Rectangle(xy=(-4,-3),width=8,height=2.8,facecolor="gainsboro"))
```

```python id="zbhMfbYR0wNu"
def add_pendulum_patches(thisfigure):

    thisaxis = thisfigure.axes[0]

    thisaxis.add_patch(patches.Rectangle(xy=(-5, 2),width=10,height=1,facecolor="gainsboro"))
```

```python id="QZFeCJJ50wNu"
def add_polar_coordinates(df):
    if 'x1' in df:
        alltheta1 = df.x1
        df['sin1'] = np.sin(alltheta1)*L1
        df['cos1']  = np.cos(alltheta1)*L1
    if 'x2' in df:
        alltheta2 = df.x2
        df['sin2'] = np.sin(alltheta2)*L2
        df['cos2'] = np.cos(alltheta2)*L2
    if 'x3' in df:
        alltheta3 = df.x3
        df['sin3'] = np.sin(alltheta3)*L3
        df['cos3'] = np.cos(alltheta3)*L3
    return df
```

<!-- #region id="2k2ibu0a0wNu" -->
## 2. Introduction to the simple harmonic oscillator
<!-- #endregion -->

```python id="g2TNrczY0wNu" outputId="ff417113-9800-43ab-bbc6-48a5cbc58745"
Image(url= "https://upload.wikimedia.org/wikipedia/commons/thumb/2/22/Harmonic_oscillator.svg/200px-Harmonic_oscillator.svg.png", width=200)
```

<!-- #region id="L6ASh6Y60wNu" -->
The harmonic oscillator is a basic concept in physics that can be extended from its simplest form to address many complex problems (as we will see later). It is also a test case for working with differential equations and common formalisms of quantum mechanics such as the Hamiltonian formalism.

Let's start with the most basic version of the harmonic oscillator: a mass attached to a string and moving horizontally (therefore not affected by gravity) in an environment that is assumed to have no friction.

This system is not particularly interesting if the initial condition is $x=0$ i.e. the mass is in a position where the spring is not compressed or expanded. In that case the system will not do anything. However, if the initial condition is different, then the spring will either expand or contract which creates an oscillation between two forces: an acceleration and a reverse acceleration.

In this notebook we will use the following variables. For simplicity many of them are 1.
<!-- #endregion -->

```python id="ye3bcmkD0wNu"
m1 = 1
k1 = 0.5

m2 = 1 # second mass and spring for later
k2 = 0.5

kc = 0.1 # coupling spring for later

c = 0.1 # damping factor for later
```

```python id="EQhdK-IJ0wNu"
t_end = 50 # time for simulations
allt = np.arange(0,t_end,0.1) # time vector size 500
```

<!-- #region id="b5rwaLVg0wNu" -->
## 2.1. Equations of motion from force balance (Newtonian approach)
<!-- #endregion -->

<!-- #region id="eSdLGrGK0wNu" -->
The two forces will be of equal magnitude at the extremes (turning points) which leads to the expression:

$$F = ma = -kx$$
$$m\ddot {x} = -kx$$
This can be rewritten as
$$m\ddot {x} + kx = 0$$
$$\ddot {x} + \frac{k}{m} x = 0$$
for which a solution exists in the form of
$$x(t)=A\sin(\omega t)$$
$$\dot x(t)=B\cos(\omega t)$$
where $\omega ={\sqrt {\frac {k}{m}}}$ and where $A=x_0$ and $B=\frac{p_0}{m}$ are determined by initial conditions.

And remember from the waves notebook: This can also be expressed as

$$\operatorname{Re}({e^{i\omega t}})$$
<!-- #endregion -->

<!-- #region id="2R2F6aUJ0wNu" -->
We can plot this solution easily:
<!-- #endregion -->

```python id="ngHZlt600wNu"
omega = np.sqrt(k1/m1)

allx = 1*np.sin(omega*allt)
allv = omega*np.cos(omega*allt)
```

```python id="kWzT0n-K0wNu" outputId="5d232318-5b8d-4858-c6c8-968bf84602ec"
plt.plot(allt,allx)
plt.plot(allt,allv)
plt.grid()
```

<!-- #region id="ylouRVe40wNu" -->
To develop some more intuition about it, and also to introduce the animation process used extensively below, we will now animation this motion:
<!-- #endregion -->

```python id="ZTXhlhRH0wNu"
# create position vectors for the first animated graph
data1 = []
for index,t in enumerate(allt): # step through every time step

    position = allx[index]
    data1.append([[-4,position-1],[0,0]])
```

```python id="SGHjo4Ix0wNv"
# create vector of position function up to respective point for the second animated graph
data2,data3 = ([],[])
for index,t in enumerate(allt): # step through every time step

    position_sofar = allx[:index+1]
    velocity_sofar = allv[:index+1]
    time_sofar = allt[:index+1]
    data2.append([time_sofar,position_sofar])
    data3.append([time_sofar,velocity_sofar])
```

```python id="5kJ8J91T0wNv" outputId="26398aaa-77df-477e-d824-03f0eae757b0"
# call the animation function
subplot1_data = [np.asarray(data1)]
subplot2_data = [np.asarray(data2),np.asarray(data3)]

thisfig = create_fig2(20)
add_massspring_patches(thisfig)

ani = animation.FuncAnimation(thisfig,animate2,frames=200,interval=50,fargs=(subplot1_data,subplot2_data))
rc('animation', html='jshtml')
ani
```

<!-- #region id="3oTTVNt30wNv" -->
## Intermezzo: Numerical time evolution of coupled differential equations
<!-- #endregion -->

<!-- #region id="h3N7Do4A0wNv" -->
In the example above, as is done in many textbooks, we "guessed" the solution functions. In other words, we found them "by inspection." This is not very satisfying because it does not generalize well. In reality, most differential equations are going to be more complicated and no guess can be made easily. So we will introduce here numerical methods that can be applied to a much larger number of systems.

One of the simplest and oldest numerical technique is known as Euler's method. Like in many numerical approaches, we first need to write a second-order differential equation as a coupled system of first-order differential equations.

Our equation
$$\ddot {x} + \frac{k}{m} x = 0$$
can also be written as

$$\ddot x = -\frac{k}{m} x = \dot v = \frac{dv}{dt}$$
$$\dot x = v = \frac{dx}{dt}$$

Basic it means starting from the initial conditions and slowing growing all relevant functions by adding a $\Delta t$ to $x_i$, which turns it into $x_{i+1}$, and using this as an input to calculate other functions that depend on it, e.g. $p_{i+1}(x_{i+1})$

Again, we want to calculate `allx` and `allv` starting from initial conditions `xi = allx[0]` and `vi = allv[0]`:
<!-- #endregion -->

```python id="OuRhyzC30wNv"
allx[0] = 1 # initial conditions
allv[0] = 0

dt = 0.1

for j, thist in enumerate(allt):
    if j+1 < len(allt):

        dx = allv[j] * dt
        allx[j+1] = allx[j] + dx

        dv = allx[j] * (-k1/m1) * dt
        allv[j+1] = allv[j] + dv
```

```python id="juzbVlG60wNv" outputId="7f42a039-5de1-4414-a007-79a63080fadf"
plt.plot(allt,allx)
plt.plot(allt,allv)
plt.grid()
```

<!-- #region id="5o3-f9y10wNv" -->
We can see that the amplitude unexpectedly grows. This is a problem intrinsic to the Euler method. It can be alleviated by choosing very small dt increments, but that means a lot of computing cycles. An improvement is the Euler-Cromer method where the simple change is made that
`dv = allx[j+1] * (-k/m) * num_dt` instead of `dv = allx[j+1] * (-k/m) * num_dt`  
(see for more details http://physics.ucsc.edu/~peter/242/leapfrog.pdf). This small change implemented:
<!-- #endregion -->

```python id="d0-iLXqN0wNv"
allx[0] = 1 # initial conditions
allv[0] = 0

dt = 0.1

for j, thist in enumerate(allt):
    if j+1 < len(allt):

        dx = allv[j] * dt
        allx[j+1] = allx[j] + dx

        dv = allx[j+1] * (-k1/m1) * dt
        allv[j+1] = allv[j] + dv
```

```python id="HhzBo3hu0wNv" outputId="13c7c5e2-f168-4dc2-b8c1-63d1f7bef7fc"
plt.plot(allt,allx)
plt.plot(allt,allv)
plt.grid()
```

<!-- #region id="3Qlh4J-U0wNv" -->
This looks a lot more like what we would expect and satisfies us for now. We will use the Euler-Cromer method again for more complicated problems later. We will also move it to a dedicated function to keep the code more readable.
<!-- #endregion -->

<!-- #region id="CLNGidRg0wNv" -->
## 2.2. Equations of motion from energy balance (Hamiltonian approach)
<!-- #endregion -->

<!-- #region id="3kSx-6AX0wNv" -->
Deriving the equations of motion via force balance equations, like introduced above, is known as the Newtonian approach, centered around forces.

An alternative approach is the Hamiltonian approach. It may be considered overkill for a simple problem like this but employing it gives us a sense of the formalism. In the Hamiltonian approach, we create an expression that keeps track of all energy in the system. In this case, that will be kinetic and potential energy:

$$H = E_{kin} + E_{pot}$$

At $x=0$ the system will have only kinetic energy and now potential energy. At the turning point, it will have only potential energy (stored in the spring) and no kinetic energy. The potential energy of the spring is $\frac{1}{2} k x^2$ so the expression becomes

$$H = \frac{1}{2} m \dot{x}^2 + \frac{1}{2} k x^2$$

which can also be written as

$$H(x,p) = \frac{p^2}{2m} + \frac{1}{2} k x^2$$

In a classical mechanics system, the time evolution of such a Hamiltonian is governed by Hamilton's coupled equations aka the evolution equation of the Hamiltonian (see https://en.wikipedia.org/wiki/Hamiltonian_system):

$$\dot{x} = \frac {d{x}}{dt} = +{\frac {\partial H}{\partial {p}}}$$
$$\dot{p} = \frac {d{p}}{dt} = -{\frac {\partial H}{\partial {x}}}$$

The derivations of the Hamiltonian lead to:

$$\dot{x} = \frac {d{x}}{dt} = \frac{p}{m}$$
$$\dot{p} = \frac {d{p}}{dt} = -kx$$

These can then also be time-evolved numerically by stepwise growing t at small increment, as discussed above (e.g. Euler's method or Runge-Kutta method).

Using Euler's method, we get:
$$x_{j+1} = x_j + \Delta t \frac{p_j}{m}$$
$$p_{j+1} = p_j - \Delta t k{x_j}$$

In our examples, m=1, so p=v and we get the same results as above.
<!-- #endregion -->

<!-- #region id="lzehyb6k0wNv" -->
## 3. Damped harmonic oscillator
<!-- #endregion -->

<!-- #region id="jci65FVP0wNv" -->
Next, let's see what happens if we consider damping. In this case our equation

$$\ddot {x} + \frac{k}{m} x = 0$$

will be added by a damping term that is proportional to the velocity

$$\ddot {x} + c \dot {x} + \frac{k}{m} x = 0$$

The equations of motion are then:

$$\ddot {x} = \dot v = \frac{dv}{dt} = - c \dot {x} - \frac{k}{m} x$$
$$\dot x = v = \frac{dx}{dt}$$

The Hamiltonian becomes:

$$H(x,p) = \frac{p^2}{2m} + \frac{1}{2}kx^2 - c p x $$

Following Hamilton's equations

$$\dot{x} = \frac {d{x}}{dt} = +{\frac {\partial H}{\partial {p}}}$$
$$\dot{p} = \frac {d{p}}{dt} = -{\frac {\partial H}{\partial {x}}}$$

we get

$$\dot{x} = \frac{p}{m} - c x$$

$$\dot{p} = - k x + c p$$

The Hamiltonian approach is having trouble with this system, since energy is not conserved and leaks out of the system without being accounted for by the Hamiltonian. This makes this approach less suited. Different people found ways to address that but it goes beyond the scope of this notebook and we'll just refer to further reading:
* https://quantumcoffee.wordpress.com/2014/07/16/the-damped-harmonic-oscillator/
* http://www.hep.princeton.edu/~mcdonald/examples/damped.pdf
* http://astro.pas.rochester.edu/~aquillen/phy411/lecture1.pdf
* https://www.hindawi.com/journals/ijo/2010/275910/
* https://spiral.imperial.ac.uk/bitstream/10044/1/42217/6/JMPsubmissionRev.pdf
* http://core.csu.edu.cn/NR/rdonlyres/60B95A28-810C-40F2-9A35-A161AF9A44DB/0/lecnotes4.pdf
* http://www.entropy.energy/scholar/node/damped-harmonic-oscillator-energy

For now, we will see what our numerical approach gives us for this system.

Also, based on physical experience, people "guess" a solution to this differential equation of the form.
<!-- #endregion -->

<!-- #region id="mO7lnWTQ0wNy" -->
We'll use the same Euler-Comer numerical approach as above but instead of doing everything within a loop, we'll clean it up a bit and move the calculations into a function. This makes it me more suitable for more complicated systems.
<!-- #endregion -->

```python id="_F7AkV8A0wNy"
c = 0.1
```

```python id="iwE3_4-90wNy"
def get_next_state(present_state,dt):

    global m1, k1

    x1 = present_state[0]
    v1 = present_state[1]

    dx1 = v1*dt
    nextx1 = x1 + dx1

    dv1 = (nextx1 * -k1/m1 + v1 * -c)*dt
    nextv1 = v1 + dv1

    next_state = np.array([nextx1,nextv1])
    return next_state
```

```python id="nBGewL0N0wNy"
x1i=1 #initial displacement x1
v1i=0 #initial velocity
thisstate = np.array([x1i,v1i]) # initial state

dt = 0.1

allstates = []
for index,t in enumerate(allt): # step through every time step, at every time step evaluate the function

    allstates.append(thisstate)
    nextstate = get_next_state(thisstate, dt)
    thisstate = nextstate

df = pd.DataFrame(np.row_stack(allstates), columns = ['x1', 'v1'])
df.insert(0, "time", allt, True)
```

```python id="wLKQlTxd0wNy" outputId="9c3f266d-385c-4eb7-b777-dab154028d74"
plt.plot(allt,df.x1)
plt.plot(allt,df.v1)
plt.plot(allt,1*np.exp(-c*0.5*allt/m1))
```

<!-- #region id="YAd6u3vO0wNy" -->
We also ploted a negative potential of the form of the guessed solution which shows that the guess was proper. The amplitude (and the energy) of the system decays on an exponential trajectory.

We can also plot potential and kinetic energy and can see that the overall energy of the system also decays this way. The energy leaves the system into the environment which is not part of the model.
<!-- #endregion -->

```python id="L7WYxls10wNy" outputId="4be604e1-53b9-492e-ac95-9b2d3f6efdb7"
E_potential = 0.5*k1*df.x1**2
E_kinetic = 0.5*m1*df.v1**2
E_total = E_potential + E_kinetic

plt.plot(E_potential)
plt.plot(E_kinetic)
plt.plot(E_total)
```

<!-- #region id="Hpwqr5uE0wNz" -->
Finally, we can see how this pans out in a corresponding animation.
<!-- #endregion -->

```python id="2oggaiRt0wNz"
# create position vectors for the first animated graph
data1 = []
for index,t in enumerate(allt): # step through every time step

    if index % 5 == 0: # take only every nth element
        position = df.x1[index]
        data1.append([[-4,position-1],[0,0]])
```

```python id="SCuPNq6_0wNz" outputId="4ed4899d-3655-4e29-ac8c-6cc7ee4ebd58"
# call the animation function
subplot1_data = [[np.asarray(data1)]]

thisfig = create_fig1()
add_massspring_patches(thisfig)

ani = animation.FuncAnimation(thisfig,animate1,frames=100,interval=50,fargs=(subplot1_data))
rc('animation', html='jshtml')
ani
```

<!-- #region id="roS9Jipm0wNz" -->
## 4. The pendulum as a harmonic oscillator
<!-- #endregion -->

```python id="9zmD27FF0wNz" outputId="ac349ecf-19c2-4c6d-ef5a-a8d465786003"
Image(url= "https://upload.wikimedia.org/wikipedia/commons/thumb/6/66/Pendulum_gravity.svg/200px-Pendulum_gravity.svg.png", width=200)
```

<!-- #region id="w5s3drYH0wNz" -->
A common variant of the harmonic oscillator introduced above is a pendulum.

The force balance of the pendulum is:

$$F = -mg\sin \theta = m l \ddot\theta$$

where m cancels out and L is the length of the pendulum; the Hamiltonian is:

$$H=\frac{p^2}{2mL^2}+mgL(1-\cos\theta)$$

We see that this looks quite different from the mass-spring system, mainly because of the trigonometric functions showing up here. To make the two systems more equivalent, often the pendulum is treated to be considered only at "small angles" i.e. near $0°$ where the sine function is almost linear as in $sin \theta \approx \theta$.

Then the force equation becomes:

$$\ddot\theta + \frac{g}{L} \theta = 0$$

And Hamiltonian:

$$H=\frac{p^2}{2mL^2}+\frac{1}{2}mgL\theta^2$$

What's important here is that compared to the oscillator, we have the following changes:

$$x \rightarrow L \theta$$
$$v = \dot x \rightarrow L \dot \theta$$
$$\frac{k}{m} \rightarrow \frac{g}{L} = \omega^2$$

Because of the equivalence, we will continue to make calculations from the perspective of the mass-spring system harmonic oscillator, but will at times use the pendulum picture to illustrate results. For purposes of visualization, we will sometimes go beyond the small angle range which will however not change results qualitatively.
<!-- #endregion -->

<!-- #region id="3hfgKFRf0wNz" -->
Note that before we had `omega = np.sqrt(k1/m1)` where `k1=0.5` and `m1=1` which meant $\omega^2 = 0.5$. If we want to get the same frequency for the pendulum compared to the mass-spring system, then we need to get the same value here. With a realistic `g=10` for gravity on earth, this would mean `L=20` which is not convenient to animate. So we will pretend to be on another planet where our g is smaller and set `g=1` and `L=2`, so that $\omega^2 = 0.5$.
<!-- #endregion -->

```python id="4tijeXyn0wNz"
g = 1
L = 2
```

<!-- #region id="6b-khhc50wNz" -->
Let's quickly create some vectors for x and v that we know correspond to the harmonic oscillator.
<!-- #endregion -->

```python id="iVm2zgtn0wNz"
omega = np.sqrt(g/L)

allx = 1*np.sin(omega*allt)
allv = omega*np.cos(omega*allt)
```

```python id="w_0yMl4O0wNz" outputId="29b611b7-b3b5-4951-edd8-22abe18ff494"
plt.plot(allt,allx)
plt.plot(allt,allv)
plt.grid()
```

```python id="DswniriP0wNz"
alltheta = allx
allsin = np.sin(alltheta)*L
allcos = np.cos(alltheta)*L
```

```python id="fzVy-XdU0wNz"
# create position vectors for the first animated graph
data1 = []
for index,t in enumerate(allt): # step through every time step

    if index % 5 == 0: # take only every nth element
        position = allx[index]
        data1.append([[0,0+allsin[index]],[2,2-allcos[index]]]) # first number: x anchor; second number: y anchor
```

```python id="ZXb6ZW--0wNz" outputId="2fb7d6b9-a39c-4a06-f5f5-05085f2534c2"
# call the animation function
subplot1_data = [[np.asarray(data1)]]

thisfig = create_fig1()
add_pendulum_patches(thisfig)

ani = animation.FuncAnimation(thisfig,animate1,frames=100,interval=50,fargs=(subplot1_data))
rc('animation', html='jshtml')
ani
```

<!-- #region id="pABAqOBq0wNz" -->
## 4. Coupled harmonic oscillators
<!-- #endregion -->

```python id="xoi7FzK30wNz" outputId="c9a393c3-210f-4478-f86f-43817f63f99a"
Image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Coupled_Harmonic_Oscillator.svg/500px-Coupled_Harmonic_Oscillator.svg.png")
```

<!-- #region id="nf7faTB20wN0" -->
We will now consider coupled harmonic oscillators, as in the picture above. We will call the couplings of the left and right springs k1 and k2, and the coupling in the center kc.

Let's again start with a consideration of forces:


$$F_1 = m \ddot x_1 = -k_1x_1 -k_c(x_1-x_2)$$

$$F_2 = m \ddot x_2 = -k_2x_2 -k_c(x_2-x_1)$$

Organizing in terms of x1 and x2 gives us:

$$\ddot x_1 = \dot v_1 = x_1 \frac{-k_1-k_c}{m_1} + x_2 \frac{k_c}{m_1}$$
$$\ddot x_2 = \dot v_2 = x_1 \frac{k_c}{m_2} + x_2 \frac{-k_2-k_c}{m_2}$$
<!-- #endregion -->

```python id="EzuRS7gk0wN0"
k1 = 0.5
k2 = 0.5
kc = 0.1
```

```python id="JEhjxf3K0wN0"
t_end = 150 # time for simulations
allt = np.arange(0,t_end,0.1) # time vector size 2000
```

```python id="eKn8QofG0wN0"
def get_next_state(present_state,dt):

    global m, ka, ks, kb

    x1 = present_state[0]
    v1 = present_state[1]
    x2 = present_state[2]
    v2 = present_state[3]

    dx1 = v1*dt
    nextx1 = x1 + dx1

    dx2 = v2*dt
    nextx2 = x2 + dx2

    dv1 = (nextx1 * -(k1+kc)/m1 + nextx2 * kc/m1)*dt
    nextv1 = v1 + dv1

    dv2 = (nextx1 * kc/m2 + nextx2 * -(k2+kc)/m2)*dt
    nextv2 = v2 + dv2

    next_state = np.array([nextx1,nextv1,nextx2,nextv2])
    return next_state
```

```python id="ZoFjtUae0wN0"
x1i = 1 #initial displacement m1
v1i = 0 #initial velocity m1
x2i = 0 #initial displacement m2
v2i = 0 #initial velocity m2
thisstate = np.array([x1i,v1i,x2i,v2i]) # initial state

dt = 0.1

allstates = []
for index,t in enumerate(allt): # step through every time step

    allstates.append(thisstate)

    nextstate = get_next_state(thisstate, dt)
    thisstate = nextstate

df = pd.DataFrame(np.row_stack(allstates), columns = ['x1', 'v1', 'x2', 'v2'])
df.insert(0, "time", allt, True)
```

```python id="4I2CXqD_0wN0" outputId="e9978592-31fb-4cc6-c55b-30c8cdf96f0a"
plt.subplot(211)
plt.plot(df.time,df.x1)
plt.subplot(212)
plt.plot(df.time,df.x2)
```

```python id="71UBYj_s0wN0"
# create position vectors for the first animated graph
data1,data2,data3 = ([],[],[])
for index,t in enumerate(allt): # step through every time step

    if index % 5 == 0: # take only every nth element
        position1 = df.x1[index]
        position2 = df.x2[index]
        data1.append([[-4,position1-2],[0,0]])
        data2.append([[4,position2+2],[0,0]])
        data3.append([[position1-2,position2+2],[0,0]])
```

```python id="ScxA59Jt0wN0"
# create vector of position function up to respective point for the second animated graph
data4,data5 = ([],[])
for index,t in enumerate(allt): # step through every time step

    if index % 5 == 0: # take only every nth element
        position1_sofar = df.x1[:index+1]
        position2_sofar = df.x2[:index+1]
        time_sofar = allt[:index+1]
        data4.append([time_sofar,position1_sofar])
        data5.append([time_sofar,position2_sofar])
```

```python id="VXdW56cb0wN0" outputId="41b949e2-bf32-4e6d-9af2-571c276055ea"
# call the animation function
subplot1_data = [np.asarray(data1),np.asarray(data2),np.asarray(data3)]
subplot2_data = [np.asarray(data4),np.asarray(data5)]

thisfig = create_fig2(50)
add_massspring_patches(thisfig)

ani = animation.FuncAnimation(thisfig,animate2,frames=100,interval=100,fargs=(subplot1_data,subplot2_data))
rc('animation', html='jshtml')
ani
```

<!-- #region id="kpxnejFq0wN0" -->
There is an alternative way of solving the above equations which gets us deeper into the intricacies of oscillatory systems.

Because the equations of motion are linear, we can expect them to be solved by a superposition of complex exponentials of the form:

$$x_1 = A e^{iωt}$$
$$x_2 = B e^{iωt}$$

For a solution in this form, we know about the second derivative that

$$\ddot x = -ω^2 x$$
<!-- #endregion -->

<!-- #region id="CyBR4CM90wN0" -->
This information allows us to reframe the above equations of motion into an eigenvalue problem:

$$\begin{pmatrix} \frac{(k_1+k_c)}{m} & -\frac{k_c}{m} \\ -\frac{k_c}{m} & \frac{(k_2 + k_c)}{m} \end{pmatrix} \begin{pmatrix} x_1 \\ x_2 \end{pmatrix}  = -ω^2 \begin{pmatrix} x_1 \\ x_2 \end{pmatrix} $$

which has solutions:

$$\omega^2 = \frac{k_b}{m},\frac{k_b+2k_s}{m}$$

Which corrspond to eigenvectors

$$\begin{pmatrix} x_1 \\ x_2 \end{pmatrix} = \mathinner|+\rangle =  \frac{1}{\sqrt{2}} \begin{pmatrix} 1 \\ 1 \end{pmatrix}$$

and

$$\begin{pmatrix} x_1 \\ x_2 \end{pmatrix} = \mathinner|-\rangle = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 \\ -1 \end{pmatrix}$$
<!-- #endregion -->

```python id="-gO35eoC0wN0"
m1 = m2 = 1
```

```python id="5OfjQgio0wN0"
M = np.array([
    [(k1+kc)/m1, -kc/m1],
    [-kc/m2, (k2+kc)/m2]
])

eigvals, eigvecs = linalg.eig(M)
```

```python id="nkdxJ0Sy0wN0" outputId="04826963-63a1-4d17-ae91-d9c60090dd9a"
eigvals
```

```python id="jOrUp_qF0wN0" outputId="44bc6bbc-16da-4ca1-e3ac-8a900a3f2ffa"
eigvecs
```

```python id="5l8Elddt0wN0"
dt = 0.1
```

```python id="aAOEwEVF0wN0"
x1i = 1 #initial displacement m1
v1i = 0 #initial velocity m1
x2i = 1 #initial displacement m2
v2i = 0 #initial velocity m2
thisstate = np.array([x1i,v1i,x2i,v2i]) # initial state

allstates = []
for index,t in enumerate(allt): # step through every time step

    allstates.append(thisstate)
    nextstate = get_next_state(thisstate, dt)
    thisstate = nextstate

df = pd.DataFrame(np.row_stack(allstates), columns = ['x1', 'v1', 'x2', 'v2'])
df.insert(0, "time", allt, True)
```

```python id="-AhsWXdw0wN1"
# create position vectors for the first animated graph
data1,data2,data3 = ([],[],[])
for index,t in enumerate(allt): # step through every time step

    if index % 5 == 0: # take only every nth element
        position1 = df.x1[index]
        position2 = df.x2[index]
        data1.append([[-4,position1-2],[0,0]])
        data2.append([[4,position2+2],[0,0]])
        data3.append([[position1-2,position2+2],[0,0]])
```

```python id="RiNnICx_0wN1" outputId="5b182e76-441f-48e3-c65b-77c2fb21d343"
# call the animation function
subplot1_data = [[np.asarray(data1),np.asarray(data2),np.asarray(data3)]]

thisfig = create_fig1()
add_massspring_patches(thisfig)

ani = animation.FuncAnimation(thisfig,animate1,frames=100,interval=50,fargs=(subplot1_data))
rc('animation', html='jshtml')
ani
```

<!-- #region id="N-WqfFCk0wN1" -->
Finally, we will revisit the pendulum picture.
<!-- #endregion -->

```python id="c9IE381A0wN1"

```

<!-- #region id="SiB2yPKQ0wN1" -->
## 5. Three coupled oscillators
<!-- #endregion -->

```python id="Zl4NrOGp0wN1"
# see http://spiff.rit.edu/classes/phys283/lectures/n_coupled/n_coupled.html
```

<!-- #region id="xqk-vpwR0wN1" -->
For three coupled oscillators we can go through the same motions. In this system, we assume that each oscillator has its own spring constant k1, k2, and k3 and in addition, we have two equal coupling constants k_c between the oscillators.

Starting with the force equations:

$$F_1 = m \ddot x_1 = -k_1x_1 -k_c(x_1-x_2)$$

$$F_2 = m \ddot x_2 = -k_2x_2 -k_c(x_2-x_1) -k_c(x_2-x_3)$$

$$F_3 = m \ddot x_2 = -k_3x_3 -k_c(x_3-x_2)$$
<!-- #endregion -->

<!-- #region id="aQhahk5T0wN1" -->
Organizing in terms of x1 and x2 gives us:

$$\ddot x_1 = \dot v_1 = x_1 \frac{-k_1-k_c}{m_1} + x_2 \frac{k_c}{m_1}$$
$$\ddot x_2 = \dot v_2 = x_1 \frac{k_c}{m_2} + x_2 \frac{-k_2-2k_c}{m_2} + x_3 \frac{k_c}{m_2}$$
$$\ddot x_3 = \dot v_3 = x_2 \frac{k_c}{m_3} + x_3 \frac{-k_3-k_c}{m_3}$$
<!-- #endregion -->

```python id="opZqTXLN0wN1"
def get_next_state_3CO(present_state,dt):

    global m1, m2, m3, k1, k1, k3, kc

    x1 = present_state[0]
    v1 = present_state[1]
    x2 = present_state[2]
    v2 = present_state[3]
    x3 = present_state[4]
    v3 = present_state[5]

    dx1 = v1*dt
    nextx1 = x1 + dx1

    dx2 = v2*dt
    nextx2 = x2 + dx2

    dx3 = v3*dt
    nextx3 = x3 + dx3

    dv1 = (nextx1 * -(k1+kc)/m1 + nextx2 * kc/m1)*dt
    nextv1 = v1 + dv1

    dv2 = (nextx1 * kc/m2 + nextx2 * -(k2+2.0*kc)/m2 + nextx3 * kc/m2)*dt
    nextv2 = v2 + dv2

    dv3 = (nextx2 * kc/m3 + nextx3 * -(k3+kc)/m3)*dt
    nextv3 = v3 + dv3

    next_state = np.array([nextx1,nextv1,nextx2,nextv2,nextx3,nextv3])
    return next_state
```

```python id="aHdXapcZ0wN1"
L1 = L2 = L3 = 1
```

```python id="yBBfTKW10wN1"
m3 = 1
k3 = 1
```

```python id="Y-LUYuV10wN1"
kc = 0.02
```

```python id="6oRuDuJB0wN1"
x1i = -1 #initial displacement m1
v1i = 0 #initial velocity m1
x2i = 0 #initial displacement m2
v2i = 0 #initial velocity m2
x3i = 0 #initial displacement m3
v3i = 0 #initial velocity m3
thisstate = np.array([x1i,v1i,x2i,v2i,x3i,v3i]) # initial state

allstates = []
for index,t in enumerate(allt): # step through every time step

    allstates.append(thisstate)

    nextstate = get_next_state_3CO(thisstate, dt)
    thisstate = nextstate

df = pd.DataFrame(np.row_stack(allstates), columns = ['x1', 'v1', 'x2', 'v2', 'x3', 'v3'])
df.insert(0, "time", allt, True)
```

```python id="XD9NbXri0wN1" outputId="d893d9d9-d3bd-4215-f814-6666a1c31e3c"
plt.subplot(311)
plt.plot(df.time,df.x1)
plt.subplot(312)
plt.plot(df.time,df.x2)
plt.subplot(313)
plt.plot(df.time,df.x3)
```

```python id="JcWQqbau0wN1"
df = add_polar_coordinates(df)
```

```python id="dTJplkgI0wN1"
# create position vectors for the first animated graph
data1,data2,data3,data4,data5 = ([],[],[],[],[])
for index,t in enumerate(allt): # step through every time step

    if index % 5 == 0: # take only every nth element
        data1.append([[-3,-3+df.sin1[index]],[2,2-df.cos1[index]]]) # first number: x anchor; second number: y anchor
        data2.append([[0,0+df.sin2[index]],[2,2-df.cos2[index]]]) # first number: x anchor; second number: y anchor
        data3.append([[3,3+df.sin3[index]],[2,2-df.cos3[index]]]) # first number: x anchor; second number: y anchor
        data4.append([[-3+df.sin1[index],0+df.sin2[index]],[2-df.cos1[index],2-df.cos2[index]]])
        data5.append([[0+df.sin2[index],3+df.sin3[index]],[2-df.cos2[index],2-df.cos3[index]]])
```

```python id="KiHzXYid0wN1" outputId="1a8cd951-cf7a-49ce-98d1-e47343e03fea"
# call the animation function
subplot1_data = [[np.asarray(data1),np.asarray(data2),np.asarray(data3),np.asarray(data4),np.asarray(data5)]]

thisfig = create_fig1()
add_pendulum_patches(thisfig)
lines1A[3].set_lw(1)
lines1A[4].set_lw(1)
lines1A[3].set_marker('')
lines1A[4].set_marker('')
lines1A[3].set_color('brown')
lines1A[4].set_color('brown')

ani = animation.FuncAnimation(thisfig,animate1,frames=100,interval=100,fargs=(subplot1_data))
rc('animation', html='jshtml')
ani
```

<!-- #region id="tjlIu3uZ0wN1" -->
## 6. A damped coupled system
<!-- #endregion -->

<!-- #region id="iq3KtraF0wN1" -->
Now let's look at a damped coupled system:
<!-- #endregion -->

<!-- #region id="R4qtfVH20wN1" -->
For three coupled oscillators we can go through the same motions. In this system, we assume that each oscillator has its own spring constant k1, k2, and k3 and in addition, we have two equal coupling constants k_c between the oscillators.

Starting with the force equations:

$$F_1 = m \ddot x_1 = -k_1x_1 -k_c(x_1-x_2) - c \dot {x_1}$$

$$F_2 = m \ddot x_2 = -k_2x_2 -k_c(x_2-x_1) -k_c(x_2-x_3) - c \dot {x_2}$$

$$F_3 = m \ddot x_2 = -k_3x_3 -k_c(x_3-x_2) - c \dot {x_3}$$
<!-- #endregion -->

<!-- #region id="IK6YemM20wN1" -->
$$\ddot {x} + c \dot {x} + \frac{k}{m} x = 0$$
<!-- #endregion -->

<!-- #region id="yCwbe7at0wN1" -->
Organizing in terms of x1 and x2 gives us:

$$\ddot x_1 = \dot v_1 = x_1 \frac{-k_1-k_c}{m_1} + x_2 \frac{k_c}{m_1} - c v_1$$
$$\ddot x_2 = \dot v_2 = x_1 \frac{k_c}{m_2} + x_2 \frac{-k_2-2k_c}{m_2} + x_3 \frac{k_c}{m_2} - c v_2$$
$$\ddot x_3 = \dot v_3 = x_2 \frac{k_c}{m_3} + x_3 \frac{-k_3-k_c}{m_3} - c v_3$$
<!-- #endregion -->

```python id="cW_Ufa390wN1" outputId="f1304a42-d11f-4e06-cfd3-ce9f850bdfcc"
c
```

```python id="HKoYhcvb0wN2"
def get_next_state_3CD(present_state,dt):

    global m1, m2, m3, k1, k1, k3, kc

    x1 = present_state[0]
    v1 = present_state[1]
    x2 = present_state[2]
    v2 = present_state[3]
    x3 = present_state[4]
    v3 = present_state[5]

    dx1 = v1*dt
    nextx1 = x1 + dx1

    dx2 = v2*dt
    nextx2 = x2 + dx2

    dx3 = v3*dt
    nextx3 = x3 + dx3

    dv1 = (nextx1 * -(k1+kc)/m1 + nextx2 * kc/m1 - v1*c)*dt
    nextv1 = v1 + dv1

    dv2 = (nextx1 * kc/m2 + nextx2 * -(k2+2.0*kc)/m2 + nextx3 * kc/m2 - v2*c)*dt
    nextv2 = v2 + dv2

    dv3 = (nextx2 * kc/m3 + nextx3 * -(k3+kc)/m3 - v3*c)*dt
    nextv3 = v3 + dv3

    next_state = np.array([nextx1,nextv1,nextx2,nextv2,nextx3,nextv3])
    return next_state
```

```python id="71BSFXRx0wN2"
c = 0.02
kc = 0.02
```

```python id="zErMJHAI0wN2"

```

```python id="l8AyuwhQ0wN2" outputId="6eb21726-c90e-4c8f-c9b2-3406a26208c3"
x1i = 0 #initial displacement m1
v1i = 0 #initial velocity m1
x2i = -1 #initial displacement m2
v2i = 0 #initial velocity m2
x3i = 0 #initial displacement m3
v3i = 0 #initial velocity m3
thisstate = np.array([x1i,v1i,x2i,v2i,x3i,v3i]) # initial state

allstates = []
for index,t in enumerate(allt): # step through every time step

    allstates.append(thisstate)

    nextstate = get_next_state_3CO(thisstate, dt)
    thisstate = nextstate

df = pd.DataFrame(np.row_stack(allstates), columns = ['x1', 'v1', 'x2', 'v2', 'x3', 'v3'])
df.insert(0, "time", allt, True)

plt.subplot(311)
plt.plot(df.time,df.x1)
plt.subplot(312)
plt.plot(df.time,df.x2)
plt.subplot(313)
plt.plot(df.time,df.x3)
```

```python id="omZ_ISss0wN2"
# how to know when one is on resonance with three pendula?
```

```python id="w9ozPKGB0wN2"
df = add_polar_coordinates(df)

# create position vectors for the first animated graph
data1,data2,data3,data4,data5 = ([],[],[],[],[])
for index,t in enumerate(allt): # step through every time step

    if index % 5 == 0: # take only every nth element
        data1.append([[-3,-3+df.sin1[index]],[2,2-df.cos1[index]]]) # first number: x anchor; second number: y anchor
        data2.append([[0,0+df.sin2[index]],[2,2-df.cos2[index]]]) # first number: x anchor; second number: y anchor
        data3.append([[3,3+df.sin3[index]],[2,2-df.cos3[index]]]) # first number: x anchor; second number: y anchor
        data4.append([[-3+df.sin1[index],0+df.sin2[index]],[2-df.cos1[index],2-df.cos2[index]]])
        data5.append([[0+df.sin2[index],3+df.sin3[index]],[2-df.cos2[index],2-df.cos3[index]]])
```

```python id="QHaT4ZrB0wN2" outputId="3f2e4f0c-7171-4e59-cee7-df7b1d5dc83b"
# call the animation function
subplot1_data = [[np.asarray(data1),np.asarray(data2),np.asarray(data3),np.asarray(data4),np.asarray(data5)]]

thisfig = create_fig1()
add_pendulum_patches(thisfig)
lines1A[3].set_lw(1)
lines1A[4].set_lw(1)
lines1A[3].set_marker('')
lines1A[4].set_marker('')
lines1A[3].set_color('brown')
lines1A[4].set_color('brown')

ani = animation.FuncAnimation(thisfig,animate1,frames=100,interval=100,fargs=(subplot1_data))
rc('animation', html='jshtml')
ani
```

<!-- #region id="DfrOHApH0wN2" -->
To understand where resonances occur, we need to find the modes and eigen frequencies of the system.
<!-- #endregion -->

<!-- #region id="AK6cza9l0wN2" -->
$$\begin{pmatrix} \frac{-(k_1+k_c)}{m_1} & \frac{k_c}{m_1} & 0 \\ \frac{k_c}{m_2} & \frac{-(k_2 + 2k_c)}{m_2} & \frac{k_c}{m_2} \\ 0 & \frac{k_c}{m_3} & \frac{-(k_3+k_c)}{m_3} \end{pmatrix} \begin{pmatrix} x_1 \\ x_2 \\ x_3 \end{pmatrix}  = -ω^2 \begin{pmatrix} x_1 \\ x_2 \\ x_3 \end{pmatrix} $$
<!-- #endregion -->

```python id="UPhRcCIn0wN2"
M = np.array([
    [-(k1+kc)/m1, kc/m1, 0],
    [kc/m2, -(k2+2*kc)/m2, kc/m2],
    [0, kc/m3, -(k3+kc)/m3]
])

M = M*-1

eigvals, eigvecs = linalg.eig(M)
```

```python id="_Wyz45_30wN2" outputId="4b0b5942-ac59-478c-df6c-7bd1c48db881"
eigvals
```

```python id="hfJrVVZK0wN2" outputId="cc0368c6-5e1d-4e73-a49c-e81cd6c5fb7b"
eigvecs
```

<!-- #region id="X4f23NPo0wN2" -->
## Appendix A: Another note on vibrations, atoms and nuclei
<!-- #endregion -->

```python id="DXoogX2c0wN2" outputId="10fd858c-d231-47b5-b918-af430b6ff81a"
from IPython.display import HTML

# Youtube
#HTML('<iframe width="560" height="315" src="https://www.youtube.com/embed/S_f2qV2_U00?rel=0&amp;controls=0&amp;showinfo=0" frameborder="0" allowfullscreen></iframe>')
HTML('<iframe width="560" height="315" src="https://www.youtube.com/embed/pCWbb4c2SEo?rel=0&showinfo=0" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>')

```

<!-- #region id="EETRzkg20wN2" -->
## Appendix B: Comparison of numerical time evolution algorithms
<!-- #endregion -->

<!-- #region id="-QRzwXF70wN2" -->
Finally a comparison between Euler-Cohen and Runge-Kutta 4:
<!-- #endregion -->

```python id="1pVvE-OZ0wN2"
def derivs_2m(s):
    x1=s[0]
    v1=s[1]
    x2=s[2]
    v2=s[3]
    a1 = (-(k1+kc)*x1 + kc*x2)/m1
    a2 = (-(k2+kc)*x2 + kc*x1)/m2
    return np.array([v1, a1, v2, a2])

def get_next_stateRK4(present_state, dt):
    """
    Take a single RK4 step.
    """
    f1 = derivs_2m(present_state)
    f2 = derivs_2m(present_state+f1*dt/2.0)
    f3 = derivs_2m(present_state+f2*dt/2.0)
    f4 = derivs_2m(present_state+f3*dt)
    return present_state + (f1+2*f2+2*f3+f4)*dt/6.0
```

```python id="2KqLd4hJ0wN2"
x1i = 1 #initial displacement m1
v1i = 0 #initial velocity m1
x2i = 0 #initial displacement m2
v2i = 0 #initial velocity m2
thisstate = np.array([x1i,v1i,x2i,v2i]) # initial state

dt = 0.1

allstates = []
for index,t in enumerate(allt): # step through every time step

    allstates.append(thisstate)

    nextstate = get_next_state(thisstate, dt)
    thisstate = nextstate

df1 = pd.DataFrame(np.row_stack(allstates), columns = ['x1', 'v1', 'x2', 'v2'])
df1.insert(0, "time", allt, True)
```

```python id="CAJN53kw0wN2"
x1i = 1 #initial displacement m1
v1i = 0 #initial velocity m1
x2i = 0 #initial displacement m2
v2i = 0 #initial velocity m2
thisstate = np.array([x1i,v1i,x2i,v2i]) # initial state

dt = 0.1

allstates = []
for index,t in enumerate(allt): # step through every time step

    allstates.append(thisstate)

    nextstate = get_next_stateRK4(thisstate, dt)
    thisstate = nextstate

df2 = pd.DataFrame(np.row_stack(allstates), columns = ['x1', 'v1', 'x2', 'v2'])
df2.insert(0, "time", allt, True)
```

```python id="SsHskwUT0wN2" outputId="5f8c1a11-7692-47d7-e775-b2b45e26c75f"
plt.plot(df1.time,df1.x1)
plt.plot(df2.time,df2.x1)
```

<!-- #region id="J7yasGoJ0wN2" -->
## Appendix C: Does excitation transfer happen even if the coupling is very weak?
<!-- #endregion -->

```python id="6pTjWp5_0wN2"
t_end = 4000 # time for simulations
allt = np.arange(0,t_end,0.1) # time vector size 2000
dt = 0.1
```

```python id="1k1EmNdO0wN2"
kc = 0.001

x1i = 1 #initial displacement m1
v1i = 0 #initial velocity m1
x2i = 0 #initial displacement m2
v2i = 0 #initial velocity m2
thisstate = np.array([x1i,v1i,x2i,v2i]) # initial state

allstates = []
for index,t in enumerate(allt): # step through every time step

    allstates.append(thisstate)

    nextstate = get_next_stateRK4(thisstate, dt)
    thisstate = nextstate

df2 = pd.DataFrame(np.row_stack(allstates), columns = ['x1', 'v1', 'x2', 'v2'])
df2.insert(0, "time", allt, True)
```

```python id="zfUhHhh70wN3" outputId="f0c1cc78-9949-4b3f-a44a-6d0a2f88a9de"
plt.plot(df2.x1)
```

```python id="uuZ2kfrn0wN3" outputId="737f6f99-65cb-4ad0-dd3a-8014684b0aa0"
plt.plot(df2.x1[20000:25000])
```

```python id="jvAEwVQW0wN3"

```

```python id="dbMmZ6wN0wN3"

```
