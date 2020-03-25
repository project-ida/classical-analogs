---
jupyter:
  jupytext:
    formats: ipynb,src//md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.4.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# A classical analog to demonstrate decay rate enhancement via couplings


In this tutorial we will create a classical analog to illustrate how decay rates can be enhanced through couplings and resulting excitation transfer. 


## 1. Python helper functions

```python
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

```python
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

```python
def animate1(i,data_graph1): # animation function that is called once for every frame (for fig 1)
    global lines1A
    
    for lineno,_ in enumerate(data_graph1): # strep through the list and the data for each line and assign data for frame i
        
        lines1A[lineno].set_xdata(data_graph1[lineno][i][0]) 
        lines1A[lineno].set_ydata(data_graph1[lineno][i][1])
```

```python
def add_pendulum_patches(thisfigure):
    
    thisaxis = thisfigure.axes[0]
    
    thisaxis.add_patch(patches.Rectangle(xy=(-5, 2),width=10,height=1,facecolor="gainsboro"))
```

```python
def add_polar_coordinates(df):
    alltheta1 = df.x1
    df['sin1'] = np.sin(alltheta1)*L1
    df['cos1']  = np.cos(alltheta1)*L1
    alltheta2 = df.x2
    df['sin2'] = np.sin(alltheta2)*L2
    df['cos2'] = np.cos(alltheta2)*L2
    alltheta3 = df.x3
    df['sin3'] = np.sin(alltheta3)*L3
    df['cos3'] = np.cos(alltheta3)*L3   
    return df
```

## 2. REVIEW: a damped coupled system


Here is a brief review of the last part of the previous notebook:


For three coupled oscillators we can get the equations of motion as follows. In this system, we assume that each oscillator has its own spring constant k1, k2, and k3 and in addition, we have two equal coupling constants k_c between the oscillators. 

Starting with the force equations:

$$F_1 = m \ddot x_1 = -k_1x_1 -k_c(x_1-x_2) - c \dot {x_1}$$

$$F_2 = m \ddot x_2 = -k_2x_2 -k_c(x_2-x_1) -k_c(x_2-x_3) - c \dot {x_2}$$

$$F_3 = m \ddot x_2 = -k_3x_3 -k_c(x_3-x_2) - c \dot {x_3}$$


$$\ddot {x} + c \dot {x} + \frac{k}{m} x = 0$$


Organizing in terms of x1 and x2 gives us:

$$\ddot x_1 = \dot v_1 = x_1 \frac{-k_1-k_c}{m_1} + x_2 \frac{k_c}{m_1} - c v_1$$
$$\ddot x_2 = \dot v_2 = x_1 \frac{k_c}{m_2} + x_2 \frac{-k_2-2k_c}{m_2} + x_3 \frac{k_c}{m_2} - c v_2$$
$$\ddot x_3 = \dot v_3 = x_2 \frac{k_c}{m_3} + x_3 \frac{-k_3-k_c}{m_3} - c v_3$$


From this, we can create the time evolution function based on the Euler-Cromer method:

```python
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

## 4. Creating the illustration


## 4.1. Setting the main parameters

```python
t_end = 250 # time for simulations
allt = np.arange(0,t_end,0.1) # time vector size 2000
```

```python
dt = 0.1
```

```python
k1 = k2 = k3 = 1 
m1 = m2 = m3 = 1
c = 0.02
kc = 0.1 #0.5
```

```python
L1 = L3 = 1
L2 = 2
```

```python
x1i = 0 #initial displacement m1
v1i = 0 #initial velocity m1
x2i = -1 #initial displacement m2
v2i = 0 #initial velocity m2
x3i = 0 #initial displacement m3
v3i = 0 #initial velocity m3
```

## 4.2. Uncoupled damped oscillator

```python
kc = 0
```

```python
thisstate = np.array([x1i,v1i,x2i,v2i,x3i,v3i]) 
thisstate
```

```python
allstates = []
for index,t in enumerate(allt): # step through every time step

    allstates.append(thisstate)    
    nextstate = get_next_state_3CD(thisstate, dt)  
    thisstate = nextstate    
    
df1 = pd.DataFrame(np.row_stack(allstates), columns = ['x1', 'v1', 'x2', 'v2', 'x3', 'v3']) 
df1.insert(0, "time", allt, True) 
df1 = add_polar_coordinates(df1)

plt.subplot(311)
plt.plot(df1.x1)
plt.subplot(312)
plt.plot(df1.x2)
plt.subplot(313)
plt.plot(df1.x3)
```

```python
df1[310:320] # index 314 is a peak for x2
```

```python
dfA = df1[:315]
dfA
```

```python
offset1 = len(dfA)-1
x2i_newB = dfA.x2[offset1]
x2i_newB # this will be the new initial condition for the next section
```

```python
df = df1

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

```python
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

ani = animation.FuncAnimation(thisfig,animate1,frames=200,interval=100,fargs=(subplot1_data))
rc('animation', html='jshtml')
ani        
```

## 4.3. Coupled damped oscillator

```python
kc = 0.1 # now rerun with the coupling and with the new initial condition
```

```python
thisstate = np.array([x1i,v1i,x2i_newB,v2i,x3i,v3i])
thisstate
```

```python
allstates = []
for index,t in enumerate(allt): # step through every time step

    allstates.append(thisstate)    
    nextstate = get_next_state_3CD(thisstate, dt)  
    thisstate = nextstate    
    
df2 = pd.DataFrame(np.row_stack(allstates), columns = ['x1', 'v1', 'x2', 'v2', 'x3', 'v3']) 
df2.insert(0, "time", allt, True) 
df2 = add_polar_coordinates(df2)

plt.subplot(311)
plt.plot(df2.x1)
plt.subplot(312)
plt.plot(df2.x2)
plt.subplot(313)
plt.plot(df2.x3)
```

```python
plt.plot(df2.x1[:500])
plt.plot(df2.x2[:500])
np.max(df2temp.x1[:500]) # index 220, value 0.390761
df2temp[215:225] 
```

```python
dfB = df2[:221]
dfB
```

```python
offset2 = len(dfB)-1
x2i_newC = dfB.x2[offset2]
x2i_newC # this will be the new initial condition for the next section
```

```python
df = df2

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

```python
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

ani = animation.FuncAnimation(thisfig,animate1,frames=200,interval=100,fargs=(subplot1_data))
rc('animation', html='jshtml')
ani        
```

## 4.4. Uncoupled damped oscillator

```python
kc = 0
c = 0.2
```

```python
thisstate = np.array([x1i,v1i,x2i_newC,v2i,x3i,v3i])
thisstate
```

```python
allstates = []
for index,t in enumerate(allt): # step through every time step

    allstates.append(thisstate)    
    nextstate = get_next_state_3CD(thisstate, dt)  
    thisstate = nextstate    
    
df3 = pd.DataFrame(np.row_stack(allstates), columns = ['x1', 'v1', 'x2', 'v2', 'x3', 'v3']) 
df3.insert(0, "time", allt, True) 
df3 = add_polar_coordinates(df3)

plt.subplot(311)
plt.plot(df1.time,df3.x1)
plt.subplot(312)
plt.plot(df1.time,df3.x2)
plt.subplot(313)
plt.plot(df1.time,df3.x3)
```

```python
dfC = df3
```

```python
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

```python
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

ani = animation.FuncAnimation(thisfig,animate1,frames=200,interval=100,fargs=(subplot1_data))
rc('animation', html='jshtml')
ani        
```

## 4.5. Bringing it all together

```python
df_all = pd.concat([dfA, dfB, dfC], ignore_index=True, sort=False)
```

```python
df_all
```

```python
plt.plot(df_all.x2[:1000])
plt.plot(df_all.x1[:1000])
#plt.plot(df_all.x3[:1000])
```

## 4.6. Final touches (e.g. incoherent decay of receiver states)

```python
df_all['anchor1x'] = -3
df_all['anchor1y'] = 2
df_all['anchor3x'] = 3
df_all['anchor3y'] = 2
```

```python
df_all
```

```python
offset = offset1+offset2+2-10
offset
```

```python
df_all[offset-5:offset+5]
```

```python
remainingrows = df_all.sin1[offset:].shape[0]
remainingrows
```

```python
newsin1 = 0.380893 * np.linspace(1, 200, num=remainingrows)
newsin1.shape
```

```python
newcos1 = 0.924619 * np.linspace(1, 200, num=remainingrows)
newcos1.shape
```

```python
newsin3 = 0.380893 * np.linspace(1, 200, num=remainingrows)
newsin3.shape
```

```python
newcos3 = 0.924619 * np.linspace(1, 200, num=remainingrows)
newcos3.shape
```

```python
df_all.sin1[offset:] = 0 
df_all.cos1[offset:] = 0 
df_all.sin3[offset:] = 0 
df_all.cos3[offset:] = 0 
```

```python
df_all.anchor1x[offset:] = -3+newsin1
df_all.anchor1y[offset:] = 2-newcos1
df_all.anchor3x[offset:] = 3+newsin3
df_all.anchor3y[offset:] = 2-newcos3
```

## 4.7. Creating the final animations

```python
df = df_all

# create position vectors for the first animated graph
data1,data2,data3,data4,data5 = ([],[],[],[],[])
for index,t in enumerate(allt): # step through every time step

    if index % 5 == 0: # take only every nth element    
        data1.append([[df.anchor1x[index],df.anchor1x[index]+df.sin1[index]],[df.anchor1y[index],df.anchor1y[index]-df.cos1[index]]]) 
        data2.append([[0,0+df.sin2[index]],[2,2-df.cos2[index]]]) # first number: x anchor; second number: y anchor 
        data3.append([[df.anchor3x[index],df.anchor3x[index]+df.sin1[index]],[df.anchor3y[index],df.anchor3y[index]-df.cos1[index]]]) 
        data4.append([[-3+df.sin1[index],0+df.sin2[index]],[2-df.cos1[index],2-df.cos2[index]]]) 
        data5.append([[0+df.sin2[index],3+df.sin3[index]],[2-df.cos2[index],2-df.cos3[index]]]) 
```

```python
def animate1special(i,data_graph1): # animation function that is called once for every frame (for fig 1)
    global lines1A,offset1,offset2,thisfig
      
    if i == int(offset1/5):
        lines1A[3].set_lw(1)
        lines1A[4].set_lw(1)          
        thisaxis = thisfig.axes[0]
        thisaxis.add_patch(patches.Rectangle(xy=(-5, -3),width=10,height=5,facecolor="red",alpha=0.05))

    if i == int((offset1+offset2)/5):
        lines1A[3].set_lw(0)
        lines1A[4].set_lw(0)           
        
    for lineno,_ in enumerate(data_graph1): # strep through the list and the data for each line and assign data for frame i
        
        lines1A[lineno].set_xdata(data_graph1[lineno][i][0]) 
        lines1A[lineno].set_ydata(data_graph1[lineno][i][1])
```

```python
# call the animation function 
subplot1_data = [[np.asarray(data1),np.asarray(data2),np.asarray(data3),np.asarray(data4),np.asarray(data5)]]

thisfig = create_fig1()
add_pendulum_patches(thisfig)
lines1A[3].set_lw(0)
lines1A[4].set_lw(0)
lines1A[3].set_marker('')
lines1A[4].set_marker('')
lines1A[3].set_color('brown')
lines1A[4].set_color('brown')

ani = animation.FuncAnimation(thisfig,animate1special,frames=200,interval=100,fargs=(subplot1_data)) # blit=True,
rc('animation', html='jshtml')
ani        
```

```python
plt.subplot(311)
plt.plot(df_all.x1[:2500])
plt.subplot(312)
plt.plot(df_all.x2[:2500])
plt.subplot(313)
plt.plot(df_all.x3[:2500])
```

```python
df = df1

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

```python
# call the animation function 
subplot1_data = [[np.asarray(data1),np.asarray(data2),np.asarray(data3),np.asarray(data4),np.asarray(data5)]]

thisfig = create_fig1()
add_pendulum_patches(thisfig)
lines1A[3].set_lw(0)
lines1A[4].set_lw(0)
lines1A[3].set_marker('')
lines1A[4].set_marker('')
lines1A[3].set_color('brown')
lines1A[4].set_color('brown')

ani = animation.FuncAnimation(thisfig,animate1,frames=250,interval=100,fargs=(subplot1_data)) # blit=True,
rc('animation', html='jshtml')
ani        
```

```python
plt.plot(df1.x2)
plt.plot(df_all.x2[:2500])
```

```python

```
