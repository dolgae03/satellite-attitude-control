from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import math
# Inital setting

def M(t):
    Mx = 0.2*math.sin(2*math.pi/300*t + 0.5*math.pi) + 0.1*math.cos(2*math.pi/200 * t + math.pi)
    My = 0.05*math.sin(2*math.pi/500*t + 1.5*math.pi) + 0.1*math.cos(2*math.pi/400 * t + 0.5*math.pi)
    Mz = 0.01*math.sin(2*math.pi/300*t) #양수의 토크를 주었더니 시계방향으로 회전(-축으로 회전)

    return np.array([[Mx],[My],[Mz]])


def f_v(t, w):
    trans_w = np.transpose(w)
    trans_h = np.transpose(I@w)
    res = np.cross(trans_w, trans_h)

    return I_inv@(M(t) - np.transpose(res))

def f_q(t, q, w):
    w1, w2, w3 = np.transpose(w)[0]
    omega = np.array([[0, w3, -w2, w1],
                      [-w3, 0, w1, w2],
                      [w2, -w1, 0, w3],
                      [-w1, -w2, -w3, 0]])
    
    return 1/2*omega@q

def e_t_q(li):
    a, b, c = li
    a, b, c = a/2, b/2, c/2

    return np.array([[math.sin(a)*math.cos(b)*math.cos(c) + math.cos(a)*math.sin(b)*math.sin(c)],
                     [-math.sin(a)*math.cos(b)*math.sin(c) + math.cos(a)*math.sin(b)*math.cos(c)],
                     [math.sin(a)*math.sin(b)*math.cos(c) + math.cos(a)*math.cos(b)*math.sin(c)],
                     [-math.sin(a)*math.sin(b)*math.sin(c) + math.cos(a)*math.cos(b)*math.cos(c)]])

def q_t_e(li):
    q1, q2, q3, q4 = li

    return np.array([[math.atan(4*(q2*q3+q1*q4)/(-(q1**2)-(q2**2)+(q3**2)+(q4**2)))],
                     [math.acos(-2*(q1*q3-q2*q4))],
                     [math.atan(4*(q1*q2+q3*q4)/((q1**2)-(q2**2)-(q3**2)+(q4**2)))]])

def q_t_d(li):
    q1, q2, q3, q4 = np.transpose(li)[0]

    return np.array([[(q4**2 + q1**2 -q2**2-q3**2), 2*(q1*q2+q3*q4), 2*(q1*q3-q2*q4)],
                     [2*(q1*q2-q3*q4), (q4**2-q1**2 + q2**2 - q3**2), 2*(q2*q3 + q1*q4)],
                     [2*(q1*q3+q2*q4), 2*(q2*q3 - q1*q4), (q4**2 - q1**2 - q2**2 + q3**2)]]) 


def rungekutha(t, w, q, h):
    k1 = h * f_v(t, w)
    k2 = h * f_v((t+h/2), (w + k1/2))
    k3 = h * f_v((t+h/2), (w + k2/2))
    k4 = h * f_v((t+h), (w + k3))
    
    new_w = w + (k1 + 2*k2 + 2*k3 + k4) / 6

    k1 = h * f_q(t, q, w)
    k2 = h * f_q((t+h/2), (q + k1/2), w)
    k3 = h * f_q((t+h/2), (q + k2/2), w)
    k4 = h * f_q((t+h), (q + k3), w)
    
    new_q = q + (k1 + 2*k2 + 2*k3 + k4) / 6

    return new_w, new_q

def unveal(li):
    new_li = []
    for i in li:
        new_li.append(i[0])

    return new_li


I = np.array([[128.22, 0.26, -0.11],[0.26, 131.81, 0.74],[-0.11, 0.74, 71.93]])
#I = np.array([[100,0,0],[0,100,0],[0,0,100]])
I_inv = np.linalg.inv(I)

t = 0
tend = 300
h = 0.01
time = [0]

inital_dcm = q_t_d(e_t_q(np.array([[math.pi*30/180],[0],[math.pi*20/180]])))
a = np.array([[1,0,0],[0,1,0],[0,0,1]])

quaternion = [e_t_q(np.array([[math.pi*30/180],[0],[math.pi*20/180]]))]
anguler_velocity = [np.array([[math.pi*0.1/180],[0.0],[0.0]])]
dcm = []

#implementation

while time[-1] < tend:
    t = time[-1]
    w = anguler_velocity[-1]
    q = quaternion[-1]

    #print()

    new_w, new_q = rungekutha(t, w, q, h)
    q_norm = (new_q[0] ** 2 + new_q[1] ** 2 + new_q[2] ** 2 + new_q[3] ** 2)**(1/2)
    new_q = new_q/q_norm

    anguler_velocity.append(new_w)
    quaternion.append(new_q)
    dcm.append(q_t_d(new_q))
    time.append(t+h)

'''
anguler_velocity and qurter
'''    

fig, ax = plt.subplots(3,1,figsize=(15, 7), layout='constrained')

new_av = []

for ele in anguler_velocity:
    new_av.append(ele * 180 / math.pi)


wx, wy, wz = zip(*(new_av))
wx, wy, wz = unveal(wx), unveal(wy), unveal(wz)

ax[0].plot(time, wx, label = 'w_x')
ax[0].plot(time, wy, label = 'w_y')
ax[0].plot(time, wz, label = 'w_z')
ax[0].set_xlabel('time(s)')  # Add an x-label to the axes.
ax[0].set_ylabel('anguler velocity(deg/s)')  # Add a y-label to the axes.
ax[0].set_title("w - t")  # Add a title to the axes.
ax[0].legend();  # Add a legend.
ax[0].set_xlim(0,300)

q1, q2, q3, q4 = zip(*quaternion)

ax[1].plot(time, q1, label = 'q1')
ax[1].plot(time, q2, label = 'q2')
ax[1].plot(time, q3, label = 'q3')
ax[1].plot(time, q4, label = 'q4')
ax[1].set_xlabel('time(s)')  # Add an x-label to the axes.
ax[1].set_ylabel('quaternion(dimesionless)')  # Add a y-label to the axes.
ax[1].set_title("q - t")  # Add a title to the axes.
ax[1].legend()
ax[1].set_xlim(0,300)
ax[1].set_ylim(-1,1)

plt.savefig('w and q')

def animate(dc):
    VecStart = np.array([[0,0,0],[0,0,0],[0,0,0]])
    j2k = np.array([[1,0,0],[0,1,0],[0,0,1]])
    VecEnd = np.transpose(dc) @ j2k


    for i in range(3):
        line3d[i].set_data_3d([VecStart[0][i], VecEnd[0][i]], [VecStart[1][i],VecEnd[1][i]] , [VecStart[2][i],VecEnd[2][i]])


fig_ani = plt.figure()
ax_ani = fig_ani.add_subplot(111, projection='3d')

ax_ani.set_xlim(-1,1)
ax_ani.set_ylim(-1,1)
ax_ani.set_zlim(-1,1)
ax_ani.set_xticks([])
ax_ani.set_yticks([])
ax_ani.set_zticks([])
ax_ani.set_xlabel("X")
ax_ani.set_ylabel("Y")
ax_ani.set_zlabel("Z")
ax_ani.view_init(30,-45)

line3d = []
col = ['red','blue','green']
lab = ['x','y','z']
for i in range(3):
    line3d.append(ax_ani.plot([0, 0], [0, 0] ,zs = [0, 0], color = col[i], label = lab[i])[0])

anim = FuncAnimation(fig_ani, animate, frames=dcm[::10], interval = 10)
#anim.save('trajectory.gif', writer='imagemagick', fps=30, dpi=100)

plt.show()