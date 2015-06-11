from matplotlib import pyplot as plt
import numpy as np
from scipy.integrate import odeint
from IPython.html.widgets import interact, interactive, fixed
from IPython.html import widgets
from IPython.display import display
from IPython.display import Image,HTML,SVG

gamma=4.4936287*(10**(-8))
def derivs(rvec,t,M,S):
    rx=rvec[0]
    ry=rvec[1]
    vx=rvec[2]
    vy=rvec[3]
    Rx=rvec[4]
    Ry=rvec[5]
    Vx=rvec[6]
    Vy=rvec[7]
    
    R=np.sqrt(Rx**2+Ry**2)
    r=np.sqrt(rx**2+ry**2)
    rho = np.sqrt((Rx-rx)**2 + (Ry-ry)**2)
    
    drx=vx
    dry=vy
    dvx=-gamma*((M/(r**3))*rx+(S/(R**3))*Rx-(S/(rho**3))*(Rx-rx))
    dvy=-gamma*((M/(r**3))*ry+(S/(R**3))*Ry-(S/(rho**3))*(Ry-ry))
    dRx=Vx
    dRy=Vy
    dVx=-gamma*((M+S)/(R**3))*Rx
    dVy=-gamma*((M+S)/(R**3))*Ry
    return np.array([drx,dry,dvx,dvy,dRx,dRy,dVx,dVy])

def create_ic_stars(rmin,s):
    ics = []
    
    # loop
    for n in range(0,36,1):
        vr=np.sqrt(4493.6287/(0.6*rmin))
        x=(0.6*rmin*np.cos(np.pi*2*n/36))
        y=(0.6*rmin*np.sin(np.pi*2*n/36))
        vx=(vr*np.sin(np.pi*2*n/36+np.pi))
        vy=(vr*np.cos(np.pi*2*n/36))
        ic = np.array([x,y,vx,vy])
        ics.append(ic)
    for n in range(0,30,1):
        vr=np.sqrt(4493.6287/(0.5*rmin))
        x=(0.5*rmin*np.cos(np.pi*2*n/30))
        y=(0.5*rmin*np.sin(np.pi*2*n/30))
        vx=(vr*np.sin(np.pi*2*n/30+np.pi))
        vy=(vr*np.cos(np.pi*2*n/30))
        ic = np.array([x,y,vx,vy])
        ics.append(ic)
    for n in range(0,24,1):
        vr=np.sqrt(4493.6287/(0.4*rmin))
        x=(0.4*rmin*np.cos(np.pi*2*n/24))
        y=(0.4*rmin*np.sin(np.pi*2*n/24))
        vx=(vr*np.sin(np.pi*2*n/24+np.pi))
        vy=(vr*np.cos(np.pi*2*n/24))
        ic = np.array([x,y,vx,vy])
        ics.append(ic)
    for n in range(0,18,1):
        vr=np.sqrt(4493.6287/(0.3*rmin))
        x=(0.3*rmin*np.cos(np.pi*2*n/18))
        y=(0.3*rmin*np.sin(np.pi*2*n/18))
        vx=(vr*np.sin(np.pi*2*n/18+np.pi))
        vy=(vr*np.cos(np.pi*2*n/18))
        ic = np.array([x,y,vx,vy])
        ics.append(ic)
    for n in range(0,12,1):
        vr=np.sqrt(4493.6287/(0.2*rmin))
        x=(0.2*rmin*np.cos(np.pi*2*n/12))
        y=(0.2*rmin*np.sin(np.pi*2*n/12))
        vx=(vr*np.sin(np.pi*2*n/12+np.pi))
        vy=(vr*np.cos(np.pi*2*n/12))
        ic = np.array([x,y,s*vx,s*vy])
        ics.append(ic)

    return ics

def create_ic_gala(Sy,rmin,M,S):
    Sx=-(1/(4*rmin))*Sy**2+rmin
    vr=np.sqrt(2*gamma*(M+S)/(np.sqrt(Sx**2+Sy**2)))
    angle=np.arctan(2*rmin/Sy)
    Vx=vr*np.cos(angle)
    Vy=-vr*np.sin(angle)
    return np.array([Sx,Sy,Vx,Vy])

def solve_one_star(ic_star, ic_gala, tmax, ntimes,M,S):
    t = np.linspace(0,tmax,ntimes)
    ic = np.hstack([ic_star, ic_gala])
    soln = odeint(derivs, ic, t,(M,S))
    return soln

def solve_all_stars(ic_stars, ic_gala, tmax, ntimes,M,S):
    solns = []
    for ic_star in ic_stars:
        soln = solve_one_star(ic_star, ic_gala, tmax, ntimes,M,S)
        solns.append(soln)
    return solns

ic_stars1 = create_ic_stars(25,1)
ic_gala1 = create_ic_gala(100,25,10**11,10**11)
soln1 = solve_all_stars(ic_stars1, ic_gala1, 20,401,10**11,10**11)

def subplot_1(t,lim):
    plt.figure(figsize=(20,10))
    
    plt.subplot(2,4,1)
    plt.xlim(-lim,lim)
    plt.ylim(-lim,lim)
    for n in range(0,120,1):
        plt.scatter(soln1[n][t*20][0],soln1[n][(t*20)][1],15)
    plt.scatter(soln1[0][t*20][4],soln1[0][t*20][5],80,'g')
    plt.scatter(0,0,50,'r')
    plt.tight_layout()
    plt.xticks(visible=False)
    
    plt.subplot(2,4,2)
    plt.xlim(-lim,lim)
    plt.ylim(-lim,lim)
    plt.xticks(visible=False)
    plt.yticks(visible=False)
    for n in range(0,120,1):
        plt.scatter(soln1[n][(t+1)*20][0],soln1[n][((t+1)*20)][1],15)
    plt.scatter(soln1[0][(t+1)*20][4],soln1[0][(t+1)*20][5],80,'g')
    plt.scatter(0,0,50,'r')
    plt.tight_layout()
    
    plt.subplot(2,4,3)
    plt.xlim(-lim,lim)
    plt.ylim(-lim,lim)
    plt.xticks(visible=False)
    plt.yticks(visible=False)
    for n in range(0,120,1):
        plt.scatter(soln1[n][(t+2)*20][0],soln1[n][((t+2)*20)][1],15)
    plt.scatter(soln1[0][(t+2)*20][4],soln1[0][(t+2)*20][5],80,'g')
    plt.scatter(0,0,50,'r')
    plt.tight_layout()
    
    plt.subplot(2,4,4)
    plt.xlim(-lim,lim)
    plt.ylim(-lim,lim)
    plt.xticks(visible=False)
    plt.yticks(visible=False)
    for n in range(0,120,1):
        plt.scatter(soln1[n][(t+3)*20][0],soln1[n][((t+3)*20)][1],15)
    plt.scatter(soln1[0][(t+3)*20][4],soln1[0][(t+3)*20][5],80,'g')
    plt.scatter(0,0,50,'r')
    plt.tight_layout()
    
    plt.subplot(2,4,5)
    plt.xlabel('$X(t)$',size=20)
    plt.ylabel('$Y(t)$',size=20)
    plt.xlim(-lim,lim)
    plt.ylim(-lim,lim)
    for n in range(0,120,1):
        plt.scatter(soln1[n][(t+4)*20][0],soln1[n][((t+4)*20)][1],15)
    plt.scatter(soln1[0][(t+4)*20][4],soln1[0][(t+4)*20][5],80,'g')
    plt.scatter(0,0,50,'r')
    plt.tight_layout()
    plt.subplot(2,4,6)
    plt.xlim(-lim,lim)
    plt.ylim(-lim,lim)
    plt.yticks(visible=False)
    for n in range(0,120,1):
        plt.scatter(soln1[n][(t+5)*20][0],soln1[n][((t+5)*20)][1],15)
    plt.scatter(soln1[0][(t+5)*20][4],soln1[0][(t+5)*20][5],80,'g')
    plt.scatter(0,0,50,'r')
    plt.tight_layout()
    plt.subplot(2,4,7)
    plt.xlim(-lim,lim)
    plt.ylim(-lim,lim)
    plt.yticks(visible=False)
    for n in range(0,120,1):
        plt.scatter(soln1[n][(t+6)*20][0],soln1[n][((t+6)*20)][1],15)
    plt.scatter(soln1[0][(t+6)*20][4],soln1[0][(t+6)*20][5],80,'g')
    plt.scatter(0,0,50,'r')
    plt.tight_layout()
    plt.subplot(2,4,8)
    plt.xlim(-lim,lim)
    plt.ylim(-lim,lim)
    plt.yticks(visible=False)
    for n in range(0,120,1):
        plt.scatter(soln1[n][(t+7)*20][0],soln1[n][((t+7)*20)][1],15)
    plt.scatter(soln1[0][(t+7)*20][4],soln1[0][(t+7)*20][5],80,'g')
    plt.scatter(0,0,50,'r')
    plt.tight_layout()
    
def subplot_2(t,lim):
    plt.figure(figsize=(20,10))
    
    plt.subplot(2,4,1)
    plt.xlim(-lim,lim)
    plt.ylim(-lim,lim)
    for n in range(0,120,1):
        plt.scatter(soln1[n][t*20][0]-soln1[0][t*20][4],soln1[n][(t*20)][1]-soln1[0][t*20][5],15)
    plt.scatter(0,0,80,'g')
    plt.scatter(0-soln1[0][t*20][4],0-soln1[0][t*20][5],50,'r')
    plt.tight_layout()
    plt.xticks(visible=False)
    
    plt.subplot(2,4,2)
    plt.xlim(-lim,lim)
    plt.ylim(-lim,lim)
    plt.xticks(visible=False)
    plt.yticks(visible=False)
    for n in range(0,120,1):
        plt.scatter(soln1[n][(t+1)*20][0]-soln1[0][(t+1)*20][4],soln1[n][((t+1)*20)][1]-soln1[0][(t+1)*20][5],15)
    plt.scatter(0,0,80,'g')
    plt.scatter(0-soln1[0][(t+1)*20][4],0-soln1[0][(t+1)*20][5],50,'r')
    plt.tight_layout()
    
    plt.subplot(2,4,3)
    plt.xlim(-lim,lim)
    plt.ylim(-lim,lim)
    plt.xticks(visible=False)
    plt.yticks(visible=False)
    for n in range(0,120,1):
        plt.scatter(soln1[n][(t+2)*20][0]-soln1[0][(t+2)*20][4],soln1[n][((t+2)*20)][1]-soln1[0][(t+2)*20][5],15)
    plt.scatter(0,0,80,'g')
    plt.scatter(0-soln1[0][(t+2)*20][4],0-soln1[0][(t+2)*20][5],50,'r')
    plt.tight_layout()
    
    plt.subplot(2,4,4)
    plt.xlim(-lim,lim)
    plt.ylim(-lim,lim)
    plt.xticks(visible=False)
    plt.yticks(visible=False)
    for n in range(0,120,1):
        plt.scatter(soln1[n][(t+3)*20][0]-soln1[0][(t+3)*20][4],soln1[n][((t+3)*20)][1]-soln1[0][(t+3)*20][4],15)
    plt.scatter(0,0,80,'g')
    plt.scatter(0-soln1[0][(t+3)*20][4],0-soln1[0][(t+3)*20][5],50,'r')
    plt.tight_layout()
    
    plt.subplot(2,4,5)
    plt.xlabel('$X(t)$',size=20)
    plt.ylabel('$Y(t)$',size=20)
    plt.xlim(-lim,lim)
    plt.ylim(-lim,lim)
    for n in range(0,120,1):
        plt.scatter(soln1[n][(t+4)*20][0]-soln1[0][(t+4)*20][4],soln1[n][((t+4)*20)][1]-soln1[0][(t+4)*20][5],15)
    plt.scatter(0,0,80,'g')
    plt.scatter(0-soln1[0][(t+4)*20][4],0-soln1[0][(t+4)*20][5],50,'r')
    plt.tight_layout()
    
    plt.subplot(2,4,6)
    plt.xlim(-lim,lim)
    plt.ylim(-lim,lim)
    plt.yticks(visible=False)
    for n in range(0,120,1):
        plt.scatter(soln1[n][(t+5)*20][0]-soln1[0][(t+5)*20][4],soln1[n][((t+5)*20)][1]-soln1[0][(t+5)*20][5],15)
    plt.scatter(0,0,80,'g')
    plt.scatter(0-soln1[0][(t+5)*20][4],0-soln1[0][(t+5)*20][5],50,'r')
    plt.tight_layout()
    
    plt.subplot(2,4,7)
    plt.xlim(-lim,lim)
    plt.ylim(-lim,lim)
    plt.yticks(visible=False)
    for n in range(0,120,1):
        plt.scatter(soln1[n][(t+6)*20][0]-soln1[0][(t+6)*20][4],soln1[n][((t+6)*20)][1]-soln1[0][(t+6)*20][5],15)
    plt.scatter(0,0,80,'g')
    plt.scatter(0-soln1[0][(t+6)*20][4],0-soln1[0][(t+6)*20][5],50,'r')
    plt.tight_layout()
    
    plt.subplot(2,4,8)
    plt.xlim(-lim,lim)
    plt.ylim(-lim,lim)
    plt.yticks(visible=False)
    for n in range(0,120,1):
        plt.scatter(soln1[n][(t+7)*20][0]-soln1[0][(t+7)*20][4],soln1[n][((t+7)*20)][1]-soln1[0][(t+7)*20][5],15)
    plt.scatter(0,0,80,'g')
    plt.scatter(0-soln1[0][(t+7)*20][4],0-soln1[0][(t+7)*20][5],50,'r')
    plt.tight_layout()

def subplot_3(t,lim):
    plt.figure(figsize=(20,10))
    
    plt.subplot(2,4,1)
    plt.xlim(-lim,lim)
    plt.ylim(-lim,lim)
    for n in range(0,120,1):
        plt.scatter(soln1[n][t*20][0]-0.5*soln1[0][t*20][4],soln1[n][(t*20)][1]-0.5*soln1[0][t*20][5],15)
    plt.scatter(0.5*soln1[0][t*20][4],0.5*soln1[0][t*20][5],80,'g')
    plt.scatter(0-0.5*soln1[0][t*20][4],0-0.5*soln1[0][t*20][5],50,'r')
    plt.tight_layout()
    plt.xticks(visible=False)
    
    plt.subplot(2,4,2)
    plt.xlim(-lim,lim)
    plt.ylim(-lim,lim)
    plt.xticks(visible=False)
    plt.yticks(visible=False)
    for n in range(0,120,1):
        plt.scatter(soln1[n][(t+1)*20][0]-0.5*soln1[0][(t+1)*20][4],soln1[n][((t+1)*20)][1]-0.5*soln1[0][(t+1)*20][5],15)
    plt.scatter(0.5*soln1[0][(t+1)*20][4],0.5*soln1[0][(t+1)*20][5],80,'g')
    plt.scatter(0-0.5*soln1[0][(t+1)*20][4],0-0.5*soln1[0][(t+1)*20][5],50,'r')
    plt.tight_layout()
    
    plt.subplot(2,4,3)
    plt.xlim(-lim,lim)
    plt.ylim(-lim,lim)
    plt.xticks(visible=False)
    plt.yticks(visible=False)
    for n in range(0,120,1):
        plt.scatter(soln1[n][(t+2)*20][0]-0.5*soln1[0][(t+2)*20][4],soln1[n][((t+2)*20)][1]-0.5*soln1[0][(t+2)*20][5],15)
    plt.scatter(0.5*soln1[0][(t+2)*20][4],0.5*soln1[0][(t+2)*20][5],80,'g')
    plt.scatter(0-0.5*soln1[0][(t+2)*20][4],0-0.5*soln1[0][(t+2)*20][5],50,'r')
    plt.tight_layout()
    
    plt.subplot(2,4,4)
    plt.xlim(-lim,lim)
    plt.ylim(-lim,lim)
    plt.xticks(visible=False)
    plt.yticks(visible=False)
    for n in range(0,120,1):
        plt.scatter(soln1[n][(t+3)*20][0]-0.5*soln1[0][(t+3)*20][4],soln1[n][((t+3)*20)][1]-0.5*soln1[0][(t+3)*20][4],15)
    plt.scatter(0.5*soln1[0][(t+3)*20][4],0.5*soln1[0][(t+3)*20][5],80,'g')
    plt.scatter(0-0.5*soln1[0][(t+3)*20][4],0-0.5*soln1[0][(t+3)*20][5],50,'r')
    plt.tight_layout()
    
    plt.subplot(2,4,5)
    plt.xlabel('$X(t)$',size=20)
    plt.ylabel('$Y(t)$',size=20)
    plt.xlim(-lim,lim)
    plt.ylim(-lim,lim)
    for n in range(0,120,1):
        plt.scatter(soln1[n][(t+4)*20][0]-0.5*soln1[0][(t+4)*20][4],soln1[n][((t+4)*20)][1]-0.5*soln1[0][(t+4)*20][5],15)
    plt.scatter(0.5*soln1[0][(t+4)*20][4],0.5*soln1[0][(t+4)*20][5],80,'g')
    plt.scatter(0-0.5*soln1[0][(t+4)*20][4],0-0.5*soln1[0][(t+4)*20][5],50,'r')
    plt.tight_layout()
    
    plt.subplot(2,4,6)
    plt.xlim(-lim,lim)
    plt.ylim(-lim,lim)
    plt.yticks(visible=False)
    for n in range(0,120,1):
        plt.scatter(soln1[n][(t+5)*20][0]-0.5*soln1[0][(t+5)*20][4],soln1[n][((t+5)*20)][1]-0.5*soln1[0][(t+5)*20][5],15)
    plt.scatter(0.5*soln1[0][(t+5)*20][4],0.5*soln1[0][(t+5)*20][5],80,'g')
    plt.scatter(0-0.5*soln1[0][(t+5)*20][4],0-0.5*soln1[0][(t+5)*20][5],50,'r')
    plt.tight_layout()
    
    plt.subplot(2,4,7)
    plt.xlim(-lim,lim)
    plt.ylim(-lim,lim)
    plt.yticks(visible=False)
    for n in range(0,120,1):
        plt.scatter(soln1[n][(t+6)*20][0]-0.5*soln1[0][(t+6)*20][4],soln1[n][((t+6)*20)][1]-0.5*soln1[0][(t+6)*20][5],15)
    plt.scatter(0.5*soln1[0][(t+6)*20][4],0.5*soln1[0][(t+6)*20][5],80,'g')
    plt.scatter(0-0.5*soln1[0][(t+6)*20][4],0-0.5*soln1[0][(t+6)*20][5],50,'r')
    plt.tight_layout()
    
    plt.subplot(2,4,8)
    plt.xlim(-lim,lim)
    plt.ylim(-lim,lim)
    plt.yticks(visible=False)
    for n in range(0,120,1):
        plt.scatter(soln1[n][(t+7)*20][0]-0.5*soln1[0][(t+7)*20][4],soln1[n][((t+7)*20)][1]-0.5*soln1[0][(t+7)*20][5],15)
    plt.scatter(0.5*soln1[0][(t+7)*20][4],0.5*soln1[0][(t+7)*20][5],80,'g')
    plt.scatter(0-0.5*soln1[0][(t+7)*20][4],0-0.5*soln1[0][(t+7)*20][5],50,'r')
    plt.tight_layout()
    
def plot_1(t,lim):
    plt.figure(figsize=(8,8))
    plt.xlim(-lim,lim)
    plt.ylim(-lim,lim)
    for n in range(0,120,1):
        plt.scatter(soln1[n][t*20][0],soln1[n][(t*20)][1],15)
    plt.scatter(soln1[0][t*20][4],soln1[0][t*20][5],80,'g')
    plt.scatter(0,0,50,'r')
    plt.tight_layout()
    plt.xlabel('$X(t)$',size=20)
    plt.ylabel('$Y(t)$',size=20)
    plt.title('$Y(t)$ vs $X(t)$ Centered on Galaxy M',size=20)
    
def interact_1():
    interact(plot_1,t=(0,20,.05),lim=(50,200,5))

def plot_2(t,lim):
    plt.figure(figsize=(8,8))
    plt.xlim(-lim,lim)
    plt.ylim(-lim,lim)
    for n in range(0,120,1):
        plt.scatter(soln1[n][t*20][0]-soln1[0][t*20][4],soln1[n][(t*20)][1]-soln1[0][t*20][5],15)
    plt.scatter(0,0,80,'g')
    plt.scatter(0-soln1[0][t*20][4],0-soln1[0][t*20][5],50,'r')
    plt.tight_layout()
    plt.xlabel('$X(t)$',size=20)
    plt.ylabel('$Y(t)$',size=20)
    plt.title('$Y(t)$ vs $X(t)$ Centered on Galaxy S',size=20)
    
def interact_2():
    interact(plot_2,t=(0,20,.05),lim=(50,200,5))
    
def plot_3(t,lim):
    plt.figure(figsize=(8,8))
    plt.xlim(-lim,lim)
    plt.ylim(-lim,lim)
    for n in range(0,120,1):
        plt.scatter(soln1[n][t*20][0]-soln1[0][t*20][4]*.5,soln1[n][(t*20)][1]-soln1[0][t*20][5]*.5,15)
    plt.scatter(soln1[0][t*20][4]-0.5*soln1[0][t*20][4],soln1[0][t*20][5]-0.5*soln1[0][t*20][5],80,'g')
    plt.scatter(0-soln1[0][t*20][4]*.5,0-soln1[0][t*20][5]*.5,50,'r')
    plt.tight_layout()
    plt.xlabel('$X(t)$',size=20)
    plt.ylabel('$Y(t)$',size=20)
    plt.title('$Y(t)$ vs $X(t)$ About the Center of Mass',size=20)
    
def interact_3():
    interact(plot_3,t=(0,20,.05),lim=(50,200,5))
    
ic_stars2 = create_ic_stars(25,-1)
ic_gala2 = create_ic_gala(80,25,10**11,1*10**11)
soln2 = solve_all_stars(ic_stars2, ic_gala2, 20,401,10**11,10**11)

def dplot_1(t,lim):
    plt.figure(figsize=(8,8))
    plt.xlim(-lim,lim)
    plt.ylim(-lim,lim)
    for n in range(0,120,1):
        plt.scatter(soln2[n][t*20][0],soln2[n][(t*20)][1],15)
    plt.scatter(soln2[0][t*20][4],soln2[0][t*20][5],80,'g')
    plt.scatter(0,0,50,'r')
    plt.tight_layout()
    plt.xlabel('$X(t)$',size=20)
    plt.ylabel('$Y(t)$',size=20)
    plt.title('$Y(t)$ vs $X(t)$ Centered on Galaxy M',size=20)
    
def dinteract_1():
    interact(dplot_1,t=(0,20,.05),lim=(50,200,5))
    
ic_stars3 = create_ic_stars(25,-1)
ic_gala3 = create_ic_gala(80,25,10**11,3*10**11)
soln3 = solve_all_stars(ic_stars3, ic_gala3, 20,401,10**11,3*10**11)

def hplot_1(t,lim):
    plt.figure(figsize=(8,8))
    plt.xlim(-lim,lim)
    plt.ylim(-lim,lim)
    for n in range(0,120,1):
        plt.scatter(soln3[n][t*20][0],soln3[n][(t*20)][1],15)
    plt.scatter(soln3[0][t*20][4],soln3[0][t*20][5],80,'g')
    plt.scatter(0,0,50,'r')
    plt.tight_layout()
    plt.xlabel('$X(t)$',size=20)
    plt.ylabel('$Y(t)$',size=20)
    plt.title('$Y(t)$ vs $X(t)$ Centered on Galaxy M',size=20)
    
def hinteract_1():
    interact(hplot_1,t=(0,20,.05),lim=(50,200,5))