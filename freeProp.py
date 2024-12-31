###################################################################
# Created By        : ZAP Physics (zapphysics@gmail.com), 2024
# File Name         : freeProp.py
# Description       : Generates animations for free, massive scalar propagator and compares to Schroedinger equation
##################################################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import scipy.integrate
import os

class freeProp1d:

    # General class for the free, massive scalar propagator in one spatial dimension
    # assuming an initial-state Gaussian distribution for position and momentum

    def __init__(self, mass, p0 = 0, x0 = 0, sigma = 1):

        self.m = mass     # Mass of particle
        self.x0 = x0      # Initial location of center of initial Gaussian
        self.p0 = p0      # Initial central momentum of Gaussian
        self.sig = sigma  # Width of initial momentum-space Gaussian

    def integrand(self, p, y, t):

        # Analytic integrand for two-point correlation function

        return 1/(2*np.pi)*(
                1/2/np.sqrt(p**2 + self.m**2)
                *1/np.sqrt(2*np.pi)/self.sig*np.exp(- (p - self.p0)**2/2/self.sig**2)
                *np.exp(-1j*np.sqrt(p**2 + self.m**2)*t)*np.exp(1j*p*(y - self.x0))
                )

    def prop(self, y, t):

        # Numerically evaluates the momentum integral for propagator

        reint = lambda u: np.real(self.integrand(u, y, t))
        imint = lambda u: np.imag(self.integrand(u, y, t))

        return (
                scipy.integrate.quad(reint, -np.inf, np.inf)[0]
                + 1j*scipy.integrate.quad(imint, -np.inf, np.inf)[0]
                )

    def schroedProp(self, y, t):

        # Analytic solution to Schroedinger equation with same initial conditions as propagator

        A = 1/2/self.sig**2 + 1j*t/2/self.m
        B = (self.p0/self.sig**2 + 1j*(y - self.x0))/2/A

        return 1/2/self.m/np.sqrt(2*np.pi)/self.sig*np.exp(A*B**2 - self.p0**2/2/self.sig**2)*1/2/np.sqrt(np.pi*A)

    def createAnimation(self, xvals, times, directory = r'./', filename = r'freeProp1D.gif', schroedOverlay = False):

        # Generates animation of two-point correlation function at location directory + filename.
        # Spatial range given by xvals, animates frames for times, and schroedOverlay toggles the 
        # overlay of Schroedinger solution.

        fig, ax = plt.subplots()

        dt = times[1] - times[0] # define time step

        SoLx = np.array([self.x0, self.x0])
        SoLy = np.array([0, 1])

        SoL, = ax.plot(SoLx, SoLy, '-b', label = 'Speed of Light') # Plots speed of light

        # Initializes propagator
        data = np.array([np.abs(self.prop(y, times[0]))**2 for y in xvals])
        norm = np.sum(data)*(xvals[1] - xvals[0])   # Normalization of data over x-range
        line, = ax.plot(xvals, data/norm, '-k', label = 'QFT')

        if schroedOverlay:
            # Initializes Schroedinger solution
            datSchroed = np.array([np.abs(self.schroedProp(y, times[0]))**2 for y in xvals])
            normSchroed = np.sum(datSchroed)*(xvals[1] - xvals[0])
            lineS, = ax.plot(xvals, datSchroed/normSchroed, '--r', label = 'Schroedinger')

        ax.set_xlabel('Position')
        ax.set_ylim([0.0, 1.5*np.max(data / norm)])
        ax.set_yticks([])
        ax.legend(loc = 'upper right')

        def animate(frame):

            # Generates animation

            data = np.array([np.abs(self.prop(y, times[frame]))**2 for y in xvals])
            line.set_ydata(data/norm)

            SoL.set_xdata([self.x0 + frame*dt, self.x0 + frame*dt])

            if schroedOverlay:
                datSchroed = np.array([np.abs(self.schroedProp(y, times[frame]))**2 for y in xvals])
                lineS.set_ydata(datSchroed/normSchroed)

            print('{:.2f}'.format(times[frame]/times[-1]*100), '% Done', end = '\r')

            if schroedOverlay:
                return line, lineS, SoL,

            return line, SoL,

        anim = animation.FuncAnimation(fig, animate, frames = len(times), interval = 1, blit = True)

        if not os.path.exists(directory): os.makedirs(directory)

        writergif = animation.PillowWriter(fps = 20)
        anim.save(directory + filename, writer = writergif, dpi = 250)

        print('Done! Result location: {}'.format(directory + filename))

if __name__ == '__main__':

    # Example code

    xvals = np.arange(-10, 10, 0.1)
    tvals = np.arange(0, 20, 0.3)

    prop1d = freeProp1d(3, 0, -5, 1)
    prop1d.createAnimation(xvals, tvals, filename = r'prop1D.gif', schroedOverlay = True)
