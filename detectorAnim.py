###################################################################
# Created By        : ZAP Physics (zapphysics@gmail.com), 2024
# File Name         : detectorAnim.py
# Description       : Generates animation for 2D detector simulation based on model discussed in PDF
##################################################################

from scipy.optimize import root
from scipy.integrate import quad
from scipy.special import loggamma
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patch
from matplotlib import animation
from matplotlib.lines import Line2D
import vegas
import gvar as gv
import os

def L(r, m1, m2): # L function as defined in PDF

    if np.sqrt(np.abs(r)) <= m1 + m2:
        if r >= 0:
            return 1/np.sqrt(r + 1e-8)*np.log((m1 + m2 + np.sqrt(r))/(m1 + m2 - np.sqrt(r + 1e-8)))
        else:
            return -1j/np.sqrt(-r + 1e-8)*np.log((m1 + m2 + 1j*np.sqrt(-r))/(m1 + m2 - 1j*np.sqrt(-r + 1e-8)))
    else:
        if r >= 0:
            return 1/np.sqrt(r + 1e-8)*(1j*np.pi + np.log((m1 + m2 + np.sqrt(r))/(np.sqrt(r + 1e-8) - m1 - m2)))
        else:
            return -1j/np.sqrt(-r + 1e-8)*(1j*np.pi + np.log((m1 + m2 + 1j*np.sqrt(-r))/(1j*np.sqrt(-r + 1e-8) - m1 - m2)))

def Ldot(A, B): # Lorentz contraction between two 3-vectors in 2+1 dimensions

    eta = np.diag([1, -1, -1])
    return np.matmul(A, np.matmul(eta, B))

def Lsq(A): return Ldot(A, A) # Lorentz vector magnitude

def lammom(m1, m2, m3): # lambda function as defined in PDF
    return 1/2/m1*np.sqrt((m1**2 - (m2 + m3)**2)*(m1**2 - (m2 - m3)**2))

class detector: # Class for building detector simulation

    def __init__(self, p, psi, mphi = 0.1, mth = 1, sigb = 0.1, n = 10, lam = 0.1):

        self.p0ex = p      # Initial-state beam momentum
        self.psi = psi     # Relative beam angle
        self.mph = mphi    # Mass of phi particle
        self.mth = mth     # Mass of theta particle
        self.sigb = sigb   # Beam width
        self.n = n         # Beam number density
        self.lam = lam     # phi^2*theta^2 quadratic coupling constant

        self.p0com = p*np.cos(self.psi/2) # Center of mass momentum
        self.E0ex = np.sqrt(p**2 + mphi**2) # Energy in lab frame
        self.E0com = np.sqrt(self.p0com**2 + mphi**2) # Center of mass energy
        self.s = 4*self.E0com**2 # Mandelstam variable s

        self.L = 2*n**2*self.p0com*self.E0com*sigb**2/self.E0ex**2/np.sin(psi) # Beam luminosity

        # Initial-state 3-momenta in center of mass frame
        self.p1 = np.array([self.E0com, self.p0com, 0]) 
        self.p2 = np.array([self.E0com, - self.p0com, 0])

    def Mphi2toth2(self, t, u): # phi^2 -> theta^2 matrix element to O(lambda^2)

        M0 = - 1j*self.lam
        M1 = 1j*self.lam**2/8/np.pi*(L(t, self.mph, self.mth) + L(u, self.mph, self.mth))

        return M0 + M1

    def Mphi2tophi2(self, s, t, u): # phi^2 -> phi^2 matrix element to O(lambda^2)

        return 1j*self.lam**2/8/np.pi*(L(s, self.mth, self.mth) + L(t, self.mth, self.mth) + L(u, self.mth, self.mth))

    def Mphi2toth4(self, t1, t2, t3, u1, u2, u3): # phi^2 -> theta^4 matrix element to O(lambda^2)

        Dph = lambda x: 1/(x - self.mph**2)

        return - 1j*self.lam**2*(Dph(t1) + Dph(t2) + Dph(t3) + Dph(u1) + Dph(u2) + Dph(u3))

    def Mphi2toth2ph2(self, s1, s2, t1, t2, u1, u2): # phi^2 -> phi^2 theta^2 matrix element to O(lambda^2)

        Dth = lambda x: 1/(x - self.mth**2)

        return - 1j*self.lam**2*(Dth(s1) + Dth(s2) + Dth(t1) + Dth(t2) + Dth(u1) + Dth(u2))

    def dsigphi2toth2(self, xi): # phi^2 -> theta^2 differential cross-section

        t = - 2*self.p0com**2*(1 - np.cos(xi))
        u = - 2*self.p0com**2*(1 + np.cos(xi))

        if self.E0com >= self.mth:
            return np.abs(self.Mphi2toth2(t, u))**2/256/np.pi/self.p0com/self.E0com

        else: return 0

    def dsigphi2tophi2(self, xi): # phi^2 -> phi^2 differential cross-section

        s = 4*self.E0com
        t = - 2*self.p0com**2*(1 - np.cos(xi))
        u = - 2*self.p0com**2*(1 + np.cos(xi))

        return np.abs(self.Mphi2tophi2(s, t, u))**2/256/np.pi/self.p0com/self.E0com

    def dsigphi2toth4(self, x): # phi^2 -> theta^4 differential cross-section

        if self.E0com < 2*self.mth: return 0
        else:

            f12, s34, xi12, xi1, xi3 = x # Vector of kinematical variables

            # Note: s12 has been shifted to f12 (defined below) to eliminate s34-dependence of bounds

            detJ = 1/((np.sqrt(self.s) - np.sqrt(s34))**2 - 4*self.mth**2)
            s12 = 4*self.mth**2 + f12/detJ

            lam0 = lammom(np.sqrt(self.s), np.sqrt(s12), np.sqrt(s34))
            lam12 = lammom(np.sqrt(s12), self.mth, self.mth)
            lam34 = lammom(np.sqrt(s34), self.mth, self.mth)

            Rmat = np.array([[1, 0, 0], [0, np.cos(xi12), - np.sin(xi12)], [0, np.sin(xi12), np.cos(xi12)]])
            b12 = lam0/np.sqrt(lam0**2 + s12)
            b34 = - lam0/np.sqrt(lam0**2 + s34)

            L12 = np.array([[1/np.sqrt(1 - b12**2), b12/np.sqrt(1 - b12**2), 0],
                            [b12/np.sqrt(1 - b12**2), 1/np.sqrt(1 - b12**2), 0],
                            [0, 0, 1]])
            L34 = np.array([[1/np.sqrt(1 - b34**2), b34/np.sqrt(1 - b34**2), 0],
                            [b34/np.sqrt(1 - b34**2), 1/np.sqrt(1 - b34**2), 0],
                            [0, 0, 1]])

            q112 = np.array([np.sqrt(lam12**2 + self.mth**2), lam12*np.cos(xi1), lam12*np.sin(xi1)])
            q212 = np.array([np.sqrt(lam12**2 + self.mth**2), - lam12*np.cos(xi1), - lam12*np.sin(xi1)])
            q334 = np.array([np.sqrt(lam34**2 + self.mth**2), lam34*np.cos(xi3), lam34*np.sin(xi3)])

            q1 = np.matmul(Rmat, np.matmul(L12, q112))
            q2 = np.matmul(Rmat, np.matmul(L12, q212))
            q3 = np.matmul(Rmat, np.matmul(L34, q334))

            t1 = Lsq(self.p1 - q1 - q3)
            t2 = Lsq(self.p2 - q2 - q3)
            t3 = Lsq(self.p1 - q1 - q2)
            u1 = Lsq(self.p1 - q2 - q3)
            u2 = Lsq(self.p2 - q1 - q3)
            u3 = Lsq(self.p2 - q1 - q2)

            return np.abs(self.Mphi2toth4(t1, t2, t3, u1, u2, u3))**2/48/self.p0com/self.E0com/(2*np.pi)**2/(8*np.pi)**3/np.sqrt(self.s*s12*s34)

    def dsigphi2toth2ph2(self, x): # phi^2 -> phi^2 theta^2 differential cross-section

        if self.E0com < self.mth + self.mph: return 0
        else:

            f12, s34, xi12, xi1, xi3 = x # Vector of kinematical variables

            # Note: s12 has been shifted to f12 (defined below) to eliminate s34-dependence of bounds

            detJ = 1/((np.sqrt(self.s) - np.sqrt(s34))**2 - 4*self.mth**2)
            s12 = 4*self.mth**2 + f12/detJ

            lam0 = lammom(np.sqrt(self.s), np.sqrt(s12), np.sqrt(s34))
            lam12 = lammom(np.sqrt(s12), self.mth, self.mth)
            lam34 = lammom(np.sqrt(s34), self.mph, self.mph)

            Rmat = np.array([[1, 0, 0], [0, np.cos(xi12), - np.sin(xi12)], [0, np.sin(xi12), np.cos(xi12)]])
            b12 = lam0/np.sqrt(lam0**2 + s12)
            b34 = - lam0/np.sqrt(lam0**2 + s34)

            L12 = np.array([[1/np.sqrt(1 - b12**2), b12/np.sqrt(1 - b12**2), 0],
                            [b12/np.sqrt(1 - b12**2), 1/np.sqrt(1 - b12**2), 0],
                            [0, 0, 1]])
            L34 = np.array([[1/np.sqrt(1 - b34**2), b34/np.sqrt(1 - b34**2), 0],
                            [b34/np.sqrt(1 - b34**2), 1/np.sqrt(1 - b34**2), 0],
                            [0, 0, 1]])

            q112 = np.array([np.sqrt(lam12**2 + self.mth**2), lam12*np.cos(xi1), lam12*np.sin(xi1)])
            q212 = np.array([np.sqrt(lam12**2 + self.mth**2), - lam12*np.cos(xi1), - lam12*np.sin(xi1)])
            q334 = np.array([np.sqrt(lam34**2 + self.mph**2), lam34*np.cos(xi3), lam34*np.sin(xi3)])

            q1 = np.matmul(Rmat, np.matmul(L12, q112))
            q2 = np.matmul(Rmat, np.matmul(L12, q212))
            q3 = np.matmul(Rmat, np.matmul(L34, q334))

            s1 = Lsq(self.p1 + self.p2 - q1)
            s2 = Lsq(self.p1 + self.p2 - q1)
            t1 = Lsq(self.p1 - q1 - q3)
            t2 = Lsq(self.p2 - q2 - q3)
            u1 = Lsq(self.p1 - q2 - q3)
            u2 = Lsq(self.p2 - q1 - q3)

            return np.abs(self.Mphi2toth2ph2(s1, s2, t1, t2, u1, u2))**2/8/self.p0com/self.E0com/(2*np.pi)**2/(8*np.pi)**3/np.sqrt(self.s*s12*s34)

    def rates(self, select = []): # Calculates rates for all selected processes. Computes all four if none specified

        # The accepted values in the list given to select are:
        # 0: phi^2 -> phi^2
        # 1: phi^2 -> theta^2
        # 2: phi^2 -> theta^4
        # 3: phi^2 -> phi^2 theta^2

        # Two-particle final state interactions can be evaluated numerically, but four-particle final state interactions
        # are evaluated with Monte Carlo methods using the Vegas package

        if select == []:

            Rphi2phi2 = self.L*(quad(self.dsigphi2tophi2, 1e-4, np.pi - 1e-4)[0] + quad(self.dsigphi2tophi2, np.pi + 1e-4, 2*np.pi - 1e-4)[0])

            if self.E0com < self.mth:

                Rphi2th2 = 0
                Rphi2th4 = 0
                Rphi2th2phi2 = 0

                return [Rphi2phi2, Rphi2th2, Rphi2th4, Rphi2th2phi2]

            elif self.E0com < self.mth + self.mph and self.E0com < 2*self.mth:

                Rphi2th2 = self.L*(quad(self.dsigphi2toth2, 1e-4, np.pi - 1e-4)[0] + quad(self.dsigphi2toth2, np.pi + 1e-4, 2*np.pi - 1e-4)[0])
                Rphi2th4 = 0
                Rphi2th2phi2 = 0

                return [Rphi2phi2, Rphi2th2, Rphi2th4, Rphi2th2phi2]

            elif self.E0com > self.mth + self.mph and self.E0com < 2*self.mth:

                Rphi2th2 = self.L*(quad(self.dsigphi2toth2, 1e-4, np.pi - 1e-4)[0] + quad(self.dsigphi2toth2, np.pi + 1e-4, 2*np.pi - 1e-4)[0])
                Rphi2th4 = 0
                integ2 = vegas.Integrator([[0, 1], [4*self.mph**2, (np.sqrt(self.s) - 2*self.mth)**2], [0, 2*np.pi], [0, 2*np.pi], [0, 2*np.pi]])
                Rphi2th2phi2 = self.L*gv.mean(integ2(self.dsigphi2toth2ph2, nitn = 20, neval = 10000))

                return [Rphi2phi2, Rphi2th2, Rphi2th4, Rphi2th2phi2]

            elif self.E0com > 2*self.mth and self.E0com < self.mth + self.mph:

                Rphi2th2 = self.L*(quad(self.dsigphi2toth2, 1e-4, np.pi - 1e-4)[0] + quad(self.dsigphi2toth2, np.pi + 1e-4, 2*np.pi - 1e-4)[0])
                integ1 = vegas.Integrator([[0, 1], [4*self.mth**2, (np.sqrt(self.s) - 2*self.mth)**2], [0, 2*np.pi], [0, 2*np.pi], [0, 2*np.pi]])
                Rphi2th4 = self.L*gv.mean(integ1(self.dsigphi2toth4, nitn = 20, neval = 10000))
                Rphi2th2phi2 = 0

                return [Rphi2phi2, Rphi2th2, Rphi2th4, Rphi2th2phi2]

            else:

                Rphi2th2 = self.L*(quad(self.dsigphi2toth2, 1e-4, np.pi - 1e-4)[0] + quad(self.dsigphi2toth2, np.pi + 1e-4, 2*np.pi - 1e-4)[0])
                integ1 = vegas.Integrator([[0, 1], [4*self.mth**2, (np.sqrt(self.s) - 2*self.mth)**2], [0, 2*np.pi], [0, 2*np.pi], [0, 2*np.pi]])
                Rphi2th4 = self.L*gv.mean(integ1(self.dsigphi2toth4, nitn = 20, neval = 10000))
                integ2 = vegas.Integrator([[0, 1], [4*self.mph**2, (np.sqrt(self.s) - 2*self.mth)**2], [0, 2*np.pi], [0, 2*np.pi], [0, 2*np.pi]])
                Rphi2th2phi2 = self.L*gv.mean(integ2(self.dsigphi2toth2ph2, nitn = 20, neval = 10000))

                return [Rphi2phi2, Rphi2th2, Rphi2th4, Rphi2th2phi2]

        else:

            vals = []
            for i in select:
                
                try: np.zeros(4)[i]
                except IndexError:
                    raise IndexError('There are only four scattering processes!') from None

                if i == 0: vals.append(self.L*(quad(self.dsigphi2tophi2, 1e-4, np.pi - 1e-4)[0] + quad(self.dsigphi2tophi2, np.pi + 1e-4, 2*np.pi - 1e-4)[0]))
                elif i == 1: vals.append(self.L*(quad(self.dsigphi2toth2, 1e-4, np.pi - 1e-4)[0] + quad(self.dsigphi2toth2, np.pi + 1e-4, 2*np.pi - 1e-4)[0]))
                elif i == 2:
                    integ1 = vegas.Integrator([[0, 1], [4*self.mth**2, (np.sqrt(self.s) - 2*self.mth)**2], [0, 2*np.pi], [0, 2*np.pi], [0, 2*np.pi]])
                    vals.append(self.L*gv.mean(integ1(self.dsigphi2toth4, nitn = 20, neval = 10000)))
                else:
                    integ2 = vegas.Integrator([[0, 1], [4*self.mph**2, (np.sqrt(self.s) - 2*self.mth)**2], [0, 2*np.pi], [0, 2*np.pi], [0, 2*np.pi]])
                    vals.append(self.L*gv.mean(integ2(self.dsigphi2toth2ph2, nitn = 20, neval = 10000)))

            return vals

    def createAnimation(self, time, fps = 30, directory = r'./', filename = r'detector.gif'):
        # Generates animation for detector simulation for given experimental parameters

        dt = 1/fps # Time-step
        t = np.arange(0, time, dt)

        print(' Calculating Rates...(this may take a bit)', end = '\r')

        totalRate = np.array(self.rates())
        totRateS = 1.54e24*totalRate # Converts rate in GeV to 1/seconds

        print(' Rates:                                   ')
        print('   R(phi phi -> theta theta) = {:.2f} s^-1'.format(totRateS[1]))
        print('   R(phi phi -> phi phi) = {:.2f} s^-1'.format(totRateS[0]))
        print('   R(phi phi -> theta theta phi phi) = {:.2f} s^-1'.format(totRateS[3]))
        print('   R(phi phi -> theta theta theta theta) = {:.2f} s^-1'.format(totRateS[2]))

        if totRateS.sum()*dt > 1e3: # A warning message just in case. Otherwise can be VERY slow.

            userInput = input('\n Warning: this will generate ~{:.2f} events per frame.\n'\
                    ' This will likely be VERY slow. Continue? (y or n): '.format(totRateS.sum()*dt))

            while userInput != 'y':
                if userInput == 'n':
                    print('\n Tip: To reduce rates it is recommended to decrease beam width,\n'\
                            ' decrease beam density, increase relative beam angle, or\n'
                            ' increase beam momentum.')
                    exit()
                else: userInput = input(' Please input \'y\' to continue or \'n\' to exit: ')

        print(' ')

        #The following functions generate CDFs based on the corresponding differential cross sections:

        def th2CDF(x):
            if x < 0.0 or x > np.pi:
                return 0.0
            vals = np.array([self.dsigphi2toth2(y) / (totalRate[1]/self.L) for y in np.arange(1e-5, x, 0.01*np.sqrt(2))])
            return 2*vals.sum()*0.01*np.sqrt(2)

        def phi2CDF(x):
            if x < 0.0 or x > np.pi:
                return 0.0
            vals = np.array([self.dsigphi2tophi2(y) / (totalRate[0]/self.L) for y in np.arange(1e-5, x, 0.01*np.sqrt(2))])
            return 2*vals.sum()*0.01*np.sqrt(2)

        def th2ph2CDF(x):
            f12, s34, xi12, xi1, xi3 = x
            integ = vegas.Integrator([[0, f12], [4*self.mph**2, s34], [0.0, xi12], [0.0, xi1], [0.0, xi3]])
            return gv.mean(integ(self.dsigphi2toth2ph2, nitn = 10, neval = 100)) / (totalRate[3]/self.L)

        def th4CDF(x):
            f12, s34, xi12, xi1, xi3 = x
            integ = vegas.Integrator([[0, f12], [4*self.mth**2, s34], [0.0, xi12], [0.0, xi1], [0.0, xi3]])
            return gv.mean(integ(self.dsigphi2toth4, nitn = 10, neval = 100)) / (totalRate[2]/self.L)

        # The following functions use the previously-defined CDFs to generate a set of kinematical variables
        # when an interaction occurs from the differential probability distributions extracted from the cross-sections.
        # This is accomplished by throwing a "dart" at a value between 0.0 and 1.0 and choosing an intersection point
        # with the CDF. Note that in some cases, there may be more than one intersection. In this case, one set of
        # values is "randomly" generated.

        def get2Th():

            dart = np.random.uniform(low = 0.0, high = 1.0)
            test = lambda x: (th2CDF(x) - dart)

            xisol = root(test, 0.0).x[0]

            q = np.sqrt(self.E0com**2 - self.mth**2) # Magnitude of final-state two-momentum (from conservation of momentum/energy)

            return np.array([[q*np.cos(xisol), self.E0ex/self.E0com*q*np.sin(xisol) + self.p0ex*np.sin(self.psi/2)],
                             [- q*np.cos(xisol), - self.E0ex/self.E0com*q*np.sin(xisol) + self.p0ex*np.sin(self.psi/2)]])

        def get2Phi():

            dart = np.random.uniform(low = 0.0, high = 1.0)
            test = lambda x: (phi2CDF(x) - dart)

            xisol = root(test, 0.0).x[0]

            q = np.sqrt(self.E0com**2 - self.mph**2) # Magnitude of final-state two-momentum (from conservation of momentum/energy)

            return np.array([[q*np.cos(xisol), self.E0ex/self.E0com*q*np.sin(xisol) + self.p0ex*np.sin(self.psi/2)],
                             [- q*np.cos(xisol), - self.E0ex/self.E0com*q*np.sin(xisol) + self.p0ex*np.sin(self.psi/2)]])

        def get2Th2Ph():

            dart = np.random.uniform(low = 0.0, high = 1.0)
            pows = np.random.uniform(low = 1.0, high = 2.0, size = 5) # Randomly weights kinematic variables to pull away from extreme values
            test = lambda x: (th2ph2CDF(x) - dart)
            run = lambda t: np.array([t**pows[0], 4*self.mph**2 + ((np.sqrt(self.s) - 2*self.mth)**2 - 4*self.mph**2)*t**(pows[1]),
                                      2*np.pi*t**(pows[2]), 2*np.pi*t**(pows[3]), 2*np.pi*t**(pows[4])])
            t = 1.0 - 1e-5
            xold = run(t)
            # This part is a bit sketchy: it is essentially runs a by-hand iterator for the test function until
            # it determines it has hit the "dart" surface. It has a tendency to favor extreme values of variables, but it is tricky
            # to find a numerical solver that doesn't do this...
            while True:
                xnew = run(t)
                for i in range(len(xnew)):
                    if xnew[i] < run(0)[i]: xnew[i] = xold[i]
                tval = test(xnew)
                if tval < 0.0:
                    break
                xold = xnew
                t += - 0.01

            f12, s34, xi12, xi1, xi3 = xold
            s12 = ((np.sqrt(self.s) - np.sqrt(s34))**2 - 4*self.mth**2)*f12 + 4*self.mth**2

            lam0 = lammom(np.sqrt(self.s), np.sqrt(s12), np.sqrt(s34))
            lam12 = lammom(np.sqrt(s12), self.mth, self.mth)
            lam34 = lammom(np.sqrt(s34), self.mph, self.mph)

            Rmat = np.array([[1, 0, 0], [0, np.cos(xi12), - np.sin(xi12)], [0, np.sin(xi12), np.cos(xi12)]])
            b12 = lam0/np.sqrt(lam0**2 + s12)
            b34 = - lam0/np.sqrt(lam0**2 + s34)

            # Performs a series of boosts to get particle momenta in lab frame

            L12 = np.array([[1/np.sqrt(1 - b12**2), b12/np.sqrt(1 - b12**2), 0],
                            [b12/np.sqrt(1 - b12**2), 1/np.sqrt(1 - b12**2), 0],
                            [0, 0, 1]])
            L34 = np.array([[1/np.sqrt(1 - b34**2), b34/np.sqrt(1 - b34**2), 0],
                            [b34/np.sqrt(1 - b34**2), 1/np.sqrt(1 - b34**2), 0],
                            [0, 0, 1]])

            q112 = np.array([np.sqrt(lam12**2 + self.mth**2), lam12*np.cos(xi1), lam12*np.sin(xi1)])
            q212 = np.array([np.sqrt(lam12**2 + self.mth**2), - lam12*np.cos(xi1), - lam12*np.sin(xi1)])
            q334 = np.array([np.sqrt(lam34**2 + self.mph**2), lam34*np.cos(xi3), lam34*np.sin(xi3)])
            q434 = np.array([np.sqrt(lam34**2 + self.mph**2), - lam34*np.cos(xi3), - lam34*np.sin(xi3)])

            q1 = np.matmul(Rmat, np.matmul(L12, q112))
            q2 = np.matmul(Rmat, np.matmul(L12, q212))
            q3 = np.matmul(Rmat, np.matmul(L34, q334))
            q4 = np.matmul(Rmat, np.matmul(L34, q434))

            bex = - self.p0ex*np.sin(self.psi / 2)/(np.sqrt(self.p0ex**2 + self.mph**2))
            gam = 1/np.sqrt(1 - bex**2)

            Lex = np.array([[gam, 0, - gam*bex], [0, 1, 0], [- gam*bex, 0, gam]])

            return np.array([np.matmul(Lex, q1)[1:], np.matmul(Lex, q2)[1:], np.matmul(Lex, q3)[1:], np.matmul(Lex, q4)[1:]])

        def get4Th():

            dart = np.random.uniform(low = 0.0, high = 1.0)
            pows = np.random.uniform(low = 0.0, high = 1.0, size = 5) # Randomly weights kinematic variables to pull away from extreme values
            test = lambda x: (th4CDF(x) - dart)
            run = lambda t: np.array([t**pows[0], 4*self.mth**2 + ((np.sqrt(self.s) - 2*self.mth)**2 - 4*self.mth**2)*t**(pows[1]),
                                      2*np.pi*t**(pows[2]), 2*np.pi*t**(pows[3]), 2*np.pi*t**(pows[4])])
            t = 1.0 - 1e-5
            xold = run(t)
            # This part is a bit sketchy: it is essentially runs a by-hand iterator for the test function until
            # it determines it has hit the "dart" surface. It has a tendency to favor extreme values of variables, but it is tricky
            # to find a numerical solver that doesn't do this...
            while True:
                xnew = run(t)
                for i in range(len(xnew)):
                    if xnew[i] < run(0)[i]: xnew[i] = xold[i]
                tval = test(xnew)
                if tval < 0.0:
                    break
                xold = xnew
                t += - 0.01

            f12, s34, xi12, xi1, xi3 = xold
            s12 = ((np.sqrt(self.s) - np.sqrt(s34))**2 - 4*self.mth**2)*f12 + 4*self.mth**2

            lam0 = lammom(np.sqrt(self.s), np.sqrt(s12), np.sqrt(s34))
            lam12 = lammom(np.sqrt(s12), self.mth, self.mth)
            lam34 = lammom(np.sqrt(s34), self.mph, self.mph)

            Rmat = np.array([[1, 0, 0], [0, np.cos(xi12), - np.sin(xi12)], [0, np.sin(xi12), np.cos(xi12)]])
            b12 = lam0/np.sqrt(lam0**2 + s12)
            b34 = - lam0/np.sqrt(lam0**2 + s34)

            # Performs a series of boosts to get particle momenta in lab frame

            L12 = np.array([[1/np.sqrt(1 - b12**2), b12/np.sqrt(1 - b12**2), 0],
                            [b12/np.sqrt(1 - b12**2), 1/np.sqrt(1 - b12**2), 0],
                            [0, 0, 1]])
            L34 = np.array([[1/np.sqrt(1 - b34**2), b34/np.sqrt(1 - b34**2), 0],
                            [b34/np.sqrt(1 - b34**2), 1/np.sqrt(1 - b34**2), 0],
                            [0, 0, 1]])

            q112 = np.array([np.sqrt(lam12**2 + self.mth**2), lam12*np.cos(xi1), lam12*np.sin(xi1)])
            q212 = np.array([np.sqrt(lam12**2 + self.mth**2), - lam12*np.cos(xi1), - lam12*np.sin(xi1)])
            q334 = np.array([np.sqrt(lam34**2 + self.mth**2), lam34*np.cos(xi3), lam34*np.sin(xi3)])
            q434 = np.array([np.sqrt(lam34**2 + self.mth**2), - lam34*np.cos(xi3), - lam34*np.sin(xi3)])

            q1 = np.matmul(Rmat, np.matmul(L12, q112))
            q2 = np.matmul(Rmat, np.matmul(L12, q212))
            q3 = np.matmul(Rmat, np.matmul(L34, q334))
            q4 = np.matmul(Rmat, np.matmul(L34, q434))

            bex = - self.p0ex*np.sin(self.psi / 2)/(np.sqrt(self.p0ex**2 + self.mph**2))
            gam = 1/np.sqrt(1 - bex**2)

            Lex = np.array([[gam, 0, - gam*bex], [0, 1, 0], [- gam*bex, 0, gam]])

            return np.array([np.matmul(Lex, q1)[1:], np.matmul(Lex, q2)[1:], np.matmul(Lex, q3)[1:], np.matmul(Lex, q4)[1:]])

        poissonDist = lambda r, k: np.exp(k*np.log(r) - r - loggamma(k + 1))
        nPois = lambda r: quad(lambda y: poissonDist(r, y), 0.0, np.inf)[0] + 1e-5

        def poissonCDF(r, x):
            if x < 0.0: return 0
            return quad(lambda y: poissonDist(r, y), 0.0, x)[0]/nPois(r)

        def getN(r): # Generates number of interactions given an expectation following a Poisson distribution

            if r > 0:

                dart = np.random.uniform(low = 0.0, high = 1.0)
                test = lambda x: (poissonCDF(r, x) - dart)
                ksol = root(test, r).x[0]

                return int(np.round(ksol))

            else: return 0

        fig, ax = plt.subplots(figsize = (12, 12))

        # Static elements making up "detector"
        cal = patch.Annulus((0.0, 0.0), 2.5, 1.5, color = 'grey', alpha = 0.7)
        blackBox = patch.Circle((0.0, 0.0), 1.0, color = 'k')
        inBeamLine1 = patch.Rectangle((-2.5, -0.25), 2.5, 0.15, color = 'k')
        inBeamLine2 = patch.Rectangle((0.0, -0.25), 2.5, 0.15, color = 'k')
        outBeamLine1 = patch.Rectangle((0.0, 0.1), 2.5, 0.15, color = 'k')
        outBeamLine2 = patch.Rectangle((-2.5, 0.1), 2.5, 0.15, color = 'k')
        whitespace1 = patch.Wedge(0.0, 2.51, 0.925*np.pi*(180/np.pi), -0.925*np.pi*(180/np.pi), color = 'w')
        whitespace2 = patch.Wedge(0.0, 2.51, - 0.075*np.pi*(180/np.pi), 0.075*np.pi*(180/np.pi), color = 'w')
        inArrow1 = patch.Arrow(-3.0, -0.175, 0.45, 0.0, width = 0.1, color = 'r')
        inArrow2 = patch.Arrow(3.0, -0.175, - 0.45, 0.0, width = 0.1, color = 'r')
        outArrow1 = patch.Arrow(-2.55, 0.175, - 0.45, 0.0, width = 0.1, color = 'r')
        outArrow2 = patch.Arrow(2.55, 0.175, 0.45, 0.0, width = 0.1, color = 'r')

        def drawDet(): # Draws the detector on the figure

            ax.set_xlim([-3.05, 3.05])
            ax.set_ylim([-3.05, 3.05])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect(1)
            ax.axis('off')

            ax.add_patch(cal)
            ax.add_patch(whitespace1)
            ax.add_patch(whitespace2)
            ax.add_patch(blackBox)
            ax.add_patch(inBeamLine1)
            ax.add_patch(inBeamLine2)
            ax.add_patch(outBeamLine1)
            ax.add_patch(outBeamLine2)
            ax.add_patch(inArrow1)
            ax.add_patch(inArrow2)
            ax.add_patch(outArrow1)
            ax.add_patch(outArrow2)

        maxMom = self.p0ex/0.8 # Normalizes the max penetration depth of each line so that everything stays in the detector.

        lineCuts = 10 # Number of segments in a track
        lines = []
        alphas = []

        def animate(frame): # Generates animation

            ax.clear()
            drawDet()

            i = 0
            while i < len(alphas): # Fades each track every time step; removes if alpha is zero
                if alphas[i] - 0.1 <= 0:
                    lines.pop(i)
                    alphas.pop(i)
                else:
                    alphas[i] = alphas[i] - 0.1
                    lines[i].set_alpha(alphas[i])
                    i += 1

            numInts = [getN(totRateS[i]*dt) for i in range(4)] # Number of each interactions per frame
            qPhi = np.array([]) # Final-state phi momenta
            qTh = np.array([]) # Final-state theta momenta
            # The following generate all phi's and theta's given the number of each type of interaction per frame
            for n in range(numInts[0]):
                lst = get2Phi()
                if qPhi.size == 0: qPhi = np.array([x for x in lst])
                else: 
                    for i in range(2):
                        qPhi = np.vstack((qPhi, lst[i]))
            for n in range(numInts[1]):
                lst = get2Th()
                if qTh.size == 0: qTh = np.array([x for x in lst])
                else: 
                    for i in range(2):
                        qTh = np.vstack((qTh, lst[i]))
            for n in range(numInts[2]):
                lst = get4Th()
                if qTh.size == 0: qTh = np.array([x for x in lst])
                else: 
                    for i in range(4):
                        qTh = np.vstack((qTh, lst[i]))
            for n in range(numInts[3]):
                lst = get2Th2Ph()
                if qPhi.size == 0: qPhi = np.array([lst[2], lst[3]])
                else: 
                    qPhi = np.vstack((qPhi, lst[2]))
                    qPhi = np.vstack((qPhi, lst[3]))
                if qTh.size == 0: qTh = np.array([lst[0], lst[1]])
                else:
                    qTh = np.vstack((qTh, lst[0]))
                    qTh = np.vstack((qTh, lst[1]))

            for q in qPhi: # Adds track for each phi which enters the detector body

                if 0.075*np.pi < np.abs(np.arctan(q[1]/q[0])) < 0.925*np.pi or 1.075*np.pi < np.abs(np.arctan(q[1]/q[0])) < 1.925*np.pi:

                    depth = 1.5*np.sqrt(q[0]**2 + q[1]**2)/maxMom + 1.0 # Penetration depth (determined by momentum: higher momentum, more penetration)
                    direct = [q[0]/np.sqrt(q[0]**2 + q[1]**2), q[1]/np.sqrt(q[0]**2 + q[1]**2)] # Track direction

                    for i in range(lineCuts):
                        lines.append(Line2D([(1.0 + (depth - 1.0)*i/lineCuts)*direct[0], ((1.0 + (depth - 1.0)*(i + 1)/lineCuts) - 0.001)*direct[0]],
                            [(1.0 + (depth - 1.0)*i/lineCuts)*direct[1], ((1.0 + (depth - 1.0)*(i + 1)/lineCuts) - 0.001)*direct[1]], color = 'r',
                            alpha = (0.1/(1.1 - (i + 1)/lineCuts))**1.5))
                        alphas.append((0.1/(1.1 - (i + 1)/lineCuts))**1.5) # Line opacity determined by 1/p^1.5 relation (1.5 chosen for visualization)

            for q in qTh: # Adds track for each theta which enters the detector body

                if 0.075*np.pi < np.abs(np.arctan(q[1]/q[0])) < 0.925*np.pi or 1.075*np.pi < np.abs(np.arctan(q[1]/q[0])) < 1.925*np.pi:

                    depth = 1.5*np.sqrt(q[0]**2 + q[1]**2)/maxMom + 1.0 # Penetration depth (determined by momentum: higher momentum, more penetration)
                    direct = [q[0]/np.sqrt(q[0]**2 + q[1]**2), q[1]/np.sqrt(q[0]**2 + q[1]**2)] # Track direction

                    for i in range(lineCuts):
                        lines.append(Line2D([(1.0 + (depth - 1.0)*i/lineCuts)*direct[0], ((1.0 + (depth - 1.0)*(i + 1)/lineCuts) - 0.001)*direct[0]],
                            [(1.0 + (depth - 1.0)*i/lineCuts)*direct[1], ((1.0 + (depth - 1.0)*(i + 1)/lineCuts) - 0.001)*direct[1]], color = 'b',
                            alpha = (0.1/(1.1 - (i + 1)/lineCuts))**1.5))
                        alphas.append((0.1/(1.1 - (i + 1)/lineCuts))**1.5) # Line opacity determined by 1/p^1.5 relation (1.5 chosen for visualization)

            for i in range(len(lines)): ax.add_line(lines[i])

            print(' Animating: {:.2f} % Done'.format(float(frame / len(t))*100), end = '\r')

            return lines, alphas,

        anim = animation.FuncAnimation(fig, animate, frames = len(t), interval = 1, blit = True)

        if not os.path.exists(directory): os.makedirs(directory)

        writergif = animation.PillowWriter(fps = fps)
        anim.save(directory + filename, writer = writergif, dpi = 150)

        print(' Finished! Saved as {}'.format(directory + filename))

if __name__ == '__main__':

    # Example code 

    beamangle = np.pi/12
    m1 = 0.1 # GeV: phi mass
    m2 = 1.0 # GeV: theta mass
    beamwidth = 1e13 # GeV^-1: corresponds to ~mm beam width
    beamdens = 1e-21 # GeV^2: corresponds to ~10^6 particles per cm^2 in beam
    coup = 0.1 # quadratic coupling between theta and phi
    p = 2.1 # GeV: beam momentum

    test = detector(p, beamangle, mphi = m1, mth = m2, sigb = beamwidth, n = beamdens, lam = coup)
    test.createAnimation(15)
