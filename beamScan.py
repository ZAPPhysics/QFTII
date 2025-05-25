from detectorAnim import *
import numpy as np
import matplotlib.pyplot as plt

beamangle = np.pi/12
m1 = 0.1 # GeV: phi mass
m2 = 1.0 # GeV: theta mass
m3 = 5.0 # GeV: psi mass
beamwidth = 1e13 # GeV^-1: corresponds to ~mm beam width
beamdens = 1e-21 # GeV^2: corresponds to ~10^6 particles per cm^2 in beam
coup = 0.1 # quadratic coupling between theta and phi
g1 = 0.1 # trilinear theta^2*psi coupling
g2 = 0.01 # trilinear phi^2*psi coupling

beammom = np.linspace(0.75, 4.0, 100) # Beam momentum scan range
b1 = beammom
b2 = beammom
b3 = beammom
b4 = beammom
rate1 = []
rate2 = []
rate3 = []
rate4 = []

for p in beammom: # Factor of 1.54e24 converts rates to 1/seconds

    rs = detector(p, beamangle, mphi = m1, mth = m2, 
                sigb = beamwidth, n = beamdens, lam = coup,
                gth = g1, gph = g2, mps = m3).rates()
    rate1.append(1.54e24*rs[0])
    if rs[1] == 0.0: b2 = np.delete(b2, 0, 0)
    else: rate2.append(1.54e24*rs[1])
    if rs[2] == 0.0: b3 = np.delete(b3, 0, 0)
    else: rate3.append(1.54e24*rs[2])
    if rs[3] == 0.0: b4 = np.delete(b4, 0, 0)
    else: rate4.append(1.54e24*rs[3])

    print('Basic Scan: {:.2f}'.format(p/beammom[-1]*100), '% Complete', end = '\r')

print('Finished Basic Scan                  ')

if m2 > m1:

    b1ext = np.arange(b1[0],
            np.sqrt((m2**2 - m1**2)/np.cos(beamangle/2)**2) + 0.3, 0.001)

    for p in b1ext:
        rate1.append(1.54e24*detector(p, beamangle, mphi = m1, mth = m2, 
                sigb = beamwidth, n = beamdens, lam = coup,
                gth = g1, gph = g2, mps = m3).rates([0])[0])
        print('Two-Theta Resonance Region: {:.2f}'.format(
                (p - b1ext[0] + 0.001)/(b1ext[-1] - b1ext[0])*100), 
                '% Complete', end = '\r')

    print('Finished Two-Theta Resonance Region                  ')

    b1 = np.concatenate((b1, b1ext))

    b2ext = np.arange(np.sqrt((m2**2 - m1**2)/np.cos(beamangle/2)**2) - 1e-4, 
                np.sqrt((m2**2 - m1**2)/np.cos(beamangle/2)**2) + 0.1, 0.001)

    for p in b2ext:
        rate2.append(1.54e24*detector(p, beamangle, mphi = m1, mth = m2, 
                sigb = beamwidth, n = beamdens, lam = coup,
                gth = g1, gph = g2, mps = m3).rates([1])[0])
        print('Two-Theta Threshold Region: {:.2f}'.format(
                (p - b1ext[0] + 0.001)/(b1ext[-1] - b1ext[0])*100), 
                '% Complete', end = '\r')

    print('Finished Two-Theta Threshold Region                  ')

    b2 = np.concatenate((b2, b2ext))

if 2*m2 > m1:

    b3ext = np.arange(np.sqrt((4*m2**2 - m1**2)/np.cos(beamangle/2)**2) + 0.001, 
                np.sqrt((4*m2**2 - m1**2)/np.cos(beamangle/2)**2) + 0.25, 0.01)

    for p in b3ext:
        rate3.append(1.54e24*detector(p, beamangle, mphi = m1, mth = m2, 
                sigb = beamwidth, n = beamdens, lam = coup,
                gth = g1, gph = g2, mps = m3).rates([2])[0])
        print('Four-Theta Threshold Region: {:.2f}'.format(
                (p - b3ext[0] + 0.01)/(b3ext[-1] - b3ext[0])*100), 
                '% Complete', end = '\r')

    print('Finished Four-Theta Threshold Region                  ')

    b3 = np.concatenate((b3, b3ext))

b4ext = np.arange(np.sqrt(((m2 + m1)**2 - m1**2)/np.cos(beamangle/2)**2) + 0.001, 
            np.sqrt(((m2 + m1)**2 - m1**2)/np.cos(beamangle/2)**2) + 0.5, 0.01)

for p in b4ext:
    rate4.append(1.54e24*detector(p, beamangle, mphi = m1, mth = m2, 
            sigb = beamwidth, n = beamdens, lam = coup,
                gth = g1, gph = g2, mps = m3).rates([3])[0])
    print('Two-Theta, Two-Phi Threshold Region: {:.2f}'.format(
            (p - b4ext[0] + 0.01)/(b4ext[-1] - b4ext[0])*100), 
            '% Complete', end = '\r')

print('Finished Two-Theta, Two-Phi Threshold Region                  ')

b4 = np.concatenate((b4, b4ext))

bres = np.arange(m3/np.cos(beamangle/2)/2 - 0.25, m3/np.cos(beamangle/2)/2 + 0.25, 0.01)

for p in bres:
    rs = detector(p, beamangle, mphi = m1, mth = m2, 
                sigb = beamwidth, n = beamdens, lam = coup,
                gth = g1, gph = g2, mps = m3).rates()
    rate1.append(1.54e24*rs[0])
    if m3 > 2*m2: rate2.append(1.54e24*rs[1])
    if m3 > 2*m2 + 2*m1: rate3.append(1.54e24*rs[2])
    if m3 > 4*m2: rate4.append(1.54e24*rs[3])

    print('Psi Resonance Region: {:.2f}'.format(p/bres[-1]*100), '% Complete', end = '\r')

print('Finished Resonance Region, Two-Phi Threshold Region                  ')

b1 = np.concatenate((b1, bres))
if m3 > 2*m2: b2 = np.concatenate((b2, bres))
if m3 > 2*m2 + 2*m1: b3 = np.concatenate((b3, bres))
if m3 > 4*m2: b4 = np.concatenate((b4, bres))

ord1 = np.argsort(b1)
ord2 = np.argsort(b2)
ord3 = np.argsort(b3)
ord4 = np.argsort(b4)

non_zero_1 = [x if x != 0 else 1e6 for x in rate1]
non_zero_2 = [x if x != 0 else 1e6 for x in rate2]
non_zero_3 = [x if x != 0 else 1e6 for x in rate3]
non_zero_4 = [x if x != 0 else 1e6 for x in rate4]

maxes = [max(rate1), max(rate2), max(rate3), max(rate4)
        ]

mins = [min(non_zero_1), min(non_zero_2), min(non_zero_3), min(non_zero_4)
        ]

fig, ax = plt.subplots(figsize = (14, 8))

ax.set_yscale('log')

ax.set_ylim([1e-3, max(maxes)*10])

ax.plot(b2[ord2], np.array(rate2)[ord2], 'k-', 
        label=r'$\phi\phi\to\theta\theta$', linewidth = 2.0)
ax.plot(b1[ord1], np.array(rate1)[ord1], 'r-', 
        label=r'$\phi\phi\to\phi\phi$', linewidth = 2.0)
ax.plot(b3[ord3], np.array(rate3)[ord3], 'k--', 
        label=r'$\phi\phi\to\theta\theta\theta\theta$', linewidth = 2.0)
ax.plot(b4[ord4], np.array(rate4)[ord4], 'r--', 
        label=r'$\phi\phi\to\theta\theta\phi\phi$', linewidth = 2.0)

ax.set_title(r'$\psi = {:.3f}$, $m_\phi = {:.2f}$ GeV, $m_\theta = {:.2f}$ GeV, $m_\psi = {:.2f}$ GeV, $n = {:.2E}$ GeV$^2$, $\sigma_b = {:.2E}$ 1/GeV'.format(
            beamangle, m1, m2, m3, beamdens, beamwidth), fontsize = 16)

ax.set_xlabel(r'$\mathrm{Beam\;Momentum\;[GeV]}$', fontsize = 16)
ax.set_ylabel(r'$\mathrm{Scattering\;Rate}\;[\mathrm{s}^{-1}]$', fontsize = 16)

ax.legend()

plt.savefig(r'./beamScanRes.pdf')

