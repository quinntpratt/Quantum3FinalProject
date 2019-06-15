#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 11:31:42 2019

@author: Quinn Pratt
All Rights Reserved
"""

import numpy as np
from math import sin,sqrt,pi
import scipy.special as sp
import matplotlib.pyplot as plt

def classicalKR(K,l0=0,th0=0,N=100):
    dT= 1e-3
    t = np.linspace(0,N*dT,N)
    theta = np.zeros_like(t)
    l = np.zeros_like(t)
    
    theta[0] = th0
    l[0] = l0
    
    for i in range(N-1):
        l[i+1] = l[i] + K*sin(theta[i])
        theta[i+1] = theta[i] + l[i+1]
        if theta[i+1] >= 2*np.pi:
            theta[i+1] = theta[i+1] - 2*np.pi
        
    return theta,l

def quantumKR(K,tau,l0=0,N=20):
    ''' Quantum kicked-rotor: 
    1. Prepare an initial wavefunction (in the eigen-basis)
    as a pure-state of the rotational-term in the hamiltonian.
    2. Evolve this state according to the iteration routine by evolving the
    vector of coefficients of which modes are components of the wavefunction.
    3. Use this matrix of coefficients (1 vector of coeffs for a given n) to evolve a wavefunction.
    '''
    # Number of kick-iterations: N
    
    # Preallocate a ranges of l-values, this sets the dimensionality of the matricies.
    lmax = 200
    l = np.arange(-lmax,lmax+1)
    m = np.arange(-lmax,lmax+1)
    # Preapare a theta-basis
    theta = np.linspace(0,2*pi,750)
    # preallocate a wavefunction vs. time matrix:
    psi = np.zeros(shape=(len(theta),N),dtype=complex)
    # preallocate a complex coefficients matrix vs. time matrix:
    A = np.zeros(shape=(len(l),N),dtype=complex)
    # Initialize the coefficients in a pure l = 0 state.
    A[np.where(l==l0),0] = 1.
    # Build the unitary evolution matrix:
    U = evolutionMatrix(l,m,K,tau)
    
    # loop through n-steps, propogating the coefficients along using the Unitary operator
    for i in range(N-1):
        A[:,i+1] = np.matmul(U,A[:,i])
    
    # For each n-step...
    for i in range(N):
        # For each l... sum the contributions from all modes.
        # For a given N, we sum over all the A's and eigenfunctions
        psi_un = np.zeros(len(theta),dtype=complex)
        for li in range(len(l)):
            psi_un += (1/sqrt(2*pi))*A[li,i]*np.exp(1j*l[li]*theta)
        # Normalize the wavefunction after taking the superposition of these modes.
        psi[:,i] = psi_un/np.sqrt(np.sum(np.conj(psi_un)*psi_un))
    
    return U,A,theta,psi


def evolutionMatrix(l,m,K,tau):
    # preallocate
    U = np.zeros(shape=(len(l),len(m)),dtype=complex)
    for li in range(len(l)):
        for mi in range(len(m)):
            U[li,mi] = (1j)**(l[li] - m[li])*np.exp(-1j*tau*(l[li]**2.)/2)*sp.jv(abs(l[li] - m[mi]),K)
    return U



K = 0.1
# Number of Kicked-rotators in classical ensemble
M = 10
# Number of kicks
N = 300
# Initial Condition for l
l0 = 0
# Range of initial theta conditions
th0 = np.array([2*(m - 1/2)*pi/M for m in range(M)])
l0 = np.linspace(0,8,M)
# Preallocate energy result for each kick
E = np.zeros(N)
fig = plt.figure('K = {}'.format(K))
ax = plt.gca()
for i in range(M):
    for j in range(M):
        theta, l = classicalKR(K,th0=th0[i],l0=l0[j],N=N)
        E += np.square(l)
        ax.plot(theta,l,'.',markersize=0.75)

Ecl = E/(2*M)
 
ax.set_xlim([0,2*np.pi])
ax.set_ylim([0,2*np.pi])
ax.set_xlabel(r'$\theta$',fontsize=16)
ax.set_ylabel(r'$l$',fontsize=16)
ax.set_title('K = {}'.format(K),fontsize=20)
ax.set_aspect('equal')
plt.tight_layout()


'''
plt.figure('Energy')
plt.plot(range(N),Ecl)
plt.xlabel('N')
plt.ylabel(r'$\langle E \rangle$')
'''

'''
Quantum Kicked Rotor
'''

K = 5
tau = 1.1
# Initial Angular momentum eigenstate
l0 = 0.
# Number of kicks
N = 100
U, A, theta, psi = quantumKR(K,tau,l0=l0,N=N)

plt.figure()
ax = plt.gca()
colors = plt.cm.brg(np.linspace(0,1,N))
for i in range(0,N,10):
    ax.plot(theta,psi[:,i],color=colors[i])

ax.set_xlabel(r'$\theta$',fontsize=16)
ax.set_ylabel(r'Re($\psi$)',fontsize=16)
ax.set_title('k = {0}, '.format(K) + r'$\tau = 1$',fontsize=20)
plt.tight_layout()

## Now, we actually need to compute the expectation value of something... using these psi's
# we will follow the book and compute the expected value of the energy.

# Loop over time, compute expected values...
exp_theta = np.zeros(N,dtype=complex)
Eqm = np.zeros(N,dtype=complex)
for i in range(N):
    exp_theta[i] = np.trapz(np.conj(psi[:,i])*theta*np.conj(psi[:,i]),x=theta)
    #l2psi = -(1/np.sin(theta))*np.gradient(np.sin(theta)*np.gradient(psi[:,i])) 
    Eqm[i] = np.trapz(np.conj(psi[:,i])*0.5*-np.gradient(np.gradient(psi[:,i])))
    #exp_l2[i] = np.sum(np.conj(psi[:,i])*0.5*l2psi)

  
plt.figure('QM Energy')
plt.plot(range(N),Eqm,label=r'$\tau$ = {:.2f}'.format(tau),linewidth=2)
#plt.plot([0,100],[0,0.01],'k--') 
plt.xlabel('N',fontsize=16)
plt.ylabel(r'$\langle E \rangle$ (a.u.)',fontsize=16)
plt.legend(fontsize=16)
plt.tight_layout()


'''
Attempt at cerating a quantum phase-space portrait:
'''

K = 1
tau = np.linspace(5, 6, 10)
plt.figure('QM Phase')
ax = plt.gca()
for i in range(len(tau)):
    _, _, _, psi = quantumKR(K,tau[i],l0=l0,N=N)
    for i in range(N):
        exp_theta[i] = np.trapz(np.conj(psi[:,i])*theta*np.conj(psi[:,i]),x=theta)
        Eqm[i] = np.trapz(np.conj(psi[:,i])*0.5*-np.gradient(np.gradient(psi[:,i])))
    ax.plot(exp_theta,np.sqrt(2*Eqm),'.')

ax.set_xlabel(r'$\langle\theta\rangle$',fontsize=16)
ax.set_ylabel(r'$\langle l \rangle$',fontsize=16)
ax.set_title(str(tau[0]*K) + r' < $k\tau$ < ' + str(tau[-1]*K),fontsize=20)
#plt.axis('equal')
plt.tight_layout()
