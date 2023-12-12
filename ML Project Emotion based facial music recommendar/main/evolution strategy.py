import numpy as np
import copy
import matplotlib.pyplot as plt

'''Objective Function'''


def myfunc(x):
    z = 0
    for n in range(len(x)):
        z = z + x[n] ** 2
    y = np.sqrt(z)
    return y


def recombine(X, lper=1.0, r=1.0, method='intermediate'):
    L = int(np.floor(np.size(X, 0) * lper))
    '''Recombination in Evolution Strategy'''
    Q = copy.deepcopy(X)
    M = int(np.floor(np.size(Q, 0) * r))
    P = np.zeros(shape=(M, np.size(Q, 1)))
    if method.lower() == 'intermediate':
        for m in range(L):
            p = np.random.choice(np.size(Q, 0), M)
            q = Q[p, :]
            for n in q:
                P[m, :] = P[m, :] + (1 / M) * n
    elif method.lower() == 'discrete':
        for m in range(L):
            p = np.random.choice(range(np.size(Q, 0)), M - 1)
            q = Q[p, :]
            pp = np.zeros(np.size(Q, 1))
            for n in range(np.size(Q, 1)):
                pos = np.random.choice(range(M))
                pp[n] = q[pos, n]
            P[m, :] = pp
        # pass
    else:
        KeyError("Method should either be 'intermediate' or 'discrte'. Check the spelling instead")
    return P


def evostrat(npop, nvar, xlims, maxiteration, stdx=1.0):
    stdprof = np.zeros(maxiteration)
    '''Evolutionary Strategy Program'''
    X = np.array(np.random.uniform(low=xlims[0], high=xlims[1], size=(npop, nvar)))  # Initialization of solutions
    eval_x = np.array(np.zeros(shape=(npop, 1)))  # Evaluation
    for m in range(npop):
        eval_x[m] = myfunc(X[m, :])
    minc = np.ones(maxiteration) * 2000.0
    iter = 0
    while iter < maxiteration:
        ymin = np.min(eval_x)
        # Xr = recombine(X, 1.0, 1.0)
        Xr = copy.deepcopy(X)
        for n in range(npop):
            Xr[n] += np.random.normal(loc=0.0, scale=stdx, size=nvar)  # Mutate
            y = myfunc(Xr[n, :])  # Evaluate
            if y <= eval_x[n]:
                X[n] = copy.copy(Xr[n])
                eval_x[n] = copy.copy(y)
                if y < ymin:
                    ymin = copy.copy(ymin)
            if ymin < minc[iter]:
                minc[iter] = copy.copy(ymin)
            stdprof[iter] = np.std(eval_x)
        print(f"Iteration: {iter}, minimum objective value {np.min(eval_x)}")
        iter += 1

    return X, eval_x, minc, stdprof


inputs, objvals, minc, std_profile = evostrat(100, 5, [-5.0, 5.0], 1000, stdx=0.5)
# plt.semilogy(range(1000),minc)
plt.semilogy(range(1000), std_profile)
plt.xlabel('iteration count $(iter)$')
plt.ylabel('$y_{\min}(iter)$')
plt.title('Convergence Profile')
plt.show()