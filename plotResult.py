import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker

def doPlots(N):
    		    # 50      100     250     500     750     1000
    heuristics = [66540,  103436, 405662, 78579,  134752, 75446 ]
# Init NN         ____sec ___sec  ___sec  ___sec  70sec   ___sec
# Init NN2        1.69sec ___sec  ___sec   32sec  70sec   132sec
# Init valid_NN2  ___sec ___sec  ___sec   ___sec  69sec   ___sec
# alpha                                0.2     0.1

    if N == 50 : repeatNum = heuristics[0]
    if N == 100: repeatNum = heuristics[1]
    if N == 250: repeatNum = heuristics[2]
    if N == 500: repeatNum = heuristics[3]
    if N == 750: repeatNum = heuristics[4]
    if N == 1000: repeatNum = heuristics[5]

    # if set at 0 that means I am displaying the raw values meanOb and bestF without optimize
    skip_Inf = 1

    file = open('result.csv')
    metaData = np.loadtxt(file, dtype = 'float', delimiter="," ,skiprows = 1, max_rows = 1)
    popSize        = metaData[0]
    offsprSize     = metaData[1]
    k_tour         = metaData[2]
    alpha          = metaData[3]
    its            = metaData[4]
    lso_ON         = metaData[5]
    lso_percen     = metaData[6]
    greedy_percen  = metaData[7]
    file.close()

    if lso_ON > 0: lso_ON = f'{lso_percen*100:.0f}% ON'
    else         : lso_ON = 'OFF'

    numGreedyInd = f'({round(popSize * greedy_percen)} ind)'
    greedy_percen = f'{greedy_percen*100:.0f}%'

    metaData_rows = 2 # !!! DONT CHANGE
    startData = 2 + skip_Inf # !!! DONT CHANGE
    file = open('result.csv')
    Iterations = np.loadtxt(file, dtype = 'int', delimiter="," ,skiprows = startData + metaData_rows, usecols = 0)
    file.close()

    file = open('result.csv')
    meanFit = np.loadtxt(file, dtype = 'float', delimiter="," ,skiprows = startData + metaData_rows, usecols = 1)
    file.close()

    file = open('result.csv')
    bestFit = np.loadtxt(file, delimiter="," ,skiprows = startData+ metaData_rows, usecols = 2)
    file.close()

    heuristicY = [repeatNum] * len(Iterations)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    # Including mu
    # fig.suptitle(f"TSP = [ Tour-{N} Size ]  ·  Parameters = [ λ: {popSize:.0f}   μ: {offsprSize:.0f}   k-tour: {k_tour:.0f}   α: {alpha:.2f}   lso: {lso_ON}    pGrdy: {greedy_percen} {numGreedyInd} ]" , fontsize=15.5)
    fig.suptitle(f"TSP = [ Tour-{N} Size ]  ·  Parameters = [ λ: {popSize:.0f}    pGrdy: {greedy_percen} {numGreedyInd}   k-tour: {k_tour:.0f}   α: {alpha:.2f}   lso: {lso_ON} ]" , fontsize=15.5)
    fig.set_figheight(5)
    fig.set_figwidth(14)
    fig.subplots_adjust(top=0.85)

    ''' Semilogy'''
    ax1.semilogy(Iterations, meanFit, label = 'Mean Fitness')
    ax1.semilogy(Iterations, bestFit, label = 'Best Fitness')
    ax1.semilogy(Iterations, heuristicY, label = 'Greedy Fitness')
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))
    ax1.yaxis.set_minor_formatter(ticker.FormatStrFormatter('%.0f'))

    ax1.set_title(f"Logarithmic Scale")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Fitness Value")
    ax1.legend()
    ax1.grid(True, which="both",linewidth = 0.20)
    # plt.show()

    ''' Normal Plot'''
    # fig, ax = plt.subplots()
    # fig.set_figheight(4)
    # fig.set_figwidth(6)
    ax2.plot(Iterations, meanFit, label = 'Mean Fitness')
    ax2.plot(Iterations, bestFit, label = 'Best Fitness')
    ax2.plot(Iterations, heuristicY, label = 'Greedy Fitness')

    ax2.ticklabel_format(style='plain')
    ax2.set_title(f"Normal Scale")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Fitness Value")
    ax2.legend()
    plt.subplots_adjust( wspace=0.335)
    ax2.grid( linestyle = '-', linewidth = 0.20)
    plt.show()

'''
Initialization:
- en el 1000 the more and more ind are mutated the more they loose their respective good fitness
- Implement if spend more than 2 sec return ramd indiv

PARAMTERS:
init time
70 do k torunament, 30 fitness sharing


Tour 1000
((298953)/(1000*1000))*100 -> 30% infinity
Tour 750
((112702)/(750*750))*100   -> 20% infinity
Tour 500
((112702)/(750*750))*100   -> 10% infinity
'''
doPlots(750)
