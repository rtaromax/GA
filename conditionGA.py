#!/usr/bin/env python3

# Random paring and one-point crossover continuous GA


from scipy.integrate import odeint
from pylab import *
from math import exp,log,erfc,sqrt
import matplotlib.pyplot as plt
from random import random, sample, uniform, shuffle
import time
from scipy.stats import chisquare
from statistics import variance
import multiprocessing as mp

def simulation(individual):
    weight = 75
    BMI = 25
    D = 75000
    Vgb = (70/sqrt(BMI/22))*weight/100
    G0 = individual[26]#124.2
    Gb = G0
    I0 = individual[27]#5
    Ib = I0
    Gbday = [Gb]
    Iday = [Ib]
    
    Vgk = Vgb*0.15
    Qgk = Vgk*0.85
    Vgbra = Vgb*0.15
    Qgbra = Vgbra*0.85
    Vgh = Vgb*0.05
    Qgh = Vgh*3
    Vgpr = Vgb*0.44
    Qgpr = Vgpr*0.85
    Vgl = Vgb*0.18
    Qgl = Vgl*0.85
    Qgha = 0.25*Vgl/2
    
    R1 = individual[0]           #0.25      Primary rate of glucose excretion
    R2 = individual[1]           #0.1       Secondary rate of glucose excretion
    R3 = individual[2]           #17        Glucose concentration required before glucose excretion started
    R4 = individual[3]           #0         Glucose concentration basal threshold

    B1 = individual[4]           #0.5       Maximum glucose uptake
    B2 = individual[5]           #0.4       Rate of glucose uptake

    PE1 = individual[6]          #19.17     Maximum glucose uptake
    PE2 = individual[7]          #7.79      Primary rate of glucose uptake
    PE3 = individual[8]          #0.03      Secondary rate of glucose uptake
    PE4 = individual[9]          #251.97    Insulin concentration effect
    PE5 = individual[10]         #1         Sensitivity of glucose uptake to insulin

    G1 = individual[11]          #0.5       Maximum glucose uptake
    G2 = individual[12]          #7.79      Rate of glucose uptake

    P1 = individual[13]          #100       beta-Cell function of the islets
    P2 = individual[14]          #1254.64   Maximum output of insulin
    P3 = individual[15]          #3         Glucose concentration required before insulin release started
    P4 = individual[16]          #3.16      Rate of insulin secretion modifier 1
    P5 = individual[17]          #10        Rate of insulin secretion modifier 2

    L1 = individual[18]          #2.15      Initial glucose output
    L2 = individual[19]          #8.9       Insulin concentration effect
    L3 = individual[20]          #0.89      Maximum glucose output
    L4 = individual[21]          #0         Rate of glucose output
    L5 = individual[22]          #1         Sensitivity of glucose output to insulin>

    mealtime = [100]

    kmin = 0.013+individual[28] #uniform(-0.002,0.002)  
    kmax = 0.045+individual[29] #uniform(-0.004,0.004)
    kgri = kmax
    kabs = 0.205+individual[30] #uniform(-0.022,0.022)
    b = 0.85+individual[31] #uniform(-0.02,0.02)
    Qsto1 = 0
    Qsto2 = 0
    Qgut = 0
    F = 1

    Iin = individual[23]        #insulin infusion rate (0-2)
    ICmax = 150    #uU/ml maximum insulin clearance rate
    EHC = 2300     #enzyme's half- saturation value

    ##### GLUCOSE #####

    ### kidney ###
    def kidglu(gk,t):
        GE = kidge(Gb)
        return (Qgk*(Gb-gk)-GE)/Vgk

    ### gastro ###
    def gasglu(ggt,t):
        global Gas
        GA = F*kabs*Qgut
        Gas = GA
        return (Qggt*(Gb-ggt)+GA-GGU(Gb))/Vggt

    ### brain ###
    def braglu(gbra,t):
        return (Qgbra*(Gb-gbra)-BGU(Gb))/Vgbra

    ### heart ###
    def heaglu(gh,t):
        return (Qgh*(Gb-gh))/Vgh

    ### periphery ###
    def perglu(gpr,t):
        return (Qgpr*(Gb-gpr)-PRGU(Gb,Ib))/Vgpr

    ### liver ###
    def livglu(gl,t):
        return (Qgha*Gb+Qggt*Ggt-Qgl*gl+LGP(Gb,Ib))/Vgl
    
    ### blood ###
    def bloglu(gb,t):
        return (Qgh*(gb-Gh)+Qgbra*(Gbra-gb)+Qgk*(Gk-gb)+Qgpr*(Gpr-gb)+Qgl*Gl-(Qgha+Qggt)*gb)/Vgb



    ##### GLUCOSE ABSORBTION #####

    def kempt(qsto):
        return kmin+(kmax-kmin)/2*(tanh((5/(2*D*(1-b)))*(qsto-b*D))+1)
    def impulse(time):
        if time in mealtime:
            return 1
        else:
            return 0
    def qsto1(q1,t):
        return -kgri*q1+D*impulse(iteration)
    def qsto2(q2,t):
        return -Kempt*q2+kgri*Qsto1
    def qgut(qg,t):
        return -kabs*qg+Kempt*Qsto2




    ##### LIVER GLUCOSE PRODUCTION #####

    def LGP(Gb,Ib):
        eg = ((Gb*0.055)**2)/4.85
        uh = Ib*L5
        return L1*(1-tanh(L4*uh))*(1-((L3*eg)/(eg+L2)))



    ##### GLUCOSE UTILIZATION #####

    ### kidney ###
    def invErfCht(inp):
        pi = 3.141592653589793;
        if inp < 0.845:
            return 0.5*pi**0.5*(inp+inp**3)*(pi/12);
        else:
            flog = 2/(pi * ((inp-1)*(inp-1)));
            slog = log(flog);
            return 1/sqrt(2) * sqrt(log(flog)-log(slog));

    def kidge(Gb):
        t1 = R3-invErfCht(1-10*R2)/R1
        Kgrad = R1*exp(-R1**2)*(t1-R3)**2
        Kint = R2-t1*Kgrad
        k1 = erfc(-1*(Gb*0.055-R3)*R1)*0.1    # Gb: 1 mg/dl = 0.055 mmol/L
        k2 = Gb*0.055*Kgrad +Kint
        if k2<R2:
            return (k1+R4)/0.055
        else:
            return (k2+R4)/0.055

    ### brain ###
    def BGU(Gb):
        return B1*tanh(B2*Gb*0.055)/0.055

    ### periphery ###
    def PRGU(Gb,Ib):
        eg = (Gb*0.055)*((P1/100)**0.25)
        u = Ib*PE5
        return PE1*(eg/(eg+PE2))*(PE3+(u/(u+PE4))**2)/0.055

    ### gut ###
    def GGU(Gb):
        return G2*(Gb*0.055/(Gb*0.055+G1))/0.055



    ##### INSULIN #####
    def bloisu(i,t):
        return Iin+Ip-ICmax*i/(EHC+i)
    
    #### INSULIN RELEASE ####
    def isurel(Gb):
        eg = (Gb*0.055)*((P1/100)**0.25)
        return ((P1/100)**0.75)*(P2*(eg-P3)**P4)/((P5**P4)+((eg-P3)**P4))



    ##### MAIN CIRCLE #####

    for iteration in range(220):
        t = [0,1]

        dtan = (Gb-G0)/G0*4
        Vggt = (individual[24]*tanh(dtan)+individual[25])*Vgb*0.1
        Qggt = Vggt*0.85

        Qsto = Qsto1+Qsto2
        Kempt = kempt(Qsto)
        Qsto1 = float(odeint(qsto1,Qsto1,t)[1])
        Qsto2 = float(odeint(qsto2,Qsto2,t)[1])
        Qgut = float(odeint(qgut,Qgut,t)[1])

        Ggt = float(odeint(gasglu,Gb,t)[1])
        Gl = float(odeint(livglu,Gb,t)[1])
        Gh = float(odeint(heaglu,Gb,t)[1])
        Gk = float(odeint(kidglu,Gb,t)[1])
        Gpr = float(odeint(perglu,Gb,t)[1])
        Gbra = float(odeint(braglu,Gb,t)[1])

        Gb = float(odeint(bloglu,Gb,t)[1])
        Ip = sqrt(abs(isurel(Gb)**2))/7.175/4.5
        Ib = float(odeint(bloisu,Ib,t)[1])
        Gbday.append(Gb)
        Iday.append(Ib)

    return (Gbday,Iday)


##############    GA    #################

def population(Nind, imin, imax):
    # Set up initial population matrix
    pOR = list([] for _ in range(32))
    for z in range(32):
        pOR[z] = [uniform(imin[z], imax[z])]
        while len(pOR[z]) < Nind:
            iOR = uniform(imin[z], imax[z])
            pOR[z].append(iOR)
            for c in range(len(pOR[z])-1):
                if iOR == pOR[z][c]:
                    pOR[z].pop()
                    break
    
    populationall = list(zip(*pOR))
    return populationall


def fitness(individual):

    Gbday,Iday = simulation(individual)

    Df = 124.2
    D0 = 109.8
    D15 = 142.2
    D30 = 180
    D45 = 216
    D60 = 219.6
    D90 = 230.4
    D120 = 207
    # GD = [135,131.4,127,160.2,183.6,208.8,223.3,212.4]

    If = 5
    I0 = 4
    I15 = 7
    I30 = 13
    I45 = 21
    I60 = 21
    I90 = 28
    I120 = 26
    # IB = [5,4,7,13,21,21,28,26]
    
    ###### Cost function
    
    # cost1 = (abs(Gbday[50]-Df) + abs(Gbday[100]-D0) + abs(Gbday[115]-D15) + abs(Gbday[130]-D30) + abs(Gbday[145]-D45) + abs(Gbday[160]-D60) + abs(Gbday[190]-D90) + abs(Gbday[220]-D120))/8
    # cost2 = (abs(Iday[50]-If) + abs(Iday[100]-I0) + abs(Iday[115]-I15) + abs(Iday[130]-I30) + abs(Iday[145]-I45) + abs(Iday[160]-I60) + abs(Iday[190]-I90) + abs(Iday[220]-I120))/8
    # cost = cost1 + cost2

    cost = (((Gbday[50]-Df)/Df)**2 + ((Gbday[100]-D0)/D0)**2 + ((Gbday[115]-D15)/D15)**2 + ((Gbday[130]-D30)/D30)**2 + ((Gbday[145]-D45)/D45)**2 + ((Gbday[160]-D60)/D60)**2 + ((Gbday[190]-D90)/D90)**2 + ((Gbday[220]-D120)/D120)**2)/8*0.9 + (((Iday[50]-If)/If)**2 + ((Iday[100]-I0)/I0)**2 + ((Iday[115]-I15)/I15)**2 + ((Iday[130]-I30)/I30)**2 + ((Iday[145]-I45)/I45)**2 + ((Iday[160]-I60)/I60)**2 + ((Iday[190]-I90)/I90)**2 + ((Iday[220]-I120)/I120)**2)/8*0.1
    
    return cost


def tuple_of_population(listout,populationall):
    # Build a tuple of list with a chromosome and a cost value
    # in every element of tuple
    poptup = list(zip(listout, populationall))
    #print(poptup)
    return poptup


def selection(poptup, Nkeep):
    parents = []
    # Elite
    for candidate in range(2):
        parents.append(list(sorted(poptup)[candidate][1]))
    #Tournament selection
    while len(parents) < Nkeep:
        parents.append(list(sorted(sample(poptup, 3))[0][1]))
    return parents


def mating(parents, Nind, Nkeep):
    while len(parents) < Nind:
        # Random pairing
        famo = sample(range(0, Nkeep), 2)
        fa = parents[famo[0]]
        mo = parents[famo[1]]
        # The blending method
        offspring1 = []
        offspring2 = []
        for x in range(32):
            beta = random()
            offspring1.append(float(mo[x])*beta + float(fa[x])*(1-beta))
            offspring2.append(float(fa[x])*beta + float(mo[x])*(1-beta))
        parents.append(offspring1)
        parents.append(offspring2)
        if len(parents) > Nind:
            parents.pop()
    return parents


def mutation(u, parents, Nkeep, imin, imax):
    for individual in parents[Nkeep:]:
        for cond in range(32):
            if u > random():
                individual[cond] = uniform(imin[cond], imax[cond])

    return parents


def average(poptup):
    total = 0
    for i in range(len(poptup)):
        total += poptup[i][0]
        average = total/len(poptup)
    return average


######## MULTIPROCESS #########

def multiprocess(processes, processfunction, argurerange):
    pool = mp.Pool(processes = processes)
    outputs = [pool.apply_async(processfunction, args=(something,)) for something in argurerange]
    results = [p.get() for p in outputs]
    pool.terminate()
    return results


############# MAIN #############

for i in range(1):
    imin = [0]*28+[-0.002,-0.004,-0.022,-0.02]
    imax = [0.5,0.2,34,0.3,1,0.8,40,16,0.05,25197,2,1,16,200,2509,20,6.3,20,4.31,17.8,1.77,0.3,2,2,2,2,200,100,0.002,0.004,0.022,0.02]
    oriind = [0.25,0.1,17,0.15,0.5,0.4,19.17,7.79,0.03,251.97,1,0.5,7.79,100,1254.64,3,3.16,10,2.15,8.9,0.89,0.15,1,1,0.5,0.5]
    normrate = [0]*32
    #Nbit = 2
    Nindi = 1000
    Nind = 75
    Xrate = 0.5
    u = 0.04

    ax = []

    start_time = time.time()
    print(start_time)
    popl = population(Nindi, imin, imax)
    Nkeep = int(Xrate * len(popl))
    generation = 0
    eachopt = multiprocess(2, fitness, popl)

    # print(eachopt)
    # history = [sorted(eachopt).pop(0)]
    # print(history)
    # mean = [average(tuple_of_population(eachopt, popl))]

    subp = list(highcand[1] for highcand in list(sorted(list(zip(eachopt, popl)))[:300]))
    shuffle(subp)
    subp = list(map(list, subp))
    sub0 = subp[0:75]
    sub1 = subp[75:150]
    sub2 = subp[150:225]
    sub3 = subp[225:300]

    print(i,"--", 0, "--", time.time() - start_time)

    history = list([] for _ in range(4))
    solution = list([] for _ in range(4))
        
    for generation in range(99):
        if generation%20 == 19:
            shuffle(sub0)
            shuffle(sub1)
            shuffle(sub2)
            shuffle(sub3)
            for elepos in range(2):
                sub0.append(sub3.pop(elepos))
                sub1.append(sub0.pop(elepos))
                sub2.append(sub1.pop(elepos))
                sub3.append(sub2.pop(elepos))

        # variances = [] 
        start_time = time.time()
    

        def iterationmain(subs):
            popl = subs
            Nkeep = int(Xrate * len(popl))
            eachopt = list(fitness(individual) for individual in popl)
            subs = list(mutation(u, mating(selection(tuple_of_population(eachopt, popl), Nkeep), Nind, Nkeep), Nkeep, imin, imax))
            history = sorted(eachopt).pop(0)
            solution = list(sorted(list(zip(eachopt, subs)))[0][1])

            return [subs,history,solution]
            # subnum += 1
            # if subnum > 3:
            #     subnum = 0
                
        results = multiprocess(2, iterationmain,[sub0, sub1, sub2, sub3])

        sub0 = results[0][0]
        history[0].append(results[0][1])
        solution[0] = results[0][2]

        sub1 = results[1][0]
        history[1].append(results[1][1])
        solution[1] = results[1][2]

        sub2 = results[2][0]
        history[2].append(results[2][1])
        solution[2] = results[2][2]

        sub3 = results[3][0]
        history[3].append(results[3][1])
        solution[3] = results[3][2]
        
        
        # subnum = 0
        # for subs in [sub0, sub1, sub2, sub3]:
        #     popl = subs
        #     Nkeep = int(Xrate * len(popl))
        #     eachopt = list(fitness(individual) for individual in popl)
        #     subs = list(mutation(u, mating(selection(tuple_of_population(eachopt, popl), Nkeep), Nind, Nkeep), Nkeep, imin, imax))
        #     history[subnum].append(sorted(eachopt).pop(0))
        #     solution[subnum] = list(sorted(list(zip(eachopt, subs)))[0][1])
        #     subnum += 1
        #     if subnum > 3:
        #         subnum = 0

            
        # history.append(sorted(eachopt).pop(0))
        # solution = list(sorted(list(zip(eachopt, popl)))[0][1])
        # mean.append(average(tuple_of_population(eachopt, popl)))
        
        gen = generation + 1
        ax.append(gen)
        print(i,"--", gen, "--", time.time() - start_time)

        # if generation%100 == 0:
        #     for elem in range(28):
        #         van = []
        #         for indi in popl:
        #             van.append(indi[elem])
        #         variances.append(variance(van))
        #     print(list(float('%.5f' % something) for something in variances))
        #     print(variance(eachopt))

            
    # for record in history:
    #     print(record)

    
    # print(solution)
    

    ################
    G1day, I1day = simulation(solution[3])

    print(G1day[100])
    print(G1day[100])
    print(G1day[115])
    print(G1day[130])
    print(G1day[145])
    print(G1day[160])
    print(G1day[190])
    print(G1day[220])
    #print(G1day[250])

    print(I1day[100])
    print(I1day[100])
    print(I1day[115])
    print(I1day[130])
    print(I1day[145])
    print(I1day[160])
    print(I1day[190])
    print(I1day[220])
    #print(I1day[250])
    ################
    # plt.figure(0)
    # plt.subplot(4, 1, 1)
    # plt.plot(ax, history)
    # plt.ylabel('Minimum')

    # plt.subplot(4, 1, 2)
    # plt.plot(ax, mean)
    # plt.xlabel('generation')
    # plt.ylabel('Mean')

    # plt.subplot(4, 1, 3)
    # plt.plot(G1day)
    # plt.xlabel('Time(min)')
    # plt.ylabel('Glucose')

    # plt.subplot(4, 1, 4)
    # plt.plot(I1day)
    # plt.xlabel('Time(min)')
    # plt.ylabel('Insulin')

    # plt.figure(2)
    # plt.plot(ax, history)
    # plt.xlabel('generation')
    # plt.ylabel('Minimum')

    # plt.figure(3)
    # plt.plot(G1day)
    # plt.xlabel('Time(min)')
    # plt.ylabel('Glucose (mg/ml)')

    # plt.figure(4)
    # plt.plot(I1day)
    # plt.xlabel('Time(min)')
    # plt.ylabel('Insulin (uU/ml)')

    ################################
    for solnum in range(4):
        print(solution[solnum])
    
    plt.figure(0)
    plt.subplot(4,1,1)
    plt.plot(ax, history[0])
    plt.ylabel('Minimum')

    plt.subplot(4,1,2)
    plt.plot(ax, history[1])
    plt.ylabel('Minimum')

    plt.subplot(4,1,3)
    plt.plot(ax, history[2])
    plt.ylabel('Minimum')

    plt.subplot(4,1,4)
    plt.plot(ax, history[3])
    plt.ylabel('Minimum')
    plt.xlabel('generation')

    del popl
    del eachopt
    del history
    del solution

# plt.axis([0, 100, -50, 50])
plt.show()
