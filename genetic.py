'''
Genetic Algorithm from Scratch
Author : Viswajit V Nair

**Parameters**
Single Point Probability: 1
Mutation Probability : 0.01
Population Size : 10
Bit Length  : 6
Survival selection : 2
'''

import random
import sys
import math
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# Fitness function of one variable x
def fitness_function(x):
    if isinstance(x,str):
        x = int(x,2)
    return (x**3)+9

#Using Roulette Select to obtain 2 parents
def roulette_select(fitness,population):
    fit_sum = np.sum(fitness)
    fit_prob = [fit/fit_sum for fit in fitness]
    rotation = np.random.rand(1)[0]
    for i in range(len(fit_prob)):
        rotation -= fit_prob[i]
        if rotation < 0:
            return i
    return 0

#Flipping 1s to 0s and vice versa
def flip(i,individual):
    individual = list(individual)
    if(individual[i] =='0'):
        individual[i] = '1'
    else:
        individual[i] = '0'
    print("FLIPPPPED")
    return ''.join(individual)

#Implementing mutation
def mutation(population):
    for individual in population: 
        for i in range(len(individual)):
            rand = random.randint(1,100)
            if rand ==1:
                print(rand)
                individual = flip(i,individual)
            
    return population

#Single point crossover to obtain 2 children
def crossover(parent1, parent2):
    child1 = list(parent1)
    child2 = list(parent2)

    crossover_point = random.randint(0,5)

    for  i in range(crossover_point,6):
        temp = child1[i]
        child1[i] = child2[i]
        child2[i] = temp
    
    return ''.join(child1),''.join(child2)


#Applying  survival selection of 0.2
def survival_selection(children,parents):
    children.sort()
    parents.sort(reverse=True)
    children[0] = parents[0]
    children[1] = parents[1]

    return children

'''
def getChange(children,parents):

    childSum,parentSum = 0,0
    for i in range(len(parents)):
        childSum+= int(children[i],2)
        parentSum+= int(parents[i],2)
    return childSum - parentSum
'''
#Running the algorithrm
def start_algo():
    child_history = {}
    population = []
    for i in range(10):
        population.append(random.randint(0,63))
    isTerm  = False
    #prevChange = 10000
    children = []
    generations = 0
    child_history.update({generations:max(population)})
    population = [bin(individual)[2:].zfill(6) for individual in population]

    while generations<30:
        fitness = [fitness_function(individual) for individual in population]    
        children  = []
        parents = []
        for i in range(0,5):
            parent1 = (population[roulette_select(fitness,population)])
            parent2 = (population[roulette_select(fitness,population)])
            
            child1,child2 = crossover(parent1,parent2)

            children.append(child1)
            children.append(child2)
            
        #Applying mutation and Survival Selection after crossover
        children = mutation(children)
        children = survival_selection(children,population)

        #isTerm = True
        #change = getChange(children,population)
        #if change <= prevChange and change!=0:
        #    isTerm = False
        #    prevChange = change
    
        population = children
        generations+=1

        child_history.update({generations : max([int(child,2) for child in children])})
    return children,generations,child_history

#Starting the algorithm and reaching convergence
final_generation,generations,child_history= start_algo()



print("The Final Generation is: ", final_generation)
print('\n')
print("History of Best Children: ",child_history)
print('\n')
maxm = max([fitness_function(individual) for individual in final_generation])
print("MAXIMIZED VALUE OF FITNESS FUNCTION : ",maxm)

plt.plot(*zip(*sorted(child_history.items())))
plt.savefig('Genetic.png')