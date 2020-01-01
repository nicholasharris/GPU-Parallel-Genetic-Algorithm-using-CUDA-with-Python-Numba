#-------- Basic implementation of Genetic Algorithm for Python CUDA using Numba  ---------#

import numpy as np
from numba import cuda
import time
import random
import math

#-------- Verify CUDA access  ---------#
print(cuda.gpus)


#-------- Parallel kernel function using CUDA  ---------#
@cuda.jit
def eval_genomes_kernel(chromosomes, fitnesses, pop_length, chrom_length):
  # Thread id in a 1D block
  tx = cuda.threadIdx.x
  # Block id in a 1D grid
  ty = cuda.blockIdx.x
  # Block width, i.e. number of threads per block
  bw = cuda.blockDim.x
  # Compute flattened index inside the array
  pos = tx + ty * bw
  if pos < pop_length:  # Check array boundaries
  # in this example the fitness of an individual is computed by an arbitary set of algebraic operations on the chromosome
    for i in range(3000):
      fitnesses[pos] += chromosomes[pos*chrom_length + 1] # do the fitness evaluation
    for i in range(3000):
      fitnesses[pos] -= chromosomes[pos*chrom_length + 2]
    for i in range(3000):
      fitnesses[pos] += chromosomes[pos*chrom_length + 3]

    if (fitnesses[pos] < 0):
      fitnesses[pos] = 0
    

#-------- Plain evaluation function, not parallel  ---------#
def eval_genomes_plain(chromosomes, fitnesses):
  for i in range(len(chromosomes)):
    # in this example the fitness of an individual is computed by an arbitary set of algebraic operations on the chromosome
    for j in range(3000):
      fitnesses[i] += chromosomes[i][1] # do the fitness evaluation
    for j in range(3000):
      fitnesses[i] -= chromosomes[i][2]
    for j in range(3000):
      fitnesses[i] += chromosomes[i][3]

    if (fitnesses[i] < 0):
      fitnesses[i] = 0

#-------- Function to compute next generation in Genetic Algorithm  ---------#
#-------- Performs Selection, Crossover, and Mutation operations  ---------#
def next_generation(chromosomes, fitnesses):
  fitness_pairs = []
  fitnessTotal = 0.0
  for i in range(len(chromosomes)):
    fitness_pairs.append( [chromosomes[i], fitnesses[i]] )
    fitnessTotal += fitnesses[i]

  fitnesses = list(reversed(sorted(fitnesses))) #fitnesses now in descending order
  sorted_pairs = list(reversed(sorted(fitness_pairs, key=lambda x: x[1])))

  

  new_chromosomes = np.zeros(shape=(pop_size, chrom_size), dtype = np.float32)
  #new_brains_fitnesses = []


  #create roulette wheel from relative fitnesses for fitness-proportional selection
  rouletteWheel = []
  fitnessProportions = []
  for i in range(len(chromosomes)):
      fitnessProportions.append( float( fitnesses[i]/fitnessTotal ) )
      if(i == 0):
          rouletteWheel.append(fitnessProportions[i])
      else:
          rouletteWheel.append(rouletteWheel[i - 1] + fitnessProportions[i])

  #Generate new population with children of selected chromosomes

  for i in range(len(chromosomes)):

      #Fitness Proportional Selection
      spin1 = random.uniform(0, 1)      # A random float from 0.0 to 1.0
      spin2 = random.uniform(0, 1)      # A random float from 0.0 to 1.0

      j = 0
      while( rouletteWheel[j] <= spin1 ):
          j += 1

      k = 0
      while( rouletteWheel[k] <= spin2 ):
          k += 1

      genome_copy = sorted_pairs[j][0]    #Genome of parent 1
      genome_copy2 = sorted_pairs[k][0]   #Genome of parent 2
      
      #create child genome from parents (crossover)
      index = random.randint(0, len(genome_copy) - 1)
      index2 = random.randint(0, len(genome_copy2) - 1)

      child_sequence = []

      for y in range(math.floor(len(genome_copy) / 2)):
          child_sequence.append( genome_copy[ (index + y) % len(genome_copy) ] )

      for y in range(math.floor(len(genome_copy2)/ 2)):
          child_sequence.append( genome_copy2[ (index2 + y) % len(genome_copy2) ] )


      child_genome = np.zeros(len(chromosomes[0]), dtype=np.float32)

      #mutate genome
      for a in range(len(child_sequence)):
        if random.uniform(0,1) < 0.01: # 1% chance of a random mutation
          child_genome[a] = random.uniform(0,1)
        else:
          child_genome[a] = child_sequence[a]

      #Add add new chromosome to next population
      new_chromosomes[i] = child_genome

  #Replace old chromosomes with new
  chromosomes = new_chromosomes

  #Erase fitnesses
  fitnesses = np.zeros(len(chromosomes), dtype = np.float32)
      
  
#-------- Initialize Population  ---------#
random.seed(1111)
pop_size = 5000
chrom_size = 10
num_generations = 5
fitnesses = np.zeros(pop_size, dtype=np.float32)
chromosomes = np.zeros(shape=(pop_size, chrom_size), dtype = np.float32)
for i in range(pop_size):
  for j in range(chrom_size):
    chromosomes[i][j] = random.uniform(0,1) #random float between 0.0 and 1.0


#-------- Measure time to perform 10 generations of the Genetic Algorithm without CUDA  ---------#

print("NO CUDA:")
start = time.time()
# Genetic Algorithm on CPU
for i in range(num_generations):
  print("Gen " + str(i) + "/" + str(num_generations))
  eval_genomes_plain(chromosomes, fitnesses)
  next_generation(chromosomes, fitnesses) #Performs selection, mutation, and crossover operations to create new generation
  
end = time.time()
print("time elapsed: " + str((end-start)))
print("First chromosome: " + str(chromosomes[0]))


#-------------------------------------------------------#
#-------- Prepare kernel ---------#
# Set block & thread size
threads_per_block = 256
blocks_per_grid = (chromosomes.size + (threads_per_block - 1))

#--------- Initialize population again for a new run -------------- #
random.seed(1111)
fitnesses = np.zeros(pop_size, dtype=np.float32)
chromosomes = np.zeros(shape=(pop_size, chrom_size), dtype = np.float32)
for i in range(pop_size):
  for j in range(chrom_size):
    chromosomes[i][j] = random.uniform(0,1) #random float between 0.0 and 1.0

#-------- Measure time to perform 10 generations of the Genetic Algorithm with CUDA  ---------#
print("CUDA:")
start = time.time()
# Genetic Algorithm on GPU
for i in range(num_generations):
  print("Gen " + str(i) + "/" + str(num_generations))
  chromosomes_flat = chromosomes.flatten()
  eval_genomes_kernel[blocks_per_grid, threads_per_block](chromosomes_flat, fitnesses, pop_size, chrom_size)
  next_generation(chromosomes, fitnesses) #Performs selection, mutation, and crossover operations to create new generation
  
end = time.time()
print("time elapsed: " + str((end-start)))
print("First chromosome: " + str(chromosomes[0]))
#-------------------------------------------------------# 
