import retro
import numpy as np
import cv2
import neat
import pickle

resume = True
restore_file = 'neat-checkpoint-10330'
timeLimit = 120 #in frames
levelChangeLimit = 1200
render = False

env = retro.make(game='SuperMarioBros-Nes')
env.reset()
print(env.buttons)
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, 'marioConfig')
print(env.action_space)

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        ob = env.reset()
        ac = env.action_space.sample()
        inx, iny, inc = env.observation_space.shape
        inx = int(inx/6)
        iny = int(iny/6)
        net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
        current_max_fitness = 0
        fitness_current = 0
        frame = 0
        counter = 0
        x = 0
        xmax = 0
        done = False
        while not done:
            if render:
                env.render()
            frame += 1
            ob = cv2.resize(ob, (inx, iny))
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            ob = np.reshape(ob, (inx, iny))
            imgarray = np.ndarray.flatten(ob)
            imgarray = np.interp(imgarray, (0, 254), (-1, +1))
            nnOutput = net.activate(imgarray)
            ob, rew, done, info = env.step(nnOutput)
            #print(nnOutput)

            fitness_current += rew
            if info['levelHi'] > 0:
                fitness_current += 100000
                done = True
            if fitness_current > current_max_fitness:
                current_max_fitness = fitness_current
                counter = 0
            else:
                counter += 1
            if (fitness_current < 3100 and counter > timeLimit) or (fitness_current >= 3100 and counter > levelChangeLimit) or info['lives'] < 2:
                done = True
            #if done:
                #print(genome_id, fitness_current)
            genome.fitness = fitness_current

if resume == True:
    p = neat.Checkpointer.restore_checkpoint(restore_file)
else:
    p = neat.Population(config)

p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(50))

winner = p.run(eval_genomes)

with open('winner.pkl', 'wb') as output:
    pickle.dump(winner, output, 1)