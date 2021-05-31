import retro
import numpy as np
import cv2
import neat
import pickle

resume = True
restore_file = 'neat-checkpoint-10839'

#env = retro.make(game='SuperMarioBros-Nes')
#env.reset()
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, 'marioConfig')

class Worker(object):
    def __init__(self, genome, config):
        self.genome = genome
        self.config = config
    def work(self):
        self.env = retro.make(game='SuperMarioBros-Nes')
        ob = self.env.reset()
        inx, iny, inc = self.env.observation_space.shape
        inx = int(inx/6)
        iny = int(iny/6)
        net = neat.nn.recurrent.RecurrentNetwork.create(self.genome, self.config)

        current_max_fitness = 0
        fitness_current = 0
        frame = 0
        counter = 0
        x = 0
        xmax = 0
        done = False

        while not done:
            ob = cv2.resize(ob, (inx, iny))
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            ob = np.reshape(ob, (inx, iny))
            imgarray = np.ndarray.flatten(ob)
            imgarray = np.interp(imgarray, (0, 254), (-1, +1))
            nnOutput = net.activate(imgarray)
            ob, rew, done, info = self.env.step(nnOutput)

            '''x = info['xscrollLo']
            if x >= 65664:
                fitness_current += 100000
                done = True'''
            
            fitness_current += rew
            if info['levelLo'] > 1 or fitness_current >= 4305:
                fitness_current += 100000
                done = True
            
            if fitness_current > current_max_fitness:
                current_max_fitness = fitness_current
                counter = 0
            else:
                counter += 1
            if counter > 1200 or info['lives'] < 2:
                done = True
        return fitness_current

def eval_genomes(genome, config):
    worker = Worker(genome, config)
    return worker.work()

if __name__ == "__main__":
    if resume == True:
        p = neat.Checkpointer.restore_checkpoint(restore_file)
    else:
        p = neat.Population(config)
    pe = neat.ParallelEvaluator(50, eval_genomes)

    p.add_reporter(neat.StdOutReporter(True))
    #stats = neat.StatisticsReporter()
    #p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(50))

    winner = p.run(pe.evaluate)

    with open('winner.pkl', 'wb') as output:
        pickle.dump(winner, output, 1)