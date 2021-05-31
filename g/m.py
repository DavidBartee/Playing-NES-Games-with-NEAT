import retro
import numpy as np
import cv2
import neat
import pickle

resume = False
restore_file = 'neat-checkpoint-10330'
timeLimit = 120 #in frames
render = True

env = retro.make(game='SuperMarioBros-Nes', state='1-2.state')
env.reset()
#print(env.buttons)
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, 'pconfig')
#print(env.action_space)

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
		prevPipeValue = 0
		prevXValue = -1
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
			#Reward function, with time penalty
			fitness_current += rew - 1
			'''if fitness_current >= 5366:
				fitness_current += 100000
				done = True'''
			if info['subroutine'] == 5:
				#1680 = 28 seconds at 60FPS, add rewards from pipes (if applicable)
				if fitness_current >= 3100 - 1680:
					fitness_current += 100000
				done = True
			#Check that Mario is in control and not in a pipe
			if info['pipe'] == 0 and info['subroutine'] == 8:
				counter += 1
				prevPipeValue = 0
				#If Mario has not been in a scroll-locked room yet, initialize the
				#x scroll value here before rewarding increases in that value
				if info['scrollLock'] == 1 and prevXValue == -1:
					prevXValue = info['xrel']
				elif info['scrollLock'] == 1 and info['xrel'] > prevXValue:
					#Reward Mario for moving towards the pipe in bonus rooms,
					#including a reward for being at the right x position for
					#going into the exit pipe
					'''if info['xrel'] < 194:
						fitness_current += info['xrel'] - prevXValue
					elif info['xrel'] > 194:
						fitness_current += prevXValue - info['xrel']
					else:
						fitness_current += 1'''
					#Reward Mario for moving to the right in a scroll-locked room (Bowser)
					fitness_current += info['xrel'] - prevXValue
					prevXValue = info['xrel']
			else:
				#Reward Mario for going through pipes
				if prevPipeValue == 0 and info['pipe'] != 0:
					fitness_current += 5000
					prevPipeValue = info['pipe']
				#Don't use the timeout counter while he is going through pipes
				#or if Mario is not controllable
				if info['subroutine'] != 8 or prevPipeValue != 0:
					counter = 0
			#Also stop the counter if Mario is making progress
			if fitness_current > current_max_fitness:
				current_max_fitness = fitness_current
				counter = 0
			#Check if Mario's been stuck for too long or has died
			if counter >= 120 or info['yposHi'] >= 2 or info['subroutine'] == 11:
				done = True
			if done:
				print(genome_id, fitness_current)
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