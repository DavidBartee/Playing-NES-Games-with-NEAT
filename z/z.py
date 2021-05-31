import retro
import numpy as np
import cv2
import neat
import pickle

resume = False
restore_file = 'neat-checkpoint-1699'
timeOutFrames = 480 #in frames
render = False
'''IMPORTANT RAM VALUES
$EB(235) World map position, x location + 0x10 * y location
$12(18) Game mode:
	0=Title/transitory    1=Selection Screen
	5=Normal              6=Preparing Scroll
	7=Scrolling           4=Finishing Scroll;
	E=Registration        F=Elimination
	17=dying
	16=Entering cave
	11=In cave
	10=Exiting cave (from inside)
	4=Exiting cave (from outside)
$10(16) Current level (dungeon), 0 = overworld
$657(1623) Current Sword, 0 = none
Note: the heart values work in strange ways
$66F(1647) Hearts
$670(1648) Partial Hearts
$70(112) x position
$84(132) y position

Coordinates for objectives:
1st cave: 64,77
Sword: 112,213
Exit cave: 120,221
'''
env = retro.make(game='LegendOfZelda-Nes', state='start_no_sword')
env.reset()
#print(info['gameMode'])
#print(env.buttons)
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, 'zConfig')
#print(env.action_space)

def eval_genomes(genomes, config):
	for genome_id, genome in genomes:
		ob = env.reset()
		inx, iny, inc = env.observation_space.shape
		inx = int(inx/6)
		iny = int(iny/6)
		print(inx, iny)
		net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
		current_max_fitness = 0
		fitness_current = 0
		frame = 0
		counter = 0
		done = False
		prevPipeValue = 0
		prevXValue = -1
		prevDistanceFromCave = -1
		prevDistanceFromSword = -1
		prevDistanceFromExit = -1
		enteredCave = False
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
			nnOutput = [round(x) for x in nnOutput]
			try:
				ob, rew, done, info = env.step(nnOutput)
			except:
				#for x in nnOutput:
				#	print(type(x))
				done = True
			#Reward function
			#Enter the cave
			if enteredCave == False:
				if info['gameMode'] == 16:
					fitness_current += 100
					enteredCave = True
				else:
					distanceFromCave = abs(info['xPos'] - 64) + abs(info['yPos'] - 77)
					if prevDistanceFromCave != -1:
						fitness_current += distanceFromCave - prevDistanceFromCave
					prevDistanceFromCave = distanceFromCave
			elif info['sword'] == 0:
				#Get the sword
				distanceFromSword = abs(info['xPos'] - 112) + abs(info['yPos'] - 213)
				if prevDistanceFromSword != -1:
					fitness_current += distanceFromSword - prevDistanceFromSword
				prevDistanceFromSword = distanceFromSword
			elif info['gameMode'] != 5:
				#Exit the cave
				distanceFromExit = abs(info['xPos'] - 120) + abs(info['yPos'] - 221)
				if prevDistanceFromExit != -1:
					fitness_current += distanceFromExit - prevDistanceFromExit
				prevDistanceFromExit = distanceFromExit
			else:
				#For now, just try to complete this sequence
				fitness_current += 1000000
			#Also stop the counter if progress is made
			if fitness_current > current_max_fitness:
				current_max_fitness = fitness_current
				counter = 0
			else:
				counter += 1
			#Check if genome's been stuck for too long or has died (death check is in scenario.json)
			if counter >= timeOutFrames:
				done = True
			if done:
				#Do some one-time evaluations
				if info['sword'] == 1: fitness_current += 100
				if info['currentLevel'] == 1: fitness_current += 1000000
				#print(genome_id, fitness_current)
			genome.fitness = fitness_current

if resume == True:
	p = neat.Checkpointer.restore_checkpoint(restore_file)
else:
	p = neat.Population(config)

p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(100))

winner = p.run(eval_genomes)

with open('winner.pkl', 'wb') as output:
	pickle.dump(winner, output, 1)