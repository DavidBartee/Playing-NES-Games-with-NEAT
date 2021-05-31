import retro
import numpy as np
import cv2
import neat
import pickle

resume = True
restore_file = 'neat-checkpoint-12134'
numWorkers = 7
#Changes starting state to start Link from the exit of the cave right after getting the sword
startingWithSword = True
#Use this to make time more/less important in the reward function
timeWeight = 0
timeOutFrames = 480
bigTimeout = 10000

'''IMPORTANT RAM VALUES
$EB(235) World map position, x location + 0x10 * y location
$12(18) Game mode:
	0=Title/transitory	  1=Selection Screen
	5=Normal			  6=Preparing Scroll
	7=Scrolling			  4=Finishing Scroll;
	E=Registration		  F=Elimination
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
'''

pathToDungeon = [
	(128,61,7,7),
	(240,141,7,6),
	(48,61,8,6),
	(112,61,8,5),
	(112,61,8,4),
	(0,133,8,3)
]

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, 'zConfig')

class Worker(object):
	def __init__(self, genome, config):
		self.genome = genome
		self.config = config
	def work(self):
		#Zelda states: start_no_sword, start_cave_exit
		if startingWithSword:
			self.env = retro.make(game='LegendOfZelda-Nes', state='start_cave_exit')
		else:
			self.env = retro.make(game='LegendOfZelda-Nes', state='start_no_sword')
		ob = self.env.reset()
		inx, iny, inc = self.env.observation_space.shape
		inx = int(inx/6)
		iny = int(iny/6)
		net = neat.nn.recurrent.RecurrentNetwork.create(self.genome, self.config)
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
		prevDistanceFromLevel = -1
		prevDistanceFromWaypoint = -1
		furthestPointOnPath = 0
		#Special case for starting with the sword
		enteredCave = startingWithSword
		exitedCave = startingWithSword
		while not done:
			frame += 1
			ob = cv2.resize(ob, (inx, iny))
			ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
			ob = np.reshape(ob, (inx, iny))
			imgarray = np.ndarray.flatten(ob)
			imgarray = np.interp(imgarray, (0, 254), (-1, +1))
			nnOutput = net.activate(imgarray)
			nnOutput = [round(x) for x in nnOutput]
			try:
				ob, rew, done, info = self.env.step(nnOutput)
			except:
				#for x in nnOutput:
				#	print(type(x))
				done = True
			'''if fitness_current >= 400:
				fitness_current += 1000000
				done = True'''
			if frame >= bigTimeout:
				print("infinite loop detected")
				fitness_current += 1000000
				done = True
			#Reward function
			worldX = info['worldPos'] % 16
			worldY = int(info['worldPos'] / 16)
			#Enter the cave
			if enteredCave == False:
				if info['gameMode'] == 16:
					fitness_current += 100
					enteredCave = True
				else:
					distanceFromCave = abs(info['xPos'] - 64) + abs(info['yPos'] - 77) + (abs(worldX - 7) + abs(worldY - 7)) * 256
					if prevDistanceFromCave != -1:
						fitness_current += prevDistanceFromCave - distanceFromCave
					prevDistanceFromCave = distanceFromCave
			elif info['sword'] == 0:
				#Get the sword
				#Cut off gameplay if the cave is left without the sword
				if info['gameMode'] == 5:
					done = True
				distanceFromSword = abs(info['xPos'] - 112) + abs(info['yPos'] - 213) + (abs(worldX - 7) + abs(worldY - 7)) * 256
				if prevDistanceFromSword != -1:
					fitness_current += prevDistanceFromSword - distanceFromSword
				prevDistanceFromSword = distanceFromSword
			elif exitedCave == False:
				#Exit the cave
				if info['gameMode'] == 5:
					exitedCave = True
				else:
					distanceFromExit = abs(info['xPos'] - 120) + abs(info['yPos'] - 221) + (abs(worldX - 7) + abs(worldY - 7)) * 256
					if prevDistanceFromExit != -1:
						fitness_current += prevDistanceFromExit - distanceFromExit
					prevDistanceFromExit = distanceFromExit
			elif (worldX != 7 or worldY != 3) and furthestPointOnPath < len(pathToDungeon):
				#Get to the bridge before the level
				if info['gameMode'] != 5:
					#Don't enter any caves before the level
					if info['gameMode'] == 16:
						done = True
					#Reset the waypoint distance between screens
					prevDistanceFromWaypoint = -1
				else:
					onPath = False
					#This index is used to ensure that Link makes progress on the path without backtracking
					index = 0
					for x in pathToDungeon:
						if x[2] == worldX and x[3] == worldY and index >= furthestPointOnPath:
							targetX = x[0]
							targetY = x[1]
							onPath = True
							furthestPointOnPath = index
							break
						index += 1
					if onPath:
						distanceFromWaypoint = abs(info['xPos'] - targetX) + abs(info['yPos'] - targetY)
						if prevDistanceFromWaypoint != -1:
							fitness_current += prevDistanceFromWaypoint - distanceFromWaypoint
						prevDistanceFromWaypoint = distanceFromWaypoint
					else:
						done = True
			elif info['gameMode'] == 5:
				furthestPointOnPath = len(pathToDungeon) + 1
				#Enter the level
				if worldX != 7 or worldY != 3 or info['currentLevel'] == 1:
					done = True
				else:
					distanceFromLevel = abs(info['xPos'] - 112) + abs(info['yPos'] - 125)
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
				#if info['currentLevel'] == 1 or fitness_current >= 1085: fitness_current += 1000000
		return fitness_current

def eval_genomes(genome, config):
	worker = Worker(genome, config)
	return worker.work()

if __name__ == "__main__":
	if resume == True:
		p = neat.Checkpointer.restore_checkpoint(restore_file)
	else:
		p = neat.Population(config)
	pe = neat.ParallelEvaluator(numWorkers, eval_genomes)
	p.add_reporter(neat.StdOutReporter(True))
	#stats = neat.StatisticsReporter()
	#p.add_reporter(stats)
	p.add_reporter(neat.Checkpointer(50))
	winner = p.run(pe.evaluate)
	with open('winner.pkl', 'wb') as output:
		pickle.dump(winner, output, 1)
