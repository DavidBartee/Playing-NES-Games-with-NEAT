import retro
import numpy as np
import cv2
import neat
import pickle

resume = False
restore_file = 'neat-checkpoint-96150'
#Use this to make time more/less important in the reward function
timeWeight = 0
timeOutFrames = 180

'''IMPORTANT RAM VALUES
$0752(1874) Detects pipes (0 default, 1 entering pipe, 2 exiting pipe)
includes vertical and horizontal pipes, but not level intro pipes (like 1-2)
$0e(14) GameEngineSubroutine when this = 5, Mario has finished sliding down the flagpole
11 = Mario is dead to an enemy or the timer running out, 8 = Mario is in control
NOTE: Game stays on subroutine 8 when Mario hits the axe in a castle level
$b5(181) Mario's y coordinate (higher bit) when >=2, he has fallen down a pit and died
(Except in cloud areas, but climbing vines is not rewarded so that shouldn't matter)
$723(1827) ScrollLock (also used for flagpole for some reason) 0 = normal, 1 = locked
$86(134) Player/Sprite X position, use this for the scroll-locked pipe rooms
'''
prevPipeValue = 0
prevXValue = -1
#env = retro.make(game='SuperMarioBros-Nes')
#env.reset()
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, 'newConfig')

class Worker(object):
	def __init__(self, genome, config):
		self.genome = genome
		self.config = config
	def work(self):
		self.env = retro.make(game='SuperMarioBros-Nes', state='1-2.state')
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
		prevPipeValue = 0
		prevXValue = -1

		while not done:
			ob = cv2.resize(ob, (inx, iny))
			ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
			ob = np.reshape(ob, (inx, iny))
			imgarray = np.ndarray.flatten(ob)
			imgarray = np.interp(imgarray, (0, 254), (-1, +1))
			nnOutput = net.activate(imgarray)
			nnOutput = [round(x) for x in nnOutput]
			#nnOutput = [max(0, min(round(x), 1)) for x in nnOutput]
			try:
				ob, rew, done, info = self.env.step(nnOutput)
			except:
				#for x in nnOutput:
				#	print(type(x))
				done = True
			#Reward function, with time penalty
			fitness_current += rew - timeWeight
			'''if fitness_current >= 2062:
				fitness_current += 100000
				done = True'''
			if info['subroutine'] == 5:
				#1680 = 28 seconds at 60FPS, add rewards from pipes (if applicable)
				#if fitness_current >= 2295 - 1080 * timeWeight:
				#	fitness_current += 100000
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
			if counter >= timeOutFrames:
				done = True
			elif info['yposHi'] >= 2 or info['subroutine'] == 11:
				fitness_current -= timeOutFrames - counter
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
	pe = neat.ParallelEvaluator(14, eval_genomes)

	p.add_reporter(neat.StdOutReporter(True))
	#stats = neat.StatisticsReporter()
	#p.add_reporter(stats)
	p.add_reporter(neat.Checkpointer(50))

	winner = p.run(pe.evaluate)

	with open('winner.pkl', 'wb') as output:
		pickle.dump(winner, output, 1)