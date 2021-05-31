import retro
import numpy as np
import cv2
import neat
import pickle
import time

startingWithSword = False

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, 'zConfig')

def main(config, winner):
	genome = pickle.load(open(winner, 'rb'))
	#Zelda states: start_no_sword, start_cave_exit
	if startingWithSword:
		env = retro.make(game='LegendOfZelda-Nes', state='start_cave_exit')
	else:
		env = retro.make(game='LegendOfZelda-Nes', state='start_no_sword')
	net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
	ob = env.reset()
	inx, iny, inc = env.observation_space.shape
	inx = int(inx/6)
	iny = int(iny/6)
	done = False
	totalReward = 0
	time.sleep(1)
	prevXValue = -1
	while not done:
		env.render()
		ob = cv2.resize(ob, (inx, iny))
		ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
		ob = np.reshape(ob, (inx, iny))
		imgarray = np.ndarray.flatten(ob)
		imgarray = np.interp(imgarray, (0, 254), (-1, +1))
		nnOutput = net.activate(imgarray)
		nnOutput = [round(x) for x in nnOutput]
		#print(nnOutput)
		ob, rew, done, info = env.step(nnOutput)
		#totalReward += rew
		'''fit = 0
		if info['scrollLock'] == 1 and prevXValue == -1:
			prevXValue = info['xrel']
		elif info['scrollLock'] == 1 and info['xrel'] > prevXValue:
			fit = info['xrel'] - prevXValue if info['xrel'] <= 194 else prevXValue - info['xrel']
			prevXValue = info['xrel']
		print(fit)'''
		#print('yposHi: ', info['yposHi'], ' pipe: ', info['pipe'], ' subroutine: ', info['subroutine'])
		'''if info['lives'] < 2:
			done = True'''
		#worldX = info['worldPos'] % 16
		#worldY = int(info['worldPos'] / 16)
		#print('x: ', worldX, ', y: ', worldY)
		time.sleep(0.001)
	env.close()

if __name__ == "__main__":
	main(config, "winner.pkl")
