import retro
import numpy as np
import cv2
import neat
import pickle
import time

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, 'parConfig')

def main(config, winner):
	genome = pickle.load(open(winner, 'rb'))
	env = retro.make(game='SuperMarioBros-Nes')
	net = neat.nn.recurrent.RecurrentNetwork.create(genome, config)
	ob = env.reset()
	inx, iny, inc = env.observation_space.shape
	inx = int(inx/6)
	iny = int(iny/6)
	done = False
	totalReward = 0
	prevXValue = -1
	firstPass = True
	while not done:
		env.render()
		#Make the script wait so that OBS can find the window
		if firstPass:
			time.sleep(2)
			firstPass = False
		ob = cv2.resize(ob, (inx, iny))
		ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
		#Uncomment the following line to see what the neural net sees (lowers performance)
		#cv2.imshow("view", cv2.resize(ob, (inx * 6, iny * 6)))
		ob = np.reshape(ob, (inx, iny))
		imgarray = np.ndarray.flatten(ob)
		imgarray = np.interp(imgarray, (0, 254), (-1, +1))
		nnOutput = net.activate(imgarray)
		#nnOutput = [round(x) for x in nnOutput]
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
		#print((time.process_time_ns() - prevTime) * 1000000000.0)
		time.sleep(0.0001)
	env.close()

if __name__ == "__main__":
	main(config, "winner.pkl")
