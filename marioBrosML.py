import retro
import time
import neat
from cv2 import cv2
import numpy as np
import pickle

env = retro.make(game="SuperMarioBros-Nes", state="Level1-1", record=True)
image = []

def evaluation(genomes, config):

    for _, genome in genomes:

        counter = 0
        best = 0
        fitness = 0
        done = False

        ob = env.reset()
        ac = env.action_space.sample()
        net = neat.nn.RecurrentNetwork.create(genome, config)
        x, y, _ = env.observation_space.shape

        while not done:
            
            env.render()
            #time.sleep(0)

            ob = cv2.resize(np.array(ob),(int(x/8),int(y/8)))
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            ob = np.reshape(ob,(int(x/8),int(y/8)))

            image = np.ndarray.flatten(ob)
            neural_output = net.activate(image)
            ob, rew, done, _ = env.step(neural_output)

            #print("fitness: %s , rew: %s , done: %s, info: %s" % (fitness, rew, done, info))

            fitness += rew

            if fitness > best:
                best = fitness
                counter = 0
            else:
                counter += 1
            
            if done or counter == 200:
                done = True
            
            genome.fitness = fitness

config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, r"C:\Users\Pavel\Documents\python\marioBrosML\config-feedforward")

p = neat.Population(config)
p.add_reporter(neat.StdOutReporter(True))
p.add_reporter(neat.StatisticsReporter())
p.add_reporter(neat.Checkpointer(10))

output = p.run(evaluation)

with open(r"C:\Users\Pavel\Documents\python\marioBrosML\output.pk1", "wb") as files:
    pickle.dump(output, files, 1)