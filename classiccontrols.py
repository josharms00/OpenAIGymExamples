import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
import gym
import random
import numpy as np
import argparse

goal_score = 500
req_score = 50
games = 100
BATCH_SIZE = 20
epsilon = 0.8
gamma = 0.95

class Agent(object):

    def __init__(self, env):
        self.env = gym.make(env)
        self.observation_space = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space.n
        self.model = self.initialize_model()

    def initial_games(self):
        # intialize training data
        train_data = []
        scores = [] 
        actions = []
        correct_action = [1, 0]
        games_list = []

        # iterate through all games
        for _ in range(games):
            score = 0 # score is 0 at beginning of each game
            game_memory = []
            prev_observation = []
            
            # reset the game before trying to get data from it
            observation = self.env.reset()

            # iterate as many times as it should ideally last
            while 1:

                # take random action with epsilon chance
                if np.random.uniform(0, 1) <= epsilon or len(prev_observation) == 0:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.model.predict(np.reshape(prev_observation, [1, self.observation_space])))


                if action not in actions:
                    actions.append(action)

                # sample data based on the action just taken
                observation, reward, done, info = self.env.step(action)

                # add to the score
                score += reward

                if done:
                    break

                # if the list is not empty append the previous observation with the action that caused it
                if len(prev_observation) > 0:
                    game_memory.append([prev_observation, action, observation, reward, done])

                prev_observation = observation

            # game is done
            self.env.close()

            # if score is good enough append to data
            if score >= req_score:
                scores.append(score)

                # outputs need to be one hot encoded
                for data in game_memory:
                    train_data.append([data[0], data[1]])

                games_list.append(game_memory)
        
        
       # train_data = self.replay(games_list, actions)

        self.replay(games_list, actions)

        train_data = np.array(train_data)

        #np.save('cartpole_training_data.npy', train_data)

        return train_data, actions

    def replay(self, games, actions):
        train_data = []
    
        for game in games:
            for i in range(len(game)):
                action = game[i][1]
                state = game[i][0]
                reward = game[i][3]
                next_state = game[i][2]
                done = game[i][4]

                Q = self.model.predict(np.reshape(state, [1, self.observation_space]))
                Q[0][action] = reward + gamma*np.max(self.model.predict(np.reshape(next_state, [1, self.observation_space ]))[0])

                if done:
                    Q[0][action] = -reward
                    break

                state = np.array(state)

    
                game[i][1] = np.argmax(Q[[0]])
                self.model.fit(np.reshape(state, [1, self.observation_space]), Q)
        #         train_data.append([game[i][0], game[i][1]])

        # print('nice')
        # return train_data

    def initialize_model(self):
        # create model
        model = Sequential()

        # create feed forward part of neural network
        model.add(Dense(24, input_shape=(self.observation_space,)))
        model.add(Dropout(0.3))
        model.add(Activation('relu'))

        model.add(Dense(256))
        model.add(Dropout(0.3))
        model.add(Activation('relu'))

        model.add(Dense(512))
        model.add(Dropout(0.3))
        model.add(Activation('relu'))

        model.add(Dense(256))
        model.add(Dropout(0.3))
        model.add(Activation('relu'))

        model.add(Dense(128))
        model.add(Dropout(0.3))
        model.add(Activation('relu'))

        model.add(Dense(self.action_space))
        model.add(Activation('linear'))

        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

        return model

    def train(self, train_data, envname):
        x = []
        y = []

        # extract labels and data from train_data
        for i in train_data:
            print(np.array([i[0]]).reshape(1, self.observation_space))
            x.append(np.array([i[0]]).reshape(1, self.observation_space))
            y.append(np.array([i[1]]))

        # convert lists to numpy arrays
        x = np.array(x)
        y = np.array(y)

        # train and save model
        if len(x) == 0:
            print('No training data.')
        else:
            self.model.fit(x, y, epochs=2)

            # save model
            self.model.save(envname + '.h5')

    def test(self, model, games):
        prev_observation = []
        scores = []
        
        # play for required amount of games
        for _ in range(games):
            score = 0 
            self.env.reset()
            while 1:
                self.env.render()

                # if list is empty there are no actions to predict so movement is random
                if len(prev_observation) == 0:
                    action = self.env.action_space.sample()
                else:
                    # predict next action based off previous observation
                    action = np.argmax(model.predict(prev_observation.reshape(-1, len(prev_observation), 1))[0])

                # get timestep data based on action
                obver, reward, done, info = self.env.step(action)
                prev_observation = obver
                score += 1

                if done:
                    break

            scores.append(score)

        # print out the highest and average score for the games
        print('Highest score: ', max(scores))
        print('Average score: ', sum(scores)/len(scores))

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-t', '--train',
            action="store_true", dest="train",
            help="train model")

    parser.add_argument('-p', '--test',
            action="store", dest="test",
            help="test model over amount of games")

    parser.add_argument('-e', '--env',
            action="store", dest="env",
            help="choose classic environment to use")

    parser.add_argument('-i', '--investigate',
            action="store_true", dest="inv",
            help="Investigate how an environment works")

    args = parser.parse_args()

    agent = Agent(args.env)

    if args.train:
        data, actions = agent.initial_games()
        #agent.train(data, args.env)

    if args.test:
        model = tf.keras.models.load_model(args.env + '.h5')
        game = int(args.test)
        agent.test(model, game)



if __name__ == '__main__':
    main()