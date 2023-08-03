from game import Game
import torch as pyt
from torch import nn
import time

# network
class T3NN(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = pyt.nn.Sequential(
            nn.Linear(9, 32),
            nn.ReLU(),
            nn.Linear(32, 9)
            )
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        return self.net(x)

# load model
MODEL_NAME = "T3NN.pkl"
model = pyt.load(MODEL_NAME)

# begin infinite game loop
ind = 0
while True:

    # generate game
    g = Game()
    g.print_with_labels = True

    # if game number is even, user goes first
    if not ind % 2:
        
        # print game
        print("*** NEW GAME ***\nX : User | O : T3 Neural Network\n")
        print(g)

        # while there is no winner...
        while not g.game_over:

            # get input from human
            valid_move = False
            while not valid_move:

                # raw input
                user_move_raw = input("Move : ")

                # try to convert it to a coordinate tuple
                try:
                    user_move = ({"A": 0, "B": 1, "C": 2}[user_move_raw[0]], int(user_move_raw[1]) - 1)
                
                # except try again
                except KeyError:
                    print("INVALID MOVE\n")
                    continue
                
                # break loop
                if user_move in g.remaining_moves:
                    valid_move = True
                
                # try again if move is already taken
                else:
                    print("INVALID MOVE\n")
            print()

            # input user move into game and print
            g.take_turn(user_move)
            print(g)
            print()
            time.sleep(1)

            # check for winner
            if g.game_over:
                break
            
            # get raw nn output
            nn_move_raw = list(model(pyt.tensor(g.get_board_as_input(), dtype=pyt.float32)))
            
            # rotate output to match board if neccisary
            if g.orienting_corner != (0, 0):
                nn_move_rotated = g.rotate_board(nn_move_raw, 4 - g.rotation_times)
            else: 
                nn_move_rotated = nn_move_raw
            
            # index the high value of the output and convert to coordinate tuple
            nn_move_index = nn_move_rotated.index(max([i[0] for i in zip(nn_move_rotated, range(len(nn_move_rotated))) if i[1] not in g.taken_spot_indexes]))
            nn_move = list(g.board.keys())[nn_move_index]

            # take nn turn
            g.take_turn(nn_move)
            print(g)
    
    # if game number is odd, neural network goes first
    else:

        # print game
        print("*** NEW GAME ***\nX : T3 Neural Network | O : User\n")

        # while there is no winner
        while not g.game_over:

            # get raw nn output
            nn_move_raw = list(model(pyt.tensor(g.get_board_as_input(), dtype=pyt.float32)))

            # rotate output to match the board if neccisary
            if g.orienting_corner != (0, 0):
                nn_move_rotated = g.rotate_board(nn_move_raw, 4 - g.rotation_times)
            else:
                nn_move_rotated = nn_move_raw

            # index the highest value of the output and convert it to coordinate tuple
            nn_move_index = nn_move_rotated.index(max([i[0] for i in zip(nn_move_rotated, range(len(nn_move_rotated))) if i[1] not in g.taken_spot_indexes]))
            nn_move = list(g.board.keys())[nn_move_index]

            # input nn move into game
            g.take_turn(nn_move)
            print(g)

            # check for win
            if g.game_over:
                break
            
            # get input from user
            valid_move = False
            while not valid_move:

                # raw input
                user_move_raw = input("Move : ")

                # try to convert input to coordinate tuple else continue
                try:
                    user_move = ({"A": 0, "B": 1, "C": 2}[user_move_raw[0]], int(user_move_raw[1]) - 1)
                except KeyError:
                    print("INVALID MOVE\n")
                    continue
                    
                # if move is available then close loop
                if user_move in g.remaining_moves:
                    valid_move = True
                
                # else try again
                else:
                    print("INVALID MOVE\n")
            print()

            # input turn into game
            g.take_turn(user_move)
            print(g)
            print()
            time.sleep(1)

    # declare winner
    print(["O WINS!\n", "X WINS\n", "CAT'S GAME!\n"][g.winner])
        
    ind += 1






