# import statements
from sql_tools import db
from json import dumps
from datetime import datetime
from random import choice
from game import Game

# open database
database = db("T3NN_training_data.db")
database.printCmds = False

# define tables
table = database.newTable("training_data_0", ["JSON_example"])
table_basics = database.newTable("training_data_basics_0", ["JSON_example"])

"""
TABLES
    training_data_0 : winner moves from randomized games
    training_data_user_input_0 : winner moves of games played by user
    training_data_basics_0 : taken from randomized games, designed to make sure ai understands to block wins & secure wins
    """

# game count
game_count = 30000

# when to print progress
print_every = 10000

# training examples as json dump
examples = []

# for basics database
basics_examples = []

# start
start = datetime.now()

# gather data
for i in range(game_count):

    # generate game
    g = Game()
    while not g.game_over:

        # add to basics database in here
        g.take_turn(choice(g.remaining_moves))

        # rotate board if necessary
        if g.orienting_corner != (0, 0):
            rotation_times = [(0, 2), (2, 2), (2, 0)].index(g.orienting_corner) + 1

        # bools = X or O are 1 move away from winning
        x_bool, x_space, o_bool, o_space = g.is_game_point()

        # if both are 1 move away
        if x_bool and o_bool:

            # get input/rotate input if necessary
            if g.orienting_corner != (0, 0):
                input_0 = g.get_board_as_input(True)
                label_0 = g.rotate_board([1 if i == x_space[1]*3 + x_space[0] else 0 for i in range(9)], g.rotation_times)
                input_1 = g.get_board_as_input(False)
                label_1 = g.rotate_board([1 if i == o_space[1]*3 + o_space[0] else 0 for i in range(9)], g.rotation_times)
            else:
                input_0 = g.get_board_as_input(True)
                label_0 = [1 if i == x_space[1]*3 + x_space[0] else 0 for i in range(9)]
                input_1 = g.get_board_as_input(False)
                label_1 = [1 if i == o_space[1]*3 + o_space[0] else 0 for i in range(9)]
            
            # add examples of both sides securing wins
            basics_examples.append([dumps([input_0, label_0])])
            basics_examples.append([dumps([input_1, label_1])])

        # if X is 1 move away
        elif x_bool:

            # get input/rotate input if necessary
            if g.orienting_corner != (0, 0):
                input_0 = g.get_board_as_input(True)
                label_0 = g.rotate_board([1 if i == x_space[1]*3 + x_space[0] else 0 for i in range(9)], g.rotation_times)
                input_1 = g.get_board_as_input(False)
                label_1 = g.rotate_board([1 if i == x_space[1]*3 + x_space[0] else 0 for i in range(9)], g.rotation_times)
            else:
                input_0 = g.get_board_as_input(True)
                label_0 = [1 if i == x_space[1]*3 + x_space[0] else 0 for i in range(9)]
                input_1 = g.get_board_as_input(False)
                label_1 = [1 if i == x_space[1]*3 + x_space[0] else 0 for i in range(9)]
            
            # add examples of X securing/O blocking the win
            basics_examples.append([dumps([input_0, label_0])])
            basics_examples.append([dumps([input_1, label_1])])

        # if O is 1 move away
        elif o_bool:

            # get input/rotate input if necessary
            if g.orienting_corner != (0, 0):
                input_0 = g.get_board_as_input(True)
                label_0 = g.rotate_board([1 if i == o_space[1]*3 + o_space[0] else 0 for i in range(9)], g.rotation_times)
                input_1 = g.get_board_as_input(False)
                label_1 = g.rotate_board([1 if i == o_space[1]*3 + o_space[0] else 0 for i in range(9)], g.rotation_times)
            else:
                input_0 = g.get_board_as_input(True)
                label_0 = [1 if i == o_space[1]*3 + o_space[0] else 0 for i in range(9)]
                input_1 = g.get_board_as_input(False)
                label_1 = [1 if i == o_space[1]*3 + o_space[0] else 0 for i in range(9)]

            # add examples of O securing/X blocking the win
            basics_examples.append([dumps([input_0, label_0])])
            basics_examples.append([dumps([input_1, label_1])])

    # if not a cats game, get examples from all winner moves
    if not g.winner == -1:
        game_examples = g.get_training_examples()
        examples.extend(game_examples)
    
    # print game number
    if not i % print_every:
        print(i)

print(f"{len(examples)} training examples generated after {datetime.now() - start}\n")
    
# insert data into database
start = datetime.now()
table.insertRows(examples)
table_basics.insertRows(basics_examples)
print(f"Data downloaded after {datetime.now() - start}")





