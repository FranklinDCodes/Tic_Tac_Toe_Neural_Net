from random import choice
from json import dumps

# Class that keeps track of turns, the winner, the game board, etc
class Game():

    def __init__(self, orient_input = True):
        
        """ TIC TAC TOE GAME
            0 = O
            1 = X 
            -1 = CATS GAME """

        # game status
        self.game_over = False
        self.winner = None
        self.starter = 1

        # game board (column, row): 0, 1 or None
        self.board = {(0, 0): None, (1, 0): None, (2, 0): None,
                 (0, 1): None, (1, 1): None, (2, 1): None,
                 (0, 2): None, (1, 2): None, (2, 2): None}
        
        # the corner of the starting move to use to orient the game
        self.orienting_corner = None
        self.rotation_times = 0
        
        # moves [(column, rows), ... ]
        self.x_moves = []
        self.o_moves = []

        # available spots
        self.remaining_moves = list(self.board.keys())
        self.taken_spot_indexes = []

        # turn tracker
        self.turn = self.starter

        # settings
        self.print_with_labels = False
        self.orient_input = orient_input

    def __str__(self):
        prnt =  "  | A | B | C |\n"
        prnt += "--+-----------+\n"
        ind = 1
        for space in self.board.values():
            symbols = {None: " ", 0: "O", 1: "X"}
            if not ind % 3:
                prnt += symbols[space] + " |\n"
                if ind != 9:
                    prnt += "--|-----------|\n"
            elif not (ind + 2) % 3:
                prnt += str(ind//3 + 1) + " | " + symbols[space] + " | "
            else:
                prnt += symbols[space] + " | "
            ind += 1
        prnt += "--+-----------+"
        return prnt

    # record a move for either a human or AI player
    def take_turn(self, space: tuple[int]) -> int:
        
        # update moves
        {1: self.x_moves, 0: self.o_moves}[self.turn].append(space)

        # update remaining moves
        self.remaining_moves.pop(self.remaining_moves.index(space))

        # if this is the first turn...
        if len(self.remaining_moves) == 8:
            
            # if the input is being rotated for the ease of the AI
            if self.orient_input:

                # use the first turn to set the orienting corner and the number of rotations
                self.orienting_corner = {(1, 1): (0, 0), (0, 0): (0, 0), (1, 0): (0, 0), (2, 0): (2, 0), (2, 1): (2, 0), (2, 2): (2, 2), (1, 2): (2, 2), (0, 2): (0, 2), (0, 1): (0, 2)}[space]
                if self.orienting_corner != (0, 0):
                    self.rotation_times = [(0, 2), (2, 2), (2, 0)].index(self.orienting_corner) + 1
            
            # else dont
            else:
                self.orienting_corner = (0, 0)
                self.rotation_times = 0

        # add taken spot to list
        index = list(self.board.keys()).index(space)
        self.taken_spot_indexes.append(index)

        # update board
        self.board[space] = self.turn

        # check for win
        win = self.game_is_over()
        if win[0]:
            self.game_over = True
            self.winner = win[1]
            return 1

        # update turn
        self.turn = [1, 0][self.turn]

        return 0

    # returns True, winner if the game has ended
    def game_is_over(self) -> tuple[bool, int]:

        # check for win 
        for player in [0, 1]:
            
            # col occupied totals [col 0, col 1, col 2]
            col_totals = [0, 0, 0]

            # row occupied totals [row 0, row 1, row 2]
            row_totals = [0, 0, 0]

            # add up row and col totals
            player_moves = [self.o_moves, self.x_moves][player]
            for col, row in player_moves:
                
                col_totals[col] += 1
                row_totals[row] += 1
            
            # check for straight wins
            if max(col_totals) == 3 or max(row_totals) == 3:

                return True, player
            
            #check for diagonal wins
            condition_0 = (0, 0) in player_moves and (1, 1) in player_moves and (2, 2) in player_moves
            condition_1 = (0, 2) in player_moves and (1, 1) in player_moves and (2, 0) in player_moves
            if condition_0 or condition_1:

                return True, player
            
        # check for cats game
        if None not in list(self.board.values()):

            return True, -1
        
        return False, None
    
    # returns a tuple to signify whether x or o is one move away from winning
    def is_game_point(self):

        # returns x can win in 1 move, <- the move, o can win in 1 move, <- the move
        bool_value = []
        win_space = [None, None]

        # check for win 
        for player in [0, 1]:

            # return value for player
            player_game_point = False
            
            # col occupied totals [col 0, col 1, col 2]
            col_totals = [0, 0, 0]

            # row occupied totals [row 0, row 1, row 2]
            row_totals = [0, 0, 0]

            # add up row and col totals
            player_moves = [self.o_moves, self.x_moves][player]
            for col, row in player_moves:
                
                col_totals[col] += 1
                row_totals[row] += 1
            
            # check cols
            ind = 0
            for col in col_totals:

                if col == 2:
                    for i in range(3):
                        if (ind, i) in self.remaining_moves:
                            player_game_point = True
                            win_space[player] = (ind, i)

                ind += 1

            # check rows
            ind = 0
            for row in row_totals:

                if row == 2:
                    for i in range(3):
                        if (i, ind) in self.remaining_moves:
                            player_game_point = True
                            win_space[player] = (i, ind)

                ind += 1

            # diagonal win conditions and remaining moves
            conditions_0_moves = [(0, 0) in player_moves, (1, 1) in player_moves, (2, 2) in player_moves]
            conditions_0_spaces = [(0, 0) in self.remaining_moves, (1, 1) in self.remaining_moves, (2, 2) in self.remaining_moves]
            conditions_1_moves = [(0, 2) in player_moves, (1, 1) in player_moves, (2, 0) in player_moves]
            conditions_1_spaces = [(0, 2) in self.remaining_moves, (1, 1) in self.remaining_moves, (2, 0) in self.remaining_moves]
            
            # if 2 of the win conditions are met and there is one remaining space
            condition_0 = sum([i for i in conditions_0_moves if i]) == 2 and sum([i for i in conditions_0_spaces if i]) == 1
            condition_1 = sum([i for i in conditions_1_moves if i]) == 2 and sum([i for i in conditions_1_spaces if i]) == 1

            # set to return true and space needed for win
            if condition_0:
                player_game_point = True
                win_space[player] = [(0, 0), (1, 1), (2, 2)][conditions_0_spaces.index(True)]   
            if condition_1:
                player_game_point = True
                win_space[player] = [(0, 2), (1, 1), (2, 0)][conditions_1_spaces.index(True)]

            bool_value.append(player_game_point)

        # X HAS GP, X GP SPACE, O HAS GP, O GP SPACE
        return (bool_value[1], win_space[1], bool_value[0], win_space[0])

    # converts the board into input for a NN forward pass
    def get_board_as_input(self, nn_is_x: bool = True) -> list[float]:
        
        # final input for the neural network
        input = [0 for i in range(9)]
        
        # if x is the neural network
        if nn_is_x:
            
            # open values
            ind = 0
            for i in self.board.values():

                # set xs as 1 (its moves)
                if i == 1:
                    input[ind] = 1
                
                # set os as -1 (opponents moves)
                elif i == 0:
                    input[ind] = -1

                ind += 1
        
        # if o is the neural network
        else:
            
            # open board values
            ind = 0
            for i in self.board.values():
                
                # set xs as -1 (opponents moves)
                if i == 1:
                    input[ind] = -1

                # set os as 1 (its moves)
                elif i == 0:
                    input[ind] = 1

                ind += 1

        # if needed to rotate board
        if self.orienting_corner != (0, 0) and len(self.remaining_moves) != 9:

            # rotate board
            input = self.rotate_board(input, self.rotation_times)

        return input

    # rotates all the values on the board clockwise a specified number of times
    def rotate_board(self, board_as_list: list[int], rotations: int) -> list[int]:

        # open rotation count
        for i in range(rotations):
            
            # new board after rotations
            new_board = []
            
            # for each column
            for i in range(3):

                # add each value in the column into the new board as a row
                new_board.append(board_as_list[6 + i])
                new_board.append(board_as_list[3 + i])
                new_board.append(board_as_list[0 + i])
            
            # set rotated board as the new standard
            board_as_list = new_board.copy()

        # return after final rotation
        return new_board

    # takes random turns for both x and o until a winner is found
    def play_out(self, prnt: bool = False) -> int:

        # until the game ends
        while not self.game_over:
            
            # take random turn
            self.take_turn(choice(self.remaining_moves))

            # if print, then print
            if prnt:
                print(self)
                print()
        
        # return winner
        return self.winner

    # returns a training example for a nn based on the moves of the winner
    def get_training_examples(self, use_json: bool = True) -> tuple[list[float]] | None:
        
        # trains on the winners examples, or on o's examples for cat's games

        """
            For input data:
                -1 MEANS THE ENEMY PLAYER
                 0 MEANS EMPTY
                 1 MEANS THE NETWORKS PLAYER
        """

        # make sure game is over
        if self.winner is None or not self.game_over:
            return -1
        
        # train for cat's games for the person who went second
        if self.winner == -1:
            winner = 0
        else:
            winner = self.winner

        # final return value
        examples = []        

        # the winner based on turns and the number that needs to be added to sync up the players turn indexes
        if winner == 1:
            starter_won = True
            turn_factor = 0
        else:
            starter_won = False
            turn_factor = 1
            
        # recreate board previous to each move as an accumulative 1D list
        move_board = [0, 0, 0, 0, 0, 0, 0, 0, 0]

        # define moves
        winner_moves = [self.o_moves, self.x_moves][winner]
        loser_moves = [self.x_moves, self.o_moves][winner]

        # open moves
        ind = 0
        for col, row in winner_moves:

            # apply first move if non start won
            if not starter_won and ind == 0:
                loser_move = loser_moves[ind]
                move_board[loser_move[1]*3 + loser_move[0]] = -1

            # define label
            label = [0 for i in range(9)]
            correct_move_col, correct_move_row = winner_moves[ind]
            label[correct_move_row*3 + correct_move_col] = 1

            # create_example
            if use_json:

                # rotate board
                if self.orienting_corner != (0, 0):

                    input = self.rotate_board(move_board.copy(), self.rotation_times)
                    label = self.rotate_board(label, self.rotation_times)
                    examples.append([dumps([input, label])])

                else:
                    input = move_board.copy()
                    examples.append([dumps([input, label])])
            else:
                # rotate board
                if self.orienting_corner != (0, 0):

                    input = self.rotate_board(move_board.copy(), self.rotation_times)
                    label = self.rotate_board(label, self.rotation_times)
                    examples.append([input, label])
                
                else:

                    input = move_board.copy()
                    examples.append([input, label])

            # update for next time
            move_board[row*3 + col] = 1

            # apply loser move if not last move
            if not ind + 1 == len(winner_moves):
                loser_move = loser_moves[ind + turn_factor]
                move_board[loser_move[1]*3 + loser_move[0]] = -1

            ind += 1

        return examples


