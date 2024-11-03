import warnings
import numpy as np
import random
import tkinter as tk
from tkinter import messagebox, simpledialog
import os


warnings.filterwarnings("ignore")


################################### Creating the Interface/ GUI for TicTacToe game ####################
class GameInterface:
    def __init__(self, root_window):
        self.root_window = root_window
        self.root_window.title("TicTacToe Game")
        self.environment = GameTicTacToe()
        self.agent = RILearningAgent()
        self.human_marker = 'X'
        self.computer_marker = 'O'
        self.score_board = {'Human': 0, 'AI': 0, 'Ties': 0}
        self.current_position = None
        self.gui_buttons = []
        self.build_gui()
        self.select_marker()
        self.start_new_game()

    def select_marker(self):
        while True:
            symbol = simpledialog.askstring("Select Marker", "Pick your marker (X or O):").upper()
            if symbol in ['X', 'O']:
                self.human_marker = symbol
                self.computer_marker = 'O' if symbol == 'X' else 'X'
                break
            else:
                messagebox.showwarning("Invalid Choice", "Please select either 'X' or 'O'.")

    def build_gui(self):
        for i in range(3):
            row = []
            for j in range(3):
                button = tk.Button(self.root_window, text=' ', font=('Arial', 60), width=5, height=2,
                                   command=lambda x=i, y=j: self.human_turn(x, y))
                button.grid(row=i, column=j)
                row.append(button)
            self.gui_buttons.append(row)
        self.display_scoreboard()

    def display_scoreboard(self):
        self.score_label = tk.Label(self.root_window, text=self.fetch_score(), font=('Arial', 20))
        self.score_label.grid(row=3, column=0, columnspan=3)

    def fetch_score(self):
        return f"Human: {self.score_board['Human']} | AI: {self.score_board['AI']} | Ties: {self.score_board['Ties']}"

    def update_score(self, result):
        if result == self.human_marker:
            self.score_board['Human'] += 1
        elif result == self.computer_marker:
            self.score_board['AI'] += 1
        else:
            self.score_board['Ties'] += 1
        self.score_label.config(text=self.fetch_score())

    def start_new_game(self):
        self.environment.initialize_grid()
        for row in self.gui_buttons:
            for button in row:
                button.config(text=' ', state=tk.NORMAL)
        self.agent.exploration_rate = 1.0

    def human_turn(self, row, col):
        if self.environment.place_marker((row, col), self.human_marker):
            self.current_position = (row, col)
            self.gui_buttons[row][col].config(text=self.human_marker, state=tk.DISABLED)
            winner = self.environment.check_outcome()
            if winner:
                self.end_round(winner)
            else:
                self.ai_turn()

    def ai_turn(self):
        state_key = self.agent.generate_state_key(self.environment.grid)
        move = self.agent.select_action(self.environment.grid, self.human_marker)
        self.root_window.after(1000, self.process_ai_move, move, state_key)

    def process_ai_move(self, move, state_key):
        self.environment.place_marker((move // 3, move % 3), self.computer_marker)
        self.current_position = (move // 3, move % 3)
        self.gui_buttons[move // 3][move % 3].config(text=self.computer_marker, state=tk.DISABLED)

        next_state_key = self.agent.generate_state_key(self.environment.grid)
        outcome = self.environment.check_outcome()

        reward = 0
        if outcome == self.computer_marker:
            reward = 1  # AI wins
        elif outcome == self.human_marker:
            reward = -1  # Human wins
        elif outcome == 'T':
            reward = 0.5  # Tie

        self.agent.update_value(state_key, move, reward, next_state_key)
        if outcome:
            self.root_window.after(500, lambda: self.end_round(outcome))

    def end_round(self, result):
        if result == self.human_marker:
            messagebox.showinfo("Game Over", f"You win!")
        elif result == self.computer_marker:
            messagebox.showinfo("Game Over", "AI wins!")
        else:
            messagebox.showinfo("Game Over", "It's a draw!")

        self.update_score(result)
        self.start_new_game()


class GameTicTacToe:
    def __init__(self):
        self.grid = np.full((3, 3), ' ') # Grid means a game GUI board

    def initialize_grid(self):
        self.grid = np.full((3, 3), ' ')
        return self.grid

    def possible_moves(self):
        return list(zip(*np.where(self.grid == ' ')))

    def place_marker(self, move, player):
        if self.grid[move] == ' ':
            self.grid[move] = player
            return True
        return False

    def check_outcome(self):
        for marker in ['X', 'O']:
            if any(np.all(self.grid[i, :] == marker) for i in range(3)) or any(np.all(self.grid[:, i] == marker) for i in range(3)) or \
               np.all(np.diag(self.grid) == marker) or np.all(np.diag(np.fliplr(self.grid)) == marker):
                return marker
        return 'T' if len(self.possible_moves()) == 0 else None

    def is_winning_move(self, move, marker):
        grid_copy = self.grid.copy()
        grid_copy[move] = marker
        return self.check_outcome() == marker


class RILearningAgent:
    def __init__(self):
        self.state_action_values = {}
        self.alpha = 0.3  # alpha is learning rate and we can higher the value for learning rate for faster updates
        self.gamma = 0.9  # # gamma is discount factor for future rewards
        self.exploration_rate = 1.0
        self.exploration_decay = 0.995  # for decay
        self.exploration_min = 0.01  # for minimum exploration rate
        self.load_state_action_values()

    # we are using Below 'generate_state_key'function for converting index into a Q-table, which holds the estimated values. #####
    def generate_state_key(self, grid):
        return str(grid.reshape(9))

    def select_action(self, grid, opponent_marker):
        state_key = self.generate_state_key(grid)
        if state_key not in self.state_action_values:
            self.state_action_values[state_key] = np.zeros(9)

        available_moves = [i for i in range(9) if grid.flatten()[i] == ' ']

        # This functions is for defensive strategy - It will block human/user if they are about to win
        for move in available_moves:
            row, col = divmod(move, 3)  # Convert index to (row, col)
            if self.is_winning_move((row, col), opponent_marker, grid):
                return move  # Block human from winning

        # We are using the Epsilon-greedy exploration strategy for reinforcement learning
        if random.uniform(0, 1) < self.exploration_rate:
            return random.choice(available_moves)
        else:
            # It will return the best action for the current state
            action_values = self.state_action_values[state_key][available_moves]
            return available_moves[np.argmax(action_values)]

    def is_winning_move(self, move, marker, grid):
        row, col = move
        grid_copy = grid.copy()
        grid_copy[row, col] = marker
        return np.any([np.all(grid_copy[i, :] == marker) for i in range(3)]) or \
            np.any([np.all(grid_copy[:, i] == marker) for i in range(3)]) or \
            np.all(np.diag(grid_copy) == marker) or \
            np.all(np.diag(np.fliplr(grid_copy)) == marker)

    def update_value(self, state_key, action, reward, next_state_key):
        if state_key not in self.state_action_values:
            self.state_action_values[state_key] = np.zeros(9)
        if next_state_key not in self.state_action_values:
            self.state_action_values[next_state_key] = np.zeros(9)

        future_best_value = np.max(self.state_action_values[next_state_key])
        self.state_action_values[state_key][action] += self.alpha * (
            reward + self.gamma * future_best_value - self.state_action_values[state_key][action])

    def reduce_exploration(self):
        self.exploration_rate = max(self.exploration_min, self.exploration_rate * self.exploration_decay)

    def save_state_action_values(self, file_name='state_action_values.npy'):
        np.save(file_name, self.state_action_values)

    def load_state_action_values(self, file_name='state_action_values.npy'):
        if os.path.exists(file_name):
            self.state_action_values = np.load(file_name, allow_pickle=True).item()


def agent_train_using_ril(total_episodes):
    agent_ril = RILearningAgent()
    game_env = GameTicTacToe()

    for _ in range(total_episodes):
        current_state = game_env.initialize_grid()
        game_over = False

        while not game_over:
            action = agent_ril.select_action(current_state, 'X')  # Assuming human is 'X'
            if game_env.place_marker((action // 3, action % 3), 'O'):
                win = game_env.check_outcome()
                next_state = game_env.grid
                next_state_key = agent_ril.generate_state_key(next_state)

                if win == 'O':
                    reward = 1
                    game_over = True
                elif win == 'X':
                    reward = -1
                    game_over = True
                elif win == 'T':
                    reward = 0.5
                    game_over = True
                else:
                    reward = 0

                state_key = agent_ril.generate_state_key(current_state)
                agent_ril.update_value(state_key, action, reward, next_state_key)
                current_state = next_state

        agent_ril.reduce_exploration()

    agent_ril.save_state_action_values()  # Saving all learned values/actions
    return agent_ril


# Training the Agent for game before launching the game - Please use less value for fast launch but not recommend by me. #####
agent_ril = agent_train_using_ril(100000)

# Launch the GUI for the game
root = tk.Tk()
game_app = GameInterface(root)
root.mainloop()