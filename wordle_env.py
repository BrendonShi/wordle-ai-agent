import gymnasium as gym
from gymnasium import spaces
import numpy as np
from collections import Counter
import random

# ANSI color codes for rendering
class bcolors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    GRAY = '\033[90m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

class WordleEnv(gym.Env):
    """
    A Gymnasium environment for the game of Wordle.

    last_feedback: Box(0, 3, (word_length,))
        The feedback from the previous guess.
        0: Empty (for the first turn)
        1: Gray (Letter not in word)
        2: Yellow (Letter in word, wrong position)
        3: Green (Letter in word, correct position)
    alphabet_status: Box(0, 3, (26,))
        0: Unused
        1: Gray
        2: Yellow
        3: Green
    """
    metadata = {"render_modes": ["human"], "render_fps": 1}

    # define feedback values
    EMPTY = 0
    GRAY = 1
    YELLOW = 2
    GREEN = 3

    def __init__(self, custom_words, word_length=5, max_guesses=6, render_mode=None):
        super().__init__()

        if not custom_words:
            raise ValueError("custom_words list cannot be empty.")
            
        self.word_length = word_length
        self.max_guesses = max_guesses
        self.render_mode = render_mode

        # making sure words are unique, lowercase and the right length
        self.word_list = sorted(list(set(
            [w.lower() for w in custom_words if len(w) == self.word_length]
        )))
        
        if not self.word_list:
            raise ValueError(f"No valid words of length {self.word_length} found in custom_words.")

        self.num_words = len(self.word_list)
        
        # define spaces for agent memory
        self.action_space = spaces.Discrete(self.num_words)
        
        # the observation space is a dictionary because the agent needs different types of info
        # 1. last_feedback: colors of the previous guess
        # 2. alphabet_status: which letters on the keyboard are green/yellow/gray
        # 3. guesses_remaining: how many turns are left
        self.observation_space = spaces.Dict({
            "last_feedback": spaces.Box(low=0, high=3, shape=(self.word_length,), dtype=np.int32),
            "alphabet_status": spaces.Box(low=0, high=3, shape=(26,), dtype=np.int32),
            "guesses_remaining": spaces.Discrete(self.max_guesses + 1)
        })

        # game state variables
        self.secret_word = ""
        self.current_guess_count = 0
        self.last_feedback = np.zeros((self.word_length,), dtype=np.int32)
        self.alphabet_status = np.zeros((26,), dtype=np.int32)
        self.guess_history = []
        self.feedback_history = []

    def _letter_to_int(self, letter):
        """Converts a lowercase letter 'a'-'z' to an integer 0-25"""
        return ord(letter) - ord('a')

    def _int_to_letter(self, val):
        """Converts an integer 0-25 to a lowercase letter 'a'-'z'"""
        return chr(val + ord('a'))

    def _get_obs(self):
        """Return the current state of the agent"""
        return {
            "last_feedback": self.last_feedback.copy(),
            "alphabet_status": self.alphabet_status.copy(),
            "guesses_remaining": self.max_guesses - self.current_guess_count
        }

    def _get_info(self):
        """Gets auxiliary information (for debugging/logging)"""
        return {
            "secret_word": self.secret_word,
            "guesses_made": self.current_guess_count,
            "guess_history": self.guess_history
        }

    def reset(self, seed=None, options=None):
        """Resets the environment for a new game"""
        super().reset(seed=seed)

        # pick a new secret word
        self.secret_word = self.np_random.choice(self.word_list)
        
        # reset game state
        self.current_guess_count = 0
        self.last_feedback = np.zeros((self.word_length,), dtype=np.int32)
        self.alphabet_status = np.zeros((26,), dtype=np.int32)
        self.guess_history = []
        self.feedback_history = []
        
        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), self._get_info()

    def step(self, action):
        """Performs one step in the environment (makes a guess)"""
        
        if self.current_guess_count >= self.max_guesses:
            return self._get_obs(), 0, True, False, self._get_info()

        # converting the agent's action index back into a word string
        guess_word = self.word_list[action]
        self.current_guess_count += 1
        
        # calculate feedback (core wordle logic)
        feedback = np.full((self.word_length,), self.GRAY, dtype=np.int32)
        secret_counts = Counter(self.secret_word)
        
        # pass for greens
        for i in range(self.word_length):
            if guess_word[i] == self.secret_word[i]:
                feedback[i] = self.GREEN
                secret_counts[guess_word[i]] -= 1
                self.alphabet_status[self._letter_to_int(guess_word[i])] = self.GREEN

        # pass for yellows and grays
        # we do this second so we don't double count letters that are already green
        for i in range(self.word_length):
            if feedback[i] == self.GREEN:
                continue

            letter = guess_word[i]
            lti = self._letter_to_int(letter)

            if letter in self.secret_word and secret_counts[letter] > 0:
                feedback[i] = self.YELLOW
                secret_counts[letter] -= 1
                # only update the alphabet to yellow if it's not already green
                if self.alphabet_status[lti] != self.GREEN:
                    self.alphabet_status[lti] = self.YELLOW
            else:
                # if it's not green or yellow, it's gray (unused)
                if self.alphabet_status[lti] == self.EMPTY:
                    self.alphabet_status[lti] = self.GRAY
        
        # saving the feedback for the next observation
        self.last_feedback = feedback
        self.guess_history.append(guess_word)
        self.feedback_history.append(feedback)

        # >>> calculate Reward and Termination <<<
        terminated = False
        reward = -0.1  # small penalty for each guess to encourage speed

        if guess_word == self.secret_word:
            terminated = True
            reward = 1.0  # big reward for winning
        elif self.current_guess_count == self.max_guesses:
            terminated = True
            reward = -1.0 # big penalty for losing

        truncated = False # this env doesn't have a time limit, just a game-end condition
        
        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self):
        """Renders the current state of the game"""
        if self.render_mode == "human":
            self._render_frame()

    def _render_frame(self):
        """Internal helper for rendering to console.
        Printing the game board with colors to the terminal"""
        print("\n" + "="*30)
        print(f"Guess {self.current_guess_count} of {self.max_guesses}")
        print("-" * 30)

        for i in range(len(self.guess_history)):
            guess = self.guess_history[i]
            feedback = self.feedback_history[i]
            
            render_str = ""
            for j in range(self.word_length):
                letter = guess[j].upper()
                if feedback[j] == self.GREEN:
                    render_str += f"{bcolors.GREEN}{letter}{bcolors.ENDC} "
                elif feedback[j] == self.YELLOW:
                    render_str += f"{bcolors.YELLOW}{letter}{bcolors.ENDC} "
                else: # GRAY
                    render_str += f"{bcolors.GRAY}{letter}{bcolors.ENDC} "
            print(f"  {render_str}")
        
        # print empty slots
        for _ in range(self.max_guesses - len(self.guess_history)):
            print("  " + "_ " * self.word_length)
        
        print("-" * 30)
        print("Alphabet Status:")
        
        alphabet_render = ""
        for i in range(26):
            letter = self._int_to_letter(i).upper()
            status = self.alphabet_status[i]
            if status == self.GREEN:
                alphabet_render += f"{bcolors.GREEN}{letter}{bcolors.ENDC} "
            elif status == self.YELLOW:
                alphabet_render += f"{bcolors.YELLOW}{letter}{bcolors.ENDC} "
            elif status == self.GRAY:
                alphabet_render += f"{bcolors.GRAY}{letter}{bcolors.ENDC} "
            else: # UNUSED
                alphabet_render += f"{letter} "
            if letter == 'M':
                alphabet_render += "\n"
        print(alphabet_render)
        print("="*30)

    def close(self):
        """Cleans up the environment"""
        pass
