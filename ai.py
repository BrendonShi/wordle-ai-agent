import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
import os

# import the wordle environment
from wordle_env import WordleEnv

# including txt file with words
try:
    with open("183_words.txt", "r", encoding="utf-8") as file:
        # loading the words and filtering only the 5 letter ones just in case
        words_list = {line.strip().lower() for line in file if len(line.strip()) == 5}
except FileNotFoundError:
    print("Error: file not found")
    exit()

if not words_list:
    print("Error")
    exit()
    
CUSTOM_WORDS = list(words_list)


# environment and model
LOG_DIR = "wordle_logs"
MODEL_PATH = "dqn_wordle_model"
os.makedirs(LOG_DIR, exist_ok=True)

# creating the environment
env = WordleEnv(custom_words=CUSTOM_WORDS, word_length=5, max_guesses=6)


# callback to save the model during training
# this saves a backup every 100k steps so i don't lose everything if it crashes
checkpoint_callback = CheckpointCallback(
    save_freq=100_000,
    save_path=LOG_DIR,
    name_prefix="wordle_model"
)

# defining the dqn model. using "MultiInputPolicy" because our observation space is a dictionary
model = DQN(
    "MultiInputPolicy",
    env,
    verbose=1,
    tensorboard_log=LOG_DIR,
    buffer_size=50000,
    learning_rate=1e-4, # learning rate is low to be stable (0.0001)
    batch_size=64,
    learning_starts=1000,
    gamma=0.99,
    train_freq=4,
    gradient_steps=1,
    target_update_interval=1000,
    exploration_fraction=0.1, # explore 10% of the time at first
    exploration_final_eps=0.05,
)

if not os.path.exists(MODEL_PATH + ".zip"):
    # train the agent
    print("Starting training...")
    TOTAL_TIMESTEPS = 500_000
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=checkpoint_callback,
            log_interval=100 # training progress every 100 episodes
        )
        # to save the final model
        model.save(MODEL_PATH)
        print(f"Training complete, model saved")
    except KeyboardInterrupt:
        model.save(MODEL_PATH + "_interrupted")
        print(f"Training interrupted, model saved as interrupted")


# to load the trained model
try:
    model = DQN.load(MODEL_PATH, env=env)
except FileNotFoundError:
    print(f"Could not find model")
    # fallback to load a checkpoint if the main file isn't there
    model = DQN.load(f"{LOG_DIR}/wordle_model_{TOTAL_TIMESTEPS}_steps.zip", env=env)


# set the render_mode on the environment for evaluation
env.render_mode = "human"
num_episodes = 100 # number of games to play
total_wins = 0
total_guesses_in_wins = 0

for i in range(num_episodes):
    obs, info = env.reset()
    terminated = False
    truncated = False
    episode_reward = 0
    num_guesses = 0

    print(f"\n--- Episode {i+1} / {num_episodes} ---")
    
    while not terminated and not truncated:
        # use deterministic=True for evaluation (agent picks best-known action)
        action, _states = model.predict(obs, deterministic=True)
        
        obs, reward, terminated, truncated, info = env.step(int(action))
        
        episode_reward += reward
        num_guesses += 1

        if terminated:
            if reward == 1.0: # win condition
                print(f"WIN! Secret word was: {info['secret_word'].upper()}")
                print(f"Solved in {num_guesses} guesses.")
                total_wins += 1
                total_guesses_in_wins += num_guesses
            else: # lose condition
                print(f"LOSE! Secret word was: {info['secret_word'].upper()}")
            print(f"Total Reward: {episode_reward:.2f}")

# calculating final stats
win_rate = (total_wins / num_episodes) * 100
avg_guesses = (total_guesses_in_wins / total_wins) if total_wins > 0 else 0
print(f"Win Rate: {win_rate:.1f}% ({total_wins} / {num_episodes})")
if avg_guesses > 0:
    print(f"Average Guesses (on wins): {avg_guesses:.2f}")

env.close()