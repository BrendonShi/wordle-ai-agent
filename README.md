# Wordle AI Agent

---

# Played games by AI agent with winrate percentage:

## 1 played game

![wezterm-gui_VPEXPKiTAB](https://github.com/user-attachments/assets/512ad845-5842-42da-ae2e-5cc78a7954c1)

---

# My research

Environment Design: The Wordle environment was created taking
into account the main characteristics that are needed for Reinforcement
Learning: rewards and delay of rewards, incomplete information
and the need for strategic exploration. The game uses a dictionary
of 5-letter letters, which our agent has access to. The agent is
provided with information such as:

• Last feedback, which is a vector that shows the information of the
previous guess, namely the status of the letters.

• A vector that contains 26 integers, which is the state of the alphabet
and keeps track of the status of each letter.

• Remaining guesses, which indicates the remaining number of at
tempts that the agent can use, it can be from 0 to 6.

The most important function in RL is the reward function, because
it tells the agent how to behave correctly and what actions to take.
When the agent performs an action, the environment gives a reward
or penalty, depending on the action of the agent itself. DQN uses
these rewards to update the Q-value. By adjusting the rewards and
penalties, the agent gradually becomes smarter. Reward structure:

• When the secret word is found, the agent receives 1 point (win).

• When the agent runs out of attempts, he is given a penalty of 1
point.

• Every time the agent does not guess the secret word, he is given
a penalty of-0.1 point. This affects his incentive to find the secret
word faster and not waste a large number of attempts.

### Results
The agent was trained for 1,500,000 steps on 183 words with learning 
rate of 0.0001. At the end of the training, the graph showed that
the average number of attempts was 3.55, means the agent guessed
the secret word on average in 3 or 4 attempts. The agent learned to
prioritize depending on the status of the known letters, and used the
green ones again in the same place, and tried to use the yellow ones
in a different place to turn them into green, while avoiding the letters
with gray status. The agent’s win rate is approximately 87% out of
all 100 games played.
