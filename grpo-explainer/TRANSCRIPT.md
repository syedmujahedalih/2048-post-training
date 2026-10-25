# GRPO and online RL: the 2048 explainer

Original teaching script. Numerical illustrations are not measured training results.

## How playing changes a model

You already know how to play 2048. Now imagine teaching a language model to play it. Supervised fine tuning gives it examples of a solver's moves. Online reinforcement learning lets it play, receive feedback, and change its future behavior. GRPO is one way to turn that feedback into a training signal. Let's follow one training iteration, from a board to an update.

## Board, policy, action, reward

The board is the state. Our model reads a text version of that board. Its policy describes how likely it is to choose each possible move. In our language model, those choices are produced through the tokens of a move tool call. The simulator applies the move, merges tiles, adds a random tile, and returns a new board. A sequence of these interactions is called a rollout, or trajectory.

## What makes the learning online?

Online means that training keeps collecting fresh experience from the policy being trained. It does not mean the model needs to browse the internet. First, hold a snapshot of the policy fixed while it generates a batch of gameplay. Then score those rollouts and run optimizer updates on the trainable weights. The updated policy generates the next batch. Playing alone does not change the weights.

## Start together. Explore differently.

GRPO stands for Group Relative Policy Optimization. For one starting board, sample several rollouts from the same policy. Our first experiment proposes four rollouts, each capped at sixteen decision attempts. They can choose different moves, and random tile spawns can produce different outcomes. This group supplies a local comparison: how did each attempt perform against its siblings?

## Choose what earns reward

A reward function converts each outcome into a number. For 2048, we can compute feedback directly from the simulator. Our initial proposal uses the logarithm of one plus score gained, minus a penalty for invalid attempts. That rewards progress during a short episode, but it is only a proxy for strong full game play. We still need held out games to check whether optimizing that proxy actually helps.

## Reward becomes relative advantage

Use a small numerical example: four rewards are zero, one, two, and three. Their mean is one point five. Subtract that mean from each reward. Attempts below the mean get negative advantages. Attempts above it get positive advantages. A common GRPO version also divides by the group's standard deviation. The group provides the baseline, replacing a separately learned value critic. Exact normalization depends on the implementation and configuration.

## Advantage steers the policy

Now the optimizer uses those advantages to change the model's probabilities. Positive advantages encourage the sampled action tokens. Negative advantages discourage them. With an episode level reward, many decisions share the same signal. That does not prove every move in a successful rollout was good. This is the credit assignment challenge: connecting a later outcome to earlier choices. In our project, the LoRA adapters are the trainable weights.

## Keep an update from overreacting

GRPO can use a clipped objective to limit the incentive for overly large probability changes. For a positive advantage, increasing the probability ratio beyond the clipping threshold stops improving that part of the objective. Clipping is not a hard guarantee that the probability ratio stays inside the band. An optional KL penalty can also anchor the policy to a reference model. The old rollout policy and the reference policy have different jobs.

## Three things to watch

First, equal rewards provide no group relative preference signal. Second, lucky tile spawns can make a poor strategy look good in a small sample. Third, a model can improve the short episode reward while getting worse at complete games. Track reward variation and policy behavior, then evaluate on separate game seeds. A rising training reward is evidence to investigate, not proof that the model learned a better strategy.

## Our first TRL experiment

Here is our plan. Begin with the supervised checkpoint trained on solver demonstrations. Let the student play short episodes. Use the simulator's outcomes to calculate rewards. Inspect ten updates: the sampled actions, observation masks, advantages, gradients, and changed adapter weights. Then compare against supervised fine tuning alone on held out full games. The solver teaches during SFT. It does not silently replace the student's sampled actions during RL. We have not run that training yet.

## Three ideas to remember

Before starting the GPU run, check these three ideas. Online: fresh gameplay comes from the policy being trained. Group relative: compare attempts that share a starting board. Policy optimization: rewards become advantages, and optimizer steps change the trainable weights. The experiment succeeds as a learning exercise when we can explain that whole loop and measure its results honestly. Pause, replay a chapter, and then we are ready for the first small training check.
