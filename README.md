# Q-Learning Algorithm

---

## AIM
To implement the **Q-Learning algorithm** for a reinforcement learning environment and derive the **optimal policy** and **optimal value function**, comparing its performance with the **Monte Carlo method**.

---

## PROBLEM STATEMENT
In this experiment, we aim to design and implement an **off-policy Temporal Difference (TD) learning algorithm**, known as **Q-Learning**, to help an agent learn an optimal behavior through interactions with its environment.  

The agent will learn to choose actions that maximize cumulative rewards over time using the **Q-table**, which stores action-value pairs for each state.

The problem involves:
- Understanding state–action–reward–next state transitions (s, a, r, s’)
- Updating the Q-values iteratively using the TD error
- Exploring and exploiting the environment using an ε-greedy strategy
- Converging towards the optimal policy π\* and value function V\*

---

## Q-LEARNING ALGORITHM

**Algorithm Steps:**

1. **Initialize**
   - Initialize Q(s, a) arbitrarily for all state–action pairs.
   - Set learning rate (α), discount factor (γ), and exploration rate (ε).

2. **For each episode:**
   - Initialize the starting state `s`.

3. **Repeat for each step of the episode:**
   - Choose an action `a` using ε-greedy policy derived from Q.
   - Take action `a`, observe reward `r` and next state `s'`.
   - Update the Q-value using the Bellman optimality equation:

     \[
     Q(s, a) = Q(s, a) + α [r + γ \max_{a'} Q(s', a') - Q(s, a)]
     \]

   - Set `s = s'`.

4. **Until** `s` becomes terminal.

5. **Derive the optimal policy**:
   \[
   π^*(s) = \arg\max_a Q(s, a)
   \]

6. **Output**
   - Optimal policy (π\*)
   - Optimal state-value function (V\*)
   - Success rate and learning curves

---

## ⚙️ Q-LEARNING FUNCTION

### **Name:** Dharshini K  
### **Register Number:** 212223230047  

```python
def q_learning(env,
               gamma=1.0,
               init_alpha=0.5,
               min_alpha=0.01,
               alpha_decay_ratio=0.5,
               init_epsilon=1.0,
               min_epsilon=0.1,
               epsilon_decay_ratio=0.9,
               n_episodes=3000):

    nS, nA = env.observation_space.n, env.action_space.n
    pi_track = []

    Q = np.zeros((nS, nA), dtype=np.float64)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)

    select_action = lambda state, Q, epsilon: \
        np.argmax(Q[state]) if np.random.random() > epsilon \
        else np.random.randint(len(Q[state]))

    alphas = decay_schedule(init_alpha, min_alpha,
                            alpha_decay_ratio, n_episodes)

    epsilons = decay_schedule(init_epsilon, min_epsilon,
                              epsilon_decay_ratio, n_episodes)

    for e in tqdm(range(n_episodes), leave=False):
        state, done = env.reset(), False

        while not done:
            action = select_action(state, Q, epsilons[e])
            next_state, reward, done, _ = env.step(action)

            td_target = reward + gamma * Q[next_state].max() * (not done)
            td_error = td_target - Q[state][action]
            Q[state][action] = Q[state][action] + alphas[e] * td_error

            state = next_state

        Q_track[e] = Q
        pi_track.append(np.argmax(Q, axis=1))

    V = np.max(Q, axis=1)
    pi = lambda s: {s: a for s, a in enumerate(np.argmax(Q, axis=1))}[s]

    return Q, V, pi, Q_track, pi_track
```

## OUTPUT
### Optimal policy, optimal value function, and success rate for the optimal policy
<img width="549" height="868" alt="image" src="https://github.com/user-attachments/assets/e0f3254c-608e-4b30-a7b4-fc8a3044137b" />
<img width="423" height="394" alt="image" src="https://github.com/user-attachments/assets/587ef036-ca73-4a35-9d14-000829e21b53" />
<img width="850" height="713" alt="image" src="https://github.com/user-attachments/assets/0c629594-17b5-47fd-9ec4-93fb94c40de7" />
<img width="399" height="367" alt="image" src="https://github.com/user-attachments/assets/42276753-eabb-4515-8931-de9fa348974c" />
<img width="793" height="716" alt="image" src="https://github.com/user-attachments/assets/8b35abf0-49fb-48f2-8745-afe1aaaf37ab" />


### Plot comparing the state value functions of Monte Carlo method and Q-learning
<img width="1487" height="821" alt="image" src="https://github.com/user-attachments/assets/059a1df9-538d-4998-b720-d0ae6a8d1d8c" />
<img width="1484" height="823" alt="image" src="https://github.com/user-attachments/assets/3a668f12-ce7e-42b2-a3b2-cc49f426d19d" />

## RESULT
The Q-Learning algorithm was successfully implemented. The agent learned an optimal policy that maximized cumulative rewards through iterative Q-value updates.
The comparison plot demonstrates that the Q-learning state-value function closely approximates or outperforms the Monte Carlo method, confirming efficient off-policy TD learning behavior.
