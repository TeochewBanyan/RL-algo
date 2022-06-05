# RL-algo

Codes and notes. Implement some RL algorithms.

---

### Differences between REINFORCE, Actor-critic and A2C

#### REINFORCE vs. Actor-critic

+ REINFORCE uses value function as baseline, and use MC return $G_t$ to estimate policy gradient $\grad J(\theta)$
+ Actor-critic uses value function (or Q function) to bootstrap, updates the value function step by step, and use Q function to estimate policy gradient.

~~~python
# REINFORCE
"""
params:
lr: learning rate
decay: decay rate (t)
log_pi_gradient: ln(gradient_pi(at|st))
theta: param of policy
"""
error = G-v(st)
w = w + lr*error*v_gradient
theta = theta + lr*decay*error*log_pi_gradient

# Actor-critic (Q function)
"""
params:
df: discount factor
r: r_t+1
w: param of Q
pi: policy
"""
at_next = pi(st)
error = r + df*Q(st_next, at_next) - Q(st, at)
theta = theta + lr*decay*Q(st,at)*log_pi_gradient
w = w + lr*error*Q_gradient
~~~



#### AC vs. A2C

###### In Brief:

A2C uses Advantage function instead of Q function. Since Advantage function is Q-v, and v is the baseline, A2C could be viewed as a baseline version of AC.





### DDPG

Code referenced from spinningup.
