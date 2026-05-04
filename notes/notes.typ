#let solution(body) = block(
  width: 100%,
  fill: rgb("#f4f8ff"),
  stroke: rgb("#c7d7f2"),
  inset: 12pt,
  above: 10pt,
  below: 10pt,
  [
    #strong[Solution]

    #body
  ],
)

#let code(body) = block(
  width: 100%,
  fill: rgb("#f4f4f4"),
  stroke: rgb("#c7c7c7"),
  inset: 12pt,
  above: 10pt,
  below: 10pt,
  [#body],
)

#let todo(body) = box(
  fill: rgb("#ff6666"),
  stroke: rgb("#cc4444"),
  inset: (x: 6pt, y: 4pt),
  [*TODO:* #body],
)


#align(center)[
  #text(22pt, weight: "bold")[RL Final Review]
  #v(0.3em)
  #text(14pt)[Gordon Chen]
]
#v(1em)


= DP
$v(S_t) = EE_pi [R_(t+1) + v(S_(t+1))]$

Need to explore all 1-step actions and next states to do an update.

== Policy Iteration
Policy Evaluation
Given $pi$, find $v_pi$.

$v_pi (s) = sum_a pi(a|s) q_pi (s,a)$

$q_pi (s,a) = R(s,a) + gamma sum_s' P_(s s')^a v_pi (s')$

$v_pi (s) = sum_a pi(a|s) [R(s, a) + gamma sum_s' P_(s s')^a v_pi (s')]$

$v_pi = R^pi + gamma P^pi v_pi$

$v_pi = (I - gamma P^pi)^(-1) R^pi$

+ for $s in S$: $v^"new"_pi (s) <- sum_a pi(a|s) [R(s, a) + gamma sum_s' P_(s s')^a v_pi (s')]$
+ $v_pi <- v_pi^"new"$

Greedy Policy Improvement: $pi(s) = "argmax"_a q(s, a)$

Do we need policy eval to converge?
- $epsilon$ convergence
- $k$ policy eval iters


== Value Iteration
for $s in S$: $v^*_pi (s) <- max_a [R(s, a) + gamma sum_s' P_(s s')^a v^*_pi (s')]$

Optimal policy readout: $pi(s) = "argmax"_a [R(s, a) + gamma sum_s' P_(s, s')^a v^*_pi (s')]$


= Monte Carlo Learning
$v_pi (s) = EE_pi [G_t | S_t = s]$

Empirical estimation: $V(s) = lim_(N -> infinity) 1/N sum_(i=1)^N G_i (s)$

EMA: $V(S_t) <- V(S_t) + alpha (G_t - V(S_t))$

Problem: high variance b/c $G_t$ is the sum of rewards through many timesteps.


= TD Learning
$V(S_t) <- V(S_t) + alpha (R_(t+1) + gamma V(S_(t+1)) - V(S_t))$

Biased b/c we are bootstrapping with $V(S_(t+1))$, but lower variance b/c reward from 1 future step instead of
whole trajectory.

Learning efficiency: learn from 1 step instead of rollouts in MC, takes advantage of Markov property.

#image("mc_td_dp.png", width: 100%)

== $n$-step TD Learning
$G_t^((1)) = R_(t+1) + gamma V(S_(t+1))$

$G_t^((2)) = R_(t+1) + gamma R_(t+2) + gamma^2 V(S_(t+2))$

$G_t^((n)) = sum_(i=1)^n gamma^(i-1) R_(t+i) + gamma^n V(S_(t+n))$

Tradeoff between bias and variance

== $lambda$ return
$G_t^lambda = (1-lambda) sum_(n=1)^infinity lambda^(n-1) G_t^(n)$


= Control
Monte-Carlo Learning and TD Learning are for prediction (estimating value function), now we need control (policy).

== Monte Carlo
Cannot use $V$ because then we would need to model environment dynamics for
$ pi(s) = "argmax"_a [R(s, a) + gamma sum_s' P_(s s')^a V(s')] $

So instead, estimate $Q$ using MC, then just $pi(s) = "argmax"_a Q(s, a)$.

Instead of using $"argmax"$ use $epsilon"-greedy"$.

== SARSA (TD)
$Q(s, a) <- Q(s, a) + alpha [R(s, a) + gamma Q(s', a') - Q(s, a)]$

$epsilon"-greedy"$ policy improvement.

== Q-Learning
Off-policy SARSA

$Q(s, a) <- Q(s, a) + alpha [R(s, a) + gamma max_a' Q(s', a') - Q(s, a)]$

SARSA: behavior policy = target policy = $Q(s, a)$

Q-Learning: behavior policy = $Q(s, a)$, target policy = $max_a' Q(s', a')$

Behavior policy is $epsilon"-greedy"$, target policy is greedy.


= DQN
Q-Learning Update: $Q(s, a) <- Q(s, a) + alpha [R(s, a) + gamma max_a' Q(s', a') - Q(s, a)]$

+ Sample trajectories using $epsilon"-greedy"$
+ Q-learning optim step

With replay buffer, Double DQN to not overestimate the Q-function, PER...

= Policy Gradients
$
J(theta) &= EE_(tau ~ pi_theta) [sum_(t=1)^T r_t] \

PP(tau|pi_theta) &= p(s_1) pi_theta (a_1|s_1) p(s_2|s_1, a_1) pi_theta (a_2|s_2) p(s_3|s_2, a_2)... \

&= p(s_1) product_(t=1)^T pi_theta (a_t|s_t) p(s_(t+1)|s_t, a_t) \

nabla_theta PP(tau|pi_theta) &= PP(tau|pi_theta) nabla_theta log PP(tau|pi_theta) \

&= PP(tau|pi_theta) sum_(t=1)^T nabla_theta log pi_theta (a_t|s_t) \

nabla_theta J(theta) &= EE_(tau ~ pi(theta)) [sum_(t=1)^T nabla_theta log pi_theta (a_t|s_t) (sum_(t=1)^T r_t)]
$

Rewards to go: $phi_t = sum_(t'=t)^T r_t'$

Rewards to go - baseline: $phi_t = sum_(t'=t)^T r_t' - v(s_t)$
