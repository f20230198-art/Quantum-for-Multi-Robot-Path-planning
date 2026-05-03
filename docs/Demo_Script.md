# Demo Script
## Quantum-Assisted Multi-Robot Path Coordination via QUBO + QAOA

Use this as a spoken script for your demo or viva. Keep the tone confident, but stay honest about what has been demonstrated.

---

## 1. Opening

"Good morning sir. Our project is on multi-robot path coordination using a hybrid classical-quantum approach.

The main idea is simple: instead of trying to solve the entire multi-robot routing problem in one huge search, we first use classical A* to generate good candidate paths for each robot, and then we use QAOA on a QUBO formulation to choose the best combination of those paths while avoiding conflicts.

So the contribution is not just path planning for one robot, but coordinated decision-making for multiple robots."

---

## 2. Problem Statement

"The problem becomes hard because if each robot has K candidate paths and we have N robots, then the number of combinations is K^N.

That grows exponentially, so brute-force search becomes expensive very quickly.

Also, if we choose paths independently for each robot, we can get collisions, bottlenecks, or globally bad solutions even if each single path looks good locally."

---

## 3. Why We Chose This Approach

"We chose a hybrid A* plus QAOA design because each part does what it is best at.

A* is strong for generating sensible single-robot paths in a grid.
QAOA is designed for combinatorial optimization, so it is suitable for selecting among many binary choices under constraints.

This separation keeps the system efficient and also makes the quantum part focused on the hardest part of the problem, which is coordination."

---

## 4. Why QUBO

"We formulated the coordination problem as a QUBO because QUBO is a standard binary optimization form, and QAOA naturally works on binary variables through Ising-style cost Hamiltonians.

In our case, each binary variable means whether a robot selects a specific candidate path.

The QUBO objective combines three things:
- path cost, so shorter paths are preferred
- conflict penalties, so colliding paths are discouraged
- one-hot constraints, so each robot selects exactly one path

So QUBO is the bridge that converts the robotics problem into a form that quantum optimization can handle directly."

---

## 5. Core Novelty

"The novelty of the project is the way we structure the problem.

First, we do not ask the quantum algorithm to solve raw grid navigation from scratch. That would be too large and inefficient.

Instead, we reduce the problem using A* candidate generation, then let QAOA solve the global coordination layer.

So the novelty is in the hybrid decomposition: classical search generates feasible options, and quantum optimization resolves the global multi-robot assignment."

If asked for a one-line novelty statement:

"Our novelty is a QUBO-based QAOA coordination layer on top of classical A* candidate generation for multi-robot path planning."

---

## 6. How It Works

"The pipeline works in four steps.

First, for each robot, we generate multiple candidate paths using A* and small perturbations.
Second, we build a conflict matrix that captures when two candidate paths overlap or collide.
Third, we encode the objective as a QUBO.
Fourth, QAOA searches for the minimum-energy binary assignment, which corresponds to the best coordinated set of paths.

So instead of picking one path per robot independently, we optimize the whole team together."

---

## 7. How It Beats the Baselines

"Compared to greedy selection, our method is better because greedy only looks at local cost and can easily pick conflict-heavy combinations.

Compared to brute-force, our method does not enumerate every combination explicitly. It searches the QUBO landscape using a parameterized quantum circuit, which is much more structured than exhaustive enumeration.

In our results, QAOA gives near-optimal solutions and avoids the bad collision-prone choices that greedy often makes."

If the examiner asks about quality numbers:

"In our tested benchmarks, QAOA is around 99 percent of brute-force optimum on average, so the solution quality is very close to optimal."

---

## 8. Important Honest Point About Runtime

"I want to be precise here: at the current tested simulator scale, brute-force is still faster than QAOA in wall-clock time.

So we are not claiming a current runtime win on small instances.

What we are claiming is that brute-force scales exponentially as K^N, while the quantum formulation is built for better scaling on larger hardware and larger problem instances. The practical speedup is therefore a future-scale and hardware-dependent advantage, not something we overstate today."

This is a very strong answer if the sir challenges you on honesty.

---

## 9. Why QAOA Specifically

"We chose QAOA because it is one of the most direct quantum algorithms for combinatorial optimization.

It works well with QUBO and Ising formulations, it is hybrid, and it can run on near-term quantum devices or simulators.

That makes it a practical choice for a research prototype, because we can demonstrate the mapping today even if full quantum advantage is still a longer-term goal."

---

## 10. Why This Is Better Than Brute Force as Scale Grows

"Brute-force checks every possible combination, so if the number of robots or candidate paths increases, the search space explodes exponentially.

Our method avoids that full enumeration. It compresses the problem into a quadratic objective and then uses variational optimization.

So as the problem size grows, brute-force becomes infeasible very quickly, while the quantum formulation is designed to stay tractable on structured hardware and can potentially scale better."

If pushed harder:

"In short, brute-force has guaranteed optimality but poor scaling. QAOA trades exactness for a much more structured search, which is why it is attractive for larger instances."

---

## 11. What The Results Show

"Our experiments show three main things.

First, QAOA gives solutions very close to brute-force optimum.
Second, it handles conflicts much better than greedy selection.
Third, the current simulator results show that the quantum method is not yet faster at small scale, which is expected.

So the project is successful as a proof of concept for quantum-assisted coordination, not as a claim of immediate practical speedup on tiny benchmarks."

---

## 12. Short Closing

"To conclude, this project shows that multi-robot coordination can be expressed as a QUBO and solved with QAOA on top of classical A* candidate generation.

The main contribution is the hybrid decomposition and the conflict-aware optimization layer.

Our results show near-optimal quality, and our future direction is to test larger instances and more advanced quantum backends to study scaling more deeply.

Thank you sir. I am happy to take questions."

---

# Viva / Defense Q&A

## 1. What is the novelty?

"The novelty is the A* plus QAOA hybrid architecture for multi-robot coordination, where QAOA solves the global assignment problem on a QUBO rather than planning each robot independently."

## 2. Why not use only A*?

"A* is excellent for one robot, but it does not solve global coordination across multiple robots. If we use it independently, robots can collide or create inefficient team-level solutions."

## 3. Why not use only brute-force?

"Brute-force is exact but exponential in the number of robots and candidate paths. It becomes impractical as the problem grows."

## 4. Why QUBO instead of a direct graph search?

"QUBO is a natural fit for binary decision problems and maps directly to Ising models, which QAOA can optimize. It also lets us encode costs and constraints in one objective."

## 5. Why QAOA instead of Grover or another quantum algorithm?

"QAOA is better suited for constrained combinatorial optimization. It is hybrid, flexible, and directly aligned with QUBO/Ising problems."

## 6. Does your method beat brute-force?

"On small tested instances, it does not beat brute-force in runtime. It does beat greedy in solution quality and conflict handling, and it matches brute-force closely in quality. The scaling advantage over brute-force is the long-term motivation, not the current small-scale runtime result."

## 7. Then why is it useful?

"Because it demonstrates a scalable formulation for a hard multi-robot coordination problem, and it achieves near-optimal quality without exhaustive enumeration."

## 8. How do you prove it handles collisions?

"We include conflict penalties in the QUBO objective, so combinations with overlapping or conflicting paths get penalized during optimization."

## 9. What happens if the number of robots increases?

"The brute-force space grows as K^N. Our method still needs more resources, but it does not enumerate every combination explicitly, so the formulation remains more structured and potentially more scalable."

## 10. Is the quantum advantage proven here?

"No, not experimentally at this small scale. What we show is a correct hybrid quantum formulation and strong solution quality. The advantage claim is future-oriented and hardware-dependent."

## 11. What is the role of A* exactly?

"A* generates strong candidate paths for each robot. It reduces the search space before the quantum optimization stage."

## 12. What is the role of QAOA exactly?

"QAOA chooses the best combination of candidate paths across all robots while respecting costs and collision penalties."

---

# One-Minute Version

"Our project is a hybrid classical-quantum approach to multi-robot path coordination. A* generates candidate paths for each robot, and QAOA solves a QUBO that selects the best global combination while avoiding collisions.

We chose QUBO because the problem is naturally binary: each robot either picks a candidate path or not. That makes it compatible with QAOA and Ising-style optimization.

The main novelty is the coordination layer: instead of optimizing each robot separately, we optimize the whole team together.

In experiments, QAOA gives near-optimal solutions and performs much better than greedy in conflict-heavy cases. However, I want to be honest that brute-force is still faster at our current small test sizes. The reason we use this approach is that brute-force scales exponentially as the number of robots and candidate paths grows, while the quantum formulation is designed for better scaling on larger hardware and larger instances."

---

# Tips For Delivery

- If asked about novelty, say: "The novelty is the QUBO-based QAOA coordination layer on top of classical A*."
- If asked about brute-force, say: "Quality-wise we are close to optimal; runtime-wise we are not claiming a small-scale speedup."
- If asked why QUBO, say: "Because the decision is binary and QAOA naturally optimizes binary quadratic objectives."
- If asked why this matters, say: "Because the search space grows exponentially, so exact enumeration becomes infeasible as scale increases."
- Do not overclaim quantum speedup. Be precise and defensive. That will make your demo stronger.
