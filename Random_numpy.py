import numpy as np
np.random.seed(3)

# Simulating coin flips

# Bernoulli trail : If the result of an experiment has a binary outcome.
# such a experiment is called Bernoulli trail.

# gives random numbers between 0 and 1
random_numbers = np.random.random(size=5) # flipped coin 5 times
print(random_numbers)


# say probability of heads is > 0.5

n_heads = np.sum(random_numbers > 0.5)
print(n_heads)


# Simulating 5 coin flips 10000 times

n_all_heads = 0

for i in range(100000):
    outcomes = np.random.random(size=4)
    n_heads = np.sum(outcomes > 0.5)
    if n_heads == 4:
        n_all_heads += 1

print(n_all_heads)  # the number of times they were 4 heads in 100000 trails.

# avg of the number of times we got 4 or more heads in 4 coin flips that ran 10000 times
avg_heads = n_all_heads/100000
print(avg_heads)


def bernouli_trails(n, p):
    """the outcomes of n trails with probablity of success p"""

    n_successes = 0

    for i in range(n):
        random_number = np.random.random()

        # success is getting less than or equal to success probability
        if random_number < p:
            n_successes += 1

    return n_successes

# Perform this 1000 times with probability 0.05 of success


n_total_success = np.empty(1000)

for i in range(1000):
    n_total_success[i] = bernouli_trails(100, 0.05)

print(n_total_success)

