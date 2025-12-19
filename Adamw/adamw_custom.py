import numpy as np

# Example: simple quadratic function f(x) = x^2
def f(x):
    return x**2

def grad_f(x):
    return 2*x

# Adam hyperparameters
eta = 0.1          # learning rate
beta1 = 0.9        # decay rate for first moment
beta2 = 0.999      # decay rate for second moment
epsilon = 1e-8     # small number to prevent division by zero
num_steps = 100    # number of optimization steps

# Initialize
x = np.array([5.0])  # starting point
m = np.zeros_like(x) # first moment
v = np.zeros_like(x) # second moment

print(m, v)

# Optimization loop
for t in range(1, num_steps + 1):
    g = grad_f(x) # compute gradient
    m = beta1 * m + (1 - beta1) * g # update first moment
    v = beta2 * v + (1 - beta2) * (g ** 2) # update second moment
    
    # Bias correction
    m_hat = m / (1 - beta1**t)
    v_hat = v / (1 - beta2**t)
    
    # Update parameter
    x = x - eta * m_hat / (np.sqrt(v_hat) + epsilon)
    
    print(f"Step {t}: x = {x}, f(x) = {f(x)}")

print(f"Optimized x: {x}, f(x) = {f(x)}")
