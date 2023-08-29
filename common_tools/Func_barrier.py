import numpy as np
import scipy.stats as stats


import math
from scipy.stats import norm


def dp(T, r, sigma, S):
    return (math.log(S) + (r + sigma ** 2 / 2.0) * T) / (sigma * math.sqrt(T))


def dm(T, r, sigma, S):
    return (math.log(S) + (r - sigma ** 2 / 2.0) * T) / (sigma * math.sqrt(T))


def ind(condition):
    return 1 if condition else 0


def CBBC(S, K, B, T, r, sigma):

    if T==0:
        if S <= B:
            return max(S-K,0)
        else:
            return 0
    else:
        phi = 1
        d1 = dp(T, r, sigma, S / K)
        d2 = dp(T, r, sigma, S / B)
        d3 = dp(T, r, sigma, B ** 2 / (K * S))
        d4 = dp(T, r, sigma, B / S)
        d5 = dm(T, r, sigma, S / K)
        d6 = dm(T, r, sigma, S / B)
        d7 = dm(T, r, sigma, B ** 2 / (K * S))
        d8 = dm(T, r, sigma, B / S)

        A = S * ind(S < B) * (
                    norm.cdf(d1) - norm.cdf(d2) - (B / S) ** (1 + 2 * r / sigma ** 2) * (norm.cdf(d3) - norm.cdf(d4)))
        B = K * math.exp(-r * T) * ind(S < B) * (
                    (norm.cdf(d5) - norm.cdf(d6)) - (S / B) ** (1 - 2 * r / sigma ** 2) * (norm.cdf(d7) - norm.cdf(d8)))

        return A - B





def monte_carlo_up_and_out_barrier_option(S0, K, T, r, sigma, H, num_paths, num_steps):
    dt = T / num_steps
    drift = np.exp((r - 0.5 * sigma ** 2) * dt)
    vol = sigma * np.sqrt(dt)

    payoffs = np.zeros(num_paths)
    H_shift = H* math.exp(-0.5826 * sigma * math.sqrt(dt))
    for i in range(num_paths):
        path = S0*np.cumprod(drift * np.exp(vol * np.random.normal(size=num_steps)))
        if np.max(path) <= H_shift:
            payoffs[i] = np.maximum(path[-1] - K, 0)

    monte_carlo_option_price = np.mean(payoffs) * np.exp(-r * T)

    return monte_carlo_option_price
if __name__ == "__main__":
    # Parameters
    S = 100
    K = 110
    B = 150
    T = 1
    r = 0.05
    sigma = 0.2

    # Calculate the price of the up-and-out call option
    c_uo_price = CBBC(S, K, B, T, r, sigma)
    print("Price of the up-and-out call option:", c_uo_price)
    # Parameters
    S0 = 100  # Initial stock price
    K = 110  # Strike price
    T = 1  # Time to maturity (in years)
    r = 0.05  # Risk-free rate
    sigma = 0.2  # Volatility
    H = 150  # Barrier level
    num_paths = 5000000  # Number of paths for Monte Carlo simulation
    num_steps = 100  # Number of steps for Monte Carlo simulation

    # Calculate the exact solution
    #exact_option_price = exact_up_and_out_barrier_option(S0, K, T, r, sigma, H)

    # Perform Monte Carlo simulation
    monte_carlo_option_price = monte_carlo_up_and_out_barrier_option(S0, K, T, r, sigma, H, num_paths, num_steps)

    # Compare the results
    #print("Exact Option Price:", exact_option_price)
    print("Monte Carlo Option Price:", monte_carlo_option_price)

#%%


import numpy as np
import scipy.stats as stats
import math
from scipy.stats import norm

def dp(T, r, sigma, S):
    return (math.log(S) + (r + sigma ** 2 / 2.0) * T) / (sigma * math.sqrt(T))

def dm(T, r, sigma, S):
    return (math.log(S) + (r - sigma ** 2 / 2.0) * T) / (sigma * math.sqrt(T))

def ind(condition):
    return 1 if condition else 0

def DBBC(S, K, B, T, r, sigma):
    if T == 0:
        if S >= B:
            return max(K - S, 0)
        else:
            return 0
    else:
        phi = -1
        d1 = dp(T, r, sigma, S / K)
        d2 = dp(T, r, sigma, S / B)
        d3 = dp(T, r, sigma, B ** 2 / (K * S))
        d4 = dp(T, r, sigma, B / S)
        d5 = dm(T, r, sigma, S / K)
        d6 = dm(T, r, sigma, S / B)
        d7 = dm(T, r, sigma, B ** 2 / (K * S))
        d8 = dm(T, r, sigma, B / S)

        A = phi * K * math.exp(-r * T) * ind(S >= B) * (
                    norm.cdf(d5) - norm.cdf(d6) - (B / S) ** (1 + 2 * r / sigma ** 2) * (norm.cdf(d7) - norm.cdf(d8)))
        B = phi * S * ind(S >= B) * (
                    (norm.cdf(d1) - norm.cdf(d2)) - (S / B) ** (1 - 2 * r / sigma ** 2) * (norm.cdf(d3) - norm.cdf(d4)))

        return A - B

def monte_carlo_down_and_out_barrier_option(S0, K, T, r, sigma, H, num_paths, num_steps):
    dt = T / num_steps
    drift = np.exp((r - 0.5 * sigma ** 2) * dt)
    vol = sigma * np.sqrt(dt)

    payoffs = np.zeros(num_paths)
    H_shift = H * math.exp(-0.5826 * sigma * math.sqrt(dt))
    for i in range(num_paths):
        path = S0 * np.cumprod(drift * np.exp(vol * np.random.normal(size=num_steps)))
        if np.min(path) >= H_shift:
            payoffs[i] = np.maximum(path[-1] - K, 0)

    monte_carlo_option_price = np.mean(payoffs) * np.exp(-r * T)

    return monte_carlo_option_price

if __name__ == "__main__":
    # Parameters
    S = 100
    K = 90
    B = 80
    T = 1
    r = 0.05
    sigma = 0.2

    # Calculate the price of the down-and-out call option
    p_do_price = DBBC(S, K, B, T, r, sigma)
    print("Price of the down-and-out call option:", p_do_price)

    # Parameters
    S0 = 100  # Initial stock price
    K = 90  # Strike price
    T = 1  # Time to maturity (in years)
    r = 0.05  # Risk-free rate
    sigma = 0.2  # Volatility
    H = 80  # Barrier level
    num_paths = 5000000  # Number of paths for Monte Carlo simulation
    num_steps = 100  # Number of steps for Monte Carlo simulation

    # Perform Monte Carlo simulation
    monte_carlo_option_price = monte_carlo_down_and_out_barrier_option(S0, K, T, r, sigma, H, num_paths, num_steps)

    # Print the result
    print("Monte Carlo Option Price:", monte_carlo_option_price)

