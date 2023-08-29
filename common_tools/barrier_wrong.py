import numpy as np
import scipy.stats as stats
import math
from scipy.stats import norm

class BarrierOption:
    def __init__(self, S, K, B, T, r, sigma):
        self.S = S
        self.K = K
        self.B = B
        self.T = T
        self.r = r
        self.sigma = sigma

    def dp(self, S):
        return (math.log(S) + (self.r + self.sigma ** 2 / 2.0) * self.T) / (self.sigma * math.sqrt(self.T))

    def dm(self, S):
        return (math.log(S) + (self.r - self.sigma ** 2 / 2.0) * self.T) / (self.sigma * math.sqrt(self.T))

    def ind(self, condition):
        return 1 if condition else 0

class UpAndOutBarrierOption(BarrierOption):
    def calculate_price(self):
        if self.T == 0:
            if self.S <= self.B:
                return max(self.S - self.K, 0)
            else:
                return 0
        else:
            phi = 1
            d1 = self.dp(self.S / self.K)
            d2 = self.dp(self.S / self.B)
            d3 = self.dp(self.B ** 2 / (self.K * self.S))
            d4 = self.dp(self.B / self.S)
            d5 = self.dm(self.S / self.K)
            d6 = self.dm(self.S / self.B)
            d7 = self.dm(self.B ** 2 / (self.K * self.S))
            d8 = self.dm(self.B / self.S)

            A = self.S * self.ind(self.S < self.B) * (
                norm.cdf(d1) - norm.cdf(d2) - (self.B / self.S) ** (1 + 2 * self.r / self.sigma ** 2) * (
                            norm.cdf(d3) - norm.cdf(d4)))
            B = self.K * math.exp(-self.r * self.T) * self.ind(self.S < self.B) * (
                (norm.cdf(d5) - norm.cdf(d6)) - (self.S / self.B) ** (1 - 2 * self.r / self.sigma ** 2) * (
                            norm.cdf(d7) - norm.cdf(d8)))

            return A - B

class DownAndOutBarrierOption(BarrierOption):
    def calculate_price(self):
        if self.T == 0:
            if self.S >= self.B:
                return max(self.K - self.S, 0)
            else:
                return 0
        else:
            phi = -1
            d1 = self.dp(self.S / self.K)
            d2 = self.dp(self.S / self.B)
            d3 = self.dp(self.B ** 2 / (self.K * self.S))
            d4 = self.dp(self.B / self.S)
            d5 = self.dm(self.S / self.K)
            d6 = self.dm(self.S / self.B)
            d7 = self.dm(self.B ** 2 / (self.K * self.S))
            d8 = self.dm(self.B / self.S)

            A = phi * self.K * math.exp(-self.r * self.T) * self.ind(self.S >= self.B) * (
                norm.cdf(d5) - norm.cdf(d6) - (self.B / self.S) ** (1 + 2 * self.r / self.sigma ** 2) * (
                            norm.cdf(d7) - norm.cdf(d8)))
            B = phi * self.S * self.ind(self.S >= self.B) * (
                (norm.cdf(d1) - norm.cdf(d2)) - (self.S / self.B) ** (1 - 2 * self.r / self.sigma ** 2) * (
                            norm.cdf(d3) - norm.cdf(d4)))

            return A - B

def monte_carlo_barrier_option_up_out(option_type, S0, K, T, r, sigma, H, num_paths, num_steps):
    dt = T / num_steps
    drift = np.exp((r - 0.5 * sigma ** 2) * dt)
    vol = sigma * np.sqrt(dt)

    payoffs = np.zeros(num_paths)
    H_shift = H * math.exp(0.5826 * sigma * math.sqrt(dt))
    for i in range(num_paths):
        path = S0 * np.cumprod(drift * np.exp(vol * np.random.normal(size=num_steps)))
        if option_type == "up":
            condition = np.max(path) <= H_shift
        else:
            condition = np.min(path) >= H_shift

        if condition:
            payoffs[i] = np.maximum(path[-1] - K, 0)

    monte_carlo_option_price = np.mean(payoffs) * np.exp(-r * T)

    return monte_carlo_option_price

if __name__ == "__main__":
    # S = 100
    # K = 100
    # B = 150
    # T = 1
    # r = 0.05
    # sigma = 0.2


    # up_and_out_option = UpAndOutBarrierOption(S, K, B, T, r, sigma)
    # up_and_out_price = up_and_out_option.calculate_price()
    # print("Price of the up-and-out call option:", up_and_out_price)

    S0 = 100
    K = 100
    T = 1
    r = 0.05
    sigma = 0.2
    B = 90
    num_paths = 5000000
    num_steps = 100

    down_and_out_option = DownAndOutBarrierOption(S0, K, B, T, r, sigma)
    down_and_out_price = down_and_out_option.calculate_price()
    print("Price of the down-and-out call option:", down_and_out_price)

    # monte_carlo_up_and_out_price = monte_carlo_barrier_option("up", S0, K, T, r, sigma, B, num_paths, num_steps)
    # print("Monte Carlo Up-and-Out Option Price:", monte_carlo_up_and_out_price)

    monte_carlo_down_and_out_price = monte_carlo_barrier_option_up_out("down", S0, K, T, r, sigma, B, num_paths, num_steps)
    print("Monte Carlo Down-and-Out Option Price:", monte_carlo_down_and_out_price)
