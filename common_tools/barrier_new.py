import numpy as np
from scipy.special import comb
from scipy.stats import norm
import matplotlib.pyplot as plt


class vanilla_option:

    def __init__(self, S0=1, K=1, T=1, r=0.05, sigma=0.2, type_='call'):
        """
        :param S0: Initial spot price
        :param K: Strike price
        :param T: Time to expiration
        :param r: risk-free rate
        :param sigma: Volatility
        :param type_: type of option - 'call' or 'put'
        """
        self.S0 = S0 + 1.0e-10     # avoid dividing by zero with S=0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.type_ = type_

    def d1(self):
        return ((np.log(self.S0 / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T)) / (self.sigma * np.sqrt(self.T))

    def d2(self):
        return self.d1() - self.sigma * np.sqrt(self.T)

    def price(self):
        if self.type_ == "call":
            return self.S0 * norm.cdf(self.d1()) - self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2())
        else:
            return self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2()) - self.S0 * norm.cdf(-self.d1())

    def delta(self):
        if self.type_ == "call":
            return norm.cdf(self.d1())
        else:
            return norm.cdf(self.d1()) - 1

    def gamma(self):
        return norm.pdf(self.d1()) / (self.S0 * self.sigma * np.sqrt(self.T))

    def vega(self):
        return self.S0 * norm.pdf(self.d1()) * np.sqrt(self.T)

    def rho(self):
        if self.type_ == "call":
            return self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(self.d2())
        else:
            return -self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-self.d2())

    def theta(self):
        if self.type_ == "call":
            return -((self.S0 * norm.pdf(self.d1()) * self.sigma) / (2 * np.sqrt(self.T))) - self.r * self.K * np.exp(
                -self.r * self.T) * norm.cdf(self.d2())
        else:
            return -((self.S0 * norm.pdf(-self.d1()) * self.sigma) / (2 * np.sqrt(self.T))) + self.r * self.K * np.exp(
                -self.r * self.T) * norm.cdf(-self.d2())






class barrier_option:
    # down and out call

    def __init__(self, S0=1, K=1, T=1, r=0.05, sigma=0.2, B=0.5):
        """
            :param S0: Initial spot price
            :param K: Strike price
            :param T: Time to expiration
            :param r: risk-free rate
            :param sigma: Volatility
            :param B: Barrier price
        """
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.B = B

        self.ex = (1 - 2 * r / sigma ** 2)
        self.ex2 = 4 * r / sigma ** 3


    def price(self):

        if self.S0 <= self.B:
            return 0

        V = vanilla_option(self.S0, self.K, self.T, self.r, self.sigma).price() \
            - (self.S0 / self.B) ** self.ex * vanilla_option(self.B**2/self.S0,self.K,self.T,self.r,self.sigma).price()
        return V



    def delta(self):
        delta_ = vanilla_option(self.S0, self.K, self.T, self.r, self.sigma).delta() \
        - self.ex (self.S0/self.B)**(self.ex-1) * vanilla_option(self.B**2/self.S0,self.K,self.T,self.r,self.sigma).price()/self.B\
+ (self.S0/self.B)**(self.ex-2) * vanilla_option(self.B**2/self.S0,self.K,self.T,self.r,self.sigma).delta()
        return delta_

    def gamma(self):
        gamma = vanilla_option(self.S0, self.K, self.T, self.r, self.sigma).gamma() \
        - self.ex * (self.ex-1)(self.S0/self.B) ** (self.ex-2) * vanilla_option(self.B**2/self.S0,self.K,self.T,self.r,self.sigma).price()/self.B**2\
        + 2 * (self.ex-1) * (self.S0/self.B) ** (self.ex-3) * vanilla_option(self.B**2/self.S0,self.K,self.T,self.r,self.sigma).delta()/self.B\
        - (self.S0/self.B) ** (self.ex-4) * vanilla_option(self.B**2/self.S0,self.K,self.T,self.r,self.sigma).gamma()


        return gamma

    def vega(self):
        vega = vanilla_option(self.S0, self.K, self.T, self.r, self.sigma).vega() \
        - (self.S0/self.B0) ** self.ex * vanilla_option(self.B**2/self.S0,self.K,self.T,self.r,self.sigma).vega()
        - self.ex2 * np.log(self.S0/self.B) * (self.S0/self.B) ** self.ex * vanilla_option(self.B**2/self.S0,self.K,self.T,self.r,self.sigma).price()

        return vega


if __name__ == "__main__":
    S = 100
    K = 100
    B = 90
    T = 1
    r = 0.05
    sigma = 0.2
    H = 70
    # Initialize a Vanilla option
    vanilla_option_ins = vanilla_option(S0=S, K=K, T=1, r=0.05, sigma=0.2,type_='call', )
    # We can now use the methods associated with this object, for example, calculate the price
    price = vanilla_option_ins.price()
    print("The price of the vanilla call option is:", price)

    barrier_option = barrier_option(S0=S, K=K, T=1, r=0.05, sigma=0.2, B=B)
    price = barrier_option.price()
    print("The price of the down-and-out barrier call option is:", price)
