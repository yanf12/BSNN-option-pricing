import numpy as np
import time
start_time = time.time()


def get_basket_price_mc(d = 10,S_0=100,K=100,r=0.05,mu = [0.05, 0.06, 0.07, 0.05, 0.06, 0.07, 0.05, 0.06, 0.07, 0.06],sigma =
                        [.10, .11, .12, .13, .14, .14, .13, .12, .11, .10],
                        rho = 0.1, alpha = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],N = 500000,T = 1):

    Tmat  = np.ones((d, N))*T

    """Implementation of Approximation"""
    "Construct Covariance Matrix and Decomposition"
    # Create Matrix Eta which is the matrix of Covariances
    Eta = np.eye(d) + rho * (np.ones(d) - np.eye(d))
    # Use the Cholesky Decomposition to derive L, which has the value that Eta = L*L'
    L = np.linalg.cholesky(Eta)
    "Construct Brownian Motion step"
    # Generate Delta_W trough sqrt(T)*L*Z with Z~N(0,1), Delta_W is a matrix of dim = (d,N)
    Delta_W = np.matmul(np.sqrt(T) * L, np.random.normal(0, 1, (d, N)))
    "Construct Price Procsses"
    # Generate the price processes (mu = r), dimension (d,N)
    S = S_0*np.exp((np.diag(mu) - 1/2 * np.diag(sigma)**2)@ Tmat + np.diag(sigma) @ Delta_W)
    "Construct Payoff"
    # Construct Payoff (Sum alpha_i*S_i - K)^+ for  every simulation step of Monte Carlo, dim = (N,1)
    Payoff_ = np.matmul(alpha, S)-K

    Payoff = np.sum(Payoff_[(Payoff_ > 0)])/N  # other entries would be 0 anyway, therefore its okay to use Payoff[(Payoff>0)]
    #Discount Payoff to get the fair price of the Option
    std = np.std(Payoff_[(Payoff_> 0)])*np.exp(- r * T)/np.sqrt(N)
    V = np.exp(- r * T) * Payoff
    
    return V,std


def get_basket_price_mc_flex_S0(d=10, S_0=np.array([1,1,1,1,1,1,1,1,1,1]), K=100, r=0.02, mu=[0.05, 0.06, 0.07, 0.05, 0.06, 0.07, 0.05, 0.06, 0.07, 0.06],
                        sigma=
                        [.10, .11, .12, .13, .14, .14, .13, .12, .11, .10],
                        rho=0.1, alpha=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], N=500000, T=1):
    Tmat = np.ones((d, N)) * T

    S_0 = np.expand_dims(S_0, axis=0)

    """Implementation of Approximation"""
    "Construct Covariance Matrix and Decomposition"
    # Create Matrix Eta which is the matrix of Covariances
    Eta = np.eye(d) + rho * (np.ones(d) - np.eye(d))
    # Use the Cholesky Decomposition to derive L, which has the value that Eta = L*L'
    L = np.linalg.cholesky(Eta)
    "Construct Brownian Motion step"
    # Generate Delta_W trough sqrt(T)*L*Z with Z~N(0,1), Delta_W is a matrix of dim = (d,N)
    Delta_W = np.matmul(np.sqrt(T) * L, np.random.normal(0, 1, (d, N)))
    "Construct Price Procsses"
    # Generate the price processes (mu = r), dimension (d,N)
    S = S_0.transpose() * np.exp((np.diag(mu) - 1 / 2 * np.diag(sigma) ** 2) @ Tmat + np.diag(sigma) @ Delta_W)
    # print(np.exp((np.diag(mu) - 1 / 2 * np.diag(sigma) ** 2) @ Tmat + np.diag(sigma) @ Delta_W).shape)
    "Construct Payoff"
    # Construct Payoff (Sum alpha_i*S_i - K)^+ for  every simulation step of Monte Carlo, dim = (N,1)
    Payoff = np.matmul(alpha, S) - K
    Payoff = np.sum(
        Payoff[(Payoff > 0)]) / N  # other entries would be 0 anyway, therefore its okay to use Payoff[(Payoff>0)]
    # Discount Payoff to get the fair price of the Option
    V = np.exp(- r * T) * Payoff

    return V




if __name__=="__main__":
    V,std = get_basket_price_mc()
    V2 =get_basket_price_mc_flex_S0()
    V3 = get_basket_price_mc_flex_S0(mu = [0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02,0.02])

    "Output"
    #print('In a market consisting of ' + str(d) + ' stocks, \nwe evaluated a classical basket option, \nusing ' + str(N) + ' MC samples.')
    print('The mc price of the Option is ' + str(round(V, 6)) + '\nusing the exact solution for the SDE')
    print("--- %s seconds ---" % np.round((time.time() - start_time), 2))


    print('The mc V2 price of the Option is ' + str(round(V2, 6)) + '\nusing the exact solution for the SDE')
    print("--- %s seconds ---" % np.round((time.time() - start_time), 2))
    print(V3)


