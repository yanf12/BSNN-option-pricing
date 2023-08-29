import numpy as np
from scipy.stats import norm
from scipy import  integrate
import scipy
from Func_rainbow import *
import plotly.graph_objects as go

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号 #有中文出现的情况，需要u'内容'
def two_asset_outperformance_exact_old(S1=61,S2 = 62, T = 0.5,sigma = np.array([[0.20,0.00],[0.225,0.1984313]]),r =0.05 ):
    '''
    This function returns the exact price of two-asset out-performance option
    :param S1: the initial price of the first stock
    :param S2: the initial price of the second stock
    :param T: time to expiry
    :param sigma: variance-covariance matrix
    :param r: risk free rate
    :return: price of the option
    '''
    a = np.matmul(sigma,sigma.transpose())

    d1 = (np.log(S1/S2) + 0.5*(a[0,0]-2*a[0,1] +a[1,1])* T)/np.sqrt((a[0,0]-2*a[0,1]+a[1,1])*T)

    d2 = (np.log(S2/S1) + 0.5 * (a[0, 0] - 2 * a[0, 1] + a[1, 1]) * T) / np.sqrt(
        (a[0, 0] - 2 * a[0, 1] + a[1, 1]) * T)

    V =- S1* norm.cdf(-d1)+S2* norm.cdf(d2)

    return V

def two_asset_outperformance_exact(S1=1,S2 = 1, T = 1,sigma = np.array([0.20,0.30]),r =0.05 ,rho =0.75,K = 0):
    '''
    This function returns the exact price of two-asset out-performance option
    :param S1: the initial price of the first stock
    :param S2: the initial price of the second stock
    :param T: time to expiry
    :param sigma: variance-covariance matrix
    :param r: risk free rate
    :return: price of the option
    '''


    a=np.array([[sigma[0]**2,sigma[0]*sigma[1]*rho],[sigma[0]*sigma[1]*rho,sigma[1]**2]])

    d1 = (np.log(S1/S2) + 0.5*(a[0,0]-2*a[0,1] +a[1,1])* T)/np.sqrt((a[0,0]-2*a[0,1]+a[1,1])*T)

    d2 = (np.log(S2/S1) + 0.5 * (a[0, 0] - 2 * a[0, 1] + a[1, 1]) * T) / np.sqrt(
        (a[0, 0] - 2 * a[0, 1] + a[1, 1]) * T)

    V =- S1* norm.cdf(-d1)+S2* norm.cdf(d2)

    return V
def two_asset_better_of_exact_old_version(S1=61,S2 = 62, T = 0.5,sigma = np.array([[0.20,0.00],[0.225,0.1984313]]),rho =0.75,r =0.05 ):
    '''
    This function returns the exact price of two-asset better-of option, 老版本用的是chelosky decomposition
    :param S1: the initial price of the first stock
    :param S2: the initial price of the second stock
    :param T: time to expiry
    :param sigma: chelosky decomposed matrix
    :param r: risk free rate
    :return: price of the option
    '''
    a = np.matmul(sigma,sigma.transpose()) #chelosky decomposition 再重新乘起来可以得到var-cov矩阵

    #a=np.array([[sigma[0]**2,sigma[0]*sigma[1]*rho],[sigma[0]*sigma[1]*rho,sigma[1]**2]])

    d1 = (np.log(S1/S2) + 0.5*(a[0,0]-2*a[0,1] +a[1,1])* T)/np.sqrt((a[0,0]-2*a[0,1]+a[1,1])*T)

    d2 = (np.log(S2/S1) + 0.5 * (a[0, 0] - 2 * a[0, 1] + a[1, 1]) * T) / np.sqrt(
        (a[0, 0] - 2 * a[0, 1] + a[1, 1]) * T)

    V =S1* norm.cdf(d1)+S2* norm.cdf(d2)

    return V



def two_asset_better_of_exact(S1=100,S2 = 100, T = 1,sigma = np.array([0.20,0.30]),rho =0.75,r =0.05,K=100 ):
    '''
    This function returns the exact price of two-asset out-performance option
    :param S1: the initial price of the first stock
    :param S2: the initial price of the second stock
    :param T: time to expiry
    :param sigma: variance  matrix
    :param r: risk free rate
    :return: price of the option
    '''
    #a = np.matmul(sigma,sigma.transpose())

    if S1 ==S2 == 0:
        return 0
    if S1 ==0:
        S1 += 0.000001 #取极限
    if S2 ==0:
        S2+= 0.0000001

    if rho ==1:
        rho = 1-0.000001

    a=np.array([[sigma[0]**2,sigma[0]*sigma[1]*rho],[sigma[0]*sigma[1]*rho,sigma[1]**2]])
    d1 = (np.log(S1/S2) + 0.5*(a[0,0]-2*a[0,1] +a[1,1])* T)/np.sqrt((a[0,0]-2*a[0,1]+a[1,1])*T)
    d2 = (np.log(S2/S1) + 0.5 * (a[0, 0] - 2 * a[0, 1] + a[1, 1]) * T) / np.sqrt(
        (a[0, 0] - 2 * a[0, 1] + a[1, 1]) * T)
    V =S1* norm.cdf(d1)+S2* norm.cdf(d2)

    if T ==0:
        return max(S1,S2)

    return V


def two_asset_better_of_exact_using_decomposed_matrix(S1=61,S2 = 62, T = 0.5,sigma =  np.array([[0.20,0.00],[0.225,0.1984313]]),rho =0.75,r =0.05 ):
    '''
    This function returns the exact price of two-asset out-performance option
    :param S1: the initial price of the first stock
    :param S2: the initial price of the second stock
    :param T: time to expiry
    :param sigma: variance  matrix
    :param r: risk free rate
    :return: price of the option
    '''
    a = np.matmul(sigma,sigma.transpose())
    #a=np.array([[sigma[0]**2,sigma[0]*sigma[1]*rho],[sigma[0]*sigma[1]*rho,sigma[1]**2]])
    d1 = (np.log(S1/S2) + 0.5*(a[0,0]-2*a[0,1] +a[1,1])* T)/np.sqrt((a[0,0]-2*a[0,1]+a[1,1])*T)
    d2 = (np.log(S2/S1) + 0.5 * (a[0, 0] - 2 * a[0, 1] + a[1, 1]) * T) / np.sqrt(
        (a[0, 0] - 2 * a[0, 1] + a[1, 1]) * T)
    V =S1* norm.cdf(d1)+S2* norm.cdf(d2)
    return V



def two_asset_European_correlation_call_exact(S_0,sigma,r,rho,T,K):
    '''
    :param S_0: initial price arrray
    :param sigma: volatility array (representation 2)
    :param r: risk free rate
    :param rho: correlation
    :param T: time to expiry
    :param K: stike price array
    :return: exact price of eu correlation option price
    '''
    if T ==0:
        if  S_0[0]>K[0]:
            return max(S_0[1]-K[1],0)
        else:
            return 0
    def bivariate_normal_pdf(x,y,rho):
        return (1/(2*np.pi*np.sqrt(1-np.power(rho,2)))*np.exp(-1/(2*(1-np.power(rho,2)))*(np.power(x,2)-2*rho*x*y+np.power(y,2))))

    y_1 = (np.log(S_0[0]/K[0])+T*(r-np.power(sigma[0],2)/2))/(sigma[0]*np.sqrt(T))
    y_2 = (np.log(S_0[1] / K[1]) + T * (r - np.power(sigma[1], 2) / 2)) / (sigma[1] * np.sqrt(T))
    M_1 = integrate.nquad(bivariate_normal_pdf,[[-np.inf,y_1+rho*sigma[1]*np.sqrt(T)],[-np.inf,y_2+sigma[1]*np.sqrt(T)]],
                          args=[rho])
    M_2 = integrate.nquad(bivariate_normal_pdf,[[-np.inf,y_1],[-np.inf,y_2]],
                          args=[rho])
    option_price = S_0[1]*M_1[0] -K[1]*np.exp(-r*T)*M_2[0]
    return option_price







def two_asset_outperformance_mc (S1=61,S2 = 62, T = 0.5,sigma = np.array([[0.20,0.00],[0.225,0.1984313]]),r = 0.05,n = 10000 ):
    '''
    This function uses monete carlo method to calculate the price of two-asset out-performance option
    :param S1: the initial price of the first stock
    :param S2: the initial price of the second stock
    :param T: time to expiry
    :param sigma: variance-covariance matrix
    :param r: risk free rate
    :param n: number of simulations
    :return: price of the option
    '''
    W_1_T = np.random.normal(0, np.sqrt(T), n)
    W_2_T = np.random.normal(0, np.sqrt(T), n)
    S_1_T = S1 * np.exp((r - (sigma[0,0]**2 + sigma[0,1]**2)/2 )*T+sigma[0,0]*W_1_T+sigma[0,1]*W_2_T)
    S_2_T = S2 * np.exp((r - (sigma[1,0]**2 + sigma[1,1]**2)/2 )*T+sigma[1,0]*W_1_T+sigma[1,1]*W_2_T)

    delta = S_2_T - S_1_T
    Y= np.array(list(map(lambda x: max(x,0),delta))).mean()
    V  = Y* np.exp(-r*T)

    return V








def two_asset_better_of_mc_old (S1=61,S2 = 62, T = 0.5,sigma = np.array([[0.20,0.00],[0.225,0.1984313]]),r = 0.05,n = 10000 ):
    '''
    This function uses monete carlo method to calculate the price of two-asset out-performance option
    :param S1: the initial price of the first stock
    :param S2: the initial price of the second stock
    :param T: time to expiry
    :param sigma: 老版本，用的是chelosky方法下对var-cov 的decomposition
    :param r: risk free rate
    :param n: number of simulations
    :return: price of the option
    '''

    W_1_T = np.random.normal(0, np.sqrt(T), n)
    W_2_T = np.random.normal(0, np.sqrt(T), n)
    S_1_T = S1 * np.exp((r - (sigma[0,0]**2 + sigma[0,1]**2)/2 )*T+sigma[0,0]*W_1_T+sigma[0,1]*W_2_T)
    S_2_T = S2 * np.exp((r - (sigma[1,0]**2 + sigma[1,1]**2)/2 )*T+sigma[1,0]*W_1_T+sigma[1,1]*W_2_T)

    Y_array = np.array(list(map(lambda x,y: max(x,y),S_1_T,S_2_T)))* np.exp(-r*T)
    V  = Y_array.mean()
    std = Y_array.std()/np.sqrt(n)

    return V,std


def two_asset_better_of_mc (S1=61,S2 = 62, T = 1.0,sigma = np.array([0.20,0.30]),rho=0.75,r = 0.05,n = 100000 ):
    '''
    This function uses monete carlo method to calculate the price of two-asset out-performance option
    :param S1: the initial price of the first stock
    :param S2: the initial price of the second stock
    :param T: time to expiry
    :param sigma: variance-covariance matrix
    :param r: risk free rate
    :param n: number of simulations
    :return: price of the option
    '''
    W_1_T = np.random.normal(0, np.sqrt(T), n)
    W_2_T = np.random.normal(0, np.sqrt(T), n)

    S_1_T = S1 * np.exp((r - (sigma[0]**2)/2)*T+sigma[0]*W_1_T)
    S_2_T = S2 * np.exp((r - (sigma[1]**2)/2)*T + sigma[1]*W_1_T *rho+sigma[1]*(np.sqrt(1-rho**2))*W_2_T)

    Y_array = np.array(list(map(lambda x,y: max(x,y),S_1_T,S_2_T)))* np.exp(-r*T)
    V  = Y_array.mean()
    std = Y_array.std()/np.sqrt(n)

    return V,std, S_1_T,S_2_T



def two_asset_best_of_call_mc (S1=1,S2 = 1, T = 1.0,sigma = np.array([0.20,0.30]),rho=0.75,r = 0.05,n = 10000,K = 1.05 ):
    '''
    This function uses monete carlo method to calculate the price of two-asset out-performance option
    :param S1: the initial price of the first stock
    :param S2: the initial price of the second stock
    :param T: time to expiry
    :param sigma: variance-covariance matrix
    :param r: risk free rate
    :param n: number of simulations
    :return: price of the option
    '''
    W_1_T = np.random.normal(0, np.sqrt(T), n)
    W_2_T = np.random.normal(0, np.sqrt(T), n)

    S_1_T = S1 * np.exp((r - (sigma[0]**2)/2)*T+sigma[0]*W_1_T)
    S_2_T = S2 * np.exp((r - (sigma[1]**2)/2)*T + sigma[1]*W_1_T *rho+sigma[1]*(np.sqrt(1-rho**2))*W_2_T)

    Y_array = np.array(list(map(lambda x,y: max(max(x,y)-K,0),S_1_T,S_2_T)))* np.exp(-r*T)
    V  = Y_array.mean()
    std = Y_array.std()/np.sqrt(n)

    return V,std
def two_asset_better_of_mc_multi_norm (S1=61,S2 = 62, T = 0.5,sigma = np.array([0.20,0.30]),rho=0.75,r = 0.05,n = 100 ):
    '''
    This function uses monete carlo method to calculate the price of two-asset out-performance option
    :param S1: the initial price of the first stock
    :param S2: the initial price of the second stock
    :param T: time to expiry
    :param sigma: variance-covariance matrix
    :param r: risk free rate
    :param n: number of simulations
    :return: price of the option
    '''

    cov_mat = [[(sigma[0]**2)*T,rho*sigma[0]*sigma[1]*T],[rho*sigma[0]*sigma[1]*T,T*sigma[1]**2]]

    X_1,X_2 = np.random.multivariate_normal([(r-0.5*sigma[0]**2)*T,(r-0.5*sigma[1]**2)*T],cov_mat,n).T

    S_1_T = S1 * np.exp(X_1)
    S_2_T = S2 * np.exp(X_2)

    Y_array = np.array(list(map(lambda x,y: max(x,y),S_1_T,S_2_T)))* np.exp(-r*T)
    V  = Y_array.mean()
    std = Y_array.std()/np.sqrt(n)

    return V,std




def two_asset_better_of_mc_with_antithetic_variates (S1=61,S2 = 62, T = 0.5,sigma = np.array([[0.20,0.00],[0.225,0.1984313]]),r = 0.05,n = 10000 ):
    '''
    This function uses monete carlo method to calculate the price of two-asset out-performance option
    :param S1: the initial price of the first stock
    :param S2: the initial price of the second stock
    :param T: time to expiry
    :param sigma: variance-covariance matrix
    :param r: risk free rate
    :param n: number of simulations
    :return: price of the option
    '''
    W_1_T = np.random.normal(0, np.sqrt(T), n)
    W_2_T = np.random.normal(0, np.sqrt(T), n)
    W_1_T_anti = -1* W_1_T
    W_2_T_anti = -1* W_2_T

    S_1_T = S1 * np.exp((r - (sigma[0,0]**2 + sigma[0,1]**2)/2 )*T+sigma[0,0]*W_1_T+sigma[0,1]*W_2_T)

    S_1_T_prime = S1*np.exp((r - (sigma[0,0]**2 + sigma[0,1]**2)/2 )*T+sigma[0,0]*W_1_T_anti+sigma[0,1]*W_2_T_anti)

    S_2_T = S2 * np.exp((r - (sigma[1,0]**2 + sigma[1,1]**2)/2 )*T+sigma[1,0]*W_1_T+sigma[1,1]*W_2_T)

    S_2_T_prime = S2 * np.exp((r - (sigma[1,0]**2 + sigma[1,1]**2)/2 )*T+sigma[1,0]*W_1_T_anti+sigma[1,1]*W_2_T_anti)


    Y= np.array(list(map(lambda x,y: max(x,y),S_1_T,S_2_T)))
    Y_prime = np.array(list(map(lambda x,y: max(x,y),S_1_T_prime,S_2_T_prime)))


    V_av= (0.5*(Y+Y_prime)).mean()
    V_av_std = (0.5*(Y+Y_prime)).std()/np.sqrt(n)
    return V_av*np.exp(-r*T),V_av_std





def two_asset_best_of_call_mc_with_antithetic_variates (S1=1,S2 = 1, T = 1.0,sigma = np.array([0.20,0.30]),rho=0.75,r = 0.05,n = 10000,K = 1.05 ):
    '''
    This function uses monete carlo method to calculate the price of two-asset out-performance option
    :param S1: the initial price of the first stock
    :param S2: the initial price of the second stock
    :param T: time to expiry
    :param sigma: variance-covariance matrix
    :param r: risk free rate
    :param n: number of simulations
    :return: price of the option
    '''
    W_1_T = np.random.normal(0, np.sqrt(T), n)
    W_2_T = np.random.normal(0, np.sqrt(T), n)
    W_1_T_anti = -1* W_1_T
    W_2_T_anti = -1* W_2_T
    cov_mat = [[(sigma[0]**2)*T,rho*sigma[0]*sigma[1]*T],[rho*sigma[0]*sigma[1]*T,T*sigma[1]**2]]
    sigma =scipy.linalg.cholesky(cov_mat, lower=True)
    #sigma = np.array([[0.20, 0.00], [0.225, 0.1984313]])
    S_1_T = S1 * np.exp((r - (sigma[0,0]**2 + sigma[0,1]**2)/2 )*T+sigma[0,0]*W_1_T+sigma[0,1]*W_2_T)
    S_1_T_prime = S1*np.exp((r - (sigma[0,0]**2 + sigma[0,1]**2)/2 )*T+sigma[0,0]*W_1_T_anti+sigma[0,1]*W_2_T_anti)
    S_2_T = S2 * np.exp((r - (sigma[1,0]**2 + sigma[1,1]**2)/2 )*T+sigma[1,0]*W_1_T+sigma[1,1]*W_2_T)
    S_2_T_prime = S2 * np.exp((r - (sigma[1,0]**2 + sigma[1,1]**2)/2 )*T+sigma[1,0]*W_1_T_anti+sigma[1,1]*W_2_T_anti)
    Y= np.array(list(map(lambda x,y: max(max(x,y)-K,0),S_1_T,S_2_T)))
    Y_prime = np.array(list(map(lambda x,y: max(max(x,y)-K,0),S_1_T_prime,S_2_T_prime)))
    V_av= (0.5*(Y+Y_prime)).mean()
    V_av_std = (0.5*(Y+Y_prime)).std()/np.sqrt(n)
    return V_av*np.exp(-r*T),V_av_std



def two_asset_best_of_call_mc_with_price_control_variate (S1=1,S2 = 1, T = 1.0,sigma = np.array([0.20,0.30]),rho=0.75,r = 0.05,n = 10000,K = 1.05 ,c=0.8):
    '''
    This function uses monete carlo method to calculate the price of two-asset out-performance option
    :param S1: the initial price of the first stock
    :param S2: the initial price of the second stock
    :param T: time to expiry
    :param sigma: variance-covariance matrix
    :param r: risk free rate
    :param n: number of simulations
    :return: price of the option
    '''
    W_1_T = np.random.normal(0, np.sqrt(T), n)
    W_2_T = np.random.normal(0, np.sqrt(T), n)
    W_1_T_anti = -1* W_1_T
    W_2_T_anti = -1* W_2_T
    cov_mat = [[(sigma[0]**2)*T,rho*sigma[0]*sigma[1]*T],[rho*sigma[0]*sigma[1]*T,T*sigma[1]**2]]
    sigma =scipy.linalg.cholesky(cov_mat, lower=True)
    #sigma = np.array([[0.20, 0.00], [0.225, 0.1984313]])
    S_1_T = S1 * np.exp((r - (sigma[0,0]**2 + sigma[0,1]**2)/2 )*T+sigma[0,0]*W_1_T+sigma[0,1]*W_2_T)
    #S_1_T_prime = S1*np.exp((r - (sigma[0,0]**2 + sigma[0,1]**2)/2 )*T+sigma[0,0]*W_1_T_anti+sigma[0,1]*W_2_T_anti)
    S_2_T = S2 * np.exp((r - (sigma[1,0]**2 + sigma[1,1]**2)/2 )*T+sigma[1,0]*W_1_T+sigma[1,1]*W_2_T)
    #S_2_T_prime = S2 * np.exp((r - (sigma[1,0]**2 + sigma[1,1]**2)/2 )*T+sigma[1,0]*W_1_T_anti+sigma[1,1]*W_2_T_anti)



    Y= np.array(list(map(lambda x,y: max(max(x,y)-K,0)-c*(x-S1*np.exp(r*T)),S_1_T,S_2_T)))

    V_cv= Y.mean()
    V_cv_std = Y.std()/np.sqrt(n)
    return V_cv*np.exp(-r*T),V_cv_std



def two_asset_best_of_call_mc_with_price_control_variate_two_direction (S1=1,S2 = 1, T = 1.0,sigma = np.array([0.20,0.30]),rho=0.75,r = 0.05,n = 10000,K = 1.05,c=0.2,b=0.5 ):
    '''
    This function uses monete carlo method to calculate the price of two-asset out-performance option
    :param S1: the initial price of the first stock
    :param S2: the initial price of the second stock
    :param T: time to expiry
    :param sigma: variance-covariance matrix
    :param r: risk free rate
    :param n: number of simulations
    :return: price of the option
    '''
    W_1_T = np.random.normal(0, np.sqrt(T), n)
    W_2_T = np.random.normal(0, np.sqrt(T), n)
    W_1_T_anti = -1* W_1_T
    W_2_T_anti = -1* W_2_T
    cov_mat = [[(sigma[0]**2)*T,rho*sigma[0]*sigma[1]*T],[rho*sigma[0]*sigma[1]*T,T*sigma[1]**2]]
    sigma =scipy.linalg.cholesky(cov_mat, lower=True)
    #sigma = np.array([[0.20, 0.00], [0.225, 0.1984313]])
    S_1_T = S1 * np.exp((r - (sigma[0,0]**2 + sigma[0,1]**2)/2 )*T+sigma[0,0]*W_1_T+sigma[0,1]*W_2_T)
    #S_1_T_prime = S1*np.exp((r - (sigma[0,0]**2 + sigma[0,1]**2)/2 )*T+sigma[0,0]*W_1_T_anti+sigma[0,1]*W_2_T_anti)
    S_2_T = S2 * np.exp((r - (sigma[1,0]**2 + sigma[1,1]**2)/2 )*T+sigma[1,0]*W_1_T+sigma[1,1]*W_2_T)
    #S_2_T_prime = S2 * np.exp((r - (sigma[1,0]**2 + sigma[1,1]**2)/2 )*T+sigma[1,0]*W_1_T_anti+sigma[1,1]*W_2_T_anti)


    Y_original = np.array(list(map(lambda x,y: max(max(x,y)-K,0),S_1_T,S_2_T)))
    c_optimal  =np.cov(Y_original,S_1_T)[0,0]/S_1_T.var()
    b_optimal = np.cov(Y_original, S_2_T) [0,0]/ S_2_T.var()
    # print(c_optimal)
    # print(b_optimal)

    # c = 0.5
    # b = 0.5


    Y= np.array(list(map(lambda x,y: max(max(x,y)-K,0)-c*(x-S1*np.exp(r*T))-b*(y-S2*np.exp(r*T)),S_1_T,S_2_T)))

    V_cv= Y.mean()
    V_cv_std = Y.std()/np.sqrt(n)
    return V_cv*np.exp(-r*T),V_cv_std




def two_asset_best_of_call_mc_with_price_control_variate_zero_strike (S1=1,S2 = 1, T = 1.0,sigma = np.array([0.20,0.30]),rho=0.75,r = 0.05,n = 10000,K = 1.05,c=0.75 ):
    '''
    This function uses monete carlo method to calculate the price of two-asset out-performance option
    :param S1: the initial price of the first stock
    :param S2: the initial price of the second stock
    :param T: time to expiry
    :param sigma: variance-covariance matrix
    :param r: risk free rate
    :param n: number of simulations
    :return: price of the option
    '''
    W_1_T = np.random.normal(0, np.sqrt(T), n)
    W_2_T = np.random.normal(0, np.sqrt(T), n)
    W_1_T_anti = -1* W_1_T
    W_2_T_anti = -1* W_2_T
    cov_mat = [[(sigma[0]**2)*T,rho*sigma[0]*sigma[1]*T],[rho*sigma[0]*sigma[1]*T,T*sigma[1]**2]]
    sigma =scipy.linalg.cholesky(cov_mat, lower=True)
    #sigma = np.array([[0.20, 0.00], [0.225, 0.1984313]])
    S_1_T = S1 * np.exp((r - (sigma[0,0]**2 + sigma[0,1]**2)/2 )*T+sigma[0,0]*W_1_T+sigma[0,1]*W_2_T)
    #S_1_T_prime = S1*np.exp((r - (sigma[0,0]**2 + sigma[0,1]**2)/2 )*T+sigma[0,0]*W_1_T_anti+sigma[0,1]*W_2_T_anti)
    S_2_T = S2 * np.exp((r - (sigma[1,0]**2 + sigma[1,1]**2)/2 )*T+sigma[1,0]*W_1_T+sigma[1,1]*W_2_T)
    #S_2_T_prime = S2 * np.exp((r - (sigma[1,0]**2 + sigma[1,1]**2)/2 )*T+sigma[1,0]*W_1_T_anti+sigma[1,1]*W_2_T_anti)


    Y_original = np.array(list(map(lambda x,y: max(max(x,y)-K,0),S_1_T,S_2_T)))




    Y= np.array(list(map(lambda x,y: max(max(x,y)-K,0)-c*(max(x,y)-two_asset_better_of_exact(S1,S2,T,sigma,rho,r)),S_1_T,S_2_T)))

    V_cv= Y.mean()
    V_cv_std = Y.std()/np.sqrt(n)
    return V_cv*np.exp(-r*T),V_cv_std



def two_asset_European_correlation_call_mc (S1=52,S2 = 65,K1 = 50,K2 = 70, T = 0.5,sigma = np.array([[0.20,0.00],[0.225,0.1984313]]),r = 0.10,n = 1000000 ):
    '''
    This function uses monete carlo method to calculate the price of two-asset European Correlation Call Option
    :param S1: the initial price of the first stock
    :param S2: the initial price of the second stock
    :param K1: the strike price of the first stock
    :param K2: the strike price of the second stock
    :param T: time to expiry
    :param sigma: sigma matrix in representation 1
    :param r: risk free rate
    :param n: number of simulations
    :return: price of the option
    '''
    W_1_T = np.random.normal(0, np.sqrt(T), n)
    W_2_T = np.random.normal(0, np.sqrt(T), n)
    K1_array = K1*np.ones(n)
    K2_array = K2*np.ones(n)
    S_1_T = S1 * np.exp((r - (sigma[0,0]**2 + sigma[0,1]**2)/2 )*T+sigma[0,0]*W_1_T+sigma[0,1]*W_2_T)
    S_2_T = S2 * np.exp((r - (sigma[1,0]**2 + sigma[1,1]**2)/2 )*T+sigma[1,0]*W_1_T+sigma[1,1]*W_2_T)

    def correlation_call_payoff (S_1_T,S_2_T,K1,K2):
        if S_1_T >K1:
            return max(S_2_T-K2,0)
        else:
            return 0

    Y= np.array(list(map(correlation_call_payoff,S_1_T,S_2_T,K1_array,K2_array))).mean()
    V  = Y* np.exp(-r* T)

    return V

if __name__ == "__main__":
    # import pandas as pd
    # V, std, S_1_T, S_2_T = two_asset_better_of_mc ()
    # res_ser_1 = pd.Series(S_1_T)
    # ret_ser_1 = res_ser_1/100
    # ret_ser_1 = ret_ser_1.dropna()
    # log_ret_ser_1 = ret_ser_1.apply(lambda x: np.log(x))
    # print(log_ret_ser_1.std())
    #
    # res_ser_2 = pd.Series(S_2_T)
    # ret_ser_2 = res_ser_2/100
    # ret_ser_2 = ret_ser_2.dropna()
    # log_ret_ser_2 = ret_ser_2.apply(lambda x: np.log(x))
    # print(log_ret_ser_2.std())
    # print(np.corrcoef(res_ser_1,res_ser_2))
    c_list = np.arange(0, 1, 0.1)
    std_list = []

    for c in c_list:
        v, std = two_asset_best_of_call_mc_with_price_control_variate(c=c)

        std_list.append(std)

    plt.plot(c_list, std_list, "-o")
    plt.grid()
    plt.savefig("parameteroptimization.png")
    plt.show()
