import numpy as np
import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from plotting import newfig, savefig
from torch.optim.lr_scheduler import StepLR
from scipy.special import comb
from scipy.stats import norm
from common_tools import neural_networks
from IPython.display import display, clear_output
#from Func_barrier import CBBC
from common_tools import plotplot
from barrier_new import barrier_option
plt.rcParams['font.family'] = 'Times New Roman'
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# 1D black sholes
from common_tools import neural_networks


def theoretical_barrier_option(S0=100, K=110, B = 150,T=1, r=0.05, sigma=0.4, type_='call'):
    '''
    :param S0: 股票当前价格
    :param K: 行权价格
    :param T: 到期时间（年）
    :param r: 无风险利率， 如 r = 0.05
    :param sigma: 波动率， 如 sigma = 0.20
    :param type_:  call or put
    :return: 期权价格
    '''
    barrier_object = barrier_option(S0=S0, K=K, B = B,T=T, r=r, sigma=sigma)

    return barrier_object.price()




class FBSNN(nn.Module):  # Forward-Backward Stochastic Neural Network
    def __init__(self, epochs,r,K,sigma ,Xi, T, M, N, D,B, learning_rate, gbm_scheme ,lambda_1,lambda_2,lambda_3,lambda_4,lambda_5,out_of_sample_input,out_of_sample_exact,save_N):
        super().__init__()
        self.r = r  # interest rate
        self.sigma = sigma # volatility
        self.K = K  # strike price
        self.Xi = Xi  # initial point
        self.T = T  # terminal time

        self.M = M  # number of trajectories
        self.N = N  # number of time snapshots
        self.D = D  # number of dimensions
        self.B = B # barrier
        self.fn_u = neural_networks.neural_net_specify_width(pathbatch=M, n_dim=D + 1, n_output=1, num_layers=3, width_list=[1024,512,256],activation=torch.nn.ReLU())

        self.optimizer = optim.Adam(self.fn_u.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[20000,40000,50000,70000,85000], gamma=0.5)
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=3000, eta_min=0.001)
        self.lambda_1 = lambda_1# terminal condition
        self.lambda_2 = lambda_2 # Dgtorch condition
        self.lambda_3 = lambda_3 # boundary condition 1 at X = 0
        self.lambda_4 = lambda_4 # bounday contition 2 at X = B
        self.lambda_5 = lambda_5 # weight for point at t = T, X = t specifically
        self.barrier_shift = 0.001


        self.gbm_scheme = gbm_scheme # 0:euler scheme for gbm #1: EXP scheme
        self.out_of_sample_input = out_of_sample_input
        self.out_of_sample_exact = out_of_sample_exact
        self.save_N = save_N # frequency of saving MSE and plot
        self.epochs = epochs


    def theoretical_vanilla_eu(self,S0=50, K=50, T=1, r=0.05, sigma=0.4, type_='call'):

        if T == 0:
            if type_ == "call":
                return max(S0 - K, 0)
            else:
                return max(K - S0, 0)
        # 求BSM模型下的欧式期权的理论定价
        d1 = ((np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T)) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if type_ == "call":
            c = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            return c
        elif type_ == "put":
            p = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
            return p





    def phi_torch(self, t, X, Y, DuDx,DuDt,D2uDx2 ):  # M x 1, M x D, M x 1, M x D

        res = DuDx*self.r*X+DuDt + 0.5*D2uDx2*X**2*self.sigma**2
        return  res # M x 1

    def g_torch(self, X,K,B):  # M x D

        row_max, _ = torch.max(X, dim=1)  # get maximum along each row
        option_price = torch.clamp(row_max - K, min=0)  # calculate option price
        #option_price = torch.where(row_max > B, torch.tensor([0.0]), option_price)  # set price to 0 if row_max > B
        return option_price.unsqueeze(1)  # M x 1



    def mu_torch(self, r,t, X, Y):  # 1x1, M x 1, M x D, M x 1, M x D
        return 0*torch.ones([self.M, self.D])  # M x D

    def sigma_torch(self, t, X, Y):  # M x 1, M x D, M x 1
        # print("sigma_torch")
        # print(X.shape)
        return self.sigma * torch.diag_embed(X)  # M x D x D

    def net_u_Du(self, t, X):  # M x 1, M x D

        inputs = torch.cat([t, X], dim=1)

        u = self.fn_u(inputs)



        DuDx = torch.autograd.grad(torch.sum(u), X, retain_graph=True,create_graph=True)[0]

        # print(DuDx.shape)

        #DuDt = torch.autograd.grad(torch.sum(u), t, retain_graph=True,create_graph=True)[0]
        D2uDx2 = 0
        DuDt = 0

        return u, DuDx,DuDt,D2uDx2 # M x 1, M x D, M x 1, M x D

    def Dg_torch(self, X):  # M x D
        return torch.autograd.grad(torch.sum(self.g_torch(X,self.K)), X, retain_graph=True)[0]  # M x D

    def fetch_minibatch(self):
        T = self.T
        M = self.M
        N = self.N
        D = self.D

        Dt = np.zeros((M, N + 1, 1))  # M x (N+1) x 1
        DW = np.zeros((M, N + 1, D))  # M x (N+1) x D

        dt = T / N
        Dt[:, 1:, :] = dt
        DW[:, 1:, :] = np.sqrt(dt) * np.random.normal(size=(M, N, D))

        t = np.cumsum(Dt, axis=1)  # M x (N+1) x 1
        W = np.cumsum(DW, axis=1)  # M x (N+1) x D

        return torch.from_numpy(t).float(), torch.from_numpy(W).float()

    def loss_function(self, t, W, Xi):  # M x (N+1) x 1, M x (N+1) x D, MxD
        loss = torch.zeros(1)
        X_buffer = []
        Y_buffer = []

        t0 = t[:, 0, :]  # M x 1
        W0 = W[:, 0, :]  # M x D
        # X0 = torch.tensor([np.linspace(0.5,1.5,self.M)]).transpose(-1,-2).float()  # M x D
        # X0 = torch.cat([Xi] * self.M)  # M x D
        X0 = Xi

        X0.requires_grad = True

        t0.requires_grad = True
        Y0, DuDx0,DuDt0,D2uDx20 = self.net_u_Du(t0, X0)  # M x 1, M x D

        X_buffer.append(X0)
        Y_buffer.append(Y0)
        total_weight = 0

        for n in range(0, self.N):
            t1 = t[:, n + 1, :]
            W1 = W[:, n + 1, :]

            #
            # print("t1-t0")
            # print((t1 - t0).shape)
            #
            # print("mu_torch:")
            # print(self.mu_torch(t0, X0, Y0, Z0).shape)
            # print("sigma_torch:")
            # print(self.sigma_torch(t0, X0, Y0).shape)
            # print("W1 - W0:")
            # print((W1 - W0).unsqueeze(-1).shape)
            # print(torch.matmul(self.sigma_torch(t0, X0, Y0), (W1 - W0).unsqueeze(-1)).squeeze(2).shape)

            if self.gbm_scheme ==0:
                X1 = X0 + self.r*X0*(t1-t0) + self.sigma * X0 * (W1 - W0) # Euler-M scheme
            elif self.gbm_scheme ==1:
                X1 = X0*torch.exp( (self.r-0.5*self.sigma**2)*(t1-t0) + self.sigma* (W1-W0))

            # print(X1.shape)

            t1.requires_grad = True
            T1 = torch.ones(t1.shape).float() * self.T
            T1.requres_grad = True
            Y1, DuDx1,DuDt1,D2uDx21 = self.net_u_Du(t1, X1)  # M x 1, M x D

            Y1_tilde = Y0 + self.r*Y0* (t1-t0) + DuDx0 * self.sigma*X0*(W1-W0)

            loss = loss + torch.sum((Y1 - Y1_tilde) ** 2)
            total_weight +=1

            t0 = t1
            W0 = W1
            X0 = X1
            Y0 = Y1
            DuDx0 = DuDx1 # not sure if this is correct
            DuDt0 = DuDt1
            D2uDx20 = D2uDx21

            # Y_at_X_0, _1, _2, _3 = self.net_u_Du(t1, X1 * 0)  # M x 1, M x D
            # loss = loss + self.lambda_3 * torch.sum((Y_at_X_0 - 0) ** 2)
            # total_weight += self.lambda_3


            X_B = torch.ones(size = X1.shape).float() * self.B
            X_B.requires_grad = True

            Y_B, DuDx_B, DuDt_B, D2uDx2_B = self.net_u_Du(t1, X_B)  # M x 1, M x D
            loss = loss + self.lambda_4* torch.sum((Y_B - torch.zeros(Y_B.shape)) ** 2)
            total_weight += self.lambda_4

            # X_B = torch.ones(size = X1.shape).float() * self.B
            # X_B.requires_grad = True
            #
            # Y_TB,_,_,_  = self.net_u_Du(T1,X_B)
            # loss = loss +self.lambda_5*torch.sum((Y_TB - torch.ones(Y_B.shape)*(self.B-self.K)) ** 2)

            X_buffer.append(X0)
            Y_buffer.append(Y0)

        loss = loss + self.lambda_1*torch.sum((Y1 - self.g_torch(X1,self.K,self.B)) ** 2)
        total_weight += self.lambda_1
        # loss = loss + self.lambda_2 * torch.sum((DuDx1 - self.Dg_torch(X1)) ** 2)
        # total_weight += self.lambda_2

        loss = loss/total_weight
        #loss = loss + torch.sum((Z1 - self.Dg_torch(X1)) ** 2)

        X = torch.stack(X_buffer, dim=1)  # M x N x D
        Y = torch.stack(Y_buffer, dim=1)  # M x N x 1

        return loss, X, Y, Y[0, 0, 0]

    def train(self):
        N_Iter =self.epochs
        loss_list = []
        error_list = []

        start_time = time.time()
        t = np.linspace(0, 1, 10)
        S = np.linspace(self.B, 200, 10)
        test_sample_list = []

        t_mesh, S_mesh = np.meshgrid(t, S)

        mse_list = []
        mae_list = []
        for it in range(N_Iter):


            t_batch, W_batch = self.fetch_minibatch()  # M x (N+1) x 1, M x (N+1) x D
            loss, X_pred, Y_pred, Y0_pred = self.loss_function(t_batch, W_batch, self.Xi)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            loss_list.append(loss.detach().numpy()[0])

            test_sample_list.append(self.fn_u(self.out_of_sample_input).detach().numpy()[0])



            # Print
            if it % self.save_N == 0:
                clear_output(wait=True)
                elapsed = time.time() - start_time
                print('It: %d, Time: %.2f, Loss: %.3e, Y0: %.3f' %
                      (it, elapsed, loss, Y0_pred))
                start_time = time.time()
                # plt.plot(np.log10(range(len(loss_list))),np.log10(loss_list))
                # plt.show()
                #
                # plt.plot(test_sample_list[50:])
                # plt.plot(np.ones(len(test_sample_list))[50:] * test_sample_exact, label='test sample exact price')
                # plt.show()

                t = np.linspace(0, 1, 10)
                S = np.linspace(self.B, 200, 10)

                t_mesh, S_mesh = np.meshgrid(t, S)

                NN_price_surface = np.zeros([10, 10])
                Exact_price_surface = np.zeros([10, 10])
                for i in range(10):
                    for j in range(10):
                        NN_price_surface[i, j] = model.fn_u(
                            torch.tensor([[t_mesh[i, j], S_mesh[i, j]]]).float()).detach().numpy()
                        Exact_price_surface[i, j] = theoretical_barrier_option(S0=S_mesh[i, j], K=self.K, T=1 - t_mesh[i, j],
                                                                               r=self.r,
                                                                               sigma=self.sigma, B=self.B)
                Error_measure = neural_networks.errormeasure(Exact_price_surface, NN_price_surface)
                mse = Error_measure.calculate_mse()
                mae = Error_measure.calculate_mae()

                mse_list.append(mse)
                mae_list.append(mae)
                error_surface = abs(Exact_price_surface - NN_price_surface)


                # error_list.append(np.max(error_surface))




                # ax = plt.figure()
                # ax = plt.axes(projection='3d')
                # ax.plot_surface(t_mesh, S_mesh, NN_price_surface, rstride=1, cstride=1, cmap='viridis',
                #                 edgecolor='none')
                # ax.set_title('surface: iter = %d' % it)
                # plt.show()

                ax = plt.figure()
                ax = plt.axes(projection='3d')
                ax.plot_surface(t_mesh, S_mesh, error_surface, rstride=1, cstride=1, cmap='viridis',
                                edgecolor='none')
                ax.set_title('error surface: iter = %d' % it)
                plt.show()



        self.loss_list = loss_list
        self.test_sample_list = test_sample_list
        self.mse_list = mse_list
        self.mae_list = mae_list




    def predict(self, Xi_star, t_star, W_star):
        _, X_star, Y_star, _ = self.loss_function(t_star, W_star, Xi_star)

        return X_star, Y_star


if __name__ == '__main__':
    M = 50 # number of trajectories (batch size)
    N = 10 # number of time snapshots

    learning_rate = 2*1e-3
    epoch = 3000

    r = 0.05
    K = 100.0
    T = 1.0
    sigma = 0.4
    B = 70
    D = 1  # number of dimensions
    lambda_1 = 10 # weight for terminal condition
    lambda_2 = 0 # Dgtroch BC
    lambda_3 = 0 #BC1 at X = 0
    lambda_4 = 1 #BC2 at X = B
    lambda_5 = 0 #BC3 at X = B, t = T
    save_N = 200

    out_of_sample_test_t = 0.0
    out_of_sample_test_S = 100.0

    out_of_sample_input = torch.tensor([out_of_sample_test_t, out_of_sample_test_S]).float()
    test_sample_exact = theoretical_barrier_option(out_of_sample_test_S, K, B,T - out_of_sample_test_t, r, sigma )
    gbm_scheme = 0 # in theory 1 is more accurate. 0 is accurate for large N


    if D==1:
        Xi = torch.tensor([np.linspace(B,200,M)]).transpose(-1,-2).float()
        #Xi = torch.ones([M,1])
    else:
        Xi = torch.from_numpy(np.array([1.0, 0.5] * int(D / 2))[None, :]).float()

    model = FBSNN(epoch,r,K,sigma,Xi, T, M, N, D, B,learning_rate,gbm_scheme=0,lambda_1=lambda_1,lambda_2=lambda_2,lambda_3 = lambda_3,lambda_4 = lambda_4,lambda_5 = lambda_5,out_of_sample_input=out_of_sample_input,out_of_sample_exact = test_sample_exact,save_N=save_N)
    model.train()

#%%
    model.M = 100
    M =model.M
    new_Xi = torch.ones(model.M).unsqueeze(-1)*150
    model.Xi = new_Xi

    t_test, W_test = model.fetch_minibatch()
    X_pred, Y_pred = model.predict(new_Xi, t_test, W_test)
    test_sample_list = model.test_sample_list

    def u_exact(t, X,r=0.05,K=100,sigma=0.4,B=90):  # (N+1) x 1, (N+1) x D

        res = np.zeros([t.shape[0], X.shape[1]])
        for i in range(t.shape[0]):
            for j in range(X.shape[1]):
                res[i, j] = theoretical_barrier_option(S0=X[i, j], K=K, T=T-t[i, 0], B = B,r=r, sigma=sigma)
        return   res


    t_test = t_test.detach().numpy()
    X_pred = X_pred.detach().numpy()
    Y_pred = Y_pred.detach().numpy()
    Y_test = np.reshape(u_exact(np.reshape(t_test[0:M, :, :], [-1, 1]), np.reshape(X_pred[0:M, :, :], [-1, D])),
                        [M, -1, 1])
    print(Y_test[0, 0, 0])

#%%

    t = np.linspace(0, 1, 10)
    S = np.linspace(B, 200, 10)
    t_mesh, S_mesh = np.meshgrid(t, S)
    NN_price_surface = np.zeros([10, 10])
    Exact_price_surface = np.zeros([10, 10])
    samples = 10

    for i in range(10):
        for j in range(10):
            NN_price_surface[i, j] = model.fn_u(torch.tensor([[t_mesh[i, j], S_mesh[i, j]]]).float()).detach().numpy()
            Exact_price_surface[i, j] = theoretical_barrier_option(S0=S_mesh[i, j], K=K, T=1 - t_mesh[i, j], r=r,
                                                               sigma=sigma, B = B)

#%%
    import importlib
    importlib.reload(plotplot)
    plotter = plotplot.PlottingTools(test_sample_list, test_sample_exact, model, t_test, X_pred, Y_pred, Y_test, Exact_price_surface, NN_price_surface, t_mesh, S_mesh, S)
    plotter.plot_1(fig_dir="plots")
    plotter.plot_2(fig_dir="plots")
    #plotter.plot_3(samples=model.M, fig_dir="plots")
    #plotter.plot_4(fig_dir="plots")
    #plotter.plot_5(fig_dir="plots")
    plotter.plot_6(fig_dir="plots")
    plotter.plot_7(fig_dir="plots")
    plotter.plot_8(fig_dir="plots")
    plotter.plot_9(samples=model.M, fig_dir="plots")
    plotter.plot_10(fig_dir="plots")
    plotter.plot_11(fig_dir="plots")

