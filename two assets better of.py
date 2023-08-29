import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import math
from plotting import newfig, savefig
from common_tools import basket_pricer
from common_tools import neural_networks
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import CosineAnnealingLR
from common_tools import Func_rainbow
from common_tools import plotplot
np.random.seed(42)
torch.manual_seed(42)
plt.rcParams['font.family'] = 'Times New Roman'
os.environ['KMP_DUPLICATE_LIB_OK']='True'
class FBSNN(nn.Module):  # Forward-Backward Stochastic Neural Network
    def __init__(self, epochs,save_N,r, mu, sigma, rho, K,alpha,Xi, T, M, N, D, learning_rate,gbm_scheme=1,out_of_sample_input =None,lambda_1= 100,lambda_2 = 100,out_of_sample_exact = 0 ):
        super().__init__()
        self.r = r  # interest rate
        self.mu = mu  # drift rate
        self.sigma = sigma  # vol of each underlying
        self.rho = rho  # correlation, assumed to be constant across the basket
        self.K = K # strike price
        self.alpha = alpha # weights of each underlying
        self.Xi = Xi  # initial point
        self.T = T  # terminal time

        self.M = M  # number of trajectories
        self.N = N  # number of time snapshots
        self.D = D  # number of dimensions
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.fn_u  = neural_networks.neural_net(self.M,n_dim=D+1,n_output=1,num_layers=2,width=512,activation=torch.nn.ReLU())


        self.optimizer = optim.Adam(self.fn_u.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                              milestones=[5000,10000,20000,30000,
                                                                          40000,50000], gamma=0.2)
        #self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=2000, eta_min=0.001)
        var_cov_mat = np.zeros((D, D))
        self.gbm_scheme = gbm_scheme  # 0:euler scheme for gbm #1: EXP scheme
        self.a = torch.tensor([0]) # the aij matrix in the mathematical modelling book
        self.out_of_sample_input = out_of_sample_input # the out of sample input for the neural network
        self.out_of_sample_exact = out_of_sample_exact
        self.save_N = save_N
        self.epochs = epochs
        for i in range(D):
            for j in range(D):
                if i == j:
                    var_cov_mat[i, j] = sigma[i] * sigma[j]
                else:
                    var_cov_mat[i, j] = rho * sigma[i] * sigma[j]
        var_cov_mat = torch.tensor(var_cov_mat)
        self.var_cov_mat = var_cov_mat
        L = torch.linalg.cholesky(self.var_cov_mat).float()
        L = L.unsqueeze(0).repeat(self.M, 1, 1)  # Repeat L along the first dimension
        self.L = L


    def phi_torch(self, t, X, DuDt,DuDx,D2uDx2):  # M x 1, M x D, M x 1, M x D, MxDXD
        # print(DuDt.shape)

        term_1 =torch.sum( X*DuDx*torch.tensor(self.mu), dim=1 ).unsqueeze(-1)
        # print(term_1.shape)

        i = 0

        term_2 = torch.zeros([self.M,1])

        for i in range(self.M):
            mat_1 = D2uDx2[i,:,:] * self.var_cov_mat.float()
            term_2_i_right = torch.matmul(mat_1, X[i,:])
            # print(term_2_i_right.unsqueeze(-1))
            # print(X[i,:])
            term_2_i = 0.5 * torch.matmul(X[i,:],term_2_i_right.unsqueeze(-1))
            # print(term_2_i)
            term_2[i,:] = term_2_i



        res = DuDt + term_1 + term_2
        return  res # M x 1


    def g_torch(self, X):  # M x D
        # terminal condition


        res =  torch.max(X, dim=1, keepdim=True).values
        # print(res.shape)
        # print(res)
        return res # M x 1

    def mu_torch(self, t, X, Y):  # M x 1, M x D, M x 1, M x D
        # return torch.zeros([self.M, self.D])  # M x D
        return torch.ones([self.M, self.D]) * torch.tensor(self.mu)

    def sigma_torch_old(self, t, X, Y):  # M x 1, M x D, M x 1
        # print(X.shape)
        d = self.D

        L = torch.linalg.cholesky(self.var_cov_mat)
        L = L.float()

        self.L = L

        res = torch.zeros([self.M,self.D,self.D])
        for j in range(self.M):
            res[j,:,:]  = L
        return  res  # M x D x D

    def sigma_torch(self, t, X, Y):
        # new version which is faster
        # d = self.D
        # L = torch.linalg.cholesky(self.var_cov_mat).float()
        # L = L.unsqueeze(0).repeat(self.M, 1, 1)  # Repeat L along the first dimension
        return self.L  # M x D x D

    def net_u_Du(self, t, X):  # M x 1, M x D


        inputs = torch.cat([t, X], dim=1) #M x (D+1)


        u = self.fn_u(inputs) # M x 1

        #DuDt = torch.autograd.grad(torch.sum(u), t, retain_graph=True,create_graph=True)[0]  # M x 1
        DuDx = torch.autograd.grad(torch.sum(u), X, retain_graph=True,create_graph=True)[0]  # M x D
        DuDt = 0
        # D2uDx2_list = []
        #
        # for i in range(self.M):
        #     hessian = torch.autograd.functional.hessian(lambda x: self.fn_u(x)[0], inputs[i,:],create_graph=True)
        #     D2uDx2_list.append(hessian[1:,1:]) # note that this hessian include time, we exclude the time part here
        #
        # D2uDx2 = torch.stack(D2uDx2_list,dim =0)

        D2uDx2 = 0 # we no longer need the second order derivative


        return u, DuDx,DuDt,D2uDx2 # M x 1, M x D, M x 1, M x D

    def Dg_torch(self, X):  # M x D
        return torch.autograd.grad(torch.sum(self.g_torch(X)), X, retain_graph=True)[0]  # M x D

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

    def loss_function(self, t, W, Xi):  # M x (N+1) x 1, M x (N+1) x D, 1 x D
        loss = torch.zeros(1)
        X_buffer = []
        Y_buffer = []

        t0 = t[:, 0, :]  # M x 1
        W0 = W[:, 0, :]  # M x D
        X0 = Xi  # M x D

        X0.requires_grad = True
        t0.requires_grad = True
        Y0, DuDx0,DuDt0,D2uDx20  = self.net_u_Du(t0, X0)  # M x 1, M x D

        X_buffer.append(X0)
        Y_buffer.append(Y0)

        X0.requires_grad = True

        t0.requires_grad = True
        total_weight  = 0

        for n in range(0, self.N):
            t1 = t[:, n + 1, :]
            W1 = W[:, n + 1, :]

            t1.requires_grad = True

            if self.gbm_scheme==0:
                X1 = X0 + self.mu_torch(t0, X0, Y0) * (t1 - t0)*X0 + torch.matmul(self.sigma_torch(t0, X0, Y0),
                                                                                   (W1 - W0).unsqueeze(
                                                                                       -1)).squeeze(2)*X0 # M x D



            elif self.gbm_scheme ==1:
                X1 = X0 * torch.exp(self.mu_torch(t0, X0, Y0) * (t1 - t0) - 0.5*torch.tensor(self.sigma)**2*(t1-t0)+torch.matmul(self.sigma_torch(t0, X0, Y0),
                                                                                   (W1 - W0).unsqueeze(
                                                                                       -1)).squeeze(2)) # not sure if this is right


            Y1_tilde = Y0 + self.r* Y0 * (t1 - t0) + torch.sum(
                X0 * DuDx0* torch.matmul(self.sigma_torch(t0, X0, Y0), (W1 - W0).unsqueeze(-1)).squeeze(2), dim=1).unsqueeze(1)


            Y1, DuDx1,DuDt1,D2uDx21  = self.net_u_Du(t1, X1)

            loss = loss + torch.sum((Y1 - Y1_tilde) ** 2)
            total_weight += 1


            # print(Y_diag.shape)
            # print(torch.tensor(np.linspace(0, 100, M)).unsqueeze(1).shape)

            # # Generating X_1_0
            # X_1_0 = np.zeros((M, 2))
            # X_1_0[:, 0] = np.linspace(0, 100, M)
            # X_1_0  = torch.tensor(X_1_0).float()
            # X_1_0.requires_grad = True
            #
            # # Generating X_0_1
            # X_0_1 = np.zeros((M, 2))
            # X_0_1[:, 1] = np.linspace(0, 100, M)
            # X_0_1 = torch.tensor(X_0_1).float()
            # X_0_1.requires_grad = True
            #
            # Y_1_0,_,_,_ = self.net_u_Du(t1,X_0_1)
            # Y_0_1,_,_,_ = self.net_u_Du(t1,X_1_0)
            # # print(Y_0_1.shape)
            # # print(torch.tensor(np.linspace(0,100,M)).float().shape)
            # loss += loss + self.lambda_2*torch.sum((Y_1_0 - torch.tensor(np.linspace(0,100,M)).unsqueeze(1).float()) ** 2)
            # loss += loss + self.lambda_2* torch.sum((Y_0_1 - torch.tensor(np.linspace(0,100,M)).unsqueeze(1).float()) ** 2)



            t0 = t1
            W0 = W1
            X0 = X1
            Y0 = Y1
            DuDx0 = DuDx1


            X_buffer.append(X0)
            Y_buffer.append(Y0)


        loss = loss + self.lambda_1* torch.sum((Y1 - self.g_torch(X1)) ** 2)
        total_weight += self.lambda_1
        # Generate X_diag
        X_diag = torch.zeros((M, 2))
        X_diag[:, 0] = torch.linspace(0, 100, M)
        X_diag[:, 1] = torch.linspace(0, 100, M)
        # X_diag = torch.tensor(X_diag)
        X_diag.requires_grad = True
        Y_diag, _, _, _ = self.net_u_Du(torch.ones(t1.shape), X_diag)
        loss += loss + self.lambda_2 * torch.sum(
            (Y_diag - torch.tensor(np.linspace(0, 100, M)).unsqueeze(1).float()) ** 2)


        loss = loss/total_weight
        # loss = loss + torch.sum((Z1 - self.Dg_torch(X1)) ** 2)

        X = torch.stack(X_buffer, dim=1)  # M x N x D
        Y = torch.stack(Y_buffer, dim=1)  # M x N x 1

        return loss, X, Y, Y[0, 0, 0]

    def train(self):
        N_Iter = self.epochs

        start_time = time.time()
        loss_list = []
        test_sample_list = []
        mse_list = []
        mae_list = []
        for it in range(N_Iter):

            t_batch, W_batch = self.fetch_minibatch()  # M x (N+1) x 1, M x (N+1) x D
            loss, X_pred, Y_pred, Y0_pred = self.loss_function(t_batch, W_batch, self.Xi)

            test_sample_list.append(self.fn_u(self.out_of_sample_input).detach().numpy()[0][0])

            self.optimizer.zero_grad()
            loss.backward()
            loss_list.append(loss.item())
            self.optimizer.step()
            self.scheduler.step()



            # Print
            if it % self.save_N == 0:
                elapsed = time.time() - start_time
                print('It: %d, Time: %.2f, Loss: %.3e, Y0: %.3f' %
                      (it, elapsed, loss, Y0_pred))
                start_time = time.time()

                # plt.plot(range(len(loss_list)), np.log10(loss_list))
                # plt.show()
                #
                # plt.plot(test_sample_list[100:])
                # plt.plot(np.ones(len(test_sample_list))[100:] * self.out_of_sample_exact)
                # plt.show()

                # surface error plot
                S_1 = np.linspace(0, 100., 10)
                S_2 = np.linspace(0, 100., 10)

                S_1_mesh, S_2_mesh = np.meshgrid(S_1, S_2)
                NN_price_surface = np.zeros([10, 10])
                Exact_price_surface = np.zeros([10, 10])
                for i in range(10):
                    for j in range(10):
                        NN_price_surface[i, j] = model.fn_u(
                            torch.tensor([0.5, S_1_mesh[i, j], S_2_mesh[i, j]]).float()).detach().numpy()

                        Exact_price_surface[i, j] = Func_rainbow.two_asset_better_of_exact(T=0.5, S1=S_1_mesh[i, j],
                                                                                           S2=S_2_mesh[i, j],sigma=sigma, rho=rho)

                error_surface = NN_price_surface-Exact_price_surface
                Error_measure = neural_networks.errormeasure(Exact_price_surface, NN_price_surface)
                mse = Error_measure.calculate_mse()
                mae = Error_measure.calculate_mae()
                # mape = Error_measure.calculate_mape()
                mse_list.append(mse)
                mae_list.append(mae)
                # ax = plt.axes(projection='3d')
                # ax.plot_surface(S_1_mesh,S_2_mesh, NN_price_surface, rstride=1, cstride=1, cmap='viridis',
                #                 edgecolor='none')
                # ax.set_title('surface: iter = %d' % it)
                # plt.show()
                #
                # ax = plt.axes(projection='3d')
                # ax.plot_surface(S_1_mesh,S_2_mesh, error_surface, rstride=1, cstride=1, cmap='viridis',
                #                 edgecolor='none')
                # ax.set_title('error surface: iter = %d' % it)
                # plt.show()
        self.mse_list = mse_list
        self.mae_list = mae_list
        self.test_sample_list = test_sample_list
        self.loss_list = loss_list



    def predict(self, Xi_star, t_star, W_star):
        _, X_star, Y_star, _ = self.loss_function(t_star, W_star, Xi_star)

        return X_star, Y_star


if __name__ == '__main__':
    D = 2  # no. of underlyings
    S0 = 100*np.ones([2])  # Initial Value of Assets at first of february
    K = 100  # Strike Price
    r = 0.05
    # mu = [0.05, 0.06, 0.07, 0.05, 0.06, 0.07, 0.05, 0.06, 0.07, 0.06]
    mu = [r,r]
    sigma = [0.20, 0.30]
    rho = 0.75  # Correlation between Brownian Motions
    lambda_1 = 1  # Terminal
    lambda_2 = 1 # Diag
    T = 1  # Time to Maturity
    N_STEPS, N_PATHS = 10, 100
    var_cov_mat = np.zeros((D, D))
    M = N_PATHS  # Number of paths
    N = N_STEPS  # Number of time steps
    alpha = [0.5,0.5] # useless
    save_N =200

    for i in range(D):
        for j in range(D):
            if i == j:
                var_cov_mat[i, j] = sigma[i] * sigma[j]
            else:
                var_cov_mat[i, j] = rho * sigma[i] * sigma[j]

    learning_rate = 0.0015
    x = np.linspace(0, 1, 10)
    y = np.linspace(0, 1, 10)
    X, Y = np.meshgrid(x, y)
    points = np.stack((X.flatten(), Y.flatten()), axis=-1)

    # Convert to torch.tensor
    Xi = torch.tensor(100 * points).float()

    #Xi = torch.from_numpy(100*np.array([[1.0, 1.0],[1.0, 1.0],[1.0, 0.5],[0.5, 1.0],[0.5, 0.5],[1.0,1.0],[1.0,1.0],[1.0,1.0],[1.0,1.0],[1,1]])).float()
    T = 1.0
    out_of_sample_input = torch.tensor([[0,100,100]]).float()
    out_of_sample_exact = Func_rainbow.two_asset_better_of_exact()
    # print(Xi.shape)
    epochs =50000


#%%


    # M=5
    model = FBSNN(epochs,save_N,r, mu, sigma, rho,K,alpha, Xi, T, M, N, D,learning_rate,0,out_of_sample_input,lambda_1=lambda_1,lambda_2 = lambda_2,out_of_sample_exact=out_of_sample_exact)
    model.train()
    # model.train()

    # check GBM results
    #t_test, W_test = model.fetch_minibatch()

    # loss, X, Y, Z = model.loss_function(t_test, W_test,Xi)
    # payoff = model.g_torch(X[:,-1,:]) * math.exp(-r*T)
    # print(payoff.mean())




#%%
    new_Xi = torch.ones(Xi.shape)*100
    new_Xi.requires_grad = True
    model.Xi = new_Xi
    t_test, W_test = model.fetch_minibatch()
    X_pred, Y_pred = model.predict(new_Xi, t_test, W_test)
#%%
    test_sample_list = []
    loss_list = []
    loss_list = model.loss_list
    test_sample_list = model.test_sample_list
    # fig0 = plt.figure(figsize=[10,6])
    # plt.plot(test_sample_list[50::10],color='black', linewidth=1.8)
    # plt.plot(np.ones(len(test_sample_list[50:])//10)*out_of_sample_exact, color='red', linewidth=1.8)
    # plt.xlabel("Epochs trained", fontsize=18)
    # plt.ylabel("Price", fontsize=18)
    # plt.title('Neural Network price vs exact price', fontsize=20)
    # plt.legend(loc='best', prop={'size': 12})
    # plt.grid(True)
    # plt.xticks(fontsize=16)
    # plt.yticks(fontsize=16)
    # plt.gca().spines['top'].set_visible(False)
    # plt.gca().spines['right'].set_visible(False)
    # fig0.savefig(os.path.join("plots", '2a_price'))
    # plt.show()

    #
    # print("cut mean")
    # print(np.array(test_sample_list[50:]).mean())


#%%
    #surface error plot
    S_1 = np.linspace(0, 100., 10)
    S_2 = np.linspace(0, 100., 10)

    S_1_mesh, S_2_mesh = np.meshgrid(S_1, S_2)
    NN_price_surface = np.zeros([10, 10])
    Exact_price_surface = np.zeros([10, 10])
    for i in range(10):
        for j in range(10):
            NN_price_surface[i, j] = model.fn_u(torch.tensor([0.5,S_1_mesh[i,j],S_2_mesh[i,j]]).float()).detach().numpy()

            Exact_price_surface[i, j] = Func_rainbow.two_asset_better_of_exact(T=0.5,S1 = S_1_mesh[i,j],S2 = S_2_mesh[i,j],sigma=sigma,rho=rho)

#%%
    def u_exact(t, X):  # (N+1) x 1, (N+1) x D
        Y_exact = np.zeros(t.shape)
        for i in range(t.shape[0]):
            t_ = t[i,:][0]
            X_ = X[i,:]
            print(X_)
            Y_exact[i,:] = Func_rainbow.two_asset_better_of_exact(S1=X_[0],S2=X_[1],T=1-t_,r=r)

        return Y_exact  # (N+1) x 1


    samples = 10
    # model.Xi = torch.tensor(np.ones([M, 1])).transpose(-1, -2).float()
    t_test = t_test.detach().numpy()
    X_pred = X_pred.detach().numpy()
    Y_pred = Y_pred.detach().numpy()
    Y_test = np.reshape(u_exact(np.reshape(t_test[:, :, :], [-1, 1]), np.reshape(X_pred[:, :, :], [-1, D])),
                        [M, -1, 1])
    print(Y_test[0, 0, 0])
#%%
    plt.rcParams['font.family'] = 'Times New Roman'
    import importlib

    importlib.reload(plotplot)
    plotter = plotplot.PlottingTools(test_sample_list[100:],out_of_sample_exact, model, t_test, X_pred, Y_pred, Y_test, Exact_price_surface, NN_price_surface, S_1_mesh, S_2_mesh, S_1)
    plotter.plot_1(fig_dir="plots")
    plotter.plot_2(fig_dir="plots")
    plotter.plot_3(samples=samples, fig_dir="plots")
    plotter.plot_4(fig_dir="plots",xlabel="Price ($S_1$)",ylabel="Price ($S_2$)")
    plotter.plot_5(fig_dir="plots",xlabel="Price ($S_1$)",ylabel="Price ($S_2$)")
    plotter.plot_6(fig_dir="plots")
    plotter.plot_7(fig_dir="plots",xlabel="Price ($S_1$)",ylabel="Price ($S_2$)")
    plotter.plot_8(fig_dir="plots")

    # fig1 = plt.figure(figsize=[10,6])
    # ax1 = plt.axes(projection='3d')
    # ax1.plot_surface(S_1_mesh, S_2_mesh, NN_price_surface, rstride=1, cstride=1, cmap='viridis',
    #                 edgecolor='none')
    # ax1.set_title('NN surface',fontsize = 20)
    # ax1.set_xlabel('Underlying asset 1 price ($S_1$)', fontsize=14)
    # ax1.set_ylabel('Underlying asset 1 price ($S_2$)', fontsize=14)
    # fig1.savefig(os.path.join("plots", '2a_NN'))
    # plt.show()




    # fig2 = plt.figure(figsize=[10,6])
    # ax2 = plt.axes(projection='3d')
    # ax2.plot_surface(S_1_mesh, S_2_mesh, Exact_price_surface, rstride=1, cstride=1, cmap='viridis',
    #                 edgecolor='none')
    # ax2.set_title('Exact surface',fontsize = 20)
    # ax2.set_xlabel('Underlying asset 1 price ($S_1$)', fontsize=14)
    # ax2.set_ylabel('Underlying asset 1 price ($S_2$)', fontsize=14)
    # fig2.savefig(os.path.join("plots", '2a_exact'))
    # plt.show()
    #
    #
    # error_surface = np.abs(NN_price_surface - Exact_price_surface)
    # fig3 = plt.figure(figsize=[10,6])
    # ax3 = plt.axes(projection='3d')
    # ax3.plot_surface(S_1_mesh, S_2_mesh, error_surface, rstride=1, cstride=1, cmap='viridis',
    #                 edgecolor='none')
    # ax3.set_title('Error surface',fontsize = 20)
    # ax3.set_xlabel('Underlying asset 1 price ($S_1$)', fontsize=14)
    # ax3.set_ylabel('Underlying asset 1 price ($S_2$)', fontsize=14)
    # fig3.savefig(os.path.join("plots", '2a_error'))
    # plt.show()





    # %%

    # fig4 = plt.figure(figsize=[10,6])
    # plt.plot(t_test[0:1, :, 0].T, Y_pred[0:1, :, 0].T, 'k', label=r'Learned $u(t,X_t)$')
    # plt.plot(t_test[0:1, :, 0].T, Y_test[0:1, :, 0].T, 'r--', label=r'Exact $u(t,X_t)$')
    # plt.plot(t_test[0:1, -1, 0], Y_test[0:1, -1, 0], 'ko', label=r'$Y_T = u(T,X_T)$')
    #
    # plt.plot(t_test[1:samples, :, 0].T, Y_pred[1:samples, :, 0].T, 'k')
    # plt.plot(t_test[1:samples, :, 0].T, Y_test[1:samples, :, 0].T, 'r--')
    # plt.plot(t_test[1:samples, -1, 0], Y_test[1:samples, -1, 0], 'ko')
    # plt.plot([0], Y_test[0, 0, 0], 'ks', label=r'$Y_0 = u(0,X_0)$')
    #
    # plt.xlabel(r'$t$', fontdict={'fontsize': 18})
    # plt.ylabel(r'$S_t$', fontdict={'fontsize': 18})
    # plt.title('Best of option', fontdict={'fontsize': 20})
    # plt.legend(loc='upper left', prop={'size': 12})
    # plt.grid(True)
    # plt.gca().spines['top'].set_visible(False)
    # plt.gca().spines['right'].set_visible(False)
    # plt.tick_params(axis='both', which='both', labelsize=16)
    # fig4.savefig(os.path.join("plots", '2a_path'))
    # plt.show()


    #errors = np.sqrt((Y_test - Y_pred) ** 2 / Y_test ** 2)
    # mean_errors = np.mean(errors, 0)
    # std_errors = np.std(errors, 0)
    #
    # plt.figure()
    # plt.plot(t_test[0, :, 0], mean_errors, 'b', label='mean')
    # plt.plot(t_test[0, :, 0], mean_errors + 2 * std_errors, 'r--', label='mean + two standard deviations')
    # plt.xlabel(r'$t$')
    # plt.ylabel('relative error')
    #
    # plt.title('100-dimensional Black-Scholes-Barenblatt')
    # plt.legend()
    # plt.show()


    # savefig('BSB_error.png', crop=False)
#%%
    # mse_list = model.mse_list
    # mae_list = model.mae_list
    # fig = plt.figure(figsize=(10, 6))
    # plt.plot(10 * np.array(range(len(mse_list))), np.log10(mse_list), color='red', label='log(mse)', linewidth=1.8)
    # plt.plot(10 * np.array(range(len(mae_list))), np.log10(mae_list), color='black', label='log(mae)', linewidth=1.8)
    # plt.xlabel("Epochs trained", fontsize=18)
    # plt.ylabel("Error", fontsize=18)
    # plt.title("Convergence of MSE and MAE", fontsize=20)
    # plt.legend(fontsize=14)
    # plt.xticks(fontsize=14)
    # plt.yticks(fontsize=14)
    # plt.grid(True)
    # #self.save_plot_with_timestamp(fig, "MSE convergence")
    # plt.show()