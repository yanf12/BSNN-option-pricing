import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from plotting import newfig, savefig
from torch.optim.lr_scheduler import StepLR
from scipy.special import comb
from scipy.stats import norm
from common_tools import plotplot
from IPython.display import display, clear_output
import torch
import torch.nn as nn
import numpy as np
np.random.seed(42)
torch.manual_seed(42)


def theoretical_vanilla_eu(S0=1, K=1, T=0, r=0.05, sigma=0.4, type_='call'):

    if T == 0:
        if type_ == "call":
            return max(S0 - K, 0)
        else:
            return max(K - S0, 0)
    d1 = ((np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T)) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if type_ == "call":
        c = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        return c
    elif type_ == "put":
        p = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
        return p

##resnet for vanilla option
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.fc1 = nn.Linear(in_channels, out_channels)
        self.fc2 = nn.Linear(out_channels, out_channels)
        self.relu = nn.ReLU()
    def forward(self, x):
        residual = x
        out = self.relu(self.fc1(x))
        out = self.fc2(out)
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, in_channels, num_classes, num_layers, width):
        super(ResNet, self).__init__()
        self.width = width
        self.fc1 = nn.Linear(in_channels, self.width)
        self.relu = nn.ReLU()
        self.resblocks = nn.ModuleList()
        for _ in range(num_layers):
            self.resblocks.append(ResBlock(self.width, self.width))
        self.fc2 = nn.Linear(self.width, num_classes)
    def forward(self, x):
        x = self.relu(self.fc1(x))
        for resblock in self.resblocks:
            x = resblock(x)
        x = self.fc2(x)
        return x



##back-forward neural network
class neural_net(nn.Module):
    def __init__(self, pathbatch=100, n_dim=100 + 1, n_output=1, num_layers=2, width=1024,activation =torch.tanh):
        super(neural_net, self).__init__()
        self.pathbatch = pathbatch
        self.num_layers = num_layers
        self.width = width
        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(nn.Linear(n_dim, self.width))
        for _ in range(num_layers - 1):
            self.fc_layers.append(nn.Linear(self.width, self.width))
        self.out = nn.Linear(self.width, n_output)
        self.activation = activation
        with torch.no_grad():
            for layer in self.fc_layers:
                np.random.seed(42)
                torch.manual_seed(42)
                torch.nn.init.xavier_uniform(layer.weight)
    def forward(self, state, train=False):
        for i in range(self.num_layers):
            state = self.activation(self.fc_layers[i](state))
        fn_u = self.out(state)
        return fn_u


class neural_net_specify_width(nn.Module):
    def __init__(self, pathbatch=100, n_dim=100 + 1, n_output=1, num_layers=2, width_list=None, activation=torch.nn.ReLU()):
        super(neural_net_specify_width, self).__init__()
        self.pathbatch = pathbatch
        self.num_layers = num_layers
        self.width_list = width_list if width_list else [512] * num_layers  # default width for each layer is 1024
        if len(self.width_list) != num_layers:
            raise ValueError("Number of widths in the list must equal to the number of layers")
        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(nn.Linear(n_dim, self.width_list[0]))
        for i in range(1, num_layers):
            self.fc_layers.append(nn.Linear(self.width_list[i-1], self.width_list[i]))
        self.out = nn.Linear(self.width_list[-1], n_output)
        self.activation = activation
        with torch.no_grad():
            for layer in self.fc_layers:
                np.random.seed(42)
                torch.manual_seed(42)
                torch.nn.init.xavier_uniform_(layer.weight)


    def forward(self, state, train=False):
        for i in range(self.num_layers):
            state = self.activation(self.fc_layers[i](state))
        fn_u = self.out(state)
        return fn_u


class neural_net_specify_width_cuda(nn.Module):
    def __init__(self, pathbatch=100, n_dim=100 + 1, n_output=1, num_layers=2, width_list=None,
                 activation=torch.nn.ReLU()):
        super(neural_net_specify_width_cuda, self).__init__()
        self.pathbatch = pathbatch
        self.num_layers = num_layers
        self.width_list = width_list if width_list else [512] * num_layers  # default width for each layer is 1024
        if len(self.width_list) != num_layers:
            raise ValueError("Number of widths in the list must equal to the number of layers")

        # Move the model to the GPU
        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(nn.Linear(n_dim, self.width_list[0]).cuda())
        for i in range(1, num_layers):
            self.fc_layers.append(nn.Linear(self.width_list[i - 1], self.width_list[i]).cuda())
        self.out = nn.Linear(self.width_list[-1], n_output).cuda()
        self.activation = activation

        with torch.no_grad():
            for layer in self.fc_layers:
                np.random.seed(42)
                torch.manual_seed(42)
                layer.weight = torch.nn.Parameter(torch.nn.init.xavier_uniform_(layer.weight).cuda())

    def forward(self, state, train=False):
        for i in range(self.num_layers):

            # print(state.device)
            state = self.activation(self.fc_layers[i](state))
        fn_u = self.out(state)
        return fn_u


##error measure
class errormeasure:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

    def calculate_mse(self):
        return np.mean((self.y_true - self.y_pred) ** 2)

    def calculate_mape(self):
        return np.mean(np.abs((self.y_true - self.y_pred) / self.y_true)) * 100

    def calculate_mae(self):
        return np.mean(np.abs(self.y_true - self.y_pred))


##fbsnn
class FBSNN(nn.Module):  # Forward-Backward Stochastic Neural Network
    def __init__(self,r,K,sigma,Xi,T,M,N,D,learning_rate,gbm_scheme,lambda_1,lambda_2,lambda_3,out_of_sample_input,out_of_sample_exact,width_list = [1024,512,256],num_layers=3,activation = torch.nn.ReLU(),save_N = 200,epochs = 50000):
        super().__init__()
        self.r = r  # interest rate
        self.mu = r # test drift rate
        self.sigma = sigma # volatility
        self.K = K  # strike price
        self.Xi = Xi  # initial point
        self.T = T  # terminal time
        self.M = M  # number of trajectories
        self.N = N  # number of time snapshots
        self.D = D  # number of dimensions
        self.fn_u = neural_net_specify_width(pathbatch=M, n_dim=D + 1, n_output=1, num_layers=num_layers, width_list=width_list,activation=activation)
        self.optimizer = optim.Adam(self.fn_u.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[10000,20000,25000], gamma=0.2)
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=3000, eta_min=0.001)
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.gbm_scheme = gbm_scheme # 0:euler scheme for gbm #1: EXP scheme
        self.out_of_sample_input = out_of_sample_input
        self.out_of_sample_exact = out_of_sample_exact
        self.epochs = epochs
        self.save_N = save_N # frequency to save mse error and surface plot
    def theoretical_vanilla_eu(self, S0=100, K=100, T=0, r=0.05, sigma=0.4, type_='call'):
        if T == 0:
            if type_ == "call":
                return max(S0 - K, 0)
            else:
                return max(K - S0, 0)
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

    def g_torch(self, X,K):  # M x D
        row_max, _ = torch.max(X, dim=1)  # get maximum along each row
        return torch.clamp(row_max - K, min=0).unsqueeze(1)  # M x 1

    def mu_torch(self, r,t, X, Y):  # 1x1, M x 1, M x D, M x 1, M x D
        return 0*torch.ones([self.M, self.D])  # M x D

    def sigma_torch(self, t, X, Y):  # M x 1, M x D, M x 1
        return self.sigma * torch.diag_embed(X)  # M x D x D

    def net_u_Du(self, t, X):  # M x 1, M x D
        inputs = torch.cat([t, X], dim=1)
        u = self.fn_u(inputs)
        DuDx = torch.autograd.grad(torch.sum(u), X, retain_graph=True,create_graph=True)[0]
        # DuDt = torch.autograd.grad(torch.sum(u), t, retain_graph=True,create_graph=True)[0]
        #D2uDx2 = torch.autograd.grad(torch.sum(DuDx), X, retain_graph=True,create_graph=True)[0]
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
            if self.gbm_scheme ==0:
                X1 = X0 + self.mu*X0*(t1-t0) + self.sigma * X0 * (W1 - W0) # Euler-M scheme
            elif self.gbm_scheme ==1:
                X1 = X0*torch.exp( (self.mu-0.5*self.sigma**2)*(t1-t0) + self.sigma* (W1-W0))
            t1.requires_grad = True
            Y1, DuDx1,DuDt1,D2uDx21 = self.net_u_Du(t1, X1)  # M x 1, M x D
            Y1_tilde = Y0 + self.r*Y0* (t1-t0) + (self.mu-self.r)*X0*DuDx0 + DuDx0 * self.sigma*X0*(W1-W0)
            loss = loss + torch.sum((Y1 - Y1_tilde) ** 2)
            Y_at_X_0, _1, _2, _3 = self.net_u_Du(t1, X1 * 0)  # M x 1, M x D

            total_weight +=1
            t0 = t1
            W0 = W1
            X0 = X1
            Y0 = Y1
            DuDx0 = DuDx1 # not sure if this is correct
            DuDt0 = DuDt1
            D2uDx20 = D2uDx21
            X_buffer.append(X0)
            Y_buffer.append(Y0)
            loss = loss + self.lambda_3 * torch.sum((Y_at_X_0 - 0) ** 2)
            total_weight += self.lambda_3

        loss = loss + self.lambda_1*torch.sum((Y1 - self.g_torch(X1,self.K)) ** 2)
        total_weight += self.lambda_1
        loss = loss + self.lambda_2 * torch.sum((DuDx1 - self.Dg_torch(X1)) ** 2)
        total_weight += self.lambda_2

        loss = loss / total_weight
        X = torch.stack(X_buffer, dim=1)  # M x N x D
        Y = torch.stack(Y_buffer, dim=1)  # M x N x 1

        return loss, X, Y, Y[0, 0, 0]

    def train(self):
        N_Iter =self.epochs
        loss_list = []
        error_list = []
        start_time = time.time()
        t = np.linspace(0, 1, 10)
        S = np.linspace(0, 200, 10)
        test_sample_list = []
        t_mesh, S_mesh = np.meshgrid(t, S)
        mse_list = []
        mae_list = []
        for it in range(N_Iter+1):
            t_batch, W_batch = self.fetch_minibatch()  # M x (N+1) x 1, M x (N+1) x D
            loss, X_pred, Y_pred, Y0_pred = self.loss_function(t_batch, W_batch, self.Xi)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()


            loss_list.append(loss.detach().numpy()[0])
            test_sample_list.append(self.fn_u(self.out_of_sample_input).detach().numpy()[0])

            # Print
            if it %self.save_N  == 0:
                clear_output(wait=True)
                NN_price_surface = np.zeros([10, 10])
                Exact_price_surface = np.zeros([10, 10])
                for i in range(10):
                    for j in range(10):
                        NN_price_surface[i, j] = self.fn_u(
                            torch.tensor([[t_mesh[i, j], S_mesh[i, j]]]).float()).detach().numpy()
                        Exact_price_surface[i, j] = self.theoretical_vanilla_eu(S0=S_mesh[i, j], K=100,
                                                                                T=1 - t_mesh[i, j],
                                                                                r=0.05, sigma=0.4, type_='call')
                Error_measure = errormeasure(Exact_price_surface, NN_price_surface)
                mse = Error_measure.calculate_mse()
                mae = Error_measure.calculate_mae()
                mape = Error_measure.calculate_mape()
                mse_list.append(mse)
                mae_list.append(mae)

                elapsed = time.time() - start_time
                print('It: %d, Time: %.2f, Loss: %.3e, Y0: %.3f' %
                      (it, elapsed, loss, Y0_pred))
                start_time = time.time()
                # plt.plot(range(len(loss_list)),np.log(loss_list))
                # plt.show()
                # plt.plot(test_sample_list[50:])
                # plt.plot(np.ones(len(test_sample_list))[50:] * self.out_of_sample_exact , label='test sample exact price')
                # plt.show()
                error_surface = np.abs(NN_price_surface - Exact_price_surface)
                # ax = plt.axes(projection='3d')
                # ax.plot_surface(t_mesh, S_mesh, NN_price_surface, rstride=1, cstride=1, cmap='viridis',
                #                 edgecolor='none')
                # ax.set_title('surface: iter = %d' % it)
                # plt.show()
                #
                # ax = plt.axes(projection='3d')
                # ax.plot_surface(t_mesh, S_mesh, error_surface, rstride=1, cstride=1, cmap='viridis',
                #                 edgecolor='none')
                # ax.set_title('error surface: iter = %d' % it)
                # plt.show()

        self.loss_list = loss_list
        self.test_sample_list = test_sample_list
        self.mse_list = mse_list
        self.mae_list = mae_list

    def predict(self, Xi_star, t_star, W_star):
        _, X_star, Y_star, _ = self.loss_function(t_star, W_star, Xi_star)
        return X_star, Y_star


class FBSNN_cuda(nn.Module):  # Forward-Backward Stochastic Neural Network
    def __init__(self, r,K,sigma ,Xi, T, M, N, D, learning_rate, gbm_scheme ,lambda_1,lambda_2,out_of_sample_input,out_of_sample_exact):
        super().__init__()
        self.r = r  # interest rate
        self.sigma = sigma # volatility
        self.K = K  # strike price
        self.Xi = Xi  # initial point
        self.T = T  # terminal time

        self.M = M  # number of trajectories
        self.N = N  # number of time snapshots
        self.D = D  # number of dimensions
        self.fn_u = neural_net_specify_width_cuda(pathbatch=M, n_dim=D + 1, n_output=1)

        self.optimizer = optim.Adam(self.fn_u.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[1500, 10000, 20000, 50000, 80000, 100000], gamma=0.1)

        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2

        self.gbm_scheme = gbm_scheme # 0:euler scheme for gbm #1: EXP scheme
        self.out_of_sample_input = out_of_sample_input
        self.out_of_sample_exact = out_of_sample_exact


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

    def g_torch(self, X,K):  # M x D

        row_max, _ = torch.max(X, dim=1)  # get maximum along each row
        return torch.clamp(row_max - K, min=0).unsqueeze(1)  # M x 1
    def mu_torch(self, r,t, X, Y):  # 1x1, M x 1, M x D, M x 1, M x D
        return 0*torch.ones([self.M, self.D])  # M x D

    def sigma_torch(self, t, X, Y):  # M x 1, M x D, M x 1
        # print("sigma_torch")
        # print(X.shape)
        return self.sigma * torch.diag_embed(X)  # M x D x D

    def net_u_Du(self, t, X):  # M x 1, M x D

        inputs = torch.cat([t, X], dim=1).cuda()

        u = self.fn_u(inputs)
        DuDx = torch.autograd.grad(torch.sum(u), X, retain_graph=True,create_graph=True)[0]

        # print(DuDx.shape)

        # DuDt = torch.autograd.grad(torch.sum(u), t, retain_graph=True,create_graph=True)[0]
        #
        # D2uDx2 = torch.autograd.grad(torch.sum(DuDx), X, retain_graph=True,create_graph=True)[0]
        DuDt = 0
        D2uDx2 = 0
        return u, DuDx,DuDt,D2uDx2 # M x 1, M x D, M x 1, M x D
    def Dg_torch(self, X):  # M x D
        return torch.autograd.grad(torch.sum(self.g_torch(X,self.K)), X, retain_graph=True)[0]  # M x D
    def fetch_minibatch(self):
        T = self.T
        M = self.M
        N = self.N
        D = self.D

        Dt = torch.zeros((M, N + 1, 1), device='cuda')  # M x (N+1) x 1
        DW = torch.zeros((M, N + 1, D), device='cuda')  # M x (N+1) x D

        dt = T / N
        Dt[:, 1:, :] = dt
        DW[:, 1:, :] = torch.sqrt(torch.tensor(dt)) * torch.randn((M, N, D), device='cuda')

        t = torch.cumsum(Dt, dim=1)  # M x (N+1) x 1
        W = torch.cumsum(DW, dim=1)  # M x (N+1) x D

        return t, W

    def loss_function(self, t, W, Xi):
        loss = torch.zeros(1, device='cuda')
        X_buffer = []
        Y_buffer = []
        Xi = Xi.cuda()
        t = t.cuda()
        W = W.cuda()

        t0 = t[:, 0, :]  # M x 1
        W0 = W[:, 0, :]  # M x D
        X0 = Xi

        X0.requires_grad = True
        t0.requires_grad = True
        Y0, DuDx0, DuDt0, D2uDx20 = self.net_u_Du(t0, X0)

        X_buffer.append(X0)
        Y_buffer.append(Y0)
        total_weight = 0

        for n in range(0, self.N):
            t1 = t[:, n + 1, :]
            W1 = W[:, n + 1, :]

            if self.gbm_scheme == 0:
                X1 = X0 + self.r * X0 * (t1 - t0) + self.sigma * X0 * (W1 - W0)
            elif self.gbm_scheme == 1:
                X1 = X0 * torch.exp((self.r - 0.5 * self.sigma ** 2) * (t1 - t0) + self.sigma * (W1 - W0))

            t1.requires_grad = True
            Y1, DuDx1, DuDt1, D2uDx21 = self.net_u_Du(t1, X1)

            Y1_tilde = Y0 + self.r * Y0 * (t1 - t0) + DuDx0 * self.sigma * X0 * (W1 - W0)

            loss = loss + torch.sum((Y1 - Y1_tilde) ** 2)
            Y_at_X_0, _1, _2, _3 = self.net_u_Du(t1, X1 * 0)  # M x 1, M x D
            loss += torch.sum((Y_at_X_0 - 0) ** 2)
            total_weight += 1

            t0 = t1
            W0 = W1
            X0 = X1
            Y0 = Y1
            DuDx0 = DuDx1
            DuDt0 = DuDt1
            D2uDx20 = D2uDx21

            X_buffer.append(X0)
            Y_buffer.append(Y0)

        loss = loss + self.lambda_1 * torch.sum((Y1 - self.g_torch(X1, self.K)) ** 2)
        total_weight += self.lambda_1
        loss = loss + self.lambda_2 * torch.sum((DuDx1 - self.Dg_torch(X1)) ** 2)

        total_weight +=self.lambda_2
        loss = loss / total_weight

        X = torch.stack(X_buffer, dim=1)
        Y = torch.stack(Y_buffer, dim=1)

        return loss, X, Y, Y[0, 0, 0]

    def train(self):
        N_Iter = self.epochs
        loss_list = []
        error_list = []

        start_time = time.time()
        t = np.linspace(0, 1, 10)
        S = np.linspace(0, 200, 10)
        test_sample_list = []

        t_mesh, S_mesh = np.meshgrid(t, S)

        mse_list = []
        mae_list = []

        for it in range(N_Iter):
            t_batch, W_batch = self.fetch_minibatch()
            t_batch = torch.tensor(t_batch, dtype=torch.float32, device='cuda')
            W_batch = torch.tensor(W_batch, dtype=torch.float32, device='cuda')

            loss, X_pred, Y_pred, Y0_pred = self.loss_function(t_batch, W_batch, self.Xi)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            loss_list.append(loss.item())


            test_sample_list.append(self.fn_u(self.out_of_sample_input).item())

            if it % 200 == 0:
                0
                clear_output(wait=True)
                elapsed = time.time() - start_time
                print('It: %d, Time: %.2f, Loss: %.3e, Y0: %.3f' % (it, elapsed, loss, Y0_pred))
                start_time = time.time()
                NN_price_surface = np.zeros([10, 10])
                Exact_price_surface = np.zeros([10, 10])

                for i in range(10):
                    for j in range(10):
                        input_tensor = torch.tensor([[t_mesh[i, j], S_mesh[i, j]]], dtype=torch.float32).cuda()
                        NN_price_surface[i, j] = self.fn_u(input_tensor).cpu().detach().numpy()
                        Exact_price_surface[i, j] = theoretical_vanilla_eu(
                            S0=S_mesh[i, j], K=1, T=1 - t_mesh[i, j], r=0.05, sigma=0.4, type_='call'
                        )

                Error_measure = errormeasure(Exact_price_surface, NN_price_surface)
                mse = Error_measure.calculate_mse()
                mae = Error_measure.calculate_mae()
                mape = Error_measure.calculate_mape()
                mse_list.append(mse)
                mae_list.append(mae)



                plt.plot(range(len(loss_list)), np.log10(loss_list))
                plt.show()

                plt.plot(test_sample_list)
                plt.plot(np.ones(len(test_sample_list)) * self.out_of_sample_exact, label='test sample exact price')
                plt.show()

                error_surface = np.abs(NN_price_surface - Exact_price_surface)
                ax = plt.figure()
                ax = plt.axes(projection='3d')
                ax.plot_surface(t_mesh, S_mesh, NN_price_surface, rstride=1, cstride=1, cmap='viridis',
                                edgecolor='none')
                ax.set_title('NN surface: iter = %d' % it)
                plt.show()


        self.loss_list = loss_list
        self.test_sample_list_cpu = test_sample_list
        self.mse_list = mse_list
        self.mae_list = mae_list

    def predict(self, Xi_star, t_star, W_star):
        _, X_star, Y_star, _ = self.loss_function(t_star.cuda(), W_star.cuda(), Xi_star.cuda())

        return X_star, Y_star