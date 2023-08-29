import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from scipy.stats import norm
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

import matplotlib.pyplot as plt
import numpy as np


def theoretical_vanilla_eu(S0=50, K=50, T=1, r=0.05, sigma=0.4, type_='call'):
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



class PlottingTools:
    def __init__(self, test_sample_list, test_sample_exact, model, t_test, X_pred, Y_pred, Y_test, Exact_price_surface, NN_price_surface, t_mesh, S_mesh, S, GPU = False, plot_counter=0):
        self.test_sample_list = test_sample_list
        self.test_sample_exact = test_sample_exact
        self.model = model
        self.t_test = t_test
        self.X_pred = X_pred
        self.Y_pred = Y_pred
        self.Y_test = Y_test
        self.GPU =GPU
        self.plot_counter = 0
        self.Exact_price_surface = Exact_price_surface
        self.NN_price_surface = NN_price_surface
        self.t_mesh = t_mesh
        self.S_mesh = S_mesh
        self.S = S
        self.error_surface=abs(self.NN_price_surface-self.Exact_price_surface)
        self.save_N = model.save_N
        self.epochs = model.epochs
    def save_plot_with_timestamp(cls, fig, fig_name, fig_dir='plots'):
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)

        current_time = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{fig_name}_{current_time}.png"
        fig.savefig(os.path.join(fig_dir, filename),bboxinches='tight')
        # plt.close(fig)



    def plot_1(self, fig_dir='plots'):
        fig=plt.figure(figsize=[9, 6])
        plt.plot(range(len(self.test_sample_list))[0::10],self.test_sample_list[0::10], color='black', label='neural network output price', linewidth=1.8)
        plt.plot(range(len(self.test_sample_list))[0::10],np.ones(len(self.test_sample_list[0::10]))* self.test_sample_exact, color='red',
                 label='test sample exact price', linewidth=1.8)
        plt.xlabel("epochs trained", fontsize=20)
        plt.ylabel("price", fontsize=20)
        plt.title("Convergence of the neural network output price", fontsize=22)
        plt.legend(loc='best', prop={'size': 20})
        plt.grid(True)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        self.save_plot_with_timestamp(fig, "price_convergence")
        plt.show()



    def plot_2(self, fig_dir='plots'):
        fig=plt.figure(figsize=[9, 6])
        loss_values = np.log10(self.model.loss_list[::10])

        # Compute the moving average with a window size of 10 (you can adjust this value)
        window_size = 10
        moving_average = np.convolve(loss_values, np.ones(window_size) / window_size, mode='valid')
        plt.plot(np.linspace(0,self.epochs,len(self.model.loss_list[::10])),np.log10(self.model.loss_list[::10]), 'k-', label='log(loss)')
        plt.plot(np.linspace(0,self.epochs,len(moving_average)), moving_average, 'r--', label='moving average of log(loss)')
        plt.xlabel("epochs trained", fontsize=20)
        plt.ylabel("log(loss)", fontsize=20)
        plt.title("Convergence of the loss", fontsize=22)
        plt.legend(loc='upper right', prop={'size': 20})
        plt.grid(True)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        self.save_plot_with_timestamp(fig, "loss_convergence")
        plt.show()
    def plot_3(self, samples=30, fig_dir='plots'):
        fig=plt.figure(figsize=(9, 6))
        plt.plot(self.t_test[0:1, :, 0].T, self.Y_pred[0:1, :, 0].T, 'r', label=r'learned $\hat{V}(S_t,t)$', linewidth=1.8)
        plt.plot(self.t_test[0:1, :, 0].T, self.Y_test[0:1, :, 0].T, 'k--', label=r'exact $V(S_t,t)$', linewidth=1.8)
        plt.plot(self.t_test[1:samples, :, 0].T, self.Y_pred[1:samples, :, 0].T, 'r')
        plt.plot(self.t_test[1:samples, :, 0].T, self.Y_test[1:samples, :, 0].T, 'k--')
        plt.plot(self.t_test[0:1, -1, 0], self.Y_test[0:1, -1, 0], 'ko', label=r'$V_T = V(S_T,T)$')
        plt.plot(self.t_test[1:samples, -1, 0], self.Y_test[1:samples, -1, 0], 'ko')
        plt.plot([0], self.Y_test[0, 0, 0], 'ks', label=r'$V_0 = V(S_0,0)$')
        plt.xlabel(r'$t$', fontdict={'fontsize': 20})
        plt.ylabel(r'$V_t$', fontdict={'fontsize': 20})
        plt.title(r'Path of exact $V(S_t,t)$ and learned $\hat{V}(S_t,t)$', fontdict={'fontsize': 22})
        plt.legend(loc='upper left', prop={'size': 20})
        plt.grid(True)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.tick_params(axis='both', which='both', labelsize=18)
        self.save_plot_with_timestamp(fig, "Path")
        plt.show()

    def plot_4(self, fig_dir='plots',xlabel = 'time ($t$)', ylabel = 'price ($S_t$)', title = 'Neural network price surface'):
        percentage_error_surface = np.where(self.Exact_price_surface != 0, np.abs(self.error_surface) / self.Exact_price_surface * 100, 0)
        fig = plt.figure(figsize=(6, 6))
        ax = plt.axes(projection='3d')
        surf = ax.plot_surface(self.t_mesh, self.S_mesh, self.NN_price_surface, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
        ax.set_title('Neural network price surface',fontsize=22)
        ax.set_xlabel(xlabel,fontsize=18)
        ax.set_ylabel(ylabel,fontsize=18)
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
        ax.tick_params(axis='z', labelsize=16)
        self.save_plot_with_timestamp(fig, "NN surface")
        plt.show()


    def plot_5(self, fig_dir='plots',xlabel = 'time ($t$)', ylabel = 'price ($S_t$)', title = 'Exact price surface'):
        fig = plt.figure(figsize=(6, 6))
        ax = plt.axes(projection='3d')
        surf = ax.plot_surface(self.t_mesh, self.S_mesh, self.Exact_price_surface, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
        ax.set_title('Exact price surface',fontsize=22)
        ax.set_xlabel(xlabel,fontsize=18)
        ax.set_ylabel(ylabel,fontsize=18)
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
        ax.tick_params(axis='z', labelsize=16)

        self.save_plot_with_timestamp(fig, "Exact surface")
        plt.show()
        self.error_surface = self.error_surface



    def plot_6(self, fig_dir='plots'):
        errors = np.sqrt((self.Y_test - self.Y_pred) ** 2)
        mean_errors = np.mean(errors, 0)
        std_errors = np.std(errors, 0)
        fig=plt.figure(figsize=(9, 6))
        plt.plot(self.t_test[0, :, 0], mean_errors, 'k', label='mean', linewidth=1.8)
        plt.plot(self.t_test[0, :, 0], mean_errors + 2 * std_errors, 'r--', label='mean + two standard deviations',
                 linewidth=1.8)
        plt.xlabel(r'$t$', fontsize=20)
        plt.ylabel('absolute error', fontsize=20)
        plt.title('Neural network absolute error', fontsize=22)
        plt.legend(fontsize=20)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.grid(True)
        self.save_plot_with_timestamp(fig, "relative error")
        plt.show()


    def plot_7(self, fig_dir='plots',xlabel = 'time ($t$)', ylabel = 'price ($S_t$)', title = 'Error surface'):

        # ax = plt.figure()
        fig=plt.figure(figsize=(6, 6))
        ax = plt.axes(projection='3d')
        ax.plot_surface(self.t_mesh, self.S_mesh, self.error_surface, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
        ax.set_title('Error surface',fontsize=22)
        ax.set_xlabel(xlabel,fontsize=18)
        ax.set_ylabel(ylabel,fontsize=18)
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)
        ax.tick_params(axis='z', labelsize=16)
        self.save_plot_with_timestamp(fig, "error surface")
        plt.show()


    def plot_8(self, fig_dir='plots'):
        mse_list = self.model.mse_list
        mae_list = self.model.mae_list
        fig=plt.figure(figsize=(9, 6))
        plt.plot(self.save_N*np.array(range(len(mse_list))),np.log10(mse_list), color='red', label='log(MSE)', linewidth=1.8)
        plt.plot(self.save_N*np.array(range(len(mae_list))),np.log10(mae_list), color='black', label='log(MAE)', linewidth=1.8)
        plt.xlabel("epochs trained", fontsize=20)
        plt.ylabel("error", fontsize=20)
        plt.title("Convergence of MSE and MAE", fontsize=22)
        plt.legend(fontsize=20)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.grid(True)
        self.save_plot_with_timestamp(fig, "MSE convergence")
        plt.show()

    def plot_9(self, samples=30, fig_dir='plots'):
        fig=plt.figure(figsize=(9, 6))
        plt.plot(self.t_test[0:1, :, 0].T, self.X_pred[0:1, :, 0].T, color='black', label=r'${S_t}(t)$', linewidth=1.8)
        plt.plot(self.t_test[1:samples, :, 0].T, self.X_pred[1:samples, :, 0].T, color = 'black')
        plt.xlabel(r'$t$', fontdict={'fontsize': 20})
        plt.ylabel(r'$S_t$', fontdict={'fontsize': 20})
        plt.title(r'Path of ${S_t}(t)$', fontdict={'fontsize': 22})
        plt.legend(loc='upper left', prop={'size': 20})
        plt.grid(True)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.tick_params(axis='both', which='both', labelsize=18)
        self.save_plot_with_timestamp(fig, "Path of st")
        plt.show()

    def plot_10(self, fig_dir='plots'):
     # Extract the 2D columns from Exact_price_surface at the specified time points
        extracted_columns_exact = self.Exact_price_surface[:, 0]
        extracted_columns_NN = self.NN_price_surface[:, 0]
        fig = plt.figure(figsize=(9, 6))
        plt.plot(self.S, extracted_columns_exact, 'k-', label="exact", linewidth=1.8)
        plt.plot(self.S, extracted_columns_NN, 'r--', label="neural network", linewidth=1.8)
        plt.xlabel('underlying asset price (S)', fontsize=20)
        plt.ylabel('option price (V)', fontsize=20)
        plt.title('Option price to underlying asset price at time 0', fontsize=22)
        plt.legend(fontsize=20)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.grid(True)
        self.save_plot_with_timestamp(fig, "crossection_0")
        plt.show()

    def plot_11(self, fig_dir='plots'):
    # Extract the 2D columns from Exact_price_surface at the specified time points
        extracted_columns_exact = self.Exact_price_surface[:, 5]
        extracted_columns_NN = self.NN_price_surface[:, 5]
        fig = plt.figure(figsize=(9, 6))
        plt.plot(self.S, extracted_columns_exact, 'k-', label="exact", linewidth=1.8)
        plt.plot(self.S, extracted_columns_NN, 'r--', label="neural network", linewidth=1.8)
        plt.xlabel('underlying asset price (S)', fontsize=20)
        plt.ylabel('option price (V)', fontsize=20)
        plt.title('Option price to underlying asset price at time 0.5', fontsize=22)
        plt.legend(fontsize=20)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.grid(True)
        self.save_plot_with_timestamp(fig, "crossection_5")


