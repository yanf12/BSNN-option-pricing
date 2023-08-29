import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from plotting import newfig, savefig
from torch.optim.lr_scheduler import StepLR
from scipy.special import comb
from scipy.stats import norm
from common_tools import neural_networks
from common_tools import plotplot
from IPython.display import display, clear_output
from common_tools.neural_networks import FBSNN

np.random.seed(42)
torch.manual_seed(42)
import os
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'
os.environ['KMP_DUPLICATE_LIB_OK']='True'

if __name__ == '__main__':
    M = 10 # number of trajectories (batch size)
    N = 10 # number of time snapshots
    learning_rate = 5*1e-3
    save_N = 200  # save every N epochs
    epochs = 500

    r = 0.05
    K = 100.0
    T = 1.0
    sigma = 0.4
    D = 1  # number of dimensions
    lambda_1 = 10 # weight for BC
    lambda_2 = 0 # weight for IC
    lambda_3 = 0 # weight for BC
    out_of_sample_test_t = 0.0
    out_of_sample_test_S = 100.0


    out_of_sample_input = torch.tensor([out_of_sample_test_t, out_of_sample_test_S]).float()
    test_sample_exact = neural_networks.theoretical_vanilla_eu(out_of_sample_test_S, K, T-out_of_sample_test_t, r, sigma, type_='call' )
    gbm_scheme = 1 # in theory 1 is more accurate. 0 is accurate for large N

    if D==1:
        Xi = torch.tensor([np.linspace(0,200,M)]).transpose(-1,-2).float()
    else:
        Xi = torch.from_numpy(np.array([100.0, 50] * int(D / 2))[None, :]).float()

    model = FBSNN(r,K,sigma,Xi, T, M, N, D, learning_rate,gbm_scheme=1,lambda_1=lambda_1,lambda_2=lambda_2,lambda_3=lambda_3,out_of_sample_input=out_of_sample_input,out_of_sample_exact = test_sample_exact,width_list=[1024,512,256],num_layers=3,activation=torch.nn.ReLU(),save_N=save_N,epochs=epochs)
    model.train()

#%%
    # model.train(N_Iter=10000)
#%%


    samples =10
    model.M =100
    M=100
    model.Xi = torch.tensor(np.ones([M,1])).transpose(-1,-2).float()
    t_test, W_test = model.fetch_minibatch()
    X_pred, Y_pred = model.predict(torch.tensor([np.linspace(95,105,M)]).transpose(-1,-2).float(), t_test, W_test)
    test_sample_list = model.test_sample_list


    def u_exact(t, X):  # (N+1) x 1, (N+1) x D
        r = 0.05
        sigma = 0.4
        K = 100
        T = 1
        res = np.zeros([t.shape[0], X.shape[1]])
        for i in range(t.shape[0]):
            for j in range(X.shape[1]):
                res[i, j] = neural_networks.theoretical_vanilla_eu(S0=X[i, j], K=K, T=T-t[i, 0], r=r, sigma=sigma, type_='call')
        return   res
#%%
    t_test = t_test.detach().numpy()
    X_pred = X_pred.detach().numpy()
    Y_pred = Y_pred.detach().numpy()
    Y_test = np.reshape(u_exact(np.reshape(t_test[0:M, :, :], [-1, 1]), np.reshape(X_pred[0:M, :, :], [-1, D])),
                        [M, -1, 1])
    print(Y_test[0, 0, 0])
    t = np.linspace(0, 1, 10)
    S = np.linspace(0, 200, 10)
    t_mesh, S_mesh = np.meshgrid(t, S)
    NN_price_surface = np.zeros([10, 10])
    Exact_price_surface = np.zeros([10, 10])

    for i in range(10):
        for j in range(10):
            NN_price_surface[i, j] = model.fn_u(
                    torch.tensor([[t_mesh[i, j], S_mesh[i, j]]]).float()).detach().numpy()
            Exact_price_surface[i, j] = model.theoretical_vanilla_eu(S0=S_mesh[i, j], K=100,
                                                                              T=1 - t_mesh[i, j], r=0.05,
                                                                              sigma=0.4, type_='call')
#%%
    import importlib

    importlib.reload(plotplot)
    plotter = plotplot.PlottingTools(test_sample_list, test_sample_exact, model, t_test, X_pred, Y_pred, Y_test, Exact_price_surface, NN_price_surface, t_mesh, S_mesh, S)
    plotter.plot_1(fig_dir="plots")
    plotter.plot_2(fig_dir="plots")
    plotter.plot_3(samples=samples, fig_dir="plots")
    plotter.plot_4(fig_dir="plots")
    plotter.plot_5(fig_dir="plots")
    plotter.plot_6(fig_dir="plots")
    plotter.plot_7(fig_dir="plots")
    plotter.plot_8(fig_dir="plots")
    plotter.plot_9(samples=samples, fig_dir="plots")
    plotter.plot_10(fig_dir="plots")
    plotter.plot_11(fig_dir="plots")
#%%
    # import numpy as np
    #
    # # Assuming model.mse_list is a Python list
    # mse_array_adam = np.array(model.mse_list)
    # # Save the NumPy array to a file using np.save
    # np.save('mse_list_adam.npy', mse_array_adam)
    # mae_array_adam = np.array(model.mae_list)
    # np.save('mae_list_adam.npy', mae_array_adam)
#%%
    # mse_print = model.mse_list[-1]
    # mae_print = model.mae_list[-1]
    # print(mse_print)
    # print(mae_print)

#%%
    # import pickle
    #
    # # Assuming you have a trained model named 'model'
    # # Save the model to a file using pickle.dump()
    # with open('trained_model_1d_+-2.pkl', 'wb') as file:
    #     pickle.dump(model, file)



#%%plot1
#     plt.figure(figsize=[10, 6])
#
#     plt.plot(test_sample_list[100:], color='black', label='NN output price', linewidth=1.8)
#     plt.plot(np.ones(len(test_sample_list))[100:] * test_sample_exact, color='red', label='test sample exact price',
#              linewidth=1.8)
#     # Axis labels and title
#     plt.xlabel("Epochs trained", fontsize=18)
#     plt.ylabel("Price", fontsize=18)
#     plt.title("Convergence of the price", fontsize=20)
#     # Legend
#     plt.legend(loc='best', prop={'size': 12})
#     # Grid lines
#     plt.grid(True)
#     # Adjust tick label font size
#     plt.xticks(fontsize=16)
#     plt.yticks(fontsize=16)
#     # Remove top and right spines
#     plt.gca().spines['top'].set_visible(False)
#     plt.gca().spines['right'].set_visible(False)
#     # Show the plot
#     plt.show()
#
# #%%plot2
#     plt.figure(figsize=[10, 6])
#     plt.plot(np.log10(model.loss_list), 'k', label='NN output loss')
#     # Axis labels and title
#     plt.xlabel("Epochs trained", fontsize=18)
#     plt.ylabel("Loss", fontsize=18)
#     plt.title("Convergence of the loss", fontsize=20)
#     # Legend
#     plt.legend(loc='upper right', prop={'size': 12})
#     # Grid lines
#     plt.grid(True)
#     # Remove top and right spines
#     plt.gca().spines['top'].set_visible(False)
#     plt.gca().spines['right'].set_visible(False)
#     # Adjust tick label font size
#     plt.xticks(fontsize=16)
#     plt.yticks(fontsize=16)
#     # Show the plot
#     plt.show()
# #%%plot 3
#     samples = 30
#     plt.figure(figsize=(10, 6))  # Adjust the figure size as desired
#     plt.plot(t_test[0:1, :, 0].T, Y_pred[0:1, :, 0].T, 'r', label=r'Learned $\hat{V}(S_t,t)$',linewidth=1.8)
#     plt.plot(t_test[0:1, :, 0].T, Y_test[0:1, :, 0].T, 'k--', label=r'Exact $V(S_t,t)$',linewidth=1.8)
#     plt.plot(t_test[1:samples, :, 0].T, Y_pred[1:samples, :, 0].T, 'r')
#     plt.plot(t_test[1:samples, :, 0].T, Y_test[1:samples, :, 0].T, 'k--')
#     # Plotting the markers
#     plt.plot(t_test[0:1, -1, 0], Y_test[0:1, -1, 0], 'ko', label=r'$V_T = V(S_T,T)$')
#     plt.plot(t_test[1:samples, -1, 0], Y_test[1:samples, -1, 0], 'ko')
#     plt.plot([0], Y_test[0, 0, 0], 'ks', label=r'$V_0 = V(S_0,0)$')
#
#     # Axis labels and title
#     plt.xlabel(r'$t$', fontdict={'fontsize': 18})  # Adjust the fontsize as desired
#     plt.ylabel(r'$V_t$', fontdict={'fontsize': 18})  # Adjust the fontsize as desired
#     plt.title(r'Path of exact $V(S_t,t)$ and learned $\hat{V}(S_t,t)$',
#               fontdict={'fontsize': 20})  # Adjust the fontsize as desired
#
#     # Legend
#     plt.legend(loc='upper left', prop={'size': 12})
#     # Grid lines
#     plt.grid(True)
#     # Remove top and right spines
#     plt.gca().spines['top'].set_visible(False)
#     plt.gca().spines['right'].set_visible(False)
#     plt.tick_params(axis='both', which='both', labelsize=16)
#     # Show the plot
#     plt.show()
    # %%plot4
    # t = np.linspace(0, 1, 10)
    # S = np.linspace(0, 2, 10)
    # t_mesh, S_mesh = np.meshgrid(t, S)
    # NN_price_surface = np.zeros([10, 10])
    # Exact_price_surface = np.zeros([10, 10])
    # for i in range(10):
    #     for j in range(10):
    #         NN_price_surface[i, j] = model.fn_u(torch.tensor([[t_mesh[i, j], S_mesh[i, j]]]).float()).detach().numpy()
    #         Exact_price_surface[i, j] = neural_networks.theoretical_vanilla_eu(S0=S_mesh[i, j], K=1, T=1 - t_mesh[i, j], r=0.05,
    #                                                            sigma=0.4, type_='call')
    # error_surface = NN_price_surface - Exact_price_surface
    # # Calculate percentage error, avoiding division by zero
    # percentage_error_surface = np.where(Exact_price_surface != 0, np.abs(error_surface) / Exact_price_surface * 100, 0)
    # fig = plt.figure(figsize=(16, 12))
    #
    # # First subplot for Neural Network price surface
    # ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    # surf = ax1.plot_surface(t_mesh, S_mesh, NN_price_surface, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    # ax1.set_title('Neural Network Price Surface')
    # ax1.set_xlabel('Time ($t$)')
    # ax1.set_ylabel('Price ($S_t$)')
    # # fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=5)  # add color bar
    #
    # # Second subplot for Exact price surface
    # ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    # surf = ax2.plot_surface(t_mesh, S_mesh, Exact_price_surface, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    # ax2.set_title('Exact Price Surface')
    # ax2.set_xlabel('Time ($t$)')
    # ax2.set_ylabel('Price ($S_t$)')
    # fig.colorbar(surf, ax=ax2, shrink=0.5, aspect=5)  # add color bar
#
#     # Third subplot for Error surface
#     ax3 = fig.add_subplot(2, 2, 3, projection='3d')
#     surf = ax3.plot_surface(t_mesh, S_mesh, np.abs(error_surface), rstride=1, cstride=1, cmap='viridis',
#                             edgecolor='none')
#     ax3.set_title('Absolute Error Surface')
#     ax3.set_xlabel('Time ($t$)')
#     ax3.set_ylabel('Price ($S_t$)')
#     # fig.colorbar(surf, ax=ax3, shrink=0.5, aspect=5)  # add color bar
#
#     plt.show()
#     #%%plot5
#     errors = np.sqrt((Y_test - Y_pred) ** 2)
#     mean_errors = np.mean(errors, 0)
#     std_errors = np.std(errors, 0)
#     plt.figure(figsize=(10, 6))
#     plt.plot(t_test[0, :, 0], mean_errors, 'k', label='mean', linewidth=1.8)
#     plt.plot(t_test[0, :, 0], mean_errors + 2 * std_errors, 'r--', label='mean + two standard deviations',
#              linewidth=1.8)
#     plt.xlabel(r'$t$', fontsize=18)
#     plt.ylabel('relative error', fontsize=18)
#     plt.title('100-dimensional Black-Scholes-Barenblatt', fontsize=20)
#     plt.legend(fontsize=14)
#     plt.xticks(fontsize=14)
#     plt.yticks(fontsize=14)
#     plt.grid(True)
#     plt.show()
#     #%%plot6
#     error_surface = NN_price_surface- Exact_price_surface
#     ax = plt.figure()
#     ax = plt.axes(projection='3d')
#     ax.plot_surface(t_mesh, S_mesh, error_surface, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
#     ax.set_title('Error surface')
#     plt.show()
# #%%plot7
#     mse_list = model.mse_list
#     mae_list = model.mae_list
#     plt.figure(figsize=(10, 6))
#     plt.plot(mse_list[100:], color='red', label='mse', linewidth=1.8)
#     plt.plot(mae_list[100:], color='black', label='mae', linewidth=1.8)
#     plt.xlabel("Epochs trained", fontsize=18)
#     plt.ylabel("Error", fontsize=18)
#     plt.title("Convergence of MSE and MAE", fontsize=20)
#     plt.legend(fontsize=14)
#     plt.xticks(fontsize=14)
#     plt.yticks(fontsize=14)
#     plt.grid(True)
#     plt.show()




