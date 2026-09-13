from src_deep_koopman.dynamical_systems import *
from src_deep_koopman.basis_functions import *

## Data Collection and Visualization
sys1 = SystemsAutonomous(name = 'duffing_eqn')
n = sys1.n

traj_length = 100
N = 1000 * traj_length  # Total data points
#X1,Y1 = sys1.gather_data(num_samples= N, plot_results=False)
X2,Y2 = sys1.gather_traj_data(num_traj = N // traj_length, traj_length= traj_length, plot_results=False)
print(" Data gathered. ")

#X = np.vstack((X1,X2))
#Y = np.vstack((Y1,Y2))
X = X2
Y = Y2
print(" Total data points: ", X.shape[0])

X_data = X

plt.scatter(X[:,0], X[:,1], s=1, alpha=0.5, label='Data points')
plt.xlabel('x'); plt.ylabel('y')
plt.legend()
plt.xlim([-3, 3]); plt.ylim([-3, 3])
plt.tight_layout()
plt.grid()
plt.show()
