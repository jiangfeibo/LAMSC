"""
it's used to train a mutual information system
"""

from channel_nets import MutualInfoSystem
import numpy as np
import torch
from matplotlib import pyplot as plt

def sample_batch(batch_size, sample_mode):  # used to sample data for mutual info system
    if sample_mode == "joint":  # joint sample
        index = np.random.choice(range(12799), size=batch_size, replace=False)  # replace = false means it won't select same number
        num = 0
        for i in index:
            x = np.load("MI_data/x1/" + str(i) + ".npy")
            y = np.load("MI_data/y1/" + str(i) + ".npy")
            data_x = x.reshape(-1, 128)  # -1 means python will infer the dimension automatically
            data_y = y.reshape(-1, 128)
            if num == 0:
                batch_x = data_x
                batch_y = data_y
            else:
                batch_x = np.concatenate([batch_x, data_x], axis=0)
                batch_y = np.concatenate([batch_y, data_y], axis=0)
            num += 1
    elif sample_mode == 'marginal':  # marginal sample
        joint_index = np.random.choice(range(12799), size=batch_size, replace=False)
        marginal_index = np.random.choice(range(12799), size=batch_size, replace=False)
        num = 0
        for i in range(batch_size):
            j_index = joint_index[i]
            m_index = marginal_index[i]
            x = np.load("mutual data/x1/" + str(j_index) + ".npy")
            y = np.load("mutual data/y1/" + str(m_index) + ".npy")
            data_x = x.reshape(-1, 128)
            data_y = y.reshape(-1, 128)
            if num == 0:
                batch_x = data_x
                batch_y = data_y
            else:
                batch_x = np.concatenate([batch_x, data_x], axis=0)
                batch_y = np.concatenate([batch_y, data_y], axis=0)
            num += 1
    batch = np.concatenate([batch_x, batch_y], axis=1)  # axis = 1 means that data will concat at the dim of row
    # and if axis = 0, which means that data will concat one by one
    return batch

num_epoch = 400

save_path = './checkpoints/MI.pth'
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("Using: " + str(device).upper())

net = MutualInfoSystem()
net.to(device)
optim = torch.optim.Adam(net.parameters(), lr=0.0001)

muInfo = []
for i in range(num_epoch):
    batch_joint = torch.tensor(sample_batch(40, 'joint')).to(device)
    batch_marginal = torch.tensor(sample_batch(40, 'marginal')).to(device)
    t = net(batch_joint)
    et = torch.exp(net(batch_marginal))
    loss = -(torch.mean(t) - torch.log(torch.mean(et)))

    print('epoch: {}  '.format(i + 1))
    print(-loss.cpu().detach().numpy())
    muInfo.append(-loss.cpu().detach().numpy())

    loss.backward()
    optim.step()
    optim.zero_grad()

torch.save(net.state_dict(), save_path)
plt.title('train mutual info system')
plt.xlabel('Epoch')
plt.ylabel('Mutual Info')
plt.plot(muInfo)
plt.show()
print('All done!')

