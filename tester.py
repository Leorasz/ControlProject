import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt

dev = "mps"
b_func = nn.Sequential(
    nn.Linear(1, 100),
    nn.ReLU(),
    nn.Linear(100,200),
    nn.ReLU(),
    nn.Linear(200,200),
    nn.ReLU(),
    nn.Linear(200,200),
    nn.ReLU(),
    nn.Linear(200,200),
    nn.ReLU(),
    nn.Linear(200,1)
).to(dev)

g_func = nn.Sequential(
    nn.Linear(1, 100),
    nn.ReLU(),
    nn.Linear(100,200),
    nn.ReLU(),
    nn.Linear(200,200),
    nn.ReLU(),
    nn.Linear(200,200),
    nn.ReLU(),
    nn.Linear(200,200),
    nn.ReLU(),
    nn.Linear(200,1)
).to(dev)

b_func.load_state_dict(torch.load("b_func_final1.pt"))
g_func.load_state_dict(torch.load("g_func_final1.pt"))

min_bound = torch.tensor([16.5], dtype=torch.float32, device=dev)
max_bound = torch.tensor([28], dtype=torch.float32, device=dev)
def heater(t, u, w):
    unchange = t.reshape(-1) + 5*((8e-3*(15-t.reshape(-1)))+3.6e-3*(45-t.reshape(-1))*u.reshape(-1))#+w.reshape(-1)
    return torch.max(torch.min(unchange, max_bound), min_bound)

p = 0.3
delta = 0.5 
beta = 0.005/3
betas = 1
mhat = 1.5e2
nhat = int(mhat / (delta*delta*betas))
print(nhat)
n = 1_0000
epsilon = (28-16.5)/n
std = 0.05
full_range = torch.arange(16.5, 28, epsilon).to(dev)
full_range = full_range[torch.randperm(full_range.shape[0])].reshape(-1,1)
ins = torch.arange(22,23,epsilon).to(dev)
ins = ins[torch.randperm(ins.shape[0])].reshape(-1,1)
uns = torch.cat((torch.arange(16.5,17.5, epsilon),torch.arange(27,28, epsilon))).to(dev)
uns = uns[torch.randperm(uns.shape[0])].reshape(-1,1)
ws = 0.05*torch.randn((int(nhat))).to(dev)

# plt.scatter(full_range.reshape(-1).tolist()[:1000], b_func(full_range).reshape(-1).tolist()[:1000])
# plt.show()
# exit()
# plt.scatter(ins, outs)
# plt.show()
def findL(x):
    res = 1
    for layer in x:
        if hasattr(layer, 'weight'):
            # res *= layer.weight.abs().max()
            print(layer.bias.abs().max())
    return res

print(findL(g_func))
b_func.eval()
g_func.eval()
spot = torch.tensor([24.0], device=dev).reshape(1,1)
step_shift = 0
step_size = 1
zero = torch.tensor([0], dtype=torch.float32).to(dev)
while True:
    gout = torch.where(g_func(spot) > step_shift, step_size, zero).reshape(-1)
    print(spot.item())
    print(b_func(spot).item())
    spot = heater(spot, gout, 0)
    _ = input(":")
# dots = torch.arange(22,23, 1.0).to(dev)
# paths = [[] for _ in range(1)]
# paths2 = [[] for _ in range(1)]
# with torch.no_grad():
#     testins = dots
#     for i in tqdm(range(10000)):
#         [paths[ii].append(i) for ii, i in enumerate(b_func(testins.reshape(-1,1)).reshape(-1).tolist())] 
#         [paths2[ii].append(i) for ii, i in enumerate(testins.reshape(-1).tolist())] 
#         testins = heater(testins.reshape(-1), g_func(testins.reshape(-1,1)).reshape(-1).detach(), 0.05*torch.randn((testins.shape[0])).to(dev).reshape(-1)).reshape(-1,1)

# for path in paths:
#     plt.plot(path)
# print(path)
# plt.show()

# for path in paths2:
#     plt.plot(path)
# print(path)
# plt.show()
