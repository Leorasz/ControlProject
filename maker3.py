
import torch 
import torch.nn as nn
from tqdm import tqdm

dev = "mps"

min_bound = torch.tensor([16.5], dtype=torch.float32, device=dev)
max_bound = torch.tensor([28], dtype=torch.float32, device=dev)
def heater(t, u, w):
    unchange = t.reshape(-1) + 5*((8e-3*(15-t.reshape(-1)))+3.6e-3*(45-t.reshape(-1))*u.reshape(-1))+w.reshape(-1)
    return torch.max(torch.min(unchange, max_bound), min_bound)

#constants used in paper
p = 0.2
delta = 2
beta = 0.005/3
betas = 0.045
mhat = 150
nhat = int(mhat/(delta*delta*betas))
print(nhat)
n = 900
horizon = 60
epsilon = (28-16.5)/n
std = 0.05
full_range = torch.arange(16.5, 28, epsilon).to(dev)
full_range = full_range[torch.randperm(full_range.shape[0])].reshape(-1,1)
full_range_hole = torch.cat((torch.arange(16.5,22.25, epsilon),torch.arange(22.75,28, epsilon))).to(dev)
full_range_hole = full_range_hole[torch.randperm(full_range_hole.shape[0])].reshape(-1,1)
batch_size = 256
batches = torch.split(full_range_hole, batch_size)
ins = torch.arange(22,23,epsilon).to(dev) #input states
ins = ins[torch.randperm(ins.shape[0])].reshape(-1,1)
uns = torch.cat((torch.arange(16.5,17.5, epsilon),torch.arange(27,28, epsilon))).to(dev) #unsafe states
uns = uns[torch.randperm(uns.shape[0])].reshape(-1,1)

class u(nn.Module):
    def forward(self, x):
        return torch.where(x >= 0, torch.tensor(1.0, device=x.device), torch.tensor(0.0, device = x.device))
# ws = 0.05*torch.randn((int(nhat))).to(dev)

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

b_func.load_state_dict(torch.load("b_func_final1.pt", weights_only=True))
g_func.load_state_dict(torch.load("g_func_final1.pt", weights_only=True))

#arbitrary initializations for variables that will be optimized
eta = torch.tensor([-0.001], dtype=torch.float32, device =dev)
lamba = torch.tensor([39.0004768371582], dtype=torch.float32, requires_grad=True, device = dev)
c = torch.tensor([0.10620000213384628], dtype=torch.float32, requires_grad=True, device = dev)
step_shift = torch.tensor([0], dtype=torch.float32, requires_grad=True, device = dev)
step_size = torch.tensor([1], dtype=torch.float32, requires_grad=True, device = dev)
lopt = torch.optim.Adam([lamba], lr=3e-4)
copt = torch.optim.Adam([c], lr=3e-4)
bopt = torch.optim.Adam(b_func.parameters(), lr=3e-4)
gopt = torch.optim.Adam(g_func.parameters(), lr=3e-4)
opts = [lopt, bopt, gopt,copt]

ss = 1000
gamma = 0.5
scheduler1 = torch.optim.lr_scheduler.StepLR(bopt, step_size=ss, gamma=gamma)
scheduler2 = torch.optim.lr_scheduler.StepLR(gopt, step_size=ss, gamma=gamma)

def findL(x):
    res = 1
    for layer in x:
        if hasattr(layer, 'weight'):
            res *= layer.weight.max()
    return res


epochs = 90000
lx = 12
lu = 12
h = 60
zero = torch.tensor([0], dtype=torch.float32).to(dev)
# g5 = torch.tensor([0], dtype=torch.float32).to(dev)
# outputs = []
# for batch in batches:
#     gout = torch.where(g_func(batch) > step_shift, step_size, zero) 
#     expect_in = heater(batch.repeat(1,nhat), gout.repeat(1,nhat), std*torch.randn((batch.shape[0], nhat)).to(dev)).reshape(-1,nhat,1)
#     output = b_func(expect_in).reshape(-1)
#     outputs += output.tolist()
#     expect_out = torch.mean(b_func(expect_in).reshape(-1,nhat), dim=1)
# g5 /= len(batches)
# g5 = torch.mean(g5).item()
# variance = 0
# for i in outputs:
#     variance += (i-g5)**2
# variance /= len(outputs)
# print(variance)
# exit()
for e in tqdm(range(epochs)):
    g1 = -b_func(full_range).reshape(-1) - eta
    g2 = b_func(ins).reshape(-1) - 1 - eta
    g3 = -b_func(uns).reshape(-1) + lamba - eta
    g4 = (1+c*horizon)/p - lamba - eta
    #g5
    g5 = torch.tensor([0], dtype=torch.float32).to(dev)
    for batch in batches:
        gout = torch.where(g_func(batch) > step_shift, step_size, zero) 
        expect_in = heater(batch.repeat(1,nhat), gout.repeat(1,nhat), std*torch.randn((batch.shape[0], nhat)).to(dev)).reshape(-1,nhat,1)
        expect_out = torch.mean(b_func(expect_in).reshape(-1,nhat), dim=1)
        print([(a,b) for a,b in zip(b_func(batch).reshape(-1).tolist()[:10],expect_out.tolist()[:10])])
        g5 += torch.mean(expect_out - b_func(batch).reshape(-1) - c + delta - eta)
    g5 /= len(batches)
    gs = [g1,g2,g3,g4,g5]
    for i in range(5):
        gs[i] = torch.mean(torch.max(gs[i], zero))#negslopes[i]*gs[i]))
    stack = torch.stack(gs)
    loss = torch.sum(stack).reshape(1)
    loss += torch.max(-c,zero)
    loss += 0.5*torch.max(-lamba+1, zero)
    if (loss).item() == 0:
        print("Success!!")
        torch.save(b_func.state_dict(), "b_func_final1.pt")
        torch.save(g_func.state_dict(), "g_func_final1.pt")
        print(c)
        print(lamba)
        break
    if e % 100 == 0:
        print(f"the loss for epoch {e} is {loss.item()}")
        for i in range(5):
            print(f"Loss {i+1} is {gs[i]}")
        print(f"Loss 6 (eta) is {eta.item()}")
        # print(f"Loss 7 is {torch.max(zero, eta+lb*epsilon*(lx+lu*lg +1)).item()}")
        print(f"Lamba is {lamba.item()}")
        print(f"C is {c.item()}")
        print(f"Step shift is {step_shift.item()}")
        print(f"Step size is {step_size.item()}")
        torch.save(b_func.state_dict(), "b_func_check.pt")
        torch.save(g_func.state_dict(), "g_func_check.pt")
    for opt in opts: opt.zero_grad()
    loss.backward()
    for opt in opts: opt.step()
    scheduler1.step()
    scheduler2.step()


print("Done")
torch.save(b_func.state_dict(), "b_func_mid3.pt")
torch.save(g_func.state_dict(), "g_func_mid3.pt")
print(eta)
print(c)
print(lamba)