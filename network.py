import time
import json
import torch
import torch.nn as nn
from tqdm import tqdm
import optuna

counter = 1
dev = "cpu"

def objective(trial):

    g, t, m, l = 9.8, 0.01, 1, 1

    def updatexs(x, u):
        x1 = x[:, 0]
        x2 = x[:, 1]
        x11 = x1 + t*x2
        x22 = x2 + t*((g/l)*torch.sin(x1)+(1/(m*l**2))*u.reshape(-1))
        return torch.cat((x11.reshape(-1,1),x22.reshape(-1,1)), dim=1)

    b_func = nn.Sequential(
        nn.Linear(2,20),
        nn.ReLU(),
        nn.Linear(20,1)
    ).to(dev)

    g_func = nn.Sequential(
        nn.Linear(2,10),
        nn.ReLU(),
        nn.Linear(10,1),
        nn.Hardtanh(min_val=-10,max_val=10)
    ).to(dev)

    b_func.load_state_dict(torch.load("b_func.pt"))
    g_func.load_state_dict(torch.load("g_func.pt"))

    ehat = 0.007
    lmax = 4.2
    eta = ehat*lmax

    examples = torch.load("examples.pt").to(dev)
    sett = json.load(open("sett.json","r"))

    batch_size = trial.suggest_int('batch_size', 64, 1024)
    batches = torch.split(examples, batch_size)
    lbatches = [sett[i:min(len(sett),i+batch_size)] for i in range(0,len(sett),batch_size)]
    ibatches = []
    ubatches = []
    for lbatch in lbatches:
        ibatch = []
        ubatch = []
        for i in range(len(lbatch)):
            if lbatch[i] == "initial":
                ibatch.append(i)
            elif lbatch[i] == "unsafe":
                ubatch.append(i)
        ibatches.append(torch.tensor(ibatch).to(dev))
        ubatches.append(torch.tensor(ubatch).to(dev))

    blr = trial.suggest_float('blr', 3e-5, 3e-2, log=True)
    glr = trial.suggest_float('glr', 3e-5, 3e-2, log=True)
    bopt = torch.optim.Adam(b_func.parameters(), lr=blr)
    gopt = torch.optim.Adam(g_func.parameters(), lr=glr)

    lb=2
    lg=22
    zero = torch.tensor([0], dtype=torch.float32).to(dev)
    a = trial.suggest_float('a', 0.1, 2)
    b = trial.suggest_float('b', 0.1, 2)
    c = trial.suggest_float('c', 0.1, 2)
    c /= a+b+c
    b /= a+b+c
    a /= a+b+c
    epochs = trial.suggest_int('epochs',5,100)
    for _ in tqdm(range(epochs)):
        for batch, ibatch, ubatch in zip(batches, ibatches, ubatches):
            bopt.zero_grad()
            gopt.zero_grad()
            llb = torch.max(list(b_func.parameters())[0])*torch.max(list(b_func.parameters())[2])
            if llb > lb:
                print(f"Lipschitz constant of barrier too high")
                return 3
            llg = torch.max(list(g_func.parameters())[0])*torch.max(list(g_func.parameters())[2])
            if llg > lg:
                print(f"Lipschitz constant of controoler too high")
                return 3
            b_res = b_func(batch)
            if ibatch.numel():
                loss0 = torch.mean(torch.max(b_res[ibatch]+eta,zero))
            else:
                loss0 = 0
            if ubatch.numel():
                loss1 = torch.mean(torch.max(-b_res[ubatch]+eta,zero))
            else:
                loss1 = 0
            u = g_func(batch)
            new_ex = updatexs(batch, u)
            new_b_res = b_func(new_ex)
            loss2 = torch.mean(torch.max(new_b_res-b_res+eta,zero))
            loss = a*loss0 + b*loss1 + c*loss2
            loss.backward()
            gopt.step()
            bopt.step()
    
    inits = torch.cat([batch[ibatch] for ibatch, batch in zip(ibatches,batches) if ibatch.numel()])
    unsfs = torch.cat([batch[ubatch] for ubatch, batch in zip(ubatches,batches) if ubatch.numel()])
    usf = False
    with torch.no_grad():
        o = inits
        b = b_func(o)
        if torch.sum(b > -eta).item() > 0:
            print(f"condition 1 fail {torch.sum(b > -eta).item()/inits.shape[0]}")
            usf = True
        else:
            print("condition 1 success")
        bu = b_func(unsfs)
        if torch.sum(bu < eta).item() > 0:
            print(f"condition 2 fail {torch.sum(bu < eta).item()/unsfs.shape[0]}")
            usf = True
        else:
            print("condition 2 success")
        b2 = b_func(updatexs(o, g_func(o)))
        if torch.sum(b2-b > -eta).item() > 0:
            print(f"condition 3 fail {torch.sum(b2-b>-eta).item()/inits.shape[0]}")
            usf = True
        else:
            print("condition 3 success")
        if torch.sum(b > -eta).item()/inits.shape[0]**2 + torch.sum(bu < eta).item()/unsfs.shape[0]**2 > 0:
            score = torch.sum(b > -eta).item()/inits.shape[0]**2 + torch.sum(bu < eta).item()/unsfs.shape[0]**2 + 1
        else:
            score = torch.sum(b2-b>-eta).item()/inits.shape[0]
    if usf == False:
        torch.save(b_func.state_dict(), "b_func1.pt")
        torch.save(g_func.state_dict(), "g_func1.pt")
        print("Success!!")
        print(b_func(inits)[:10])
        print((b_func(updatexs(inits, g_func(inits)))-b_func(inits))[:10])
        exit()
        
    return score

study = optuna.create_study()
start = time.time()
study.optimize(objective, n_trials = 1_0000)
end = time.time()
print(f"{end-start} elapsed")

