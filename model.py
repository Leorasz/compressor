
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(1337)
torch.manual_seed(1337)
random.seed(1337)

"""
TODO

Clean up code
Better variable names
Immediately convert to binary
"""
print("Finished imports")

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read() # get text

tokens = sorted(list(set("".join(text))))
vocab_size = len(tokens) + 1
cid = { ch:i for i,ch in enumerate(tokens)}
icd = { i:ch for i,ch in enumerate(tokens)}

end = [0] * vocab_size
end[-1] = 1

encode = lambda s: [cid[c] for c in s]
decode = lambda l: "".join([icd[i] for i in l])

limit = 10000
text = text[:limit]
inputy = []
t = []
double = False
wTFN = False
recording = False
record = []
rt = []
once = False

def oneHott(x):
    res = [0]*vocab_size
    res[cid[x]] = 1
    return res


for i in text:
    if i == "\n":
        if wTFN:
            wTFN = False
            recording = True
        if double:
            recording = False
            wTFN = True
            record.append(end)
            inputy.append(record[1:-1])
            t.append(rt[1:-1])
            record = []
            rt = []
        double = not double
    if double and i != "\n":
        double = False
    if recording:
        record.append(oneHott(i))
        rt.append(i) 

inputy = inputy[1:]
t = t[1:]
print("Finished data processing")

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.forget_gate = nn.Linear(
            input_size + hidden_size, hidden_size, bias=True, dtype=torch.float64
        )  # set up linear transforms for the gates
        self.input_gate_forget = nn.Linear(
            input_size + hidden_size, hidden_size, bias=True, dtype=torch.float64

        )
        self.input_gate_candidate = nn.Linear(
            input_size + hidden_size, hidden_size, bias=True, dtype=torch.float64

        )
        self.output_gate = nn.Linear(input_size + hidden_size, hidden_size, bias=True, dtype=torch.float64)


        self.output = nn.Linear(hidden_size, output_size, dtype=torch.float64)  # output linear layer

        self.h_s = torch.zeros(1, hidden_size)  # hidden state/short-term memory
        self.c_s = torch.zeros(1, hidden_size)  # cell state/long-term memory

    def forward(self, x):
        shortedInput = torch.cat(
            (x, self.h_s), dim=1
        )  # concatenate input and hidden state
        cellForget = torch.sigmoid(self.forget_gate(shortedInput))  # forget gate
        self.c_s = self.c_s * cellForget  # apply forgetting

        add = torch.tanh(self.input_gate_candidate(shortedInput))  # input gate
        addForget = torch.sigmoid(self.input_gate_forget(shortedInput))
        self.c_s = self.c_s + (add * addForget)  # add the input

        hiddenForget = torch.sigmoid(self.output_gate(shortedInput))  # output gate
        self.h_s = torch.tanh(self.c_s) * hiddenForget  # get new hidden state

    def getOut(self):
        return F.softmax(self.output(self.h_s), dim=1)

    def reset(self):
        self.h_s.zero_()
        self.c_s.zero_()

    def detach(self):
        self.h_s.detach_()  # reset the model between sequences
        self.c_s.detach_()

    def getHids(self):
        return (self.c_s, self.h_s)

    def setHids(self, hids):
        self.c_s, self.h_s = hids

class Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super().__init__()
        self.model = LSTM(input_size, hidden_size, input_size)
        self.iis = input_size
        self.os = output_size
        self.sprobs = [1e-7]*input_size

    def trainSequence(
        self,
        epochs,
        inputs,
        criterion=nn.BCELoss(),
        alpha=3e-4,
        scheduler=False,
        schedPatience=100,
        graph=True,
    ):
        print("Training")
        optimizer = torch.optim.Adam(self.model.parameters(), lr=alpha)

        if scheduler:
            sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, "min", patience=schedPatience
            )

        losses = []

        for epoch in range(epochs):
            loss_holder = []
            for line in inputs:
                start = True
                for char in range(len(line) - 1):
                    self.model.forward(torch.tensor(line[char], dtype=torch.float64).view(1,-1))
                    if start:
                        pos = np.argmax(line[char])
                        self.sprobs[pos] += 1
                        start = False
                    output = self.model.getOut().view(-1)
                    loss = criterion(output, torch.tensor(line[char+1], dtype=torch.float64).view(-1))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    if scheduler:
                        sched.step(loss)

                    self.model.detach()

                    loss_holder.append(loss.item())
                self.model.reset()
            mloss = np.mean(loss_holder)
            print(f"Epoch [{epoch+1}/{epochs}], loss is {mloss:.6f}")
            losses.append(mloss)

        if graph:
            plt.plot(losses)
            plt.show()
            plt.clf()
    
    def makeTest(self):
        self.model.reset()
        if sum(self.sprobs) == 0:
            start = icd[random.randint(0,vocab_size)]
        else:
            sp = [i/sum(self.sprobs[:-1]) for i in self.sprobs[:-1]]
            start = icd[np.random.choice(range(len(sp)), p=sp)]
            
        last = torch.tensor(oneHott(start), dtype=torch.float64).view(1,-1)
        out = [start]
        for _ in range(100):
            self.model.forward(last)
            output = self.model.getOut().view(1,-1)
            a = output.view(-1).tolist()
            a = [i/sum(a) for i in a]
            pos = np.random.choice(range(len(a)), p=a)
            if pos == vocab_size - 1:
                break
            letter = icd[pos]
            last = torch.tensor(oneHott(icd[pos]), dtype=torch.float64).view(1, -1)
            out.append(letter)
        print(out)
        print("".join(out))

    def generateDecimal(self, message):
        with torch.no_grad():
            self.model.eval()
            self.model.reset()
            last = cid[message[0]]
            onehots = [oneHott(i) for i in message]
            onehots.append(end)
            sp = [i / sum(self.sprobs) for i in self.sprobs]
            decimal = sum(sp[:last])
            rang = sp[last]
            self.m = 0
            for index, i in enumerate(onehots):
                if index == 0:
                    continue
                self.model.forward(torch.tensor(onehots[index-1], dtype=torch.float64).view(1,-1))
                out = self.model.getOut().view(-1).tolist()
                self.m = self.model.getHids()
                b = sum(out[:np.argmax(i)])
                decimal += b*rang
                rang *= out[np.argmax(i)]
            res = []
            nd = 0
            v = 1
            while nd < decimal or nd >= decimal + rang:

                add = (1/2**v)
                if nd + add < decimal + rang:
                    nd += add
                    res.append(1)
                else:
                    res.append(0)
                v+=1

            return res
        
    def interpretDecimal(self, decimal):
        with torch.no_grad():
            o = 0
            for index, i in enumerate(decimal):
                o += float(i) * (1/(2**(index+1)))
            self.model.reset()
            outdistro = [i / sum(self.sprobs) for i in self.sprobs]
            res = ""
            def midsearch(decimal, outidstro):
                l, r = 0, len(outidstro)

                while l < r:
                    mid = (r-l)//2 + l
                    prev = sum(outidstro[:mid])
                    small = prev <= decimal
                    big = prev + outidstro[mid] > decimal
                    if small and big:
                        return mid, prev
                    if small:
                        l = mid + 1
                    if big:
                        r = mid

            for _ in range(100):
                c, ps = midsearch(o, outdistro)
                o -= ps
                o /= outdistro[c]
                if c == vocab_size - 1:
                    break
                res += icd[c]
                print("Got a " + icd[c])
                self.model.forward(torch.tensor(oneHott(icd[c]), dtype=torch.float64).view(1,-1))
                outdistro = self.model.getOut().view(-1).tolist()

            return res


        

input_size = vocab_size
hidden_size = 64
num_epochs = 10

model = Model(input_size, input_size, hidden_size)
model.makeTest()
model.trainSequence(1, inputy, graph=False)
model.makeTest()
model.trainSequence(1, inputy, graph=False)
model.makeTest()


def log(x):
    res = 0
    while 2**res < x:
        res += 1
    return res

a = 0
for i in t[:5]:
    print("New test")
    n = len(i) * log(vocab_size)
    dec = model.generateDecimal(i)
    if model.interpretDecimal(dec) != i:
        print("There was an issue")
        print(model.interpretDecimal(dec))
        print(i)
    c = len(dec)
    a += (1-(c/n))*100
a /= 5
print("Compressed text was ", a, "% shorter than uncompressed")
#print(model.testDeterminism())

"""
How to generate in only binary
rang gets too small
"""

"""
How to interpret in only binary
"""