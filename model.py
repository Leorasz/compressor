import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import random

# np.random.seed(1337)
# torch.manual_seed(1337)
# random.seed(1337)

print("Finished imports")

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read() # get text

# text = ["bac","bab","abac","babac"]
#given ba, 50/50 
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
double = False
wTFN = False
recording = False
record = []
once = False

def oneHott(x):
    res = [0]*vocab_size
    res[cid[x]] = 1
    return res

# for i in text:
#     rec = []
#     for j in i:
#         rec.append(oneHott(j))
#     rec.append(end)
#     input.append(rec)

for i in text:
    if i == "\n":
        if wTFN:
            wTFN = False
            recording = True
        if double:
            recording = False
            wTFN = True
            record.append(end)
            inputy.append(record[1:])
            record = []
        #     if not once:
        #         once= True
        #     else:
        #         break
        double = not double
    if double and i != "\n":
        double = False
    if recording:
        record.append(oneHott(i)) if i != " " else None

inputy = inputy[1:]
print("Finished data processing")

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.forget_gate = nn.Linear(
            input_size + hidden_size, hidden_size, bias=True
        )  # set up linear transforms for the gates
        self.input_gate_forget = nn.Linear(
            input_size + hidden_size, hidden_size, bias=True
        )
        self.input_gate_candidate = nn.Linear(
            input_size + hidden_size, hidden_size, bias=True
        )
        self.output_gate = nn.Linear(input_size + hidden_size, hidden_size, bias=True)

        self.output = nn.Linear(hidden_size, output_size)  # output linear layer

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
        self.sprobs = [0]*input_size

    def trainSequence(
        self,
        epochs,
        inputs,
        criterion=nn.CrossEntropyLoss(),
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
                    if start:
                        pos = np.argmax(line[char])
                        self.sprobs[pos] += 1
                        start = False
                        continue
                    self.model.forward(torch.tensor(line[char], dtype=torch.float32).view(1,-1))
                    output = self.model.getOut().view(-1)
                    loss = criterion(output, torch.tensor(line[char+1], dtype=torch.float32).view(-1))
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
        if sum(self.sprobs) == 0:
            start = icd[random.randint(0,vocab_size)]
        else:
            sp = [i/sum(self.sprobs) for i in self.sprobs]
            print(sp)
            start = icd[np.random.choice(range(len(sp)), p=sp)]
            
        last = torch.tensor(oneHott(start), dtype=torch.float32).view(1,-1)
        out = [start]
        for _ in range(100):
            self.model.forward(last)
            output = self.model.getOut().view(1,-1)
            last = output
            a = output.view(-1).tolist()
            a = [i/sum(a) for i in a]
            pos = np.random.choice(range(len(a)), p=a)
            if pos == vocab_size - 1:
                break
            letter = icd[pos]
            out.append(letter)
        print(out)
        print("".join(out))

    def generateDecimal(self, message):
        start = cid[message[0]]
        onehots = [oneHott(i) for i in message[1:]]
        decimal = 0
        sp = [i / sum(self.sprobs) for i in self.sprobs]

        return decimal

        

input_size = vocab_size
hidden_size = 64
num_epochs = 10

model = Model(input_size, input_size, hidden_size)
model.makeTest()
while (input("Train?") == "y"):
    model.trainSequence(5, inputy)
    model.makeTest()
