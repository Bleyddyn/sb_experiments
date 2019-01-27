import csv
from matplotlib import pyplot as plt

fname = "/var/folders/k4/pz92_j6s17xd8vc_dnmbm54w0000gn/T/openai-2019-01-27-12-34-19-446263/monitor.csv"

data = []
with open(fname,'r') as csvfile:
   rd = csv.reader(csvfile)
   for row in rd:
     data.append(row)

rewards = []
ep_lens = []
for row in data[2:]:
    rewards.append( float(row[0]) )
    ep_lens.append( int(row[1]) )

fig = plt.figure(figsize=(16,10))
fig.add_subplot(2, 1, 1)
plt.plot(rewards)
plt.title("Reward")
fig.add_subplot(2, 1, 2)
plt.plot(ep_lens)
plt.title("Episode Length")
plt.show()

