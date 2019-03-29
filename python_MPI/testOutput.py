import numpy as np

D1 = np.loadtxt('D_1p')
rids1 = np.loadtxt('rids_1p')
D2 = np.loadtxt('D_2p')
rids2 = np.loadtxt('rids_2p')

D4 = np.loadtxt('D_4p')
rids4 = np.loadtxt('rids_4p')

#print("length of D1: ", len(D1))
#print("length of D2: ", len(D2))
#print("Are RIDS arrays equal: ", np.array_equal(rids1, rids2))
#print(rids1)
#print(rids2)

ind = 0
correct_count = 0
table = np.zeros([4, len(rids1)])
for i in D1:
    if  i in D2 :
        correct_count+=1
    index1 = np.argmin(np.abs(D1-i))
    index2 = np.argmin(np.abs(D2-i))
    index3 = np.argmin(np.abs(D4-i))
    #print("Absolute Index 1: ", index1)
    #print("Absolute Index 2: ", index2)
    #print("RID 1: ", rids1[index1])
    #print("RID 2: ", rids2[index2])
    table[0,ind] = i
    table[1, ind] = rids1[index1]
    table[2, ind] = rids2[index2] 
    table[3, ind] = rids4[index3]
    ind+=1
table = table[:, table[1, :].argsort()]
table = table.T

for i in range(len(rids1)):
    print(table[i, 0], (int)(table[i, 1]), (int)(table[i, 2]), (int)(table[i, 3]))

print("Norm of sorted: ", np.linalg.norm(np.sort(D1) - np.sort(D2)))

