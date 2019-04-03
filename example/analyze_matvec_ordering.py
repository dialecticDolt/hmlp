import numpy as np

processor_1_data = np.fromfile('temp.data_p1',dtype = 'float32')
processor_1_gids = np.fromfile('temp.gids_p1',dtype = 'int32') 

processor_2_data = np.fromfile('temp.data_p2',dtype = 'float32')
processor_2_gids = np.fromfile('temp.gids_p2',dtype = 'int32') 

processor_4_data = np.fromfile('temp.data_p4',dtype = 'float32')
processor_4_gids = np.fromfile('temp.gids_p4',dtype = 'int32')

index_on_1 = 0
table = np.zeros( [4, len(processor_1_gids)] )
for i in processor_1_data:
    index_on_2 = np.argmin(np.abs(processor_2_data - i))
    index_on_4 = np.argmin(np.abs(processor_4_data - i))

    table[0, index_on_1] = i
    table[1, index_on_1] = processor_1_gids[index_on_1]
    table[2, index_on_1] = processor_2_gids[index_on_2]
    table[3, index_on_1] = processor_4_gids[index_on_4]

table = table[:, table[1, :].argsort()]
table = table.T

print("Data", "GID on 1P", "GID on 2P", "GID on 4P")
for i in range(len(processor_1_gids)):
    print( table[i, 0], (int)(table[i, 1]), (int)(table[i, 2]), (int)(table[i, 3]) )


