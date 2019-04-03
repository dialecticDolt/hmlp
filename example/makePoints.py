import numpy as np

1_processor_data = np.load('temp.data_1p')
1_processor_gids = np.load('temp.gids_1p')

2_processor_data = np.load('temp.data_2p')
2_processor_gids = np.load('temp.gids_2p')

4_processor_data = np.load('temp.data_4p')
4_processor_gids = np.load('temp.gids_4p')

index_on_1 = 0
table = np.zeros( [4, len(1_processor_gids)] )
for i in 1_processor_data:
    index_on_2 = np.argmin(np.abs(2_processor_data - i))
    index_on_4 = np.argmin(np.abs(4_processor_data - i))

    table[0, index_on_1] = i
    table[1, index_on_1] = 1_processor_gids[index_on_1]
    table[2, index_on_1] = 2_processor_gids[index_on_2]
    table[3, index_on_1] = 4_processor_gids[index_on_4]

table = table[:, table[1, :].argsort()]
table = table.T

print("Data", "GID on 1P", "GID on 2P", "GID on 4P")
for i in range(len(rids1)):
    print( table[i, 0], (int)(table[i, 1]), (int)(table[i, 2]), (int)(table[i, 3]) )


