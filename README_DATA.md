## Dataset

The dataset consists of 12 designs, which are as follows:

1. mgc_edit_dist_a
2. mgc_fft_b
3. mgc_matrix_mult_b
4. mgc_pci_bridge32_b
5. mgc_superblue14
6. mgc_superblue19
7. mgc_des_perf_a
8. mgc_fft_a
9. mgc_matrix_mult_a
10. mgc_matrix_mult_c
11. mgc_superblue11_a
12. mgc_superblue16_a

The names of all these designs are stored in a `.txt` file called `all.names`. The `train.names` file contains the names of designs used for training, while the `test.names` file contains the names of designs used for testing. The parameter `-groud` indicates whether to exchange the test set and the train set.

In the directory for each design, you can find the following files:

- `edge_weights.txt`: It contains the weights of each net after clustering.
- `hpwl.txt`: Each line represents the estimated HPWL (Half-Perimeter Wirelength) for each placement.
- `labels.txt`: Each line contains the rWL (routed Wirelength), #vias, #shorts, and the score (ICCAD19 global routing contest score, not used in this work).
- `golden.txt`: It contains the HPWL of the manually placed result.
- `macro_index.txt`: It contains the list of all macros in the clustered netlist.
- `meta.txt`: It contains metadata about the original design, including the number of nodes, macros, I/O, nets, row height, site width, number of pins, number of movable pins, total movable node area, total fixed node area, and total space area. However, this information is not utilized in this work.
- `names.txt`: It contains the names of all the used placements.
- `node_size.txt`: It provides the size of each node in the clustered netlist.
- `pins.txt`: It contains information about all the pins. Each line represents a pin and includes the connected node, connected net, x offset of the pin, and y offset of the pin.
- `region.txt`: It specifies the placeable region of nodes.
- `node_pos`: It includes all the placements. Note that some placements are not used, as indicated in the `names.txt` file.