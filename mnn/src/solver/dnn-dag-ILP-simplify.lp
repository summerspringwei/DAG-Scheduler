Minimize
    value: tt

Subject To
    \ We convert the binary quadratic optimization problem to big-M form
    \ c_d_i_j convertion overhead on device d from node i to node j
    tt + M s_1_1 > 0
    tt - t_1_1 - M s_1_1 > T_1_1 - M
    tt + M s_2_1 > 0
    tt - t_2_1 - M s_2_1 > T_2_1 - M
    tt + M s_1_2 > 0
    tt - t_1_2 - (M + c_1_1_2) s_1_2 + c_1_1_2 s_1_1 > T_1_2 - M`
    tt + M s_2_2 > 0
    tt - t_2_2 - (M + c_2_1_2) s_2_2 + c_2_1_2 s_2_1 > T_2_2 - M
    tt + M s_1_3 > 0
    tt - t_1_3 - M s_1_3 > T_1_3 - M
    tt + M s_2_3 > 0
    tt - t_2_3 - M s_2_3 > T_2_3 - M
    
    \ Nodes are only executed once
    s_1_1 + s_2_1 = 1
    s_1_2 + s_2_2 = 1
    s_1_3 + s_2_3 = 1

    \ Only one op can be executed on one device
    \ (node 2 and 3)
    t_1_3 - t_1_2 + M u_1_2_3 - c_1_1_2 s_1_2 + c_1_1_2 s_1_1 > T_1_2
    t_1_2 - t_1_3 - M u_1_2_3 > T_1_3 - M
    t_2_3 - t_2_2 + M u_2_2_3 - c_2_1_2 s_2_2 + c_2_1_2 s_2_1 > T_2_2
    t_2_2 - t_2_3 - M u_2_2_3 > T_2_3 - M
    \ (node 1 and 3)
    t_1_3 - t_1_1 + M u_1_1_3 > T_1_1
    t_1_1 - t_1_3 - M u_1_1_3 > T_1_3 - M
    t_2_3 - t_2_1 + M u_2_1_3 > T_2_1
    t_2_1 - t_2_3 - M u_2_1_3 > T_2_3 - M
    \ (node 1 and 2) For parent node and child node
    t_1_2 - t_1_1 > T_1_1
    t_2_2 - t_2_1 > T_2_1

    \ Constraints of DAG
    t_1_2 - t_2_1 > T_2_1
    t_2_2 - t_1_1 > T_1_1

    \ Start time is grater than 0
    t_1_1 >= 0
    t_1_2 >= 0
    t_1_3 >= 0
    t_2_1 >= 0
    t_2_2 >= 0
    t_2_3 >= 0

Binary
    s_1_1
    s_1_2
    s_1_3
    s_2_1
    s_2_2
    s_2_3
    u_1_2_3
    u_2_2_3
    u_1_1_3
    u_2_1_3

End
