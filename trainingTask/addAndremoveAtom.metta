;;The code creates a new space, adds atoms (A, B, C, D, E) non-deterministically, and retrieves them. 
It then removes the atoms and verifies the space is empty by retrieving the atoms again
!(bind! &space (new-space))  
!(add-atom &space (superpose (list (A B C D E))))  
!(get-atoms &space) 
!(remove-atom &space (superpose (list (A B C D E))))  
!(get-atoms &space)