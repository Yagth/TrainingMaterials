;; Define a function that creates a set of atoms (A, B, C, D, E)
(= (set) (superpose (A B C D E)))

;; Step 1: Create a new space and bind it to the variable &x
! (bind! &x (new-space))  ; A new, empty space is created and stored in &x.

;; Step 2: Add all atoms from the set (A, B, C, D, E) to the space
! (add-atom &x (set))  ; The atoms A, B, C, D, E are added nondeterministically to &x.

;; Step 3: Retrieve and display the current atoms in the space
! (get-atoms &x)  ; This will return all atoms in the space (A, B, C, D, E).

;; Step 4: Remove the atoms (A, B, C, D, E) from the space
! (remove-atom &x (set))  ; The atoms A, B, C, D, E are removed nondeterministically from &x.

;; Step 5: Retrieve and display the current atoms in the space
! (get-atoms &x)  ; This will return an empty result as the atoms have been removed.

