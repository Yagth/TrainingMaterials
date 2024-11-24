 ;========================= Space Operations =========================
 ; This file contains the operations of adding and removing atoms from a space non-deterministically.
 ;to do this we need to define a space and add atoms to it
 ;we can then remove atoms from the space

! (bind! &my-space (new-space)) ; create a new space
! (add-atom &my-space (superpose (A B C D E))) ; add atoms to the space
! (get-atoms &my-space) ; get the atoms in the space : [A, B, C, D, E]

! (remove-atom &my-space (superpose (A B C D E))) ; remove atoms from the space
! (get-atoms &my-space) ; get the atoms left in the space : []