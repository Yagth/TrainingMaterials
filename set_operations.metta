 ;===========================Set_compelement===============================

 ; The complement of a set A, denoted by A', is the set of elements
 ; in the universe set that are not in A
 ; This function calculates the complement of a set within a given universe set

(= (set_compelement $universe_set $set)
    (subtraction (superpose $universe_set) (superpose $set))
)

 ; testcases
! (set_compelement (1 2 3 4 5) (1 2 3)) ; [4, 5]
! (set_compelement (1 2 3 4 5) ()) ; [1, 2, 3, 4, 5]

 ;===========================Set_Symmetric_difference===============================

 ; The symmetric difference of two sets A and B, denoted by A Δ B, is the set of elements
 ; that are in either of the sets but not in their intersection
 ; This function calculates the symmetric difference of two sets

(= (set_Symmetric_difference $set1 $set2)
    (union
        (subtraction (superpose $set1) (superpose $set2))
        (subtraction (superpose $set2) (superpose $set1))
    )
)

 ; testcases
! (set_Symmetric_difference (1 2 3 4 5) (3 4 5 6 7)) ; [1, 2, 6, 7]
! (set_Symmetric_difference (1 2 3 4 5) (1 2 3 4 5)) ; []

 ;===========================Is_subset_of_set===============================

 ; This function checks if set1 is a subset of set2
 ; A set A is a subset of a set B if all elements of A are also elements of B
 ; returns True if set1 is a subset of set2, False otherwise

(= (is_subset_of_set $set1 $set2)
    (if (== (collapse (subtraction (superpose $set1) (superpose $set2))) ())
        True
        False
    )
)

 ; testcases
! (is_subset_of_set (1 2 3) (1 2 3 4 5)) ; True
! (is_subset_of_set (1 2 3 4) (1 2 3)) ; False

 ;===========================Is_equivalent_set===============================

 ; This function checks if two sets are equivalent
 ; Two sets are equivalent if they have the same number of elements
 ; returns True if set1 is equivalent to set2, False otherwise

(= (length_list $list) (
        if (== $list ())
        0
        (+ 1 (length_list (cdr-atom $list)))
    )
)

(= (is_equivalent_set $set1 $set2) (
        if (== (length_list $set1) (length_list $set2))
        True
        False
    )
)
 ; testcases
! (is_equivalent_set (1 2 3) (A B C)) ; True
! (is_equivalent_set (1 2 3) (1 2 3 4)) ; False
