;; Complement: Returns elements that are in the universal set but not in the given set A
(= (complement $universe $A)
    (subtraction (superpose $universe) (superpose $A)))

;; !(complement (A B C D E) (A C E))

;; Symmetric difference: Returns elements that are in either set but not in both
(= (symmetric-difference $A $B)
    (union 
        (subtraction (superpose $A) (superpose $B))
        (subtraction (superpose $B) (superpose $A))))

;; !(symmetric-difference (A B C) (B C D))  ; Returns: A D

;; Subset: Checks if first set is a subset of second set
(= (subset $A $B)
    (== (collapse (subtraction 
            (superpose $A) 
            (superpose $B))) 
        ()))

;; Example usage:
;; !(subset (A B) (A B C D))  ; Returns: True
;; !(subset (A B E) (A B C D))  ; Returns: False

;; Equivalent: Checks if two sets contain exactly the same elements
(= (equivalent $set1 $set2)
    (and 
        (subset $set1 $set2)
        (subset $set2 $set1)))

;; Example usage:
!(equivalent (A B C) (C B A))  ; Returns: True
!(equivalent (A B C) (A B D))  ; Returns: False