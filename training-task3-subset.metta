;; Step 1: Calculate the intersection of $set and $union
;; 'superpose' prepares the sets to handle non-deterministic values (atoms)
;; 'intersection' finds the common elements between $set and $union

(: subset (-> Expression Expression Expression))

(= (subset $set $union)
    (if (== (collapse (intersection (superpose $set) (superpose $union))) $set)
                True
                False)
)


;; Input: (subset (L M N O P) (A B C L M N O P X Y Z))
;; This checks if the set (L M N O P) is a subset of (A B C L M N O P X Y Z).
;; Since all elements of (L M N O P) are contained in the second set, the output is True.

! (subset (L M N O P) (A B C L M N O P X Y Z))


