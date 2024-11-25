;; Complement: Elements in set A but not in set B
(= (complement $a $b)
    (subtraction $a $b)
)

;; Symmetric Difference: Elements unique to each set
(= (symmetric-difference $a $b)
    (union (complement $a $b) (complement $b $a))
)

;; Subset: Check if set A is a subset of set B
(= (subset? $a $b)
    (== (collapse (intersection $a $b)) $a)
)

;; Equivalent: Check if sets A and B are equal
(= (equivalent? $a $b)
    (and (subset? $a $b) (subset? $b $a))
)