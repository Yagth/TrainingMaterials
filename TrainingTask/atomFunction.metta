;; Optimizing the "Any" Function

;; Determines if any element in a list satisfies a condition
(= (any $list)
    (case $list
        (()
            False) ;; No elements, return False
        (($x . $xs)
            (or $x (any $xs)))
    )
)

;; Member Check Function

;; Checks if an element exists in a set
(= (member? $x $set)
    (not (== (filter $set (lambda $elem (== $x $elem))) ()))
)

;; Non-Deterministic Space Initialization

;; Adds multiple atoms to a space non-deterministically
(= (init-space $atoms)
    (fold $atoms (lambda $atom (add-atom &space $atom)) (new-space))
)

;; Non-Deterministic Removal of Multiple Atoms

;; Removes atoms added by init-space
(= (remove-atoms $atoms)
    (fold $atoms (lambda $atom (remove-atom &space $atom)) &space)
)