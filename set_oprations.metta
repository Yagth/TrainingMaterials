;;complement
;;elements in one set that are not in the other
(= (complement $set1 $set2) 
   (collapse (subtraction (superpose $set1) (superpose $set2)))
)

;;Symmetric difference
;;Elements that are in either of the two sets but not in both.
(= (symmetric-difference $set1 $set2)
   (collapse 
           (union (subtraction (superpose $set1) (superpose $set2)) 
                  (subtraction (superpose $set2) (superpose $set1))
           )
    )
)

;;Subset
;;All elements of one set exist in another.
(= (subset $set1 $set2)
   (== (collapse (intersection (superpose $set1) (superpose $set2))) $set1)
)

;;Equivalent
;;Two sets have the same elements.
(= (equivalent $set1 $set2)
   (== (collapse (subtraction (superpose $set1) (superpose $set2))) ())
)

;; test cases

;!(complement (1 2 3 4 5) (2 3)) ;; set1 - set2 (1 4 5)
;!(symmetric-difference (1 2 3) (3 4 5)) ;;  (set1 - set2) U (set2 - set1) (1 2 4 5)
;!(subset (2 3) (1 2 3 4)) ;; set1 ⊆ set2  True
;!(equivalent (1 2 3) (3 2 1)) ;; set1 - set2 = () and set2 - set1 = () or (set1 ⊆ set2) and ((set2 ⊆ set1)) True