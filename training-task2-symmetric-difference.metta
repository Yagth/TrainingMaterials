;; Define the symmetric difference function
;; (symmetric-difference $set1 $set2):
;; - $set1 and $set2 are the two sets for which we want the symmetric difference.

(: symmetric-difference (-> Expression Expression Expression))


(= (symmetric-difference $set1 $set2)
    (collapse (union (subtraction (superpose $set1) (superpose $set2)) (subtraction (superpose $set2) (superpose $set1)))
))


;; Test the function:
;; Input: $set1 = (A B C D E F), $set2 = (E F G X Y Z)
;; Expected output: (A B C D G X Y Z)
! (symmetric-difference (A B C D E F) (E F G X Y Z))


