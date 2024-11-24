 ;; This is the function to be optimized
 ;; The original function:
 ;; (= (any' $x $acc)
 ;; (if (== $x ())
 ;;      $acc
 ;;      (any' (cdr-atom $x) (or $acc (car-atom $x)))))
 ;; Optimized function:
 ;; The optimization involves adding base case if $acc is true, return $acc

(= (any' $x $acc)
    (if (or (== $x ()) $acc) ;; If the list is empty or $acc is true, return $acc
        $acc
        (let* ( ($head (car-atom $x)) ;; Store the head of the list in $head
                ($tail (cdr-atom $x))) ;; Store the tail of the list in $tail
            (any' $tail (or $acc $head)) ;; Recursively call any' with the tail and updated $acc
        )
)
)

 ;test cases
! (any' (True False) False) ;; True
! (any' (True False False True) False) ;; True
! (any' (False False False False False False False False False False False False False) False) ;; False
