;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Previous Implementation  ;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; any function using ND
;; (= (any' $x $acc) (if (== $x ()) $acc (any' (cdr-atom $x) (or $acc (car-atom $x)))))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;; Optimized one (pruned) ;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;  (if (or (== $x ()) $acc) )  Here \/
                    ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
                    ; If $acc is already True, we will stop and return 
                    ; because one True is enough.
                    ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(= (any'' $x $acc) 
        (if (or (== $x ()) $acc) 
            $acc 
            (any'' (cdr-atom $x) (or $acc (car-atom $x)))
        
        )
)


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;       TEST                          ;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

!(any'' (True False False False False) False)