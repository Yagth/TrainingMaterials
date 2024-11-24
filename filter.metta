;;condition for filter and any
(=(isEven $x)(
   if ( == (% $x 2) 0) 
       $x
       (empty)
))

;;Generic Filter Function
;;filter elements using predicate function
(= (filter $predicate $list)
   (collapse ( $predicate (superpose $list)))
)

;;optimized any function
;;check if any element satsfy a condition
(= (any $x) 
   (not (== (collapse (isEven (superpose $x))) ()))
)

;; is member
;; check if the element is a member of a list
(= (isMember $elemnt $list)
    (not (== (collapse (intersection (superpose ($elemnt)) (superpose $list))) ()))
)



;; test cases
;!(filter isEven (2 3 1 4 6 8 )) ;; filter even nembers only (2 4 6 8)
;!(isMember  3 (2 3)) ;; True
;!(any (1 3 9 5 7 2))  ;; if any even number exist (True)

