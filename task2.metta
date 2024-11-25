;; Generic filter function that takes a predicate and a list
;; Returns elements that satisfy the predicate

(= (filter $predicate $list) (
        if (== $list ()) ()
        ($predicate (superpose $list))))

(= (isNegative $x)
   (< $x 0)) 

!(filter isNegative (1 -2 3 4 5))