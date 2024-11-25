;;Generic Filter func: Filters elements of a list that satisfy a given predicate by collapsing the results of applying the predicate non-deterministically.
;; Predicate func to check if a number is negative
(: isNegative (-> Int Bool))
(= (isNegative $x)
   (< $x 0))  
;;filter elements using predicate func
(= (filter $predicate $list)
   (collapse ( $predicate (superpose $list))))

;;Any: Checks if at least one element in a list of booleans is True, using the intersection of True with the list.
(: any (-> (List Bool) Bool))
(= (any $list)
    (not (== (collapse (intersection (superpose (True)) (superpose $list))) ())))

;;isMember: Verifies if a specific element exists in a list by checking the intersection of the element with the list.
(= (isMember $x $list)
    (not (== (collapse (intersection (superpose ($x)) (superpose $list))) ())))

;;testcases
!(any (True False))
!(any (False False))
!(assertEqualToResult (any (True False)) (True))
!(assertEqualToResult (any (False False)) (False))
!(isMember 1 (1 2 3))
!(isMember 8 (1 2 3))
!(assertEqualToResult (isMember 1 (1 2 3)) (True))
!(assertEqualToResult (isMember 8 (1 2 3)) (False))
!(filter isNegative (2 -3 1 4 -6 8 ))