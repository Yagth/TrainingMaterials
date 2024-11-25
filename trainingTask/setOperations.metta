;Complement: Returns elements in the universal set not present in the given set.
(: complement (-> (List $t) (List $t) (List $t)))
(= (complement $list1 $list2)
   (subtraction (superpose $list1) (superpose $list2)))

;Symmetric Difference: Returns elements in either of the sets but not in their intersection
(: symmetricDifference (-> (List $t) (List $t) (List $t)))
(= (symmetricDifference $list1 $list2)
    (union (subtraction (superpose $list1) (superpose $list2)) (subtraction (superpose $list2) (superpose $list1))))

 Subset: Checks if all elements of one set are in another set. 
(: subset (-> (List $t) (List $t) Bool))
(= (subset $list1 $list2)
    (== (collapse (subtraction (superpose $list1) (superpose $list2))) ()))

Equivalent: Checks if two sets contain exactly the same elements.  
(: equivalent (-> (List $t) (List $t) Bool))
(= (equivalent $list1 $list2)
    (and 
        (== (collapse (subtraction (superpose $list1) (superpose $list2))) ())
        (== (collapse (subtraction (superpose $list2) (superpose $list1))) ())))

;;testcases
!(complement (1 2 3 4) (2 3))
!(complement (1 2 3 4) ())
!(assertEqualToResult (complement (1 2 3 4) (2 3)) (1 4))
!(assertEqualToResult (complement (1 2 3 4) ()) (1 2 3 4))
!(symmetricDifference (1 2 3) (3 4 5))
!(symmetricDifference (1 2 3) ())
!(assertEqualToResult (symmetricDifference (1 2 3) (3 4 5)) (1 2 4 5)) 
!(assertEqualToResult (symmetricDifference (1 2 3) ()) (1 2 3))
!(subset (1 2) (1 2 3))
!(subset (1 4) (1 2 3))
!(subset (1 2) ())
!(assertEqualToResult (subset (1 2) (1 2 3)) (True))    
!(assertEqualToResult (subset (1 4) (1 2 3)) (False))
!(assertEqualToResult (subset (1 2) ()) (False))
!(equivalent (1 2 3) (3 2 1))
!(assertEqualToResult (equivalent (1 2 3) (1 2 3)) (True))