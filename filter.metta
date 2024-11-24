 ;========================Filter==============================================================
 ; fiter function takes a function and a list as input and returns a list of elements that
 ; -satisfy the function condition non-deterministically
 ; if the list is empty, return an empty list
 ; else, return the list of elements that satisfy the function condition

(:filter  (-> Expression Expression Expression))
(= (filter $function $list) (
        if (== $list ()) ()
        ($function (superpose $list))))

 ;test cases
 ; filter even numbers from the list
(= (even $num)
    (if (== (% $num 2) 0)
        $num
        (empty)))
! (filter even (5 7 4 5 65 8 42)) ; [4, 8, 42]

 ; filter odd numbers from the list
(= (odd $num)
    (if (not (== (% $num 2) 0))
        $num
        (empty)))
! (filter odd (5 7 4 5 65 8 42)) ; [5, 7, 5, 65]