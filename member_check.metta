 ;===============================is_member===============================
 ; Check if an element is a member of a list or not non-deterministically
 ; If the element is a member of the list, returns True other wise False
 ; check_equlity: A function to check if an element is equal to a list element

(= (check_equlity $list_elem $element) (
        if (== $list_elem $element)
        $list_elem
        (empty)
    )
)
(= (is_member $element $list)
    (not (== (collapse (check_equlity (superpose $list) $element)) ()))
)

 ;test cases
! (is_member 5 (5 7 4 5 65 8 42)) ; True
! (is_member () (5 7 4 5 65 8 42)) ; False