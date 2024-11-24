;; Define a helper function to check if an element matches the target element
;; - $x: Current element being checked
;; - $element: Target element to find in the list
(= (isMember $x $element) 
    (if (== $x $element)  
        True           
        (empty)           
    )
)

;; Define the main function to check if an element exists in a list
;; - $list: The list to search in
;; - $element: The target element to find

(= (member $list $element) 
    (not (== (collapse (isMember (superpose $list) $element)) ())) 
)


; Test cases for the member function

!(member (A B C) A) ; Expected: True (A is in the list)

!(member (1 2 3 4 5) 5) ; Expected: True (5 is in the list)

!(member (1 2 3 4 5) 6) ; Expected: False (6 is not in the list)


