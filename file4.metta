
; Function: is-member
; Description: This function checks if a given element is a member of a list.
; Parameters:
;   - $element: The element to check for membership in the list.
;   - $list: The list in which to check for the element.
; Returns:
;   - Bool: Returns True if the element is a member of the list, otherwise returns False.
; Logic:
;   - The function uses a combination of `collapse`, `superpose`, and `if` constructs to determine membership.
;   - It first checks if the superposition of the list equals the element.
;   - If the result is an empty list, it returns False, indicating the element is not a member.
;   - Otherwise, it returns True, indicating the element is a member of the list.


(: is-member (-> Grounded Expression Bool))
(= (is-member $element $list) 
    (if (==
        (collapse (if (== (superpose $list) $element) True (empty)))
        ()
    )
    False
    True
    )
)


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;       TEST                          ;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

!(is-member 1 (2 1 3 4))
!(is-member 1 (2 3 4))