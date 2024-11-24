; Recursive function to check if any element in a list evaluates to True
; Parameters:
;   $x: The list to check
;   $acc: Accumulator that holds the result of logical OR operations
;
; Returns:
;   True if any element in the list evaluates to True, otherwise False.
;
; Logic:
; - Base Case: If the list is empty or $acc is already True, return $acc.
; - Recursive Case: Decompose the list into head and tail, 
;                   and evaluate the head using logical OR with $acc.

(= (any' $x $acc) 
    (if (or (== $x ()) $acc) 
        $acc 
        (let* (($header (car-atom $x))
                ($tail (cdr-atom $x)))
            (any' $tail (or $acc $header)))
    )
)

; Test Cases
!(any' (True False) False)          ; Output: True (since the first element is True)
!(any' (True False False True) False) ; Output: True (the first True makes it terminate early)
!(any' (False False False False False False False False False False False True False False True) False)
; Output: True (encounters True after several False entries)


