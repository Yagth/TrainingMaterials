; Filter elements from a list based on a given function
(= (filter-even $func $set)
    (if (== $set ())
        ()                                  ; Return empty if the list is empty
        (let* (($x (superpose $set)))
            ($func $x)                      ; Apply the function to each element
        )
    )
)

; Return $x if it is even, else empty
(= (even $x)
    (if (== (% $x 2) 0)
        $x
        (empty)
    )
)

; Return $x if it is odd, else empty
(= (odd $x)
    (if (== (% $x 2) 0)
        (empty)
        $x
    )
)

;; ---------------------------------------------
;; Example Usage:
;; Apply the `odd` filtering function to a list of numbers.
;; - Input: (5 7 4 5 65 8 42)
;; - Expected Output: (5 7 5 65)
;; ---------------------------------------------


! (filter-even odd (5 7 4 5 65 8 42))

