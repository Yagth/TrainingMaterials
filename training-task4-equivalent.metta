;; ---------------------------------------------
;; Function: length
;; Description: Calculates the number of elements in a list.
;; Parameters:
;; - $set: A list whose length is to be calculated.
;; Returns: An integer representing the length of the list.
;; ---------------------------------------------
(= (length $set)
    (if (== $set ())
        0
        (+ 1 (length (cdr-atom $set)))
    )
)

;; ---------------------------------------------
;; Function: equivalent
;; Description: Checks if two lists are equivalent in length.
;; Parameters:
;; - $set1: The first list to compare.
;; - $set2: The second list to compare.
;; Returns: True if the lengths of both lists are equal; False otherwise.
;; ---------------------------------------------

(= (equivalent $set1 $set2)
    (if (== (length $set1) (length $set2))
        True
        False
        )
)

;; Example usage:
! (equivalent (A B C) (1 2 3))       ; Expected output: True (Both have 3 elements)
! (equivalent (A B C) (A B C D))     ; Expected output: False (Different lengths)
! (equivalent (X Y) (A B))           ; Expected output: True (Both have 2 elements)


