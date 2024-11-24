;; Define the complement function
;; (compelement $set $union):
;; - $set: The subset to remove.
;; - $union: The universal set containing all elements.

(: compelement (-> Expression Expression Expression))

(= (compelement $set $union)
    (collapse (subtraction (superpose $union) (superpose $set)))
)

; Test the function:
; - Input: $set = (1 2 3 4), $union = (1 2 3 4 5 6)
; - Expected output: (5 6)

! (compelement (1 2 3 4) (1 2 3 4 5 6))


