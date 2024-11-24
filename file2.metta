; Function: wrapper
; Description: This function applies a predicate to an atom and returns the atom if the predicate is true, otherwise returns an empty expression.
; Parameters:
;   - $predicate: A function that takes an atom and returns a boolean value.
;   - $x: The atom to which the predicate is applied.
; Returns:
;   - Expression: Returns the atom if the predicate evaluates to true, otherwise returns an empty expression.
; Logic:
;   - The function uses an `if` construct to evaluate the predicate with the given atom.
;   - If the predicate returns true, the atom is returned.
;   - If the predicate returns false, an empty expression is returned.


(:wrapper (-> Atom Atom Expression))
(= (wrapper $predicate $x) (if ($predicate $x) $x (empty)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;



; Function: filter
; Description: This function filters elements of a list based on a given predicate.
; Parameters:
;   - $predicate: The predicate function to apply to each element of the list.
;   - $list: The list of elements to be filtered.
; Returns:
;   - List: A new list containing only the elements that satisfy the predicate.
; Logic:
;   - The function uses the `wrapper` function to apply the predicate to each element of the list.
;   - The `superpose` function is used to handle the list elements.

(:filter (-> Atom Expression Expression))
(= (filter $predicate $list) (
   wrapper $predicate (superpose $list))
)




;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;       TEST                          ;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(= (predicate $x) (> $x 0))
; !(predicate 2)


!(filter predicate (1 -1 -2 3 4))
