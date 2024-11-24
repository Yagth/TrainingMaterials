;;;;;;;;;;;;;;;;;;;;;; Note  ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
; While working with the functions, it is assumed that the sets have unique values,      ;;;;;
; similar to real mathematical sets. Therefore, the use of a unique function is avoided. ;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


; The `compliment` function calculates the complement of a given set within a specified space.
; 
; Inputs:
; - $U: A space containing a set of atoms.
; - $set: A non-deterministic set of patterns.
;
; Output:
; - The complement of $set within the space $U, which is obtained by subtracting $set from the set of atoms in $U.


(: compliment (-> Grounded Atom Expression) )
 (= (compliment $U $set) 
    (subtraction (get-atoms $U) 
                 $set               
    )
)
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


; The `symmetric-difference` function computes the symmetric difference between two sets.
; 
; Inputs:
; - $A: The first set of type `Atom`.
; - $B: The second set of type `Atom`.
;
; Output:
; - The symmetric difference of sets $A and $B, which is the set of elements 
;   that are in either of the sets $A or $B but not in their intersection.
;
; The function is defined as:
; - symmetric-difference(A, B) = union(subtraction(A, B), subtraction(B, A))


(: symmetric-difference (-> Atom Atom Expression) )
(= (symmetric-difference $A $B)
  (union 
    (subtraction $A $B) 
    (subtraction $B $A)
  )
)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

; This code defines a function `subset`.
; 
; Inputs:
; - $A: The first set of type `Atom`.
; - $B: The second set of type `Atom`.
;
; Output:
; - A `Bool` indicating whether $A is a subset of $B.
;
; The function works by checking if the result of subtracting $B from $A
; (using the `subtraction` function) collapses to an empty set.
; If the collapsed result is empty, it returns `True`, indicating that $A is a subset of $B.
; Otherwise, it returns `False`.


(: subset (-> Atom Atom Bool) )
(= (subset $A $B)
  (if (== (collapse (subtraction $A $B)) ()) True False)
)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


; This code defines a function `equivalent` that takes two atoms, $A and $B, 
; and returns a boolean value indicating whether the two atoms are equivalent.
; 
; Inputs:
; - $A: The first set of type `Atom`.
; - $B: The second set of type `Atom`.
;
; Output:
; - A `Bool` indicating whether $A and $B are equivalent.
; 
; The equivalence is determined by checking if $A is a subset of $B and $B is a subset of $A.


(: equivalent (-> Atom Atom Bool) )
(= (equivalent $A $B) (
    and (subset $A $B) (subset $B $A)
))




;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;            TEST                     ;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

!(add-reduct &self (= (get-space) (new-space)))

!(add-atom (get-space) A)
!(add-atom (get-space) B)
!(add-atom (get-space) C)
!(add-atom (get-space) D)
!(add-atom (get-space) E)

!(get-atoms (get-space))   ;; this function will return [A,B,C,D,E]



!(compliment (get-space) (superpose (A B C)))
!(symmetric-difference (superpose (A B C F))  (superpose (A B C D E)))
!(subset (superpose (A B C F))  (superpose (A B C D E)))
!(equivalent (superpose (A B C))  (superpose (A B C)) )
