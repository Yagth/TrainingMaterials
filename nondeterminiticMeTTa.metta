(Male Solomon)
(Male Dawit)
(Male Bereket)

(Female Helen)
(Female Rahel)

(Parent Solomon)
(Parent Helen)

(Child Bereket Solomon)
(Child Rahel Solomon)


;; (1 2 3 4 5)
;; (= (sum () $acc) $acc)
;; (1 2 3 4 5) ==> (sum (2 3 4 5) (+ 1 0))
;; (= (sum $x $acc) )

;; (= (sum $x $acc)
;;       (case $x
;;         (
;;           (() $acc)
;;           ($x (sum (cdr-atom $x) (+ (car-atom $x) $acc)))
;;         )
;;     )
;; )


;; ;; Superpose: A function to change a list of atoms into non deterministic values
;; !(superpose (A 1 2 C D E))
;; !((superpose (A B C D)) (superpose (E F G I)))
;; !((superpose (A B C D)) (superpose (A B C D)) )


;; Collapse: A function that changes a non deterministic value into a list of atom (Deterministic)
;; !(match &self (Male $x) $x)
;; !(collapse (match &self (Male $x) $x))



;; ;; Unify: A function to filter non deterministic branches using pattern matching
;; !(let $x (superpose ((A 1) (B 2) (C 3) (D 4) (A B 1) (A C 2) (A D 3)))
;;       (unify ($y $i) $x ($y $i) ())
;;  )
;; !(unify (FMale $x) &self True False)

;; ;; Match: A function to get atoms that satisfies a given pattern
;; !(match &self (Female $x) $x)

;; ;; ;; Union: A function to combine two lists of atoms into a single list
;; !(union (superpose (A B C)) (superpose (C D E)))
;; !(union (superpose (A B C)) (superpose (1 D E)))



;; ;; Unique: A function to remove duplicate atoms from a list
;; !(unique (superpose (A B A C A A E E)))



;; ;; Subtraction: A function to remove a list of atoms from another list of atoms
;; A - B
;; !(subtraction (superpose (A B C D E F)) (superpose (A B C D)))
;; !(subtraction (superpose (A A B C D E F)) (superpose (A B C D)))




;; ;; Intersection: A function to get common atoms in two lists of atoms
;; !(intersection (superpose (A B C D E)) (superpose (A D F G H)))


;; ;; Bind: A function to bind a token to a certain value
;; !(bind! &x (new-space))

;; !(match &x &y &y)

;; ;; ;; AddAtom: A function to add a list of atoms to a given space
;; !(add-atom &x A)
;; !(add-atom &x (A B))
;; !(add-atom &x (C D))
;; !(match &x $x $x)

;; ;; RemoveAtom: A function to remove a list of atoms from a given space
;; !(remove-atom &x A)
;; !(match &x $x $x)

;; Empty: A function used to return no result

;; !(let $x (superpose ((A 1) (B 2) (C 3) (D 4) (A B 1) (A C 2) (A D 3)))
;; (unify ($y $i) $x ($y $i) (empty))
;; )

;; (= (foo $x) (if (== $x 1) (Did something) (empty)))
;; !(foo 1)
;; !(foo 2)

;; Map function using ND
;; !(+ 1 (superpose (1 2 3 4)))

;; any function using ND
;; (= (any' $x $acc) (if (== $x ()) $acc (any' (cdr-atom $x) (or $acc (car-atom $x)))))

;; !(any' (True False) False)
;; !(any' (True False False True) False)
;; !(any' (False False False False False False False False False False False True False False True) False)

;; (= (isTrue $x) (if $x True (empty)))
;; (= (any $x) (not (== (collapse (isTrue (superpose (true)))) ())))

;; !(any (True False))
;; !(any (True False False True))
;; !(any (False False False False False False False False False False False True False False True))


;; TODO: Tasks

;; 1. Set operation functions
;;    - Compelement
;;    - Symmetric difference
;;    - Subset
;;    - Equivalent
;; 2. A generic filter function
;; 3. Optimize the any function
;; 4. A member check function
;; 5. Non deterministic space initialization (Adding multiple atoms into the space at once)
;; 6. Non deterministic removal of multiple atoms:
;;      This function performs non-deterministic removal of multiple atoms from a given space.
;;      The atoms that need to be removed should be added using a non-deterministic space 
;;        initialization process of the above task.
;;      If removing the atoms added in the above task with this function isn't possible,
;;        then both the above and this function are considered unfinished.
