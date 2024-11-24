;; --- Set Operations ---

;; Complement: returns all elements in B that are not in A
(= (complement $a $b)
   (collapse(subtraction (superpose $b) (superpose $a))))

;; Symmetric Difference: (A - B) ∪ (B - A)
(= (symmetric-difference $a $b)
   (collapse (union
      (subtraction (superpose $a) (superpose $b))
      (subtraction (superpose $b) (superpose $a)))))

;;Subset: Returns True if all elements of A are in B or (A n B)=A then A is a subset of B
(= (subset $a $b)
   (== (collapse (intersection (superpose $a) (superpose $b))) $a))

;; Equivalent: True if A is a subset of B and B is a subset of A
(= (equivalent $a $b)
   (and
      (subset $a $b)
      (subset $b $a)))

;; Example Usage for Set Operations:

!(subset (A)  (A B C D))  ; Output: True
!(subset (A B E) (A B C D))  ; Output: False

!(symmetric-difference (A B C) (C D E))  ; Output: (A B D E)
!(complement (A B) (A B C D))  ; Output: (B C D)
!(equivalent (A B C) (A B C))  ; Output: True
!(equivalent (A B) (A C))  ; Output: False
!(equivalent (A B) (A D))  ; Output: False



;; --- Generic Filter Function ---

;; Generic Filter function that applies a predicate to a list
(= (filter $list $predicate)
   ($predicate (superpose $list)))

; Example: Increment each number in a list by 1 using a filter function
(= (increment $x) (+ $x 1))
!(filter (1 2 3 4) increment)  ; Output: (2 3 4 5)


;; --- Optimized any Function ---

;; isTrue: Helper function to check if a value is truthy
(= (isTrue $x) (if $x True (empty)))

;; any: Returns True if any element in the list is truthy, otherwise False 
;;the previous implementaion had a problem it was returning True every time i also fixed that
(= (any $x) (not (== (collapse (isTrue (superpose $x))) ())))

; Example Usage for any function:

!(any (False False))  ;; Output: False
!(any (True False False True))  ;; Output: True
!(any (False False False False False False False False False False False True False False True))  ; Output: True



;; Member Check: Check if an element is a member of a list
(= (member $x $b)
    (not (== (collapse (intersection (superpose ($x)) (superpose $b))) ())))

!(member A (A B C D))  ;; Output: True
!(member F (A B C D))  ;;output:False


;; --- Non-Deterministic Space Operations ---

;; Bind a new space
!(bind! &x (new-space))

;; Add multiple atoms to a space non-deterministically
(= (add-atoms $space $atoms)
   (add-atom $space  (superpose $atoms)))

;; Remove multiple atoms from a space non-deterministically
(= (remove-atoms $space $remove)
   (remove-atom $space  (superpose $remove)))

;; Example Usage for Non-Deterministic Space Operations:
!(match &x $x $x) ;;output []
!(add-atoms &x ((A 1) (B 2) (C 3) (D 4) (B 1) (A C 2) (A D 3)))
!(match &x $x $x) ;;output [(A 1), (B 2), (C 3), (D 4), (B 1), (A C 2), (A D 3)]
!(remove-atoms &x ((A 1) (B 2) (C 3) (D 4) (B 1) (A C 2) (A D 3)))
!(match &x $x $x) ;;output []



