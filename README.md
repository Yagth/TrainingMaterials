
# MeTTa Non-Deterministic

This document outlines the functions and examples for the MeTTa Non-Deterministic (ND) library. It includes various operations such as superposition, union, intersection, filter, and other set operations that can be used to manipulate and interact with non-deterministic values and atoms.

## Functions

### 1. Sum Function

The `sum` function calculates the sum of a list recursively.

```metta
(= (sum $x $acc)
    (case $x
        (
            (() $acc)
            ($x (sum (cdr-atom $x) (+ (car-atom $x) $acc)))
        )
    )
)
```

### 2. Superpose

A function to change a list of atoms into non-deterministic values.

```lisp
!(superpose (A 1 2 C D E))
!(superpose (A B C D))
!(superpose (E F G I))
```

### 3. Collapse

A function that converts a non-deterministic value into a list of deterministic atoms.

```lisp
!(match &self (Male $x) $x)
!(collapse (match &self (Male $x) $x))
```

### 4. Unify

A function to filter non-deterministic branches using pattern matching.

```lisp
!(let $x (superpose ((A 1) (B 2) (C 3) (D 4) (A B 1) (A C 2) (A D 3)))
   (unify ($y $i) $x ($y $i) ())
)
```

### 5. Match

A function to retrieve atoms that satisfy a given pattern.

```lisp
!(match &self (Female $x) $x)
```

### 6. Union

A function to combine two lists of atoms into a single list.

```lisp
!(union (superpose (A B C)) (superpose (C D E)))
!(union (superpose (A B C)) (superpose (1 D E)))
```

### 7. Unique

A function to remove duplicate atoms from a list.

```lisp
!(unique (superpose (A B A C A A E E)))
```

### 8. Subtraction

A function to remove a list of atoms from another list.

```lisp
!(subtraction (superpose (A B C D E F)) (superpose (A B C D)))
!(subtraction (superpose (A A B C D E F)) (superpose (A B C D)))
```

### 9. Intersection

A function to get common atoms in two lists.

```lisp
!(intersection (superpose (A B C D E)) (superpose (A D F G H)))
```

### 10. Bind

A function to bind a token to a certain value.

```lisp
!(bind! &x (new-space))
!(match &x &y &y)
```

### 11. AddAtom

A function to add a list of atoms to a given space.

```lisp
!(add-atom &x A)
!(add-atom &x (A B))
!(add-atom &x (C D))
!(match &x $x $x)
```

### 12. RemoveAtom

A function to remove a list of atoms from a given space.

```lisp
!(remove-atom &x A)
!(match &x $x $x)
```

### 13. Empty

A function to return no result.

```lisp
!(let $x (superpose ((A 1) (B 2) (C 3) (D 4) (A B 1) (A C 2) (A D 3)))
   (unify ($y $i) $x ($y $i) (empty))
)
```

### 14. Map Function Using ND

```lisp
!(+ 1 (superpose (1 2 3 4)))
```

### 15. Any Function Using ND

```lisp
(= (any' $x $acc) (if (== $x ()) $acc (any' (cdr-atom $x) (or $acc (car-atom $x)))))
!(any' (True False) False)
!(any' (True False False True) False)
!(any' (False False False False False False False False False False False True False False True) False)
```

### 16. Task Examples

#### Complement

The complement of two sets is the set of elements that are in the first set but not in the second.

```lisp
(= (complement $a $b)
    (subtraction $a $b)
)
!(complement (superpose (A B C D E)) (superpose (B D)))
```

#### Symmetric Difference

The symmetric difference of two sets is the set of elements that are in either of the sets, but not in their intersection.

```lisp
(= (symmetric-difference $a $b)
    (union (subtraction $a $b) (subtraction $b $a))
)
!(symmetric-difference (superpose (A B C)) (superpose (B C D)))
```

#### Subset

Check if one set is a subset of another.

```lisp
(= (subset $a $b)
    (if (== (intersection $a $b) $a) True False)
)
!(subset (superpose (A B)) (superpose (A B C)))
```

#### Equivalent

Check if two sets are equivalent (i.e., they contain the same elements).

```lisp
(= (equivalent $a $b)
    (and (==(intersection $a $b) $a) (==(intersection $a $b) $b))
)
!(equivalent (superpose (A B C)) (superpose (C A B)))
```

#### Filter Function

A generic filter function that applies a predicate to a list.

```lisp
(= (filter $list $predicate)
    (case $list
        (
            (() ()) 
            ($list
                (if ($predicate (car-atom $list))
                    (cons (car-atom $list) (filter (cdr-atom $list) $predicate))
                    (filter (cdr-atom $list) $predicate)
                )
            )
        )
    )
)
!(filter (superpose (1 2 3 4 5)) (lambda ($x) (== (% $x 2) 0)))
```

#### Optimize the Any Function

```lisp
(= (any $x)
    (case $x
        (
            (() False)
            ($x (or (car-atom $x) (any (cdr-atom $x)))))
    )
)
!(any (superpose (False False True False)))
```

#### Member Check Function

Check if an element is a member of a list.

```lisp
(= (member $element $list)
    (case $list
        (
            (() False)
            ($list (or (== $element (car-atom $list)) (member $element (cdr-atom $list))))
        )
    )
)
!(member D (superpose (A B C)))
```

#### Non-Deterministic Space Initialization

```lisp
(= (nd-initialize $space $atoms)
    (foreach $atom $atoms
        (add-atom $space $atom)
    )
)
!(let &x (new-space)
    (nd-initialize &x (superpose (A B C)))
    (match &x $x $x)
)
```

#### Non-Deterministic Removal of Multiple Atoms

```lisp
(= (nd-remove $space $atoms)
    (foreach $atom $atoms
        (remove-atom $space $atom)
    )
)
!(let &x (new-space)
    (nd-initialize &x (superpose (A B C)))
    (nd-remove &x (superpose (A C)))
    (match &x $x $x)
)
```

## Conclusion

MeTTa Non-Deterministic provides a rich set of functions for working with non-deterministic values and sets, enabling powerful pattern matching, set operations, and atom manipulations. These functions are ideal for situations where multiple potential outcomes need to be considered simultaneously.
