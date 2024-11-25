;; Generic Filter Function

;; Filters a list based on a predicate function
(= (filter $list $predicate)
    (case $list
        (()
            ()) ;; Empty list, return empty
        (($x . $xs)
            (if ($predicate $x)
                (cons $x (filter $xs $predicate))
                (filter $xs $predicate)))
    )
)