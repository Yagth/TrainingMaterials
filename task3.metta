;; Optimized any function using non-deterministic operations
(= (any-optimized $list)
    (collapse 
        (case (superpose $list)
            (($x 
                (if $x
                    True
                    (empty))))))
)

!(any-optimized (True False))
!(any-optimized (True False False True))
!(any-optimized (False False False False False False False False False False False True False False True))