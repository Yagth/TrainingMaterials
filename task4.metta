;; Member check function that verifies if an element exists in a list
(Male Solomon)
(Male Dawit)
(Male Bereket)

(Female Helen)
(Female Rahel)

(Parent Solomon)
(Parent Helen)

(Child Bereket Solomon)
(Child Rahel Solomon)

(= (member $elem $list)
    (collapse 
        (case (superpose $list)
            (($x 
                (if (== $x $elem)
                    True
                    (empty)))))))

;; Example usage:
!(member A (A B C D))
!(member E (A B C D))       
!(member Solomon (Solomon Dawit Bereket))  
!(member Alex (Solomon Dawit Bereket))