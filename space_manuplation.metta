;; This function is implemented using the non-deterministic nature of Metta to manipulate the space with multiple atoms at once.
;; Parameters:
;; - operation: Specifies whether to add or remove atoms.
;; - space: The atom space where multiple atoms are added or removed.
;; - elements: A tuple of atoms to be added or removed.


;;(:manuplate-space (-> $operation $space $tuple Expression))
(= (manuplate-space $opration $space $elements)(
    (superpose ($opration)) (superpose ($space)) (superpose $elements)
))


;; add-atom test case
!(manuplate-space add-atom &self ((Male Solomon)
                                  (Male Dawit)
                                  (Male Bereket)
                                  (Female Helen)
                                  (Female Rahel)
                                  (Parent Solomon)
                                  (Parent Helen)
                                 )
)

!(match &self (Male $person) $person) ;;Get Male from match [Dawit, Bereket, Solomon]
!(match &self (Parent $person) $person) ;; Get parents from match [Helen, Solomon]


;; remove-atom test case
!(manuplate-space remove-atom &self ((Male Solomon)
                                     (Male Dawit)
                                     (Male Bereket)

                                     (Female Helen)
                                     (Female Rahel)

                                     (Parent Solomon)
                                     (Parent Helen)
                                    )
)

!(match &self (Male $person) $person) ;; [] removed atoms
!(match &self (Parent $person) $person);; [] removed atoms