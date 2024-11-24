; Non deterministic space initialization (Adding multiple atoms into the space at once)


(= (add-atoms-nd $space $atoms) 
        (add-atom $space (superpose $atoms))
)


; Non deterministic removal of multiple atoms

(= (remove-atoms-nd $space $atoms) 
        (remove-atom $space (superpose $atoms))
)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;       TEST  For Adding Atoms        ;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

!(add-reduct &self (= (get-space) (new-space)))


!(add-atoms-nd (get-space) (A B C D E))

;;;;;;;;;;;;;;;;;
;;; Proof ;;;;;;;
;;;;;;;;;;;;;;;;;

!(get-atoms (get-space))  ;; this will return the atoms we added in to our space 



;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;       TEST  For Removing Atoms      ;;;;;;;;;;;;;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

!(remove-atoms-nd (get-space) (A B C D E))


;;;;;;;;;;;;;;;;;
;;; Proof ;;;;;;;
;;;;;;;;;;;;;;;;;

!(get-atoms (get-space))  ;; this will return the empty because we remove all atome we added 


