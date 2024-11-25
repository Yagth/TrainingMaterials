;; Add multiple atoms at once
;; Bind a new space
!(bind! &newspace (new-space))

(= (nd-add-atom $space $atoms)
   (add-atom $space  (superpose $atoms)))

(= (nd-remove-multiple $space $remove)
   (remove-atom $space  (superpose $remove)))

!(nd-add-atom &newspace ((A B C D E F)))
!(match &newspace $x $x)
!(nd-remove-multiple &newspace ((A B C D E F)))
!(match &newspace $x $x)