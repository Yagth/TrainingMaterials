# **Set Operations and Non-Deterministic Space Manipulation**

## **Introduction**

This repository implements fundamental set operations, a generic filter function, optimized utility functions (any, member), and non-deterministic space manipulation. These tasks are designed for efficient handling of sets, flexible filtering, and managing non-deterministic spaces in MeTTa, providing a foundation for solving computational problems involving sets and dynamic spaces.

---

## **Implemented Features**

### **1. Set Operations**
- **Complement Function (compelement)**  
  Computes the complement of a subset from the universal set.  
  **Example:** Complement of {1, 2} in {1, 2, 3, 4} is {3, 4}.

- **Symmetric Difference Function (symmetric-difference)**  
  Returns elements that are in either of the sets but not in their intersection.  
  **Example:** Symmetric difference of {1, 2} and {2, 3} is {1, 3}.

- **Subset Check (subset)**  
  Verifies if one set is a subset of another.  
  **Example:** {1, 2} is a subset of {1, 2, 3}.

- **Equivalent Sets (equivalent)**  
  Checks if two sets contain the same elements.  
  **Example:** {1, 2, 3} is equivalent to {3, 2, 1}.

---

### **2. Generic Filter Function**
Implements a customizable filter function to extract elements from a set based on a condition.  
**Example:** Filter even numbers from {1, 2, 3, 4} → {2, 4}.

---

### **3. Optimized any Function**
A refined implementation of the any function for checking the presence of True in a list.  
**Example:** For [False, False, True], the result is True.

---

### **4. Member Check Function**
Verifies if an element is present in a given set.  
**Example:** 5 is a member of {1, 2, 3, 5} → True.

---

### **5. Non-Deterministic Space Initialization**
Adds multiple atoms non-deterministically to a newly created space in one operation.  
**Example:** Add {A, B, C} to a space.

---

### **6. Non-Deterministic Removal of Atoms**
Removes multiple atoms from a non-deterministic space.  
- This function ensures flexibility in managing space by removing specific atoms dynamically.  
- Validates the state of the space after removal.

---

## **Usage**

### **1. Initialize a New Space**
metta
!(bind! &x (new-space))  ; Create a new space
