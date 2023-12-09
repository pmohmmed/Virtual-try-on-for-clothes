# Writing Functions 😎 :

### Naming Conventions:
- Choose meaningful name.
- Functions (snake_case naming): Example: `my_func()`, `add()`

#### Inside a Class:
- Functions inside the Class:
  - **Public:** Example: `add_numbers()`, `sub()`
  - **Protected:** Example: `_add_numbers()`, ` _sub() `
  - **Private:** Example: `__add_numbers()`, ` __sub() `

### Function Structure: 

#### Define The Type of the Parameters and the Output in Python3
```python
def add_numbers(x: int, y: int) -> int:
    """
    add_numbers does blah blah blah.

    """ 
    result = x + y
    return result
```
#### Or if you don't know exactly the types of the parameters and return 
```python
def test_function(p1, p2, p3):
    """
    test_function does blah blah blah.

    :param p1: describe about parameter p1
    :param p2: describe about parameter p2
    :param p3: describe about parameter p3
    :return: describe what it returns
    """ 
```


# Writing Variables 📝 :

### Naming Convention:
- Choose meaningful name.
- Variables ( snake_case naming): Example: `my_name`, `name`
- Constant Variables (UPPER_Case) : `PI=3.14` , ` MY_VARIABLE `

#### Inside a Class:
- Attributes inside the Class:
  - **Public:** Example: `my_name`
  - **Protected:** Example: `_my_name`
  - **Private:** Example: `__my_name`

### Variable Structure: 

#### Add comment above the variable.
```python
# variable of type int to store the result 
res=0
```

---

# Writing a Class 🏛️ :

### Naming Convention:
- (CamelCase naming): Example: `MyClass`, `Person`

---

# Creating Module 📂 :

### Naming Convention:
- Modules (Module is a Python file like `main.py`):
  - Use lowercase names with underscores to separate words. Example: `my_file`, `pre_processing`, `helpers`

---

# Creating Package 📦 :

### Naming Convention:
- Packages:
  - Use lowercase names with no underscores. Example: `mypackage`

---
