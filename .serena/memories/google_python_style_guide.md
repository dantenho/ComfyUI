# Google Python Style Guide - ComfyUI Reference

**Source:** https://google.github.io/styleguide/pyguide.html  
**Added:** January 5, 2026  
**Purpose:** Comprehensive Python coding standards for ComfyUI development

## Overview

This is Google's Python Style Guide, used as a reference for maintaining high-quality, consistent Python code in the ComfyUI project. The guide covers language rules and style rules.

## Key Language Rules (Section 2)

### 2.1 Lint
- **Always run pylint** on your code using Google's pylintrc
- Suppress warnings with inline comments when appropriate: `# pylint: disable=invalid-name`
- Pylint catches typos, using-vars-before-assignment, and other common errors

### 2.2 Imports
**DO:**
- `import x` for packages and modules
- `from x import y` where x is package prefix, y is module name
- `from x import y as z` for conflicts or standard abbreviations (e.g., `import numpy as np`)
- Use full package names, no relative imports

**DON'T:**
- Import individual types, classes, or functions directly
- Use relative imports even within the same package

### 2.4 Exceptions
**Best Practices:**
- Use built-in exception classes (ValueError, TypeError, etc.)
- **Never** use bare `except:` or catch `Exception` without re-raising
- Minimize code in `try` blocks
- Don't use `assert` for production logic (only for tests)
- Document exceptions in docstrings under `Raises:` section

### 2.5 Mutable Global State
**Avoid mutable global state**
- If unavoidable, make internal with `_` prefix
- Use module-level constants freely (ALL_CAPS_WITH_UNDERSCORES)
- Examples:
  - Internal: `_MAX_HOLY_HANDGRENADE_COUNT = 3`
  - Public: `SIR_LANCELOTS_FAVORITE_COLOR = "blue"`

### 2.7 Comprehensions & Generator Expressions
**Allowed for simple cases:**
```python
# YES
result = [mapping_expr for value in iterable if filter_expr]
return {x: complicated_transform(x) for x in generator() if x is not None}

# NO - multiple for clauses
result = [(x, y) for x in range(10) for y in range(5) if x * y > 10]
```

### 2.8 Default Iterators and Operators
**Prefer default iterators:**
```python
# YES
for key in adict: ...
if obj in alist: ...
for k, v in adict.items(): ...

# NO
for key in adict.keys(): ...
for line in afile.readlines(): ...
```

### 2.10 Lambda Functions
- OK for one-liners (60-80 chars max)
- Prefer generator expressions over `map()` or `filter()` with lambda
- Use `operator` module for common operations: `operator.mul` instead of `lambda x, y: x * y`

### 2.13 Properties
- Use `@property` decorator for trivial computations
- Should match expectations of typical attribute access
- Document in the getter method

### 2.14 True/False Evaluations
**Use implicit false when possible:**
```python
# YES
if not users:
    print('no users')

if i % 10 == 0:
    self.handle_multiple_of_ten()

def f(x=None):
    if x is None:
        x = []

# NO
if len(users) == 0:
    print('no users')

if not i % 10:
    self.handle_multiple_of_ten()
```

**Caveats:**
- Always use `if foo is None:` for None checks
- Never compare boolean to False: use `if not x:` instead
- For sequences, use emptiness check: `if not seq:`
- Note: `'0'` (string) evaluates to True
- NumPy arrays may raise exception in implicit boolean context (use `.size`)

### 2.17 Decorators
- Use judiciously when there's a clear advantage
- Write unit tests for decorators
- Avoid external dependencies in decorators
- **Never use `@staticmethod`** (use module-level function instead)
- Use `@classmethod` only for named constructors or class-specific routines

### 2.21 Type Annotations
**Strongly encouraged:**
```python
def func(a: int) -> list[int]:
    ...

a: SomeType = some_func()
```
- Use PEP-484 type hints
- Type-check with pytype
- Improves readability and catches runtime errors at build time

## Key Style Rules (Section 3)

### 3.1 Semicolons
**Never use semicolons** - not at line endings, not to separate statements

### 3.2 Line Length
**Maximum: 80 characters**

**Exceptions:**
- Long import statements
- URLs, pathnames, long flags in comments
- Long string constants
- Pylint disable comments

**Use implicit line joining:**
```python
# YES
foo_bar(self, width, height, color='black', design=None, x='foo',
        emphasis=None, highlight=0)

if (width == 0 and height == 0 and
    color == 'red' and emphasis == 'strong'):

# NO - backslash continuation
if width == 0 and height == 0 and \
    color == 'red' and emphasis == 'strong':
```

### 3.4 Indentation
**4 spaces, never tabs**

```python
# YES - Aligned with opening delimiter
foo = long_function_name(var_one, var_two,
                         var_three, var_four)

# YES - 4-space hanging indent
foo = long_function_name(
    var_one, var_two, var_three,
    var_four)

# NO - 2-space hanging indent
foo = long_function_name(
  var_one, var_two, var_three,
  var_four)
```

### 3.8 Docstrings

#### 3.8.1 Format
- Use `"""triple double quotes"""`
- Summary line: one physical line, max 80 chars, ends with period
- Blank line after summary (if more content follows)
- Rest of docstring at same indentation as opening quotes

#### 3.8.2 Module Docstrings
```python
"""A one-line summary of the module or program, terminated by a period.

Leave one blank line.  The rest of this docstring should contain an
overall description of the module or program.  Optionally, it may also
contain a brief description of exported classes and functions and/or usage
examples.

Typical usage example:

  foo = ClassFoo()
  bar = foo.function_bar()
"""
```

#### 3.8.3 Function/Method Docstrings
**Required for:**
- Public API functions
- Non-trivial size functions
- Non-obvious logic

**Special sections:**
- **Args:** Parameter name, colon, description (include types if no type annotations)
- **Returns:** (or **Yields:** for generators) Describe return value semantics
- **Raises:** List relevant exceptions

```python
def fetch_smalltable_rows(
    table_handle: smalltable.Table,
    keys: Sequence[bytes | str],
    require_all_keys: bool = False,
) -> Mapping[bytes, tuple[str, ...]]:
    """Fetches rows from a Smalltable.

    Retrieves rows pertaining to the given keys from the Table instance
    represented by table_handle.  String keys will be UTF-8 encoded.

    Args:
        table_handle: An open smalltable.Table instance.
        keys: A sequence of strings representing the key of each table
          row to fetch.  String keys will be UTF-8 encoded.
        require_all_keys: If True only rows with values set for all keys will be
          returned.

    Returns:
        A dict mapping keys to the corresponding table row data
        fetched. Each row is represented as a tuple of strings.

    Raises:
        IOError: An error occurred accessing the smalltable.
    """
```

### 3.16 Naming Conventions

| Type | Convention | Example |
|------|-----------|---------|
| Module | `lower_with_under.py` | `module_name.py` |
| Package | `lower_with_under` | `package_name` |
| Class | `CapWords` | `ClassName` |
| Exception | `CapWords` (ends in Error) | `ExceptionName` |
| Function | `lower_with_under()` | `function_name()` |
| Method | `lower_with_under()` | `method_name()` |
| Global Constant | `UPPER_WITH_UNDER` | `GLOBAL_CONSTANT_NAME` |
| Global Variable | `lower_with_under` | `global_var_name` |
| Instance Variable | `lower_with_under` | `instance_var_name` |
| Local Variable | `lower_with_under` | `local_var_name` |

**Avoid:**
- Single character names (except counters/iterators)
- Dashes (`-`) in any package/module name
- `__double_leading_and_trailing_underscore__` names (reserved by Python)

**Guidelines:**
- Use `_` prefix for internal/protected module variables
- Use `.py` extension, never dashes
- Be descriptive, avoid abbreviations

### 3.17 Main
**Make files importable:**
```python
def main():
    ...

if __name__ == '__main__':
    main()
```
- Main functionality should be in `main()` function
- Prevents side effects when importing
- Enables unit testing and pydoc

### 3.19 Type Annotation Details

**General Rules:**
- Familiarize with PEP-484
- Only annotate `self`/`cls` if necessary for type information
- Use `Any` for unexpressible types
- At minimum, annotate public APIs
- Annotate error-prone code

**Line Breaking:**
```python
# One parameter per line after annotating
def my_method(
    self,
    first_var: int,
    second_var: Foo,
    third_var: Optional[Bar]
) -> int:
    ...
```

**Default Values (PEP-008):**
```python
# YES - spaces around = when both type annotation and default value
def func(a: int = 0) -> int:
    ...

# NO
def func(a:int=0) -> int:
    ...
```

## ComfyUI-Specific Application

### Integration Points
1. **All Python modules** in ComfyUI should follow these conventions
2. **Custom nodes** should adhere to naming and docstring standards
3. **API endpoints** should have complete type annotations
4. **Utility functions** should use proper error handling (no bare except)
5. **Configuration** should use module-level constants (UPPER_CASE)

### Priority Areas
- Type annotations on public APIs (`comfy_api/`, `api_server/`)
- Comprehensive docstrings on node classes
- Proper exception handling in server routes
- Import organization in large modules

### Tools Integration
- **Linting:** Use pylint with Google's pylintrc
- **Formatting:** Black or Pyink for auto-formatting
- **Type Checking:** pytype for static analysis
- **Documentation:** Sphinx with Google-style docstrings (Napoleon extension)

## Quick Reference Card

**DO:**
✅ Use type annotations on public APIs  
✅ Write docstrings for public, non-trivial, or non-obvious functions  
✅ Use implicit false: `if not users:`  
✅ 4 spaces for indentation  
✅ 80 character line limit  
✅ Import modules, not individual functions  
✅ Use comprehensions for simple cases  
✅ Default iterators: `for key in dict:`  
✅ `if __name__ == '__main__':` for executables  

**DON'T:**
❌ Use semicolons  
❌ Use tabs for indentation  
❌ Use backslash line continuation  
❌ Use bare `except:` or catch Exception without re-raising  
❌ Use mutable global state  
❌ Use `@staticmethod`  
❌ Import individual types/classes/functions  
❌ Use multiple `for` clauses in comprehensions  
❌ Use relative imports  

## Related Resources

- **PEP 8:** Python style guide baseline
- **PEP 257:** Docstring conventions
- **PEP 484:** Type hints
- **Black:** Auto-formatter
- **pylint:** Linter
- **pytype:** Type checker
- **Sphinx Napoleon:** Google-style docstring extension

---

**Note:** This guide should be consulted when writing new Python code or refactoring existing code in the ComfyUI project. When in doubt, prioritize readability and consistency with existing code in the same module.
