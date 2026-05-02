References:
  - https://x.com/i/grok/share/bMePYcq88mgJsVsQ7JN68UZOu

---

# Wat: A Pure, Strongly Typed, English-Like Lisp

Wat is a domain-specific language (DSL) designed to blend Lisp’s functional purity and homoiconicity with English-like sentence expressivity. It is purely functional—no I/O operations are permitted—ensuring all computations are side-effect-free. With a minimal core of 28 primitives, a rich userland built via `(lambda ...)`, and a strong type system enforced by traits, Wat balances simplicity, power, and usability.

This documentation is crafted for implementation, providing precise syntax, semantics, type rules, and examples for every component. It assumes a Ruby-based s-expression parser as a reference (per earlier discussions), but remains agnostic to the host language’s I/O capabilities, focusing solely on pure computation.

---

## Core Concepts

- **S-Expressions**: All constructs are parenthesized, prefix-notation expressions (e.g., `(add 3 2)` rather than `3 + 2`).
- **Purity**: No I/O—computations produce values or assert facts within a closed system.
- **Strong Typing**: Types (e.g., `Noun`, `Integer`) and traits (e.g., `Numeric`, `Assertable`) enforce semantic correctness at parse time.
- **Homoiconicity**: Code is data, manipulable via `(list ...)` and `(quote ...)`.
- **User Experience**: Optional maps (e.g., `(entity Noun "dog")`) and SVO-ordered `(Statement ...)` eliminate boilerplate and align with English readability.

---

## Core Primitives (28)

These are the built-in s-expressions forming Wat’s foundation. Each is pure, producing a value or asserting a fact without side effects.

### 1. `(entity Type value [map-expr])`
- **Purpose**: Creates a typed entity, the atomic building block of Wat.
- **Syntax**:
  - `(entity Type value)` – Minimal form, defaults to `(map)` for optional args.
  - `(entity Type value (map :key1 val1 :key2 val2 ...))` – Explicit map.
  - `(entity Type value :key1 val1 :key2 val2 ...)` – Shorthand, rewrites to `(map ...)`.
- **Arguments**:
  - `Type`: Required—One of `Noun`, `Verb`, `Time`, `Adverb`, `String`, `Integer`, `Float`, `Boolean`, `Pronoun`, `Preposition`, `Adjective`, `Error`.
  - `value`: Required—Matches `Type`:
    - `Noun`, `Verb`, `Time`, `Adverb`, `String`, `Preposition`, `Adjective`, `Error`: String (e.g., `"dog"`, `"chases"`, `"division by zero"`).
    - `Integer`: Unquoted integer (e.g., `3`, `-5`).
    - `Float`: Unquoted float (e.g., `2.5`, `-3.14`).
    - `Boolean`: `true` or `false`.
    - `Pronoun`: String (e.g., `"it"`).
  - `map-expr`: Optional—`(map :key1 val1 ...)` with:
    - `:role`: `Subject` or `Object` (defaults to `nil` if omitted).
    - `:adjective`: `(entity Adjective ...)` (defaults to `nil` if omitted).
- **Semantics**: Returns an entity object with `Type`, `value`, and optional attributes (`role`, `adjective`) stored in a map (default `(map)`).
- **Type Rules**:
  - `value` must match `Type`’s expected format.
  - `:role` must be `Subject` or `Object` if present.
  - `:adjective` must be an `Adjective` entity.
- **Examples**:
  - `(entity Noun "dog" :role Subject)` – A "dog" entity with subject role.
  - `(entity Verb "chases")` – A "chases" verb, no role or adjective (implicit `(map)`).
  - `(entity Integer 5)` – The number 5.
  - `(entity Boolean true)` – The boolean `true`.
- **Implementation Notes**:
  - Parser should rewrite `(entity Type value)` to `(entity Type value (map))`.
  - Store as a struct: `{type: Type, value: value, attrs: {role: val, adjective: val}}`.
  - Validate `value` against `Type` at parse time (e.g., `"dog"` for `Noun`, not `3`).

### 2. `(relate verb subject object [map-expr])`
- **Purpose**: Internal constructor for `(Statement ...)`, builds an SVO relationship with optional modifiers in a map.
- **Syntax**:
  - `(relate verb subject object)` – Minimal form, defaults to `(map)` for optional args.
  - `(relate verb subject object (map :key1 val1 :key2 val2 ...))` – Explicit map.
  - `(relate verb subject object :key1 val1 :key2 val2 ...)` – Shorthand, rewrites to `(map ...)`.
- **Arguments**:
  - `verb`: `(entity Verb ...)` – Required, the action or state.
  - `subject`: `(entity Noun ... :role Subject)` or `(list ...)` of such – Required, the actor.
  - `object`: `(entity Noun ... :role Object)` or `(entity Pronoun ...)` – Required, the acted-upon.
  - `map-expr`: Optional—`(map :adverb val :time val :number val)` with:
    - `:adverb`: `(entity Adverb ...)` – Optional, modifies the verb (defaults to `nil`).
    - `:time`: `(entity Time ...)` – Optional, specifies when (defaults to `nil`).
    - `:number`: `(entity Integer/Float ...)` or unquoted number – Optional, specifies frequency (defaults to `nil`).
- **Semantics**: Constructs a relation object used by `(Statement ...)`, combining required SVO elements with optional modifiers. Not typically user-facing—`(Statement ...)` wraps it with SVO order.
- **Type Rules**:
  - `verb`: Must implement `RelatableVerb`.
  - `subject`: Must implement `Relatable` with `:role Subject`.
  - `object`: Must implement `Relatable` with `:role Object`.
  - `:adverb`: Must implement `Adverbial` if present, or omitted (`nil`).
  - `:time`: Must implement `Timeable` if present, or omitted (`nil`).
  - `:number`: Must implement `Numeric` if present, or omitted (`nil`).
- **Examples**:
  - `(relate chases dog toy)` – Internal form of "dog chases toy" (implicit `(map)`).
  - `(relate chases dog toy :adverb quickly :time t-0 :number 5)` – Internal form of "dog quickly chases toy 5 times at t0".
- **Implementation Notes**:
  - Parser rewrites `(relate verb subject object)` to `(relate verb subject object (map))`.
  - Returns a `Relate` object: `{verb: v, subject: s, object: o, attrs: {adverb: a, time: t, number: n}}`.
  - Type check all slots at parse time—`(Statement ...)` delegates to this, reordering to VSO internally.
  - Pure—no side effects, just constructs a data structure.

### 3. `(assert expr)`
- **Purpose**: Declares an `Assertable` expression as a fact within the system.
- **Syntax**: `(assert expr)`
- **Arguments**:
  - `expr`: Required—Must implement `Assertable` (e.g., `(Statement ...)`, `(entity Boolean ...)`, `(gt 5 3)`). Unevaluated s-expressions (e.g., from `(quote ...)`) are not `Assertable` unless explicitly evaluated.
- **Semantics**: Evaluates `expr` to a truth value (`true` or `false`); if `true`, succeeds silently; if `false`, fails (implementation-dependent—e.g., exception or flag).
- **Type Rule**: `expr` must be `Assertable`.
- **Examples**:
  - `(assert (Statement dog chases toy))` – "Dog chases toy."
  - `(assert true)` – "It is true."
  - `(assert (gt 5 3))` – "5 is greater than 3."
- **Implementation Notes**:
  - Evaluate `expr` to a `Boolean`—fail if not `true` or if `expr` isn’t `Assertable` (e.g., `(assert (quote (add 1 2)))` → `(entity Error "not assertable")`).
  - No I/O—failure could set an internal state or throw a pure exception (caught by `(try ...)`).

### 4. `(at time stmt)`
- **Purpose**: Scopes a statement to occur at a specific time.
- **Syntax**: `(at time stmt)`
- **Arguments**:
  - `time`: `(entity Time ...)` – Required.
  - `stmt`: `(Statement ...)` or similar – Required.
- **Semantics**: Returns an `Assertable` expression asserting `stmt` happens at `time`.
- **Type Rules**:
  - `time`: `Timeable`.
  - `stmt`: `Assertable`.
- **Example**: `(at t-0 (Statement dog chases toy))` – "At t0, the dog chases the toy."
- **Implementation Notes**:
  - Returns a structure: `{type: :at, time: t, stmt: s}`.
  - Evaluates to `true` if `stmt` holds at `time` (implementation defines time logic).

### 5. `(let ((label be value) ...) body ...)`
- **Purpose**: Creates a lexical scope for bindings and local traits.
- **Syntax**: `(let ((label be value) ...) body ...)`
- **Arguments**:
  - `(label be value)`: Binding pairs—`label` is a lower-kebab-case symbol (e.g., `dog`, `t-0`), `value` is any expression.
  - `body`: Zero or more expressions evaluated in the scope.
- **Semantics**: Binds each `label` to its `value` in a new scope; evaluates `body` sequentially, returning the last result.
- **Type Rule**: No restrictions—type checking occurs in `body`.
- **Examples**:
  - `(let ((x be 5)) (add x 3))` – Returns `(entity Integer 8)`.
  - `(let ((dog be (entity Noun "dog" :role Subject))) (Statement dog chases toy))`
- **Implementation Notes**:
  - Push a new environment with bindings: `{label: value}`.
  - Support local `(impl ...)` traits in `body`—pop on exit.
  - Return last `body` expression’s value or `nil` if empty.

### 6. `(lambda ((arg as Type) ...) returns ReturnType body)`
- **Purpose**: Defines a pure function for abstraction.
- **Syntax**: `(lambda ((arg as Type) ...) returns ReturnType body)`
- **Arguments**:
  - `(arg as Type)`: Parameters with types (e.g., `(x as Integer)`, `(stmt as Statement)`).
  - `returns ReturnType`: Expected return type (e.g., `Numeric`, `Statement`, `Boolean`).
  - `body`: One expression (multi-line via nesting, e.g., `(let (...) ...)`).
- **Semantics**: Creates a closure capturing the current scope; applies `body` when called with matching arguments.
- **Type Rules**:
  - `arg`: Must match `Type` when called.
  - `body`: Must produce `ReturnType`.
- **Example**: `(lambda ((x as Integer)) returns Integer (add x 1))` – Adds 1 to an integer.
- **Implementation Notes**:
  - Store as `{params: [(arg, Type)], return: ReturnType, body: expr, env: current-env}`.
  - Type check `body` against `ReturnType` at definition; args at call time.
  - Pure—no side effects, captures scope immutably.

### 7. `(join str1 str2 ...)`
- **Purpose**: Concatenates strings into a single string.
- **Syntax**: `(join str1 str2 ...)`
- **Arguments**:
  - `str1`, `str2`, ...: `(entity String ...)` or string literals (e.g., `"dog"`).
- **Semantics**: Returns a new `(entity String ...)` with concatenated values.
- **Type Rule**: All args must be `StringValued`.
- **Example**: `(join "t-0" " to " "t-1")` → `(entity String "t-0 to t-1")`
- **Implementation Notes**:
  - Extract `value` from each `(entity String ...)` or use literal strings.
  - Concatenate into a new `(entity String result)`—pure, no mutation.

### 8. `(not expr)`
- **Purpose**: Negates an `Assertable` expression.
- **Syntax**: `(not expr)`
- **Arguments**:
  - `expr`: Required—Must be `Assertable` (e.g., `(entity Boolean ...)`, `(Statement ...)`).
- **Semantics**: Returns `(entity Boolean true)` if `expr` is `false`, `(entity Boolean false)` if `expr` is `true`.
- **Type Rule**: `expr` must be `Assertable`; returns `Boolean`.
- **Example**: `(not (gt 3 5))` → `(entity Boolean true)`
- **Implementation Notes**:
  - Evaluate `expr` to a `Boolean`, invert it—pure logic operation.
  - Return `(entity Boolean result)`.

### 9. `(and expr1 expr2)`
- **Purpose**: Conjoins two `Assertable` expressions.
- **Syntax**: `(and expr1 expr2)`
- **Arguments**:
  - `expr1`, `expr2`: Required—Must be `Assertable`.
- **Semantics**: Returns `(entity Boolean true)` if both `expr1` and `expr2` are `true`, `(entity Boolean false)` otherwise.
- **Type Rule**: Both args must be `Assertable`; returns `Boolean`.
- **Example**: `(and (gt 5 3) (lt 3 5))` → `(entity Boolean true)`
- **Implementation Notes**:
  - Short-circuit: If `expr1` is `false`, skip `expr2`.
  - Pure—computes a `Boolean`, no side effects.

### 10. `(ask expr)`
- **Purpose**: Queries the truth of an `Assertable` expression.
- **Syntax**: `(ask expr)`
- **Arguments**:
  - `expr`: Required—Must be `Assertable`.
- **Semantics**: Returns `(entity Boolean true)` or `(entity Boolean false)` by evaluating `expr`’s truth in the system’s context (e.g., logical rules or assertions), not merely comparing its value.
- **Type Rule**: `expr` must be `Assertable`; returns `Boolean`.
- **Example**: `(ask (Statement dog chases toy))` – "Does the dog chase the toy?" → `(entity Boolean true/false)`
- **Implementation Notes**:
  - Evaluate `expr` to a `Boolean`—pure, no I/O (e.g., internal truth table or context).
  - Distinct from `(eq expr true)`—`(ask ...)` evaluates broader `Assertable` forms, not just `Boolean` equality.
  - Return `(entity Boolean result)`.

### 11. `(if condition then expr else expr)`
- **Purpose**: Conditional execution within a pure context.
- **Syntax**: `(if condition then expr else expr)`
- **Arguments**:
  - `condition`: Required—Must be `Assertable`.
  - `then expr`: Required—Expression if `condition` is `true`.
  - `else expr`: Required—Expression if `condition` is `false`.
- **Semantics**: Evaluates `condition`; returns `then expr` if `true`, `else expr` if `false`.
- **Type Rule**: `condition` must be `Assertable`; `then` and `else` must match return type.
- **Example**: `(if (gt 5 3) then (entity Integer 5) else (entity Integer 3))` → `(entity Integer 5)`
- **Implementation Notes**:
  - Evaluate `condition` to `Boolean`, then one branch—pure, lazy evaluation of unused branch.
  - Type check `then` and `else` for consistency at parse time.

### 12. `(list expr1 expr2 ...)`
- **Purpose**: Creates a list of expressions.
- **Syntax**: `(list expr1 expr2 ...)`
- **Arguments**:
  - `expr1`, `expr2`, ...: Zero or more expressions, must share a compatible trait (e.g., `Listable`).
- **Semantics**: Returns a list object containing all arguments.
- **Type Rule**: All elements must be `Listable` (e.g., `Noun`, `Integer`, `Time`).
- **Example**: `(list dog cat)` – List of subjects.
- **Implementation Notes**:
  - Return `{type: :list, elements: [expr1, expr2, ...]}`.
  - Type check elements for `Listable` compatibility at parse time—fail on mismatch.

### 13. `(add num1 num2 ...)`
- **Purpose**: Adds `Numeric` values.
- **Syntax**: `(add num1 num2 ...)`
- **Arguments**:
  - `num1`, `num2`, ...: `(entity Integer/Float ...)` or unquoted numbers.
- **Semantics**: Returns the sum as an `(entity Integer ...)` or `(entity Float ...)` if any arg is `Float`.
- **Type Rule**: All args must be `Numeric`; returns `Numeric`.
- **Example**: `(add 3 2.5)` → `(entity Float 5.5)`
- **Implementation Notes**:
  - Extract `value` from each `(entity Integer/Float ...)` or use unquoted numbers.
  - Sum values, promote to `Float` if any arg is `Float`—pure arithmetic.

### 14. `(mul num1 num2 ...)`
- **Purpose**: Multiplies `Numeric` values.
- **Syntax**: `(mul num1 num2 ...)`
- **Arguments**:
  - `num1`, `num2`, ...: `(entity Integer/Float ...)` or unquoted numbers.
- **Semantics**: Returns the product as an `(entity Integer ...)` or `(entity Float ...)` if any arg is `Float`.
- **Type Rule**: All args must be `Numeric`; returns `Numeric`.
- **Example**: `(mul 3 2.5)` → `(entity Float 7.5)`
- **Implementation Notes**:
  - Multiply `value` fields, promote to `Float` if needed—pure.

### 15. `(gt num1 num2)`
- **Purpose**: Checks if `num1` is greater than `num2`.
- **Syntax**: `(gt num1 num2)`
- **Arguments**:
  - `num1`, `num2`: `(entity Integer/Float ...)` or unquoted numbers.
- **Semantics**: Returns `(entity Boolean true)` if `num1 > num2`, `(entity Boolean false)` otherwise.
- **Type Rule**: Both args must be `Numeric`; returns `Boolean`.
- **Example**: `(gt 5 3)` → `(entity Boolean true)`
- **Implementation Notes**:
  - Compare `value` fields—pure comparison.
  - Return `(entity Boolean result)`.

### 16. `(lt num1 num2)`
- **Purpose**: Checks if `num1` is less than `num2`.
- **Syntax**: `(lt num1 num2)`
- **Arguments**:
  - `num1`, `num2`: `(entity Integer/Float ...)` or unquoted numbers.
- **Semantics**: Returns `(entity Boolean true)` if `num1 < num2`, `(entity Boolean false)` otherwise.
- **Type Rule**: Both args must be `Numeric`; returns `Boolean`.
- **Example**: `(lt 3 5)` → `(entity Boolean true)`
- **Implementation Notes**:
  - Compare `value` fields—pure.

### 17. `(eq expr1 expr2)`
- **Purpose**: Checks equality between two expressions.
- **Syntax**: `(eq expr1 expr2)`
- **Arguments**:
  - `expr1`, `expr2`: `(entity Integer/Float/String ...)` or unquoted numbers/strings.
- **Semantics**: Returns `(entity Boolean true)` if `expr1` equals `expr2`, `(entity Boolean false)` otherwise.
- **Type Rule**: Both args must be `Numeric` or `StringValued`; returns `Boolean`.
- **Example**: `(eq 5 5)` → `(entity Boolean true)`, `(eq "dog" "dog")` → `(entity Boolean true)`
- **Implementation Notes**:
  - Compare `value` fields—type mismatch fails (e.g., `(eq 5 "5")` → `(entity Boolean false)`).
  - Return `(entity Boolean result)`.

### 18. `(sub num1 num2 ...)`
- **Purpose**: Subtracts `Numeric` values.
- **Syntax**: `(sub num1 num2 ...)`
- **Arguments**:
  - `num1`, `num2`, ...: `(entity Integer/Float ...)` or unquoted numbers.
- **Semantics**: Subtracts left-to-right, returns `(entity Integer ...)` or `(entity Float ...)` if any arg is `Float`.
- **Type Rule**: All args must be `Numeric`; returns `Numeric`.
- **Example**: `(sub 5 2)` → `(entity Integer 3)`
- **Implementation Notes**:
  - Subtract `value` fields sequentially—pure.
  - Fail on insufficient args (e.g., `(sub 5)` → `(entity Error "insufficient arguments")`).

### 19. `(div num1 num2 ...)`
- **Purpose**: Divides `Numeric` values.
- **Syntax**: `(div num1 num2 ...)`
- **Arguments**:
  - `num1`, `num2`, ...: `(entity Integer/Float ...)` or unquoted numbers.
- **Semantics**: Divides left-to-right, returns `(entity Float ...)` (or fails if divisor is 0—see `(try ...)`).
- **Type Rule**: All args must be `Numeric`; returns `Numeric`.
- **Example**: `(div 10 2)` → `(entity Float 5)`
- **Implementation Notes**:
  - Divide `value` fields—`(try ...)` catches division by zero (e.g., `(div 5 0)` → `(entity Error "division by zero")`).
  - Pure—fails cleanly.

### 20. `(every subject stmt)`
- **Purpose**: Asserts a statement holds for all instances of a subject.
- **Syntax**: `(every subject stmt)`
- **Arguments**:
  - `subject`: `(entity Noun ... :role Subject)` or similar – Required.
  - `stmt`: `(Statement ...)` with `subject` referenced (e.g., via `(entity Pronoun "it")`) – Required.
- **Semantics**: Returns an `Assertable` expression—`(entity Boolean true)` if `stmt` holds universally for `subject`.
- **Type Rule**: `subject` must be `Relatable`, `stmt` must be `Assertable`; returns `Boolean`.
- **Example**: `(every dog (Statement it chases toy))` – "Every dog chases the toy."
- **Implementation Notes**:
  - Evaluate `stmt` for all `subject` instances—implementation defines "all" (e.g., abstract domain or list).
  - Return `(entity Boolean result)`—pure.

### 21. `(some subject stmt)`
- **Purpose**: Asserts a statement holds for at least one instance of a subject.
- **Syntax**: `(some subject stmt)`
- **Arguments**:
  - `subject`: `(entity Noun ... :role Subject)` or similar – Required.
  - `stmt`: `(Statement ...)` with `subject` referenced – Required.
- **Semantics**: Returns an `Assertable` expression—`(entity Boolean true)` if `stmt` holds for some `subject`.
- **Type Rule**: `subject` must be `Relatable`, `stmt` must be `Assertable`; returns `Boolean`.
- **Example**: `(some dog (Statement it chases toy))` – "Some dog chases the toy."
- **Implementation Notes**:
  - Evaluate `stmt` for at least one `subject`—pure logic.
  - Return `(entity Boolean result)`.

### 22. `(map :key1 val1 :key2 val2 ...)`
- **Purpose**: Creates a key/value map for optional arguments in `(entity ...)`, `(Statement ...)`, and `(relate ...)`.
- **Syntax**: `(map :key1 val1 :key2 val2 ...)`
- **Arguments**:
  - `:key1`, `:key2`, ...: Keywords (e.g., `:role`, `:adverb`, `:time`).
  - `val1`, `val2`, ...: Values matching key semantics (e.g., `Subject`, `(entity Adverb ...)`, `(entity Time ...)`).
- **Semantics**: Returns a `Map` object storing key/value pairs.
- **Type Rule**: Must be `Mappable`; keys depend on context (`:role`, `:adjective` for `(entity ...)`, `:adverb` for `(Statement ...)`, `:adverb`, `:time`, `:number` for `(relate ...)`).
- **Example**: `(map :adverb quickly :time t-0)` – Modifier map for `(relate ...)` or `(Statement ...)`.
- **Implementation Notes**:
  - Return `{type: :map, pairs: {:key1: val1, :key2: val2}}`.
  - Validate keys/values per context at parse time (e.g., `:time` must be `Timeable`).

### 23. `(times number stmt)`
- **Purpose**: Specifies the frequency of a statement.
- **Syntax**: `(times number stmt)`
- **Arguments**:
  - `number`: `(entity Integer/Float ...)` or unquoted number – Required.
  - `stmt`: `(Statement ...)` or similar – Required.
- **Semantics**: Returns an `Assertable` expression asserting `stmt` occurs `number` times.
- **Type Rule**: `number` must be `Numeric`, `stmt` must be `Assertable`; returns `Boolean`.
- **Example**: `(times 3 (Statement dog chases toy))` – "Dog chases toy 3 times."
- **Implementation Notes**:
  - Return `{type: :times, number: n, stmt: s}`—evaluate `stmt` with frequency.
  - Pure—frequency is a logical assertion.

### 24. `(that subject stmt)`
- **Purpose**: Binds a subject to a relative clause, returning the modified subject.
- **Syntax**: `(that subject stmt)`
- **Arguments**:
  - `subject`: `(entity Noun ... :role Subject)` or similar – Required.
  - `stmt`: `(Statement ...)` with `subject` referenced (e.g., via `(entity Pronoun "it")`) – Required.
- **Semantics**: Returns a `Relatable` entity representing `subject` qualified by `stmt`.
- **Type Rule**: `subject` must be `Relatable`, `stmt` must be `Assertable`; returns `Relatable`.
- **Example**: `(Statement dog is (that dog (Statement it chases toy)) :adverb big)` – "The dog that chases the toy is big."
- **Implementation Notes**:
  - Return `{type: :that, subject: s, stmt: st}`—links `subject` to `stmt` for evaluation.
  - Pure—modifies `subject` conceptually, no side effects.

### 25. `(try expr catch (error-var) rescue-expr)`
- **Purpose**: Attempts an expression, providing a fallback if it fails.
- **Syntax**: `(try expr catch (error-var) rescue-expr)`
- **Arguments**:
  - `expr`: Any expression – Required.
  - `error-var`: Symbol (e.g., `e`) – Binds the error in `rescue-expr`.
  - `rescue-expr`: Fallback expression – Required, must match `expr`’s return type.
- **Semantics**: Evaluates `expr`; if successful, returns its result; if it fails (e.g., `(div 5 0)`), evaluates `rescue-expr` with `error-var` bound to an `(entity Error "message")`.
- **Type Rule**: `expr` and `rescue-expr` must share a return type (e.g., `Numeric`, `Boolean`).
- **Examples**:
  - `(try (div 5 0) catch (e) (entity Boolean false))` – Returns `(entity Boolean false)` on failure.
  - `(try (add 3 2) catch (e) (entity Integer 0))` – Returns `(entity Integer 5)` (success).
- **Implementation Notes**:
  - Wrap `expr` in an error trap; on failure, bind `error-var` to `(entity Error "reason")`.
  - Type check `expr` and `rescue-expr` for consistency—pure, no I/O.
  - Failures (e.g., `(div 5 0)`) produce `(entity Error "division by zero")`.

### 26. `(comment s-expr)`
- **Purpose**: Provides a way to include an unevaluated s-expression as a comment or hint for other developers, enhancing code readability and reuse without affecting execution.
- **Syntax**: `(comment s-expr)`
- **Arguments**:
  - `s-expr`: Required—Any valid s-expression (e.g., a `(list ...)`, `(Statement ...)`, or literal). Can be a single expression or a list of expressions (e.g., `(comment (list (add 1 2) (mul 3 4)))`).
- **Semantics**: A no-op at runtime—purely structural, ignored by the evaluator. The `s-expr` is preserved in the parsed structure for documentation or tooling (e.g., pretty-printers, IDEs) but has no computational effect.
- **Type Rule**: None—`s-expr` is not type-checked beyond basic syntax validity, as it’s never evaluated.
- **Examples**:
  - `(comment (Statement dog chases toy))` – Hints at a reusable "dog chases toy" statement.
  - `(comment (list (add 1 2) (mul 3 4)))` – Suggests a sequence of math ops for reuse.
  - `(let ((x be 5)) (comment (add x 10)) (mul x 2))` – Notes an alternative computation, returns `(entity Integer 10)`.
- **Implementation Notes**:
  - **Parser**: Parse as `[:comment, s-expr]`—store `s-expr` verbatim without evaluation. Allow nested lists (e.g., `[:comment, [:list, [:add, 1, 2], [:mul, 3, 4]]]`).
  - **Evaluator**: Skip entirely—return no value or propagate the next expression’s result in a sequence (e.g., `(comment ...) (mul x 2)` → `(entity Integer 10)`).
  - **Purity**: No effect—perfectly pure, just a marker for humans.
  - **Tooling**: Could be extracted by a linter or doc generator (e.g., "Hints: `(Statement dog chases toy)`").

### 27. `(inner-monologue "some-string" [s-expr])`
- **Purpose**: Attaches a human-readable "thought string" to an optional s-expression, providing guidance or intent for the next reader, akin to an inline note or mental aside, without impacting execution.
- **Syntax**:
  - `(inner-monologue "some-string")` – Minimal form with just a string.
  - `(inner-monologue "some-string" s-expr)` – Full form with a string and expression.
- **Arguments**:
  - `"some-string"`: Required—A string literal (e.g., `"Calculate the total here"`) in `(entity String ...)` form, conveying intent or instructions.
  - `s-expr`: Optional—Any valid s-expression (e.g., `(add 1 2)`), providing context or an example tied to the thought. Defaults to no expression if omitted.
- **Semantics**: A no-op at runtime—purely structural, ignored by the evaluator. Both the string and optional `s-expr` are preserved in the parsed structure for documentation purposes but do not affect computation.
- **Type Rule**:
  - `"some-string"`: Must be `(entity String ...)` or a string literal (parser converts to `(entity String "text")`).
  - `s-expr`: No type check—unevaluated, can be anything syntactically valid.
- **Examples**:
  - `(inner-monologue "Start with the base count")` – Simple note.
  - `(inner-monologue "Add 3 to the count here" (add count 3))` – Guidance with an example.
  - `(let ((count be 5)) (inner-monologue "Double the count" (mul count 2)) (add count 1))` – Notes an alternative, returns `(entity Integer 6)`.
- **Implementation Notes**:
  - **Parser**:
    - Parse as `[:inner-monologue, "some-string"]` or `[:inner-monologue, "some-string", s-expr]`.
    - Convert `"some-string"` to `[:entity, :String, "some-string"]` internally.
    - Store `s-expr` verbatim if present (e.g., `[:inner-monologue, "Add here", [:add, :count, 3]]`).
  - **Evaluator**: Skip—return no value or propagate the next expression’s result in a sequence (e.g., `(inner-monologue ...) (add count 1)` → `(entity Integer 6)`).
  - **Purity**: No effect—purely for human consumption.
  - **Tooling**: Extractable by doc tools (e.g., "Thought: 'Double the count' with `(mul count 2)`").

### 28. `(quote expr)`
- **Purpose**: Returns the input expression unevaluated as a quoted s-expression, preserving it as data for manipulation or reuse, supporting Wat’s homoiconicity.
- **Syntax**: `(quote expr)`
- **Arguments**:
  - `expr`: Required—Any valid s-expression (e.g., a literal like `5`, a list like `(add 1 2)`, or a complex form like `(Statement dog chases toy)`).
- **Semantics**: Prevents evaluation of `expr`, returning it as a quoted form (an unevaluated s-expression) that can be bound, passed, or manipulated as data. Unlike `(comment ...)` or `(inner-monologue ...)`, which are no-ops, `(quote ...)` produces a usable result.
- **Type Rule**: None—`expr` is not evaluated, so no type checking applies beyond syntactic validity.
- **Examples**:
  - `(quote (add 1 2))` – Returns the s-expression `(add 1 2)` unevaluated, not `(entity Integer 3)`.
  - `(quote (Statement dog chases toy))` – Returns `(Statement dog chases toy)` as data, not a computed assertion.
  - `(let ((quoted be (quote (mul 3 4)))) quoted)` – Binds `(mul 3 4)` as a value, returns `(mul 3 4)`.
- **Implementation Notes**:
  - **Parser**: Parse as `[:quote, s-expr]`—store `s-expr` verbatim without evaluating (e.g., `[:quote, [:add, 1, 2]]`).
  - **Evaluator**: Return `s-expr` as-is—no computation, just pass the parsed form (e.g., `(add 1 2)` as a list structure).
  - **Purity**: Pure—no side effects, simply returns the unevaluated input.
  - **Storage**: Represent as the raw s-expression (e.g., `(add 1 2)`), not a wrapped object, for direct use in `(list ...)` or bindings.
  - **Tooling**: Can be inspected or pretty-printed (e.g., "Quoted: `(Statement dog chases toy)`").

---

## Userland Functions

These are defined with `(lambda ...)` to extend Wat without core additions. All are pure, returning computed values.

### 1. `(or expr1 expr2)`
- **Definition**:

```
(define or as
  (lambda ((expr1 as Assertable) (expr2 as Assertable))
    returns Assertable
    (not (and (not expr1) (not expr2)))))
```

- **Purpose**: Logical disjunction via De Morgan’s law.
- **Example**: `(or (Statement dog chases toy) (Statement dog catches toy))` – "Dog chases or catches toy."
- **Implementation Notes**: Pure—uses `(not ...)` and `(and ...)` to compute `(entity Boolean true/false)`.

### 2. `(before t-1 t-2)`
- **Definition**:

```
(define before as
  (lambda ((t-1 as Time) (t-2 as Time))
    returns Statement
    (let
      ((precedes be (entity Verb "precedes")))
      (Statement t-1 precedes t-2))))
```

- **Purpose**: Asserts `t-1` precedes `t-2`.
- **Example**: `(before t-0 t-1)` – "t0 precedes t1."
- **Implementation Notes**: Pure—constructs a `(Statement ...)` using `(entity ...)` and `(Statement ...)`.

### 3. `(between t-1 t-2)`
- **Definition**:

```
(define between as
  (lambda ((t-1 as Time) (t-2 as Time))
    returns Time
    (entity Time (join t-1 " to " t-2))))
```

- **Purpose**: Creates a time range entity.
- **Example**: `(between t-0 t-1)` – "t0 to t1."
- **Implementation Notes**: Pure—`(join ...)` computes a string, wrapped in `(entity Time ...)`.

### 4. `(during event t-1 t-2)`
- **Definition**:

```
(define during as
  (lambda ((event as Statement) (t-1 as Time) (t-2 as Time))
    returns Statement
    (let
      ((occurs-during be (entity Verb "occurs-during")))
      (Statement event occurs-during (between t-1 t-2)))))
```

- **Purpose**: Asserts an event occurs during a time range.
- **Example**: `(during (Statement dog chases toy) t-0 t-1)` – "Dog chases toy during t0 to t1."
- **Implementation Notes**: Pure—builds a `(Statement ...)` with `(between ...)`.

### 5. `(mod x y)`
- **Definition**:

```
(define mod as
  (lambda ((x as Numeric) (y as Numeric))
    returns Numeric
    (sub x (mul y (div x y)))))
```

- **Purpose**: Computes the modulo (remainder).
- **Example**: `(mod 5 2)` → `(entity Integer 1)`
- **Implementation Notes**: Pure—uses `(sub ...)`, `(mul ...)`, `(div ...)` for arithmetic.

### 6. `(pow base exp)`
- **Definition**:

```
(define pow as
  (lambda ((base as Numeric) (exp as Integer))
    returns Numeric
    (if (eq exp 1)
      then base
      else (mul base (pow base (sub exp 1))))))
```

- **Purpose**: Computes exponentiation.
- **Example**: `(pow 2 3)` → `(entity Integer 8)`
- **Implementation Notes**: Pure—recursive `(mul ...)` and `(sub ...)`.

### 7. `(ge x y)`
- **Definition**:

```
(define ge as
  (lambda ((x as Numeric) (y as Numeric))
    returns Assertable
    (not (lt x y))))
```

- **Purpose**: Greater than or equal to.
- **Example**: `(ge 5 3)` → `(entity Boolean true)`
- **Implementation Notes**: Pure—`(not ...)` and `(lt ...)`.

### 8. `(le x y)`
- **Definition**:

```
(define le as
  (lambda ((x as Numeric) (y as Numeric))
    returns Assertable
    (not (gt x y))))
```

- **Purpose**: Less than or equal to.
- **Example**: `(le 3 5)` → `(entity Boolean true)`
- **Implementation Notes**: Pure—`(not ...)` and `(gt ...)`.

### 9. `(ne x y)`
- **Definition**:

```
(define ne as
  (lambda ((x as Numeric) (y as Numeric))
    returns Assertable
    (not (eq x y))))
```

- **Purpose**: Not equal to.
- **Example**: `(ne 5 3)` → `(entity Boolean true)`
- **Implementation Notes**: Pure—`(not ...)` and `(eq ...)`.

### 10. `(passive obj verb subj)`
- **Definition**:

```
(define passive as
  (lambda ((obj as Object) (verb as Verb) (subj as Subject))
    returns Statement
    (let
      ((is be (entity Verb "is")))
      (Statement obj is (Statement subj verb obj)))))
```

- **Purpose**: Constructs passive voice statements with SVO order.
- **Example**: `(passive toy chases dog)` – "The toy is chased by the dog."
- **Implementation Notes**: Pure—nested `(Statement ...)` construction, flipped for SVO.

### 11. `(after t1 stmt)`
- **Definition**:

```
(define after as
  (lambda ((t1 as Time) (stmt as Statement))
    returns Statement
    (let
      ((precedes be (entity Verb "precedes")))
      (Statement stmt precedes t1))))
```

- **Purpose**: Asserts a statement occurs after a time.
- **Example**: `(after t-0 (Statement dog chases toy))` – "After t0, the dog chases the toy."
- **Implementation Notes**: Pure—uses `(Statement ...)` and `(entity ...)` via `(before ...)` inversion.

---

## Trait Implementations

Global `(impl ...)` declarations define type capabilities, enforced at parse time:

1. `(impl Relatable for Noun)` – Usable as `subject` or `object` in `(Statement ...)`.
2. `(impl Relatable for Time)` – Usable in time-related statements.
3. `(impl Relatable for Pronoun)` – Usable as `object` (e.g., "it").
4. `(impl RelatableVerb for Verb)` – Usable as `verb` in `(Statement ...)`.
5. `(impl Adverbial for Adverb)` – Usable as `:adverb` in `(Statement ...)` map.
6. `(impl StringValued for Noun)` – Has a string `value`.
7. `(impl StringValued for Verb)` – Has a string `value`.
8. `(impl StringValued for Time)` – Has a string `value`.
9. `(impl StringValued for Adverb)` – Has a string `value`.
10. `(impl StringValued for String)` – Is a string.
11. `(impl Numeric for Integer)` – Usable in math ops.
12. `(impl Numeric for Float)` – Usable in math ops.
13. `(impl Assertable for Boolean)` – Usable in `(assert ...)`, `(if ...)`.
14. `(impl Assertable for Statement)` – Usable in `(assert ...)`, `(if ...)`.
15. `(impl Assertable for Relate)` – Internal `(Statement ...)` form.
16. `(impl Timeable for Time)` – Usable in `(at ...)`.
17. `(impl Listable for Noun)` – Usable in `(list ...)`.
18. `(impl Listable for Time)` – Usable in `(list ...)`.
19. `(impl Listable for Verb)` – Usable in `(list ...)`.
20. `(impl Listable for Integer)` – Usable in `(list ...)`.
21. `(impl Listable for Float)` – Usable in `(list ...)`.
22. `(impl Mappable for Map)` – Usable in `(entity ...)` and `(Statement ...)` maps.
23. `(impl Describable for Noun)` – Can take `:adjective` in `(entity ...)`.

---

## Sugar Constructs

Parser-level rewrites for concise syntax, all pure:

`(Verb value)` → `(entity Verb value)`
 - Example: `(Verb "chases")` → `(entity Verb "chases")`

`(Subject value [adjective])` → `(entity Noun value :role Subject [:adjective adjective])`
 - Example: `(Subject "dog" (entity Adjective "big"))` → `(entity Noun "dog" :role Subject :adjective (entity Adjective "big"))`

`(Object value [adjective])` → `(entity Noun value :role Object [:adjective adjective])`
 - Example: `(Object "toy")` → `(entity Noun "toy" :role Object)`

`(Time value)` → `(entity Time value)`
 - Example: `(Time "t0")` → `(entity Time "t0")`

`(Adverb value)` → `(entity Adverb value)`
 - Example: `(Adverb "quickly")` → `(entity Adverb "quickly")`

`(Adjective value)` → `(entity Adjective value)`
 - Example: `(Adjective "big")` → `(entity Adjective "big")`

`(Statement subject verb object)` → `(Statement subject verb object (map))`
 - Example: `(Statement dog chases toy)` → `(Statement dog chases toy (map))`

`(Statement subject verb object :key1 val1 ...)` → `(Statement subject verb object (map :key1 val1 ...))`
 - Example: `(Statement dog chases toy :adverb quickly)` → `(Statement dog chases toy (map :adverb quickly))`

`true` → `(entity Boolean true)`
 - Example: `true` → `(entity Boolean true)`

`false` → `(entity Boolean false)`
  - Example: `false` → `(entity Boolean false)`

`n` (e.g., `3`, `-5`) → `(entity Integer n)`
  - Example: `3` → `(entity Integer 3)`

`n.m` (e.g., `2.5`, `-3.14`) → `(entity Float n.m)`
  - Example: `2.5` → `(entity Float 2.5)`

---

## Types

- **Base Types**:
- `Noun`: `(entity Noun "dog")` – Represents entities like people, places, things.
- `Verb`: `(entity Verb "chases")` – Represents actions or states.
- `Time`: `(entity Time "t0")` – Represents time points.
- `Adverb`: `(entity Adverb "quickly")` – Modifies verbs.
- `String`: `(entity String "text")` – Raw text data.
- `Integer`: `(entity Integer 5)` – Whole numbers.
- `Float`: `(entity Float 2.5)` – Decimal numbers.
- `Boolean`: `(entity Boolean true)` – Truth values (`true`, `false`).
- `Pronoun`: `(entity Pronoun "it")` – References entities (e.g., "it").
- `Preposition`: `(entity Preposition "on")` – Spatial or relational modifiers.
- `Adjective`: `(entity Adjective "big")` – Describes nouns.
- `Error`: `(entity Error "division by zero")` – Represents computation failures.

**Structural Types**:
- `Subject`: `(entity Noun ... :role Subject)` – Subject role wrapper.
- `Object`: `(entity Noun ... :role Object)` – Object role wrapper.
- `Statement`: `(Statement subject verb object ...)` – SVO sentence structure.
- `Relate`: `(relate verb subject object ...)` – Internal SVO form.
- `Map`: `(map :key1 val1 ...)` – Key/value attribute container.

---

## Traits

`Relatable`: Can be `subject` or `object` in `(Statement ...)`.
- Implemented by: `Noun`, `Time`, `Pronoun`.

`RelatableVerb`: Can be `verb` in `(Statement ...)`.
- Implemented by: `Verb`.

`Adverbial`: Can be `:adverb` in `(Statement ...)` map.
- Implemented by: `Adverb`.

`Timeable`: Can be `time` in `(at ...)`.
- Implemented by: `Time`.

`StringValued`: Has a string `value`.
- Implemented by: `Noun`, `Verb`, `Time`, `Adverb`, `String`.

`Numeric`: Usable in `(add ...)`, `(mul ...)`, etc.
- Implemented by: `Integer`, `Float`.

`Assertable`: Usable in `(assert ...)`, `(not ...)`, `(if ...)`, etc.
- Implemented by: `Boolean`, `Statement`, `Relate`.

`Listable`: Can be an element in `(list ...)`.
- Implemented by: `Noun`, `Time`, `Verb`, `Integer`, `Float`.

- `Mappable`: Can be decomposed in `(entity ...)` or `(Statement ...)` maps.
Implemented by: `Map`.

`Describable`: Can take an `:adjective` in `(entity ...)`.
- Implemented by: `Noun`.

---

## Full Example
```
(let
  (
    (dog be (entity Noun "dog" :role Subject :adjective (entity Adjective "big")))
    (toy be (entity Noun "toy" :role Object))
    (it be (entity Pronoun "it" :role Object))
    (chases be (entity Verb "chases"))
    (catches be (entity Verb "catches"))
    (is be (entity Verb "is"))
    (rests be (entity Verb "rests"))
    (quickly be (entity Adverb "quickly"))
    (big be (entity Adverb "big"))
    (t-0 be (entity Time "t0"))
    (t-1 be (entity Time "t1"))
    (count be (entity Integer 5))
    (zero be (entity Integer 0))
    (or be (define or as (lambda ((e1 as Assertable) (e2 as Assertable)) returns Assertable (not (and (not e1) (not e2))))))
    (action be (quote (Statement dog chases toy)))
  )
  (comment (Statement dog chases toy))  ; Hint: Reusable "dog chases toy" statement
  (inner-monologue "Attempt division, handle zero case" (div count zero))  ; Thought with example
  (assert (Statement dog chases toy :adverb quickly))  ; "The big dog quickly chases the toy"
  (assert (at t-0 (times count (Statement dog chases toy))))  ; "At t0, the big dog chases the toy 5 times"
  (assert (every dog (Statement it chases toy)))  ; "Every big dog chases the toy"
  (assert (or (Statement dog chases toy) (Statement dog catches toy)))  ; "The big dog chases or catches the toy"
  (assert (try (div count zero) catch (e) (entity Boolean false)))  ; "If 5 / 0 fails, assert false"
  (assert (Statement dog is (that dog (Statement it chases toy)) :adverb big))  ; "The big dog that chases the toy is big"
  (assert (after t-0 (Statement dog chases toy)))  ; "After t0, the dog chases the toy"
  (assert (before t-0 t-1))  ; "t0 precedes t1"
  (if (gt count 3)
    then (assert (times count (Statement dog chases toy)))
    else (assert (Statement dog rests toy)))
  (assert action)  ; Asserts the quoted form—implementation-specific, likely needs (ask action)
)
```
---

## Implementation Notes

### Parser
- **S-Expression Format**: Parse into Ruby arrays (e.g., `[:entity, :Noun, "dog", [:map, :role, :Subject]]`).
- **Sugar Rewriting**:
  - `(entity Noun "dog")` → `[:entity, :Noun, "dog", [:map]]`.
  - `(Statement dog chases toy)` → `[:Statement, :dog, :chases, :toy, [:map]]`.
  - `(relate chases dog toy)` → `[:relate, :chases, :dog, :toy, [:map]]`.
  - `(comment (add 1 2))` → `[:comment, [:add, 1, 2]]`.
  - `(inner-monologue "note" (add 3 4))` → `[:inner-monologue, [:entity, :String, "note"], [:add, 3, 4]]`.
  - `(quote (add 1 2))` → `[:quote, [:add, 1, 2]]`.
- **Type Checking**: Validate `Type`, `value`, and map keys at parse time—raise errors for mismatches (e.g., `(entity Noun 5)` fails).

### Evaluator
- **Environment**: Stack of `{bindings: {label: value}, traits: {Type: [Trait]}}`.
- **Execution**:
  - `(entity ...)`: Construct `{type: Type, value: v, attrs: map}`.
  - `(relate ...)`: Construct `{verb: v, subject: s, object: o, attrs: map}`—used by `(Statement ...)`.
  - `(Statement ...)`: Delegate to `(relate verb subject object [map-expr])`—reorder SVO to VSO internally.
  - `(try ...)`: Wrap `expr` in an exception trap; return `rescue-expr` on failure with `error-var` bound.
  - `(assert ...)`: Evaluate to `Boolean`, fail if `false` or not `Assertable` (e.g., `(assert (quote ...))` → `(entity Error "not assertable")`).
  - `(comment ...)`: Skip—store in AST, no evaluation.
  - `(inner-monologue ...)`: Skip—store string and optional `s-expr` in AST, no evaluation.
  - `(quote ...)`: Return `expr` unevaluated—pure, no computation.
- **Purity**: No I/O—errors return `(entity Error ...)` objects, computations stay internal.

### Error Handling
- **Failures**: `(div 5 0)` → `(entity Error "division by zero")`.
- **Propagation**: `(try ...)` catches and binds errors—purely functional.
- **Validation**: Type mismatches (e.g., `(add "dog" 3)`) → `(entity Error "type mismatch")`.
