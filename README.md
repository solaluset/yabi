# YABI

Ever wanted to use braces instead of indentation in Python? No? Well, it's possible now!

## Is YABI better than other implementations?

It doesn't have problems with dicts.
However, it's not perfect (`match` is not supported).

If your condition starts with `{`, remember to parenthesise it:
```bython
# Wrong
if {a for a in range(n)} { print("Hello") }
# Correct
if ({a for a in range(n)}) { print("Hello") }
```

## Perfect for one-liners

Braces and semicolons allow to make any code a one-liner.
But, like, why would you do that?

## Acknowledgements

Inspired by https://github.com/mathialo/bython
