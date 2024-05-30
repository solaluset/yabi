# YABI

Ever wanted to use braces instead of indentation in Python? No? Well, it's possible now!

## Is YABI better than other implementations?

It doesn't have problems with dicts.
It can have bugs though, I'd be glad if you opened an issue in case you encounter one.

If your condition contains `{` or `lambda`, remember to parenthesise it:
```bython
# Wrong
if x := {a for a in range(n)} { print("Hello") }
# Correct
if (x := {a for a in range(n)}) { print("Hello") }
```

## Perfect for one-liners

Braces and semicolons allow to make any code a one-liner.
But, like, why would you do that?

## Converting Python files

You don't need to convert anything because `.py` files can be imported into `.by`.
YABI provides `yabi-convert` in case you still want to convert.
The output may be a little weird but should still be correct.

```bash
yabi-convert something.py > something.by
```

## Acknowledgements

Inspired by https://github.com/mathialo/bython
