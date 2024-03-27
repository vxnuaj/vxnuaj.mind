---
Last Updated: 03-27-2024 | 2:49 PM
---
When the Python Interpreter encounters a statement in the code that it can't understand.

This can be misspelled keywords, missed punctuation, invalid variable names, using the wrong operator, etc.

An example can be:

```
def function(x, y)
	for iterations in range(y):
		x += 1
	return x
```

Here, one forget the `:` after defining `function(x,y)`

The correction is: `def function(x,y):`