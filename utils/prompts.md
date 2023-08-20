# ai2_arc

```txt
prompt = """Question: Juan and LaKeisha roll a few objects down a ramp. They want to see which object rolls the farthest. What should they do so they can repeat their investigation?
Choices: A. Put the objects in groups. B. Change the height of the ramp. C. Choose different objects to roll. D. Record the details of the investigation.

Answer:"""
```

# CodeAlpaca-20k
### If input exists:
```txt
prompt = """Write a JavaScript code to print the pyramid pattern for a given number of rows.
Input: 7

Answer:"""
```
### If no input:
```txt
prompt = """What is the cosine similarity between two vectors?

Answer:"""
```

# gsm8k
### Zero-shot
```txt
prompt = """Question: Claire makes a 3 egg omelet every morning for breakfast. How many dozens of eggs will she eat in 4 weeks?

Answer:"""
```
### One-shot
```txt
prompt = """Question: Carlos is planting a lemon tree. The tree will cost $90 to plant. Each year it will grow 7 lemons, which he can sell for $1.5 each. It costs $3 a year to water and feed the tree. How many years will it take before he starts earning money on the lemon tree?

Answer: He makes $10.5 selling lemons each year because 7 x 1.5 = <<7*1.5=10.5>>10.5 He earns $7.5 each year from the lemon tree because 10.5 - 3 = <<10.5-3=7.5>>7.5 It will take 12 years to earn enough to pay off the tree because 90 / 7.5 = <<90/7.5=12>>12 He will make money in year 13 because 12 + 1 = <<12+1=13>>13 #### 13

Question: Claire makes a 3 egg omelet every morning for breakfast. How many dozens of eggs will she eat in 4 weeks?

Answer:"""
```

# SQuAD

```txt
prompt = """Context: The origin of electric and magnetic fields would not be fully explained until 1864 when James Clerk Maxwell unified a number of earlier theories into a set of 20 scalar equations, which were later reformulated into 4 vector equations by Oliver Heaviside and Josiah Willard Gibbs. These "Maxwell Equations" fully described the sources of the fields as being stationary and moving charges, and the interactions of the fields themselves. This led Maxwell to discover that electric and magnetic fields could be "self-generating" through a wave that traveled at a speed that he calculated to be the speed of light. This insight united the nascent fields of electromagnetic theory with optics and led directly to a complete description of the electromagnetic spectrum.

Question: Who discovered that magnetic and electric could self-generate?

Answer:"""
```