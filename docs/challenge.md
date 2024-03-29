# Software Engineer (ML & LLMs) Challenge

Name: Facundo M. Scasso

## Resolution

### Part I

#### Assumptions

- Did some major changes to `get_period_day()` so that all parameters are independent. Also, it could have been happening that exact borders weren't being taken into account as all conditions had strict inequality signs.
- Assuming no nulls, I also changed the last inequality for an else clause.
- In `get_rate_from_column()` I initialize the dict with 0s, as it's more efficient and later on missing keys are inputed as 0 nonetheless.
- For the barplots I recouple the x and y values, as many of them decoupled them. The x-axis went through a `.value_counts()` transformation, and so it could happen that they get reordered without updating the y-axis order.

#### Conclusion

Both models are pretty similar. I'll put the XGBoost model into production (see my changes `exploration.ipynb` for further information).

### Part II

...

### Part III

...

### Part IV

...
