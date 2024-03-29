# Software Engineer (ML & LLMs) Challenge

Name: Facundo M. Scasso

## Part I

### Some changes made to the code

- Did some major changes to `get_period_day()` so that all parameters are independent. Also, it could have been happening that exact borders weren't being taken into account as all conditions had strict inequality signs.
- Assuming no nulls are in the data, I also changed the last inequality for an else clause.
- In `get_rate_from_column()` I initialize the dict with 0s, as it's more efficient and later on missing keys are inputed as 0 nonetheless.
- For the barplots I recouple the x and y values, as many of them decoupled them. The x-axis went through a `.value_counts()` transformation, and so it could happen that they get reordered without updating the y-axis order.

### Conclusion

Both models are pretty similar. I'll put the XGBoost model into production (see my changes on `exploration.ipynb` for further information).

## Part II

The model is fitted and stays in memory as a `DelayModel` instance. I try to use the `typing` module as much as I can so that code development is as coherent as possible, and also FastAPI can use these to autodocument the input and output of the `/predict` endpoint.

## Part III

I will deploy the API using Google App Engine on my personal account. It consists of 3 endpoints:

- The root endpoint: A Hello World that states the challenge name and my own name.
- `/predict`: Base endpoint of the project. It receives the features, validates their keys and values, preprocess them (using the same steps used for training) and predicts based on them whether those flights will be delayed (`1`) or not (`0`).
- `/health`: Returns status OK with a code of 200.

Of course, as it always is with FastAPI, you can check all this information in a more interactive way by accessing the `/docs` or the `/redoc` endpoint.
Additionally, the API instance specifications can be found inside the `app.yaml` file.

## Part IV

I'll write the CI/CD files in order to use GitHub Actions + Google App Engine.
