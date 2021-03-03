def shuffle(df : pd.DataFrame, balance_classes : bool = False, 
  response : str ="Survived", frac : float = 1.0):
  """ randomly shuffle an input pandas dataframe object, optionally 
  enforcing balanced classes for a given response variable if asked """
  categories : np.array = np.unique(df[response])
  if balance_classes:
    class_counts : list[int] = [ 
      np.sum(df[response] == category) 
      for category in categories ]
    min_count = np.min(class_counts)
    df = pd.concat([ 
        df[ df[response] == c ].sample(n=min_count) 
        for c in categories ])
  else:
    df = df.sample(n=len(df))
  # wrap a pandas call to optionally split our dataframe
  # into testing / training combinations if a frac= argument
  # was applied
  if frac is not 1.0:
    train_set = df.copy().sample(frac=frac)
    test_set = df.copy().drop(train_set.index)
    return {'training': train_set, 'testing': test_set}
  else:
    return {'training': df}
 
def mean_center(df : pd.DataFrame):
  """ mean-variance center continuous variables in a dataframe as a pre-cursor
  to least-squares regression. This is important if you want to analyze effects
  sizes from the fitted regression parameters."""
  r = df.copy()
  return { 
    'mean': r.mean(),
    'sd' : r.std(),
    'scaled' : r.\
      subtract(r.mean()).\
      divide(r.std()) }
 
def string_to_ordinal(v : str):
  """ convert the individual characters in a string to 
  ordinal integer values as an alternative to working
  with categorical string values as predictors"""
  return np.sum([ ord(c) for c in str(v) ])
 
def scale(df : pd.DataFrame, scale : dict):
  """ apply a z-transformation (centering) to some dataframe"""
  return ( df - scale['mean'] ) / scale['sd']
 
def unscale(df : pd.DataFrame, scale : dict):
  """ back-transform a centered data.frame to its original values """
  pass
 
def get_interactions(df, variables, categoricals):
  pass
 
def get_weights(results_bs):
  pass
 
def mse(predicted : np.array, observed : np.array):
  return (1/len(predicted)) * np.sum((observed - predicted)**2) 
 
def accuracy(observed : np.array, predicted : np.array):
  # correct (total) / total
  return np.sum(observed == predicted) / len(observed)
 
def omission(observed : np.array, predicted : np.array):
  """ type one error estimator from observed vs. predicted values """
  # incorrect (zeros) / total (zeros)
  zeros : np.array = (observed == 0)
  return np.sum( observed[zeros] != predicted[zeros] ) / np.sum(zeros)
 
def commission(observed : np.array, predicted : np.array):
  """ type two error estimator from observed vs. predicted values """
  # incorrect (ones) / total (ones)
  ones : np.array = (observed == 1)
  return np.sum( observed[ones] != predicted[ones] ) / np.sum(ones)
 
def encode_design_matrix(df : pd.DataFrame, variables : list = [], 
  dummy_variables : list = [], normalize : bool = True):
  """ accepts an input pandas dataframe and optionally normalizes columns and 
  creates dummy variables for categoricals as a pre-cursor to fitting a model
  in scikit"""
  if normalize:
    normalized = mean_center(df[variables])
  else:
    normalized = df[variables]
  if len(dummy_variables) > 0:
    if normalize:
      x = normalized['scaled'].join(df[dummy_variables])
    else:
      x = normalized.join(df[dummy_variables])
    var_names = list(
      pd.get_dummies(x[variables + dummy_variables]).columns)
    x = pd.get_dummies(x[variables + dummy_variables]).to_numpy().\
      reshape(-1, len(var_names))
  else:
    var_names = list(normalized.columns)
    x = normalized.to_numpy().reshape(-1, len(var_names))
  
  result = {'x':x, 'variables':var_names }
  
  if normalize:
    result['scale'] = { k: normalized[k] for k in ('mean','sd') }
  
  return result
 
def fit_logistic_regression(df, variables : list = [], 
  dummy_variables : list = [], response= "Survived", normalize : bool = True):
  """ wrapper for LogisticRegression that will normalize an input
  pandas dataframe and encode dummy variables prior to fitting a regression
  model """
  training_df = df.copy().dropna()
 
  design_matrix = encode_design_matrix(
      training_df, variables, 
      dummy_variables, normalize)
 
  y = list(training_df[response])
 
  logistic_regression = LogisticRegression(random_state=0)
 
  return {
      'model': logistic_regression.fit(design_matrix['x'], y),
      'variables' : design_matrix['variables'],
      'scale': design_matrix['scale'] }
 
def fit_random_forests(df, variables : list = [], dummy_variables : list = [], 
  response= "Survived", normalize : bool = False, **kwargs):
  """ wrapper for RandomForestClassifier that will omit normalization of an 
  input pandas dataframe and encode dummy variables prior to fitting a classification
  model. This currently doesn't change any of the default options for RF under
  the hood, but could. """
  training_df = df.copy().dropna()
 
  design_matrix = encode_design_matrix(
      training_df, variables, 
      dummy_variables, normalize)  
 
  y = list(training_df[response])
  
  random_forests = RandomForestClassifier(random_state=0, **kwargs)
 
  result = {
      'model': random_forests.fit(design_matrix['x'], y),
      'variables' : design_matrix['variables'] }
 
  if normalize: 
    result['scale'] = design_matrix['scale'] 
 
  return result
 
def predict(m : None, x : pd.DataFrame, variables: list = None, scale : dict = None):
  """ converts a design matrix (dataframe) into a numpy array that we can 
  predict across """
  if isinstance(x, pd.DataFrame):
    x = x[variables].to_numpy().reshape(-1, len(variables))
  return m.predict(x)
 
def cross_validation(results : list, k_folds : list, variables : list, 
  dummy_variables : list = [], apply_scale : bool = False, response : str ="Survived"):
  """ assess accuracy and error for a series of k classification model replicates using 
  k-folds cross-validation """
 
  for i in range(len(results)):
    
    testing_df = k_folds[i]['testing'].dropna()
 
    # case: we need to rescale our variables using an external
    # parameterization 
    if apply_scale:
 
      scale_params = results[i]['scale']
 
      if len(dummy_variables) > 0:
        design_matrix = encode_design_matrix(
            scale(testing_df[variables], scale_params).join(testing_df[dummy_variables]), 
            variables=variables, dummy_variables=dummy_variables, normalize=False)
      else:
        design_matrix = encode_design_matrix(
            scale(testing_df[variables], scale_params), 
            variables=variables, normalize=False)
    # case: no application of scale parameters
    else:
        if len(dummy_variables) > 0:
          design_matrix = encode_design_matrix(
              testing_df[variables].join(testing_df[dummy_variables]), 
              variables=variables, dummy_variables=dummy_variables, normalize=False)
        else:
          design_matrix = encode_design_matrix(
              testing_df[variables], 
              variables=variables, normalize=False)
 
    observed = testing_df[response]
    predicted = predict(
        m=results[i]["model"], 
        x=design_matrix['x'])
    
    yield pd.DataFrame({ 
        'overall accuracy': [accuracy(observed, predicted)],
        'omission error rate' : [omission(observed, predicted)],
        'commission error rate' : [commission(observed, predicted)] })
 
def std_effect_sizes(results : list):
  """ accepts a bootstrapped list of models from the user and uses the fitted
  coefficients from the models to estimate mean and standard error of it's input
  parameters """
  n : int = len(results)
  effects = pd.DataFrame([ 
    pd.concat([ coefficients(r) for r in results ]).mean(), 
    pd.concat([ coefficients(r) for r in results ]).std() / sqrt(n) ]).\
      transpose()
  effects.columns = ['est', 'se']
  return effects
 
def coefficients(regression):
  """ extracts coefficients from a fitted regression model """
  # for logitic regression
  if regression['model'].coef_.shape[0] == 1:
    coefficients = np.array([
      regression['model'].intercept_[0]] + list(regression['model'].coef_[0]))
  # for standard regression
  else:
    coefficients = np.array([
      regression['model'].intercept_] + list(regression['model'].coef_))
  n : int = len(coefficients)
  coefficients = pd.DataFrame(coefficients.reshape(-1,n))
  coefficients.columns = ["Intercept"] + regression['variables']
  return coefficients
