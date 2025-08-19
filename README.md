# HynetImpute: Missing Pattern Specialized Imputation via Hypernetwork
=========================================

HynetImpute is a deep learning framework that does imputation via hypernetwork.





#### Running HynetImpute

**STEP 1: Installation**  

1. Install python and pytorch. We use Python 3.8, pytorch  2.4.1


2. Download the HynetImpute repository  

**STEP 2: Run HynetImpute**  

To start with, train the hypernetwork:
```
hynetImputer=HynetImpute(latent_dim=10)
hynetImputer.train(train_data,train_mask,train_ground_truth)
```

Then, impute the data:
```
imputed_data=hynetImputer.predict(test_data,test_mask)
```




