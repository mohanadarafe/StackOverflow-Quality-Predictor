# Stack Overflow Quality Predictor

## Setup
In order to run the code, make sure you have Conda installed in your machine in order to use our environment. From there, run the following commands

```
conda env create --name stack --file=environment.yml
conda activate stack
```

## Run Spark
```
python pyspark/run_pipeline.py
```

## Run Dask
```
python pyspark/run.py
```

### Authors
[Mohanad Arafe](https://github.com/mohanadarafe)

[Robert Beaudenon](https://github.com/RobertBeaudenon)
