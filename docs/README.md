# OVERVIEW

This doc is for notating the progress on the project as well as providing an overview for the other pieces of documentation

## Documentation

```
├── README.md(THIS DOC) => Provides and overview of docs and progress 
├── data_pipeline.md => Provides insight into how the data pipeline was built
├── model_architecture.md => Provides insight into how the models were built
├── RVAE.md => How the RVAE model was built
├── future.md => Things that can be improved on from this project
```

## NOTES

- **NOTHING YET**

## THINGS TO TEST:

- [x] The dataset that gets built(make sure it has the right shapes)

## TODO

- [x] Build input functions to feed the Estimator API
- [x] Write tests for the data pipeline
- [ ] Write the model code for RVAE
- [ ] Add options in graph building for inference mode
- [ ] Finish writing logs for Tensorboard
- [ ] Write training, eval and inference functions for the Estimator API
- [ ] Write evaluation scripts(BLEU)
