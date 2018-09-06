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
- [x] Write the model code for RVAE
- [x] Add options in graph building for inference mode
- [-] Finish writing logs for Tensorboard
- [x] Write training, eval and inference functions for the Estimator API
- [x] Remove embedding scope for the embedding tensor. As a direct consequence, check created variables and make sure decoder layer does not get it's own emebedding tensor
- [ ] Pass in feature_cols as params?
- [x] Use tf.train.Scaffold's init_fn for embedding variable initialization
- [ ] Write evaluation scripts(BLEU)
