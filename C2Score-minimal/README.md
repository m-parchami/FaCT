This is the minimal and standalone implementation of our C2-Score metric from the paper.

The only thing that you would need to change to adapt this to your codebase is the `load_top_activations()` function, which is supposed to return a list of tuples of top activating instances for each condcept. Here for the dummy example, the function reads the stats from the `dummy_dir` with only two concepts.
