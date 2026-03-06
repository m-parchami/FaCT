This is the minimal and standalone implementation of our C2-Score metric from the paper.

The only thing that you would need to change to adapt this to your codebase is the `load_top_activations(concept_dir)` function, which is supposed to return a list of tuples of top activating instances for each concept. Here for the dummy example, the function reads the image paths, activation values, and attribution maps from the `dummy_dir` for only two concepts.

Feel free to open an issue if you had any trouble porting C2-Score to your setup!
