# ProjectCosineTraslata

All files that needed some changes have been copied and modified into this folder, leaving the original files untouched.
Moreover, all newly created files are stored in this folder.

To be easily located, every change made to the code (to the scripts in this folder) is sorrounded by the following 
special comments:
```
# TODO REMOVED
# <removed code>
# TODO /REMOVED 
```
or 
```
# TODO ADDED
<added code>
# TODO /ADDED 
```
(The usage of a TODO comment is to allow the TODO management tools present in many IDEs to locate the changes easily.)

Notice that no changes have been made to the recommender systems and their core functionalities. All changes are instead 
only intended to facilitate the automatic execution of all the experiments and the collection of their results.
In particular we point out the following changes, operated to several helper functions involved in the parameter search:
- the parameter `allow_weighting` (present in several functions in `ParameterTuning/run_parameter_search.py`) can now
not only receive a boolean, as before, but also a single string, specifying the exact type of weighting we want to apply 
(`"none", "BM25", "TF-IDF"`). In this case, the parameter `feature_weighting` of the recommender system will not be
tuned, but will instead be fixed on the specified value. Analogous modifications have been carried on in 
`run_all_datasets.py`.
- minor modifications have been made to `run_all_datasets.py` to easily save the results of each different experiment.


Furthermore, some of the notebooks used in our analysis are added:
- `bias_effect_analysis.ipynb`: numerical analysis of the effects of the bias term;
- `result_analysis.ipynb`: collection and visualization (plots and Latex tables) of the experimental results.
- `run_single.ipynb`: all the necessary to tune one recommender system with specified parameters, on a specified dataset
 sand ICM
- `test_icm_content.ipynb`: displays the interval and type of the selected ICM, useful for quickly identifying the type 
of data it contains
- `report_support_figures.ipynb`: generates and shows plots used in the report

Final notices: the simulation have been run (and can be run) from the script `modified/run_all_datasets.py`, 
decommenting any dataset we want to use from the list contained.