# Machine Learning module for H2 Computing

- This is a self-contain package for teaching the ML module that is modified and expanded from the starter kit provided by CPDD.(2025 July version)
- Most of the code(75%) is re-written.
- See Issues with the started kit below.
- Notes and Codes are organised into seperate jupyter notbook files
  - teaching order, `00xx,..04xx.ipynb`
    - optional (`AI.ipynb, knn_normalisation.ipynb`)
  - `AI_TLDR.ipynb` is the summarised version of `AI_Detailed.ipynb`. This is completely new content to bridge from AI to ML. The AI definition and the details are optional, not in syllabus
  - Note that numpy is introduced first before knn and k_means, so that numpy code can be presented as alternative to pure python code.
  - plotting code and helper functions moved to `ml_lib.ipynb`
  - `01_numpy(_ans).ipynb` is modified and re-organised
  - `02_knn.ipynb, 03_k_means.ipynb` added numpy code solutions as alternative to pure python code. original code is modified for readability and teachability.
    - for knn, the concept of Overfitting and Underfitting is introduced bwfore parameter tuning. A side-discussion on Normalisation is added `knn_normalisation.ipynb`(optional).
  - `04_sklearn.ipynb` also heavily modified for readability and teachability.

# Issues with StarterKit from CPDD(TLDR)

### 1. AI definition (optional)

- No proper definition of AI
- O Level :
  > The ability of a computer to perform
  > <u>complex tasks without constant human
  > guidance and improve its performance</u>
  > as more data is collected
  >
- Most of the introductory contents is taken wholesale from O level textbook.


#### ML Taxonomy and Terms

- Not just Supervised vs Unsupervised
- Classification vs Regression task

### 2. KNN (using Python)

1. The raw data set's features are not of the same units( tempo:bpm, length : seconds), although the range is close.
   - If we use minutes instead of seconds for length, then the differences in units will be very apparent. The values of the tempo feature will mask the significance of the length feature in $ (\text{tempo difference})^2 + (\text{length difference})^2$. Order of $10^2$ vs $10^0$
   - need to perform normalisations. THIS IS NOT EVEN discuss at all which is A VERY IMPORTANT concept in ML especially with KNN.
2. bug in `calculate_accuracy`:
   - `predict(k, training_features, training_labels, test_features[i]) == test_labels[i] ` i NOT 1
3. Conceptual Error in k-fold cross validation Algorithm:
   - `random.shuffle` is done for every k value, this means that there are TWO variable parameters, k and the order of the folds in the dataset.
   - The order of the dataset must be fixed when evaluating k

### 3. K-means (Using Python)

- re-organise the flow of code
  1. choose initial centroids
  2. assign data points to clusters (nearest euclid distance to each centroids) - `form_clusters()`
  3. compute new centroids from each cluster using average of x and y values of each cluster members - `find_centroids()`
  4. Repeat 3-4 until convergence or max iterations.

### 4. sklearn
- using `cross_validate()` with `cv=5` means a StrifiedFold with no shuffle will be created on every call. The data set is ordered based on genre. This will create folds that will mainly contain data points of the same class labels. This is very bad and naive implementation and the comment is this is not in syllabus.

- Overall I find the starter kit contains code implementation that is overly simplified (for teaching) but gloss over very important real world implementattion details like the effects of feature scale and dataset ordering.
---
