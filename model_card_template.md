# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
- **Model**: RandomForestClassifier (scikit-learn, random_state=42)
- **Inputs**: Tabular demographic features from the 1994 U.S. Census.
- **Outputs**: Binary label indicating whether annual income exceeds $50K.

## Intended Use
This model demonstrates a production-ready machine learning pipeline with FastAPI. It is meant for educational use and should not be applied to high-stakes decisions such as credit, employment, or housing determination.

## Training Data
Trained on the UCI Census Income dataset (`census.csv`), which contains adult demographic records with features such as age, workclass, education, marital-status, occupation, relationship, race, sex, and native-country. The dataset was cleaned and split 80/20 into training and evaluation sets.

## Evaluation Data
A 20% holdout split from the same dataset served as evaluation data. Metrics below are computed on this unseen subset.

## Metrics
Overall test-set performance:
- Precision: 0.7419
- Recall: 0.6384
- F1: 0.6863

Slice analysis highlighted performance disparities across subgroups. Examples include:
- Gender: F1 = 0.6015 for females vs. 0.6997 for males.
- Education: F1 = 0.0000 for individuals with 7th-8th grade education, while Doctorate holders achieve F1 = 0.8793.

## Ethical Considerations
The dataset contains sensitive attributes (e.g., race, sex). The model may reinforce existing societal biases and exhibit uneven accuracy across demographic groups. Use in decision-making contexts could perpetuate unfair outcomes without additional fairness checks.

## Caveats and Recommendations
- Data originates from 1994 and may not represent current populations.
- Some demographic slices have very limited data, yielding unstable metrics.
- Before real-world deployment, retrain with more recent and balanced data, and monitor slice-level performance over time.
