
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LogisticRegression, Ridge

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


df = pd.read_csv("lecture_engagement_pattern_analyzer.csv")

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())



#  FEATURE ENGINEERING


df["completion_ratio"] = ( df["video_completion_pct"] / 100 )

df["quiz_efficiency"] = (
    df["quiz_score_pct"] / df["quiz_attempts"].replace(0, 1)
)

df["engagement_score"] = (
    0.35 * df["video_completion_pct"]
    + 0.20 * df["quiz_score_pct"]
    + 8 * df["sessions_per_week"]
    + 5 * df["questions_asked"]
    + 4 * (df["notes_taken"] == "Yes")
    - 2 * df["login_delay_min"]
    - 1.5 * df["pause_count"]
)

df["engagement_score"] = (
    df["engagement_score"].clip(0, 100)
)



#  CLASSIFICATION DATA

drop_columns = [
    "student_id",
    "performance_category",
    "next_quiz_score",
    "reference_behavior_group"
]
X = df.drop( columns=drop_columns)
y_class = df[ "performance_category" ]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_class,
    test_size=0.20,
    random_state=42,
    stratify=y_class
)




# REGRESSION DATA

X_reg = df.drop( columns=drop_columns )
y_reg = df[ "next_quiz_score" ]


X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg,
    y_reg,
    test_size=0.20,
    random_state=42
)





numeric_features = X.select_dtypes( include=["int64", "float64"] ).columns.tolist()

categorical_features = X.select_dtypes( include=["object"] ).columns.tolist()


print("\nNumerical Features:")
print(numeric_features)

print("\nCategorical Features:")
print(categorical_features)



preprocessor = ColumnTransformer(
    transformers=[( "numeric", StandardScaler(), numeric_features),
        ( "categorical", OneHotEncoder( handle_unknown="ignore"),
            categorical_features
        )])



#  CLASSIFICATION MODELS + HPO


classification_models = {
    "Logistic Regression": (LogisticRegression(max_iter=5000 ), {"model__C": [ 0.01, 0.1, 1, 10,00]}),
    "Random Forest": (RandomForestClassifier( random_state=42, n_jobs=-1),
        {
            "model__n_estimators": [100,200],
            "model__max_depth": [ None,10,20],
            "model__min_samples_split": [2,5],
            "model__min_samples_leaf": [1,2]
        }),
    "Gradient Boosting": (GradientBoostingClassifier(random_state=42),
        {
            "model__n_estimators": [50,100],
            "model__learning_rate": [0.05,0.1],
            "model__max_depth": [2,3]
        })}


classification_results = []

best_class_model = None
best_class_score = -np.inf
best_class_name = None


print("\n")
print("=" * 60)
print("CLASSIFICATION HPO")
print("=" * 60)


for name, (model, params) in classification_models.items():
    print(f"\nRunning HPO for: {name}")
    pipeline = Pipeline([ (
            "preprocessor",preprocessor ),
        ( "model",model)
    ])


    grid = GridSearchCV( estimator=pipeline,
                        param_grid=params,
                         cv=5,
                        scoring="f1_weighted",
                         n_jobs=-1,
                        verbose=1 )

    grid.fit( X_train, y_train)
    
    predictions = grid.predict(X_test)
    accuracy = accuracy_score(y_test,predictions)
    f1 = f1_score(y_test,predictions,average="weighted")

    print( f"{name} Accuracy: " f"{accuracy:.4f}")
    print( f"{name} F1: "f"{f1:.4f}")
    print("Best Parameters:")
    print(grid.best_params_)

    classification_results.append({"Model": name,
                                    "CV_F1": grid.best_score_,
                                    "Test_Accuracy": accuracy,
                                    "Test_F1": f1,
                                    "Best_Params":str(grid.best_params_)
                                    })


    if grid.best_score_ > best_class_score:
        best_class_score = (    grid.best_score_ )
        best_class_model = (  grid.best_estimator_ )
        best_class_name = name


# SAVE BEST CLASSIFICATION MODEL


joblib.dump( best_class_model, "classification_model.pkl" )


print("\nBest Classification Model:")
print(best_class_name)

print("Saved as: classification_model.pkl")



# REGRESSION MODELS + HPO

regression_models = {
    "Ridge Regression": (Ridge(),
        {
            "model__alpha": [ 0.01,0.1,1,10,100]
        }
    ),

    "Random Forest Regressor": ( RandomForestRegressor(random_state=42,n_jobs=-1),
        {
            "model__n_estimators": [100,200],
            "model__max_depth": [None,10,20],
            "model__min_samples_split": [2,5],
            "model__min_samples_leaf": [1,2]
        }
    ),

    "Gradient Boosting Regressor": (GradientBoostingRegressor(random_state=42),
        {
            "model__n_estimators": [50,100],
            "model__learning_rate": [0.05,0.1],
            "model__max_depth": [2,3]
        }
    )
}


regression_results = []

best_reg_model = None
best_reg_score = -np.inf
best_reg_name = None


print("\n")
print("=" * 60)
print("REGRESSION HPO")
print("=" * 60)


for name, (model, params) in regression_models.items():
    print(
        f"\nRunning HPO for: {name}"
    )

    pipeline = Pipeline([("preprocessor",preprocessor), ("model",model)
    ])

    grid = GridSearchCV(estimator=pipeline,
                        param_grid=params,
                        cv=5,
                        scoring="r2",
                        n_jobs=-1,
                        verbose=1
    )


    grid.fit( X_train_reg,y_train_reg )

    predictions = grid.predict( X_test_reg)
    mae = mean_absolute_error(y_test_reg,predictions)
    rmse = np.sqrt(mean_squared_error(y_test_reg,predictions ))
    r2 = r2_score( y_test_reg, predictions )

    print( f"{name} MAE: " f"{mae:.4f}")
    print( f"{name} RMSE: "f"{rmse:.4f}")
    print(f"{name} R2: "f"{r2:.4f}")
    print("Best Parameters:")
    print( grid.best_params_)
    
    regression_results.append({
        "Model": name,
        "CV_R2": grid.best_score_,
        "Test_MAE": mae,
        "Test_RMSE": rmse,
        "Test_R2": r2,
        "Best_Params":str(grid.best_params_)
    })


    if grid.best_score_ > best_reg_score:
        best_reg_score = (grid.best_score_)
        best_reg_model = (grid.best_estimator_)
        best_reg_name = name


# SAVE BEST REGRESSION MODEL

joblib.dump( best_reg_model,"regression_model.pkl")

print("\nBest Regression Model:")
print(best_reg_name)
print("Saved as: regression_model.pkl")



#  SAVE RESULTS

classification_results_df = pd.DataFrame(classification_results)
classification_results_df.to_csv( "classification_hpo_results.csv",index=False)


regression_results_df = pd.DataFrame( regression_results)
regression_results_df.to_csv( "regression_hpo_results.csv",index=False )



# FINAL RESULTS


print("\n")
print("=" * 60)
print("FINAL CLASSIFICATION RESULTS")
print("=" * 60)
print(
    classification_results_df
    .sort_values("Test_F1", ascending=False )
    .to_string(index=False)
)


print("\n")
print("=" * 60)
print("FINAL REGRESSION RESULTS")
print("=" * 60)
print(
    regression_results_df
    .sort_values("Test_R2", ascending=False)
    .to_string(index=False)
)


