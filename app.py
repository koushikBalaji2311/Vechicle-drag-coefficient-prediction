import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from scikeras.wrappers import KerasRegressor
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Add, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error
import tensorflow as tf


df = pd.read_csv('/content/combined_data.csv')

target_variable1 = 'Average Cd'
feature_columns = [col for col in df.columns if col not in [target_variable1, 'Std Cd']]

X = df[feature_columns]
y1 = df[target_variable1]


X_train, X_test, y1_train, y1_test = train_test_split(X, y1, test_size=0.2, random_state=42)


categorical_columns = X.select_dtypes(include=['object']).columns
numerical_columns = X.select_dtypes(include=[np.number]).columns

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_columns),
        ('cat', categorical_transformer, categorical_columns)
    ]
)

X_train_preprocessed = preprocessor.fit_transform(X_train)
input_dim = X_train_preprocessed.shape[1]


def create_neural_network():
    inputs = Input(shape=(input_dim,))
    x = Dense(512, activation='relu')(inputs)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(1, activation='linear')(x)

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.0003), loss='mean_squared_error', metrics=['mae'])
    return model

keras_regressor = KerasRegressor(model=create_neural_network, epochs=150, batch_size=32, validation_split=0.1)

gbr_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

from sklearn.ensemble import StackingRegressor

ensemble_model = StackingRegressor(
    estimators=[('gbr', gbr_model), ('nn', keras_regressor)],
    final_estimator=GradientBoostingRegressor(n_estimators=50, learning_rate=0.05, random_state=42)
)

model_pipeline_optimized_3 = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('ensemble', ensemble_model)
])
model_pipeline_optimized_3.fit(X_train, y1_train)

predictions_ensemble = model_pipeline_optimized_3.predict(X_test)

mse_ensemble = mean_squared_error(y1_test, predictions_ensemble)

print(f"Mean Squared Error for 'Average Cd' (Ensemble Model): {mse_ensemble:.4f}")
