import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import mlflow
import mlflow.xgboost
import mlflow.data
from mlflow.data.pandas_dataset import PandasDataset
from hyperopt import fmin, tpe, rand, atpe, anneal, hp, STATUS_OK, Trials, space_eval

import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, explained_variance_score
from sklearn.preprocessing import StandardScaler
import joblib

# vars
y_names = ['CANTIDAD_VENTAS_ALI','CANTIDAD_VENTAS_BEB'] #  ['CANTIDAD_VENTAS_ALI']  #
empresas = ['2','3','4','5','6'] #['2']  #

for empresa in empresas:
    for y_name in y_names:

        # CSV file path
        file_path = f'empresa_{empresa}_{y_name[-3:]}.csv' #[dbo].[goods_pretraining_data]
        

        # initialize MLflow
        mlflow.set_tracking_uri("mlruns/")
        mlflow.set_experiment(f'{y_name}_empresa_{empresa}')

        # define space
        space = {
            'n_estimators': hp.choice('n_estimators', range(50, 2000)),
            'max_depth': hp.choice('max_depth', range(3, 30)),
            'learning_rate': hp.uniform('learning_rate', 0.01, 0.4),
            'min_child_weight': hp.choice('min_child_weight', range(1, 20)),
            'subsample': hp.uniform('subsample', 0.5, 1),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
            'gamma': hp.uniform('gamma', 0, 1),
            'reg_alpha': hp.uniform('reg_alpha', 0, 1),
            'reg_lambda': hp.uniform('reg_lambda', 0, 1),
            'max_leaves': hp.choice('max_leaves', range(5, 100))
            }

        def load_data(file_path):
            """Load data from a CSV file and perform basic preprocessing."""
            df = pd.read_csv(file_path, index_col=0).pipe(lambda df: df.set_index(pd.to_datetime(df.index)))

            return df

        def split_data(df):
            """Split the data into training and testing sets."""
            y = df[[y_name]]
            x = df.drop(y_name, axis=1)

            # # time ranges
            train_range1_start = pd.to_datetime('2018-01-01')
            train_range1_end = pd.to_datetime('2020-03-26')
            train_range2_start = pd.to_datetime('2020-06-01')
            train_range2_end = pd.to_datetime('2020-11-01')
    
            test_range_start = pd.to_datetime('2020-11-01')
            test_range_end = pd.to_datetime('2021-05-01')
            
            # time ranges
            # train_range_start = '2018-01-01'
            # train_range_end = '2020-03-26'
            
            # test_range_start = '2020-06-01'
            # test_range_end = '2021-06-01'

            # Select data from the date ranges
            x_train = x[(df.index >= train_range1_start) & (df.index <= train_range1_end) |
                        (df.index >= train_range2_start) & (df.index <= train_range2_end)]
            y_train = y[(df.index >= train_range1_start) & (df.index <= train_range1_end) |
                        (df.index >= train_range2_start) & (df.index <= train_range2_end)]

            # x_train = x.loc[train_range_start:train_range_end]
            # y_train = y.loc[train_range_start:train_range_end]

            x_test = x.loc[test_range_start:test_range_end]
            y_test = y.loc[test_range_start:test_range_end]

            return x_train, x_test, y_train, y_test    

        def objective(params):
            
            with mlflow.start_run(nested=True):
                metric='r2' #'rmse', 'mae', 'mse', 'r2', 'mape'
                model = xgb.XGBRegressor(**params)
                model.fit(x_train, y_train)
                
                #METRICS
                result, rmse, mae, mse, r2, mape = evaluate_model(model, x_test, y_test)

                # Logging metrics
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("mse", mse)
                mlflow.log_metric("r2", r2)
                mlflow.log_metric("mape", mape)

                if metric == 'rmse':
                    error = rmse
                elif metric == 'mae':
                    error = mae
                elif metric == 'mse':
                    error = mse
                elif metric == 'r2':
                    error = r2
                elif metric == 'mape':
                    error = mape
                else:
                    raise ValueError("Invalid metric. Choose from: 'rmse', 'mae', 'mse', 'r2', 'mape'")
                
                mlflow.log_params(params)
                mlflow.log_metric('loss', error)
                
                return {'loss': -error, 'status': STATUS_OK}
            
        def evaluate_model(model, x_test, y_test):
            result = pd.DataFrame(np.exp(y_test))
            predictions_log = model.predict(x_test)
            result['PREDICCION'] = np.exp(predictions_log)
            result['ABS_ERROR'] = np.abs(result[y_name] - result.PREDICCION)

            rmse = np.sqrt(mean_squared_error(result[y_name], result['PREDICCION']))
            mae = mean_absolute_error(result[y_name], result['PREDICCION'])
            mse = mean_squared_error(result[y_name], result['PREDICCION'])
            r2 = explained_variance_score(result[y_name], result['PREDICCION'])
            mape = mean_absolute_percentage_error(result[y_name], result['PREDICCION']) * 100
            
            return result, rmse, mae, mse, r2, mape

        def generate_error_plot(result, y_name, plot_path='comparison_plot.png'):
            """Generar una gráfica comparativa entre los valores reales y las predicciones."""
            fig = plt.figure(figsize=(10, 6))
            plt.scatter(result[y_name], result['PREDICCION'], color='blue', alpha=0.5)
            plt.plot([result[y_name].min(), result[y_name].max()], [result[y_name].min(), result[y_name].max()], '--', color='red', linewidth=2)
            plt.xlabel('Valor Real')
            plt.ylabel('Predicción')
            plt.title('Comparación entre Valor Real y Predicción')
            plt.close(fig)
            
            return fig

        def generate_comparison_plot(result, y_name, plot_path='comparison_plot.png'):
            result_ = result.iloc[:, :2]
            resampled_mean = result.resample('W').mean().iloc[:, :2]
            # Plotting
            fig_1 = plt.figure(figsize=(10, 5))
            plt.plot(result_)
            plt.title('comparison_plot')
            plt.legend(resampled_mean.columns)
            plt.close(fig_1)

            fig_2 = plt.figure(figsize=(10, 5))
            plt.plot(resampled_mean)
            plt.title('comparison_plot')
            plt.legend(resampled_mean.columns)
            plt.close(fig_2)
            
            return fig_1, fig_2

        def generate_dataused_plot(y, y_train, y_test):
            # Concatenate y, y_train, and y_test, and resample by week
            concatenated = pd.concat([y, y_train, y_test], axis=1).iloc[:,:3]
            resampled = concatenated.resample('W').mean()
            
            # Plot the data
            fig_1 = plt.figure(figsize=(10, 5))
            concatenated.plot(style='-', ax=plt.gca())
            plt.title('data_used')
            
            # Plot the data
            fig_2 = plt.figure(figsize=(10, 5))
            resampled.plot(style='-', ax=plt.gca())
            plt.title('data_used')
            
            return fig_1, fig_2

        def generate_feature_importance_plot(model, feature_names):
            importance = model.feature_importances_
            sorted_idx = np.argsort(importance)
            fig = plt.figure(figsize=(10, 6))
            plt.barh(range(len(sorted_idx)), importance[sorted_idx], align='center')
            plt.yticks(range(len(sorted_idx)), np.array(feature_names)[sorted_idx])
            plt.xlabel('Feature Importance')
            plt.title('Feature Importance Plot')
            plt.close(fig)
            
            return fig


        # Print MLflow UI URL
        print("MLflow UI URL:", mlflow.get_tracking_uri())

        with mlflow.start_run():
            # Cargar datos
            df = load_data(file_path)
            mlflow_df = mlflow.data.from_pandas(df)
            mlflow.log_input(mlflow_df, context="training_test")

            # Dividir datos en conjuntos de entrenamiento y prueba
            x_train, x_test, y_train, y_test = split_data(df)
            
            # data preprocess
            scaler = StandardScaler()
            
            x_train[x_train.columns] = scaler.fit_transform(x_train)
            y_train = np.log(y_train) 
            
            x_test[x_test.columns] = scaler.transform(x_test)
            y_test = np.log(y_test) 
            
            # Save the scaler to a file
            joblib.dump(scaler, f"scaler_{empresa}_{y_name[-3:]}.pkl")
            # Log the scaler file as an artifact
            mlflow.log_artifact(f"scaler_{empresa}_{y_name[-3:]}.pkl")
                    
            # Perform hyperparameter tuning
            trials = Trials()
            best = fmin(fn=objective, #'rmse', 'mae', 'mse', 'r2', 'mape'
                        space=space,
                        algo=tpe.suggest, #tpe, rand, atpe, anneal
                        max_evals=200,
                        trials=trials,
                        )
            best_params = space_eval(space, best)

            # Train the best model
            model = xgb.XGBRegressor(**best_params)
            model.fit(x_train, y_train)

            # Guardar modelo en MLflow
            mlflow.xgboost.log_model(model, "model_xgboost")

            mlflow.log_params(best_params)
        
            #METRICS
            result, rmse, mae, mse, r2, mape = evaluate_model(model, x_test, y_test)
            # Logging metrics
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mape", mape)

            #ARTIFACT
            # resutlt table
            result.sort_values(by='ABS_ERROR').to_csv(f'result_{empresa}_{y_name[-3:]}.csv')
            mlflow.log_artifact(f'result_{empresa}_{y_name[-3:]}.csv')
            
            result.resample('W').mean().iloc[:,:2].plot(style='-', figsize=(10,5), title=f'test_empresa_2')
            y_test.shape
            
            # Generar gráfica comparativa y guardarla como un artefacto
            result_compare_error = generate_error_plot(result, y_name = y_name)
            mlflow.log_figure(figure = result_compare_error, artifact_file = "error_graph.png")
            
            # Generar gráfica comparativa y guardarla como un artefacto
            result_compare_timeline, result_compare_week_timeline = generate_comparison_plot(result, y_name = y_name)
            mlflow.log_figure(figure = result_compare_timeline, artifact_file = "comparison_graph.png")
            mlflow.log_figure(figure = result_compare_week_timeline, artifact_file = "comparison_graph_week.png")
            
            # Generar gráfica comparativa y guardarla como un artefacto
            data_used_plot, data_used_semanal_plot = generate_dataused_plot(df[y_name], y_train, y_test)
            mlflow.log_figure(figure = data_used_plot, artifact_file = "data_used_plot.png")
            mlflow.log_figure(figure = data_used_semanal_plot, artifact_file = "data_used_semanal_plot.png")
            
            # Generar gráfica comparativa y guardarla como un artefacto
            feature_importance_plot = generate_feature_importance_plot(model, x_train.columns)
            mlflow.log_figure(figure = feature_importance_plot, artifact_file = "feature_importance_plot.png")
            
        
