from datetime import datetime
from airflow.decorators import dag, task
from typing import Literal

from airflow.providers.microsoft.mssql.hooks.mssql import MsSqlHook
import pandas as pd
import numpy as np


def get_ventas_dia_df(ventas_df, empresa, y_var):

    ############ limpia #############
    ventas_df = ventas_df[~(ventas_df["ID_map_status"] == 3)]

    columns = ["cantidad_beb", "cantidad_ali"]
    for columns in columns:
        # Calcular los cuartiles y el rango intercuartílico (IQR)
        Q1 = ventas_df[columns].quantile(0.25)
        Q3 = ventas_df[columns].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        ventas_df = ventas_df[
            (ventas_df[columns] >= lower_bound) & (ventas_df[columns] <= upper_bound)
        ]

    ventas_df["fecha_hoy"] = pd.to_datetime(ventas_df["fecha_hoy"])
    ventas_df = ventas_df[ventas_df["ID_empresa"] == empresa]

    ############ dia_df #############
    ventas_df = ventas_df.copy()
    ventas_df = ventas_df.sort_values(by="fecha_hoy")
    ventas_dia_df = pd.DataFrame()
    ventas_grouped_sum = ventas_df.groupby("fecha_hoy").sum()

    if y_var == "CANTIDAD_VENTAS_ALI":
        ventas_dia_df["CANTIDAD_VENTAS_ALI"] = (
            ventas_grouped_sum.cantidad_ali.round().astype(int)
        )
    elif y_var == "CANTIDAD_VENTAS_BEB":
        ventas_dia_df["CANTIDAD_VENTAS_BEB"] = (
            ventas_grouped_sum.cantidad_beb.round().astype(int)
        )

    ventas_dia_df.index = pd.to_datetime(ventas_dia_df.index)
    ventas_dia_df["day_month"] = ventas_dia_df.index.strftime("%m-%d")

    # TIME variables
    ventas_dia_df["DIA_SEMANA"] = pd.Series(
        ventas_dia_df.index.day_of_week, index=ventas_dia_df.index
    )
    ventas_dia_df["CUARTO_ANO"] = pd.Series(
        ventas_dia_df.index.quarter, index=ventas_dia_df.index
    )
    ventas_dia_df["MES"] = pd.Series(
        ventas_dia_df.index.month, index=ventas_dia_df.index
    )
    ventas_dia_df["DIA_ANO"] = pd.Series(
        ventas_dia_df.index.dayofyear, index=ventas_dia_df.index
    )

    # Add sales of the same weekday last week
    ventas_dia_df[f"{y_var}_LAST_WEEK"] = ventas_dia_df[y_var].shift(7)
    ventas_dia_df[f"{y_var}_LAST_MONTH"] = ventas_dia_df[y_var].shift(30)
    ventas_dia_df[f"{y_var}_LAST_YEAR"] = ventas_dia_df[y_var].shift(365)
    ventas_dia_df[f"{y_var}_LAST_LAST_WEEK"] = ventas_dia_df[y_var].shift(14)

    def is_high_season(date):
        year = date.year
        # verano, asueto revolución, invierno, asueto constitución, asueto petróleo
        if (
            (
                pd.Timestamp(year=year, month=7, day=17)
                <= date
                <= pd.Timestamp(year=year, month=8, day=27)
            )
            or (
                pd.Timestamp(year=year, month=11, day=18)
                <= date
                <= pd.Timestamp(year=year, month=11, day=20)
            )
            or (
                pd.Timestamp(year=year, month=12, day=18)
                <= date
                <= pd.Timestamp(year=year + 1, month=1, day=5)
            )
            or (
                pd.Timestamp(year=year, month=2, day=3)
                <= date
                <= pd.Timestamp(year=year, month=2, day=5)
            )
            or (
                pd.Timestamp(year=year, month=3, day=16)
                <= date
                <= pd.Timestamp(year=year, month=3, day=18)
            )
            or (
                pd.Timestamp(year=year, month=3, day=25)
                <= date
                <= pd.Timestamp(year=year, month=4, day=7)
            )
        ):
            return 1
        return 0

    def get_semana_santa(date):
        # verano, asueto revolución, invierno, asueto constitución, asueto petróleo
        if (
            (
                pd.Timestamp(year=2018, month=3, day=25)
                <= date
                <= pd.Timestamp(year=2018, month=3, day=31)
            )
            or (
                pd.Timestamp(year=2019, month=4, day=14)
                <= date
                <= pd.Timestamp(year=2019, month=4, day=20)
            )
            or (
                pd.Timestamp(year=2020, month=4, day=5)
                <= date
                <= pd.Timestamp(year=2020, month=4, day=11)
            )
            or (
                pd.Timestamp(year=2021, month=3, day=28)
                <= date
                <= pd.Timestamp(year=2021, month=4, day=3)
            )
        ):
            return 1
        return 0

    def get_semana_pascua(date):
        # verano, asueto revolución, invierno, asueto constitución, asueto petróleo
        if (
            (
                pd.Timestamp(year=2018, month=4, day=1)
                <= date
                <= pd.Timestamp(year=2018, month=4, day=7)
            )
            or (
                pd.Timestamp(year=2019, month=4, day=21)
                <= date
                <= pd.Timestamp(year=2019, month=4, day=27)
            )
            or (
                pd.Timestamp(year=2020, month=4, day=12)
                <= date
                <= pd.Timestamp(year=2020, month=4, day=18)
            )
            or (
                pd.Timestamp(year=2021, month=4, day=4)
                <= date
                <= pd.Timestamp(year=2021, month=4, day=10)
            )
        ):
            return 1
        return 0

    ventas_dia_df["fecha"] = ventas_dia_df.index
    ventas_dia_df["VACACIONES"] = pd.to_datetime(ventas_dia_df["fecha"]).apply(
        is_high_season
    )
    ventas_dia_df["SEMANA_SANTA"] = pd.to_datetime(ventas_dia_df["fecha"]).apply(
        get_semana_santa
    )
    ventas_dia_df["SEMANA_PASCUA"] = pd.to_datetime(ventas_dia_df["fecha"]).apply(
        get_semana_pascua
    )
    ventas_dia_df.drop(columns="fecha", inplace=True)

    # VPROMEDIO variables
    def calculate_7days_mean(df, column_name):

        means = [np.nan] * len(df)  # Initialize a list with NaN values
        n = len(df)
        for i in range(n - 7, -1, -7):
            chunk = df[column_name].iloc[i : i + 7]
            chunk_mean = chunk.mean()
            means[i] = chunk_mean
        means = pd.Series(means)
        means = means.fillna(method="ffill")
        return means

    ventas_dia_df[f"{y_var}_PROMEDIO_7DIAS"] = calculate_7days_mean(
        ventas_dia_df, y_var
    ).values
    ventas_dia_df[f"{y_var}_PROMEDIO_7DIAS_PREVIO"] = ventas_dia_df[
        f"{y_var}_PROMEDIO_7DIAS"
    ].shift(7)
    ventas_dia_df = ventas_dia_df.drop(f"{y_var}_PROMEDIO_7DIAS", axis=1)

    # MA feature
    vprometio_7dias_unique = pd.Series(
        ventas_dia_df[f"{y_var}_PROMEDIO_7DIAS_PREVIO"].unique()
    )
    ma_ranges = [4, 8]
    for ma_range in ma_ranges:
        ma = vprometio_7dias_unique.rolling(window=ma_range).mean().repeat(7)
        n_nans = ventas_dia_df.shape[0] - len(ma)
        ma = pd.concat([pd.Series([np.nan] * n_nans), ma], ignore_index=True)
        ventas_dia_df[f"{y_var}_PROMEDIO_7DIAS_MA{ma_range}"] = ma.values

    return ventas_dia_df


def get_ocupaciones_df(ocupaciones_df):

    ocupaciones_df = ocupaciones_df.copy()
    ocupaciones_df = ocupaciones_df.sort_values(by="Fecha_hoy")
    ocupaciones_df = pd.concat(
        [
            ocupaciones_df,
            pd.get_dummies(
                ocupaciones_df.ID_Tipo_Habitacion, dtype=int, prefix="PCT_T_HAB"
            ),
        ],
        axis=1,
    )
    ocupaciones_df = pd.concat(
        [
            ocupaciones_df,
            pd.get_dummies(ocupaciones_df.ID_Paquete, dtype=int, prefix="PCT_T_PAQ"),
        ],
        axis=1,
    )
    ocupaciones_df = pd.concat(
        [
            ocupaciones_df,
            pd.get_dummies(
                ocupaciones_df.ID_Segmento_Mercado, dtype=int, prefix="PCT_T_SEGMERC"
            ),
        ],
        axis=1,
    )
    ocupaciones_dia_df = pd.DataFrame()
    ocupaciones_dia_df["CANTIDAD_REGISTOS_OCUP"] = (
        ocupaciones_df.groupby("Fecha_hoy").count().iloc[:, 0]
    )
    ocupaciones_grouped_sum = ocupaciones_df.groupby("Fecha_hoy").sum()
    ocupaciones_dummies_pct = ocupaciones_grouped_sum.iloc[:, 6:].divide(
        ocupaciones_dia_df["CANTIDAD_REGISTOS_OCUP"], axis=0
    )
    ocupaciones_dia_df = pd.concat(
        [ocupaciones_dia_df, ocupaciones_dummies_pct], axis=1
    )
    ocupaciones_dia_df["NUM_ADU"] = ocupaciones_grouped_sum.num_adu
    ocupaciones_dia_df["NUM_MEN"] = ocupaciones_grouped_sum.num_men
    ocupaciones_dia_df.index = pd.to_datetime(ocupaciones_dia_df.index)

    return ocupaciones_dia_df


def get_dia_festivo_df(dia_festivo_df):

    dia_festivo_df = dia_festivo_df.copy()
    dia_festivo_df = dia_festivo_df.drop(0)
    dia_festivo_df = dia_festivo_df.set_axis(pd.to_datetime(dia_festivo_df.fecha))
    dia_festivo_df["DIA_FESTIVO"] = 1
    dia_festivo_df = dia_festivo_df[["DIA_FESTIVO"]]
    dia_festivo_df["day_month"] = dia_festivo_df.index.strftime("%m-%d")
    dia_festivo_df = dia_festivo_df.drop_duplicates(subset="day_month")

    return dia_festivo_df


@dag(
    dag_id="weekly_goods_model_preprocessing_v1",
    description="",
    schedule="0 0 * * 0",
    start_date=datetime(2022, 5, 1),
    tags=["tsa", "weekly", "equipo2"],
    catchup=False,
)
def weekly_goods_model_preprocessing_v1():
    """
    This DAG is intended to extract the data needed for the pretraining the foods and
    drinks prediction model. The data is extracted from the TCABDFRONT2 database.
    """

    @task(multiple_outputs=True)
    def fetch_mssql_tables() -> "dict[str, pd.DataFrame]":
        mssql_hook = MsSqlHook(mssql_conn_id="mssql_tcabdfront2")
        goods_sales = mssql_hook.get_pandas_df(
            r"""
            SELECT
                [fecha_hoy],
                [cantidad_ali],
                [cantidad_beb],
                [ID_empresa],
                [ID_map_status]
            FROM
                [dbo].[iaab_Detalles_Vtas]
        """
        )
        occupations = mssql_hook.get_pandas_df(
            r"""
            SELECT
                [Fecha_hoy],
                [ID_empresa],
                [ID_Tipo_Habitacion],
                [ID_Paquete],
                [ID_Segmento_Mercado],
                [num_adu],
                [num_men]
            FROM
                [dbo].[iar_Ocupaciones3]
            """
        )
        occupations_holidays = mssql_hook.get_pandas_df(
            r"""
            SELECT
                [fecha],
                [festividad],
                [festividad_2]
            FROM
                [dbo].[iar_dias_festivos]
            """
        )

        return {
            "goods_sales": goods_sales,
            "occupations_holidays": occupations_holidays,
            "occupations": occupations,
        }

    @task()
    def data_munging(
        tables: "dict[str, pd.DataFrame]",
        empresa: int,
        y_var: Literal["CANTIDAD_VENTAS_ALI", "CANTIDAD_VENTAS_BEB"],
    ):

        goods_sales = tables["goods_sales"]
        occupations = tables["occupations"]
        occupations_holidays = tables["occupations_holidays"]

        ocupaciones_empresa_n = occupations[occupations.ID_empresa == empresa]
        ventas_dia_empresa_n = get_ventas_dia_df(goods_sales, empresa, y_var)
        ocupaciones_dia_empresa_n = get_ocupaciones_df(ocupaciones_empresa_n)
        dia_festivo_empresa_n = get_dia_festivo_df(occupations_holidays)

        empresa_n_df = ocupaciones_dia_empresa_n.join(ventas_dia_empresa_n, how="inner")
        empresa_n_df_index = empresa_n_df.index
        empresa_n_df = empresa_n_df.merge(
            dia_festivo_empresa_n, on="day_month", how="left"
        )
        empresa_n_df = empresa_n_df.set_index(empresa_n_df_index)
        empresa_n_df.index.name = "FECHA"
        empresa_n_df.fillna(0, inplace=True)
        empresa_n_df.drop("day_month", axis=1, inplace=True)

        if y_var == "CANTIDAD_VENTAS_ALI":
            empresa_n_df = empresa_n_df[empresa_n_df.CANTIDAD_VENTAS_ALI != 0]
        elif y_var == "CANTIDAD_VENTAS_BEB":
            empresa_n_df = empresa_n_df[empresa_n_df.CANTIDAD_VENTAS_BEB != 0]

        return empresa_n_df

    # @task()
    # def write_to_mssql(df: "pd.DataFrame"):
    #     mssql_hook = MsSqlHook(mssql_conn_id="mssql_tcabdfront2")
    #     print(df.columns)
    #     mssql_hook.insert_rows("dbo.goods_pretraining_data", df)

    tables_paths = fetch_mssql_tables()
    empresa_2_alimentos = data_munging(
        tables_paths, empresa=2, y_var="CANTIDAD_VENTAS_ALI"
    )
    empresa_2_bedidas = data_munging(
        tables_paths, empresa=2, y_var="CANTIDAD_VENTAS_BEB"
    )
    empresa_3_alimentos = data_munging(
        tables_paths, empresa=3, y_var="CANTIDAD_VENTAS_ALI"
    )
    empresa_3_bedidas = data_munging(
        tables_paths, empresa=3, y_var="CANTIDAD_VENTAS_BEB"
    )
    empresa_4_alimentos = data_munging(
        tables_paths, empresa=4, y_var="CANTIDAD_VENTAS_ALI"
    )
    empresa_4_bedidas = data_munging(
        tables_paths, empresa=4, y_var="CANTIDAD_VENTAS_BEB"
    )
    empresa_5_alimentos = data_munging(
        tables_paths, empresa=5, y_var="CANTIDAD_VENTAS_ALI"
    )
    empresa_5_bedidas = data_munging(
        tables_paths, empresa=5, y_var="CANTIDAD_VENTAS_BEB"
    )
    empresa_6_alimentos = data_munging(
        tables_paths, empresa=6, y_var="CANTIDAD_VENTAS_ALI"
    )
    empresa_6_bedidas = data_munging(
        tables_paths, empresa=6, y_var="CANTIDAD_VENTAS_BEB"
    )

    # write_to_mssql(empresa_2_alimentos)
    # write_to_mssql(empresa_2_bedidas)
    # write_to_mssql(empresa_3_alimentos)
    # write_to_mssql(empresa_3_bedidas)
    # write_to_mssql(empresa_4_alimentos)
    # write_to_mssql(empresa_4_bedidas)
    # write_to_mssql(empresa_5_alimentos)
    # write_to_mssql(empresa_5_bedidas)
    # write_to_mssql(empresa_6_alimentos)
    # write_to_mssql(empresa_6_bedidas)


main = weekly_goods_model_preprocessing_v1()
