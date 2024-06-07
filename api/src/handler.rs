use axum::http::StatusCode;
use axum::{
    extract::{Json, Path, Query, State},
    response::IntoResponse,
};
use chrono::{Datelike, Local, NaiveDate, Weekday};
use serde_json::Value;

use crate::app::AppState;
use crate::model::HotelGoodPrediction;
use crate::schema::{
    CreateHotelGoodPrediction, GoodsPredictionOptions, HistoricGoodsPredictionOptions,
};

pub async fn create_hotel_good_prediction(
    Path((hotel, good)): Path<(String, String)>,
    State(state): State<AppState>,
    Json(payload): Json<CreateHotelGoodPrediction>,
) -> Result<impl IntoResponse, (StatusCode, Json<Value>)> {
    let query_result = sqlx::query_as!(
        HotelGoodPrediction,
        r#"
        INSERT INTO "predictions.hotel_goods" (date, hotel, good, model_version, model_name, prediction)
        VALUES (?, ?, ?, ?, ?, ?) RETURNING *
        "#,
        payload.date,
        hotel,
        good,
        payload.model_version,
        payload.model_name,
        payload.prediction,
    )
    .fetch_one(&state.db)
    .await;

    match query_result {
        Ok(prediction) => Ok((
            StatusCode::CREATED,
            Json(serde_json::json!({"data": prediction, "success": true})),
        )),
        Err(e) => Ok((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string(), "sucess": false})),
        )),
    }
}

pub async fn get_hotel_good(
    Path((hotel, good)): Path<(String, String)>,
    Query(opts): Query<GoodsPredictionOptions>,
    State(state): State<AppState>,
) -> Result<impl IntoResponse, (StatusCode, Json<Value>)> {
    let now = Local::now();
    let now_year = now.year();
    let now_week_number = now.iso_week().week();

    let week_number = opts.week_number.unwrap_or(now_week_number); // if not provided, use current week
    let year = opts.year.unwrap_or(now_year); // if not provided, use current year

    let start_week = NaiveDate::from_isoywd_opt(year, week_number, Weekday::Mon);
    let end_week = NaiveDate::from_isoywd_opt(year, week_number, Weekday::Sun);

    let query_result = sqlx::query_as!(
        HotelGoodPrediction,
        r#"
        SELECT * FROM "predictions.hotel_goods" WHERE hotel = ? AND good = ? AND date >= ? AND date <= ?
        "#,
        hotel,
        good,
        start_week,
        end_week
    )
    .fetch_all(&state.db)
    .await;

    match query_result {
        Ok(predictions) => Ok((
            StatusCode::CREATED,
            Json(serde_json::json!({"data": predictions, "success": true})),
        )),
        Err(e) => Ok((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string(), "sucess": false})),
        )),
    }
}

// requires auth
pub async fn get_historic_hotel_goods(
    Query(opts): Query<HistoricGoodsPredictionOptions>,
    State(state): State<AppState>,
) -> Result<impl IntoResponse, (StatusCode, Json<Value>)> {
    let query_result = sqlx::query_as!(
        HotelGoodPrediction,
        r#"
        SELECT * FROM "predictions.hotel_goods" WHERE date >= ? AND date <= ?
        "#,
        opts.start_date,
        opts.end_date
    )
    .fetch_all(&state.db)
    .await;

    match query_result {
        Ok(predictions) => Ok((
            StatusCode::CREATED,
            Json(serde_json::json!({"data": predictions, "success": true})),
        )),
        Err(e) => Ok((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e.to_string(), "sucess": false})),
        )),
    }
}
