use chrono::NaiveDate;
use serde::{Deserialize, Serialize};
use sqlx::FromRow;

#[derive(Debug, Deserialize, FromRow, Serialize)]
pub struct HotelGoodPrediction {
    pub date: NaiveDate,
    pub hotel: String,
    pub good: String, // option between 'food' and 'drink'
    pub prediction: i64,
    pub model_version: String,
    pub model_name: String,
}
