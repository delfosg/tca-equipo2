use chrono::NaiveDate;
use serde::{Deserialize, Serialize};

#[derive(Deserialize, Serialize)]
pub struct CreateHotelGoodPrediction {
    pub date: NaiveDate,
    pub model_version: String,
    pub model_name: String,
    pub prediction: i32,
}

#[derive(Serialize, Deserialize)]
pub struct GoodsPredictionOptions {
    pub week_number: Option<u32>,
    pub year: Option<i32>,
}

#[derive(Serialize, Deserialize)]
pub struct HistoricGoodsPredictionOptions {
    pub start_date: NaiveDate,
    pub end_date: NaiveDate,
}
