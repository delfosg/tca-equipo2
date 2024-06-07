use axum::{routing::get, Router};
use sqlx::SqlitePool;

use crate::handler::{create_hotel_good_prediction, get_historic_hotel_goods, get_hotel_good};

#[derive(Clone)]
pub struct AppState {
    pub db: SqlitePool,
}

pub fn create_router(app_state: AppState) -> Router {
    let api_routes = Router::new()
        .route(
            "/:hotel/:good",
            get(get_hotel_good).post(create_hotel_good_prediction),
        )
        .route("/", get(get_historic_hotel_goods)); // return all hotels and all goods

    return Router::new()
        .nest("/prediction", api_routes)
        .with_state(app_state);
}
