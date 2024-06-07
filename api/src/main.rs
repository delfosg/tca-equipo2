pub mod app;
pub mod handler;
pub mod model;
pub mod schema;

use crate::app::AppState;
use sqlx::SqlitePool;
use std::env;
use std::error::Error;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let conn = SqlitePool::connect(
        &env::var("DATABASE_URL").expect("missing DATABASE_URL environment variable"),
    )
    .await?;

    sqlx::query(
        r#"
        CREATE TABLE IF NOT EXISTS "predictions.hotel_goods" (
            date DATE NOT NULL,
            hotel TEXT NOT NULL,
            good TEXT NOT NULL,
            prediction INT NOT NULL,
            model_version TEXT NOT NULL,
            model_name TEXT NOT NULL,
            PRIMARY KEY (date, hotel, good)
        );
        "#,
    )
    .execute(&conn)
    .await?;

    let state = AppState { db: conn };

    let app = app::create_router(state);
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await?;

    axum::serve(listener, app).await?;

    Ok(())
}
