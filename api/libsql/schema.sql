CREATE TABLE IF NOT EXISTS "predictions.hotel_goods" (
    date DATE NOT NULL,
    hotel TEXT NOT NULL,
    goods TEXT NOT NULL,
    prediction INT NOT NULL,
    model_version TEXT NOT NULL,
    model_name TEXT NOT NULL,
    PRIMARY KEY (date, hotel, goods)
)
