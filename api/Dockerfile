FROM --platform=linux/amd64 rust:1.78-buster as builder

WORKDIR /tmp

COPY Cargo.toml Cargo.lock ./

COPY src src

COPY .sqlx .sqlx

RUN cargo build --release

FROM --platform=linux/amd64 debian:buster-slim

WORKDIR /usr/bin

COPY --from=builder /tmp/target/release/api-equipo2 .

EXPOSE 3000

CMD ["/usr/bin/api-equipo2"]
