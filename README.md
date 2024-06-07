# tca-equipo2

## Desarrollo del proyecto para la materia Desarrollo de Proyectos de Ingeniería Matemática, Gpo. 502

### Minimum Value Product (Dashboard en PowerBI)

- Ubicado en el directorio `./mvp/AnalisisPrediccionPlatillos.pbix`. Se requieren credenciales a la base de datos de la
  organización socio formadora para consumir los gráficos.

### Consumo de REST API para obtener predicciones del modelo de comidas y bebidas

- HTTP/1.1 GET: `https://api.kmontocam.com/v1/tca/prediction?start_date=2021-04-25&end_date=2021-05-01`.
  Semana predicida a partir de la base de datos (estado actual), contiene las predicciones para todos los hoteles y
  para todas las categorías.

- HTTP/1.1 GET. `https://api.kmontocam.com/v1/tca/:hotel_n/:good?week_number=2021&year=17`.
  Donde hotel_n: número de hotel y good: 'food' o 'drink'. Permite obtener a nivel semana las predicciones de un hotel.
  Ideal para ofrecer cómo servicio a clientes.

- HTTP/1.1 POST. `https://api.kmontocam.com/v1/tca/:hotel_n/:good`.
  Donde hotel_n: número de hotel y good: 'food' o 'drink'.
  Método para crear una predicción generada por el modelo de Machine Learning. Las inferencias se tienen contempladas con una ejecución
  semanal; este habilita la conectividad y facilita el almacenamiento de resultados históricos

### Airflow

- Se tiene programada la ejecución de una tarea que extrae los datos del socio formador (en su estado constante), de tal manera que estos puedan
  ingresar directamente a un proceso de entrenamiento para los modelos de Machine Learning. Queda pendiente integrar almacenamientos persistences,
  eficientes y accesibles para otros flujos de la información, como lo son los feature store y el entrenamiento del modelo per se.
