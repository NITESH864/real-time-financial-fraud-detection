
import json
import time

import requests
from kafka import KafkaConsumer, KafkaProducer

BOOTSTRAP_SERVERS = "localhost:9092"
INPUT_TOPIC = "transactions"
OUTPUT_TOPIC = "fraud_scored"
API_URL = "http://127.0.0.1:8000/score"


def main():
    print("Starting fraud detection consumer...", flush=True)

    consumer = KafkaConsumer(
        INPUT_TOPIC,
        bootstrap_servers=BOOTSTRAP_SERVERS,
        group_id="fraud-scoring-consumer",
        enable_auto_commit=False,
        auto_offset_reset="earliest",
        value_deserializer=lambda data: json.loads(data.decode("utf-8")),
    )

    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        key_serializer=lambda key: key.encode("utf-8"),
        value_serializer=lambda value: json.dumps(value).encode("utf-8"),
        acks="all",
        retries=5,
    )

    print(f"Listening to Kafka topic: {INPUT_TOPIC}", flush=True)

    try:
        for message in consumer:
            event = message.value
            transaction_id = event["transaction_id"]

            try:
                payload = {
                    "step": event["step"],
                    "amount": event["amount"],
                    "type": event["type"],
                    "nameDest": event["nameDest"],
                }

                response = requests.post(
                    API_URL,
                    json=payload,
                    timeout=10,
                )
                response.raise_for_status()
                prediction = response.json()

                result = {
                    "transaction_id": transaction_id,
                    "step": event["step"],
                    "amount": event["amount"],
                    "type": event["type"],
                    "scored_at": time.time(),
                    **prediction,
                }

                producer.send(
                    OUTPUT_TOPIC,
                    key=transaction_id,
                    value=result,
                ).get(timeout=30)

                consumer.commit()

                print(
                    f"{transaction_id} | "
                    f"fraud_probability={prediction['fraud_probability']} | "
                    f"flagged={prediction['flagged']} | "
                    f"risk={prediction['risk_level']}",
                    flush=True,
                )

            except Exception as exc:
                print(
                    f"Error processing {transaction_id}: {exc}",
                    flush=True,
                )
                raise

    except KeyboardInterrupt:
        print("Stopping consumer...", flush=True)

    finally:
        consumer.close()
        producer.flush()
        producer.close()


if __name__ == "__main__":
    main()
