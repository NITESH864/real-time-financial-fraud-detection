
import argparse
import json
import time
from pathlib import Path

import pandas as pd
from kafka import KafkaProducer

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "transactions.csv"

BOOTSTRAP_SERVERS = "localhost:9092"
TOPIC = "transactions"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--delay", type=float, default=0.1)
    args = parser.parse_args()

    if args.limit < 1 or args.delay < 0:
        raise ValueError("Limit must be positive and delay cannot be negative.")

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH, nrows=args.limit)

    required = {"step", "amount", "type", "nameDest"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")

    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        key_serializer=lambda key: key.encode("utf-8"),
        value_serializer=lambda value: json.dumps(value).encode("utf-8"),
        acks="all",
        retries=5,
    )

    sent = 0

    try:
        for index, row in df.iterrows():
            transaction = {
                "transaction_id": f"paysim-{index}",
                "step": int(row["step"]),
                "amount": float(row["amount"]),
                "type": str(row["type"]),
                "nameDest": str(row["nameDest"]),
                "event_time": time.time(),
            }

            producer.send(
                TOPIC,
                key=transaction["transaction_id"],
                value=transaction,
            ).get(timeout=30)

            sent += 1
            print(
                f"Published {transaction['transaction_id']} | "
                f"type={transaction['type']} | "
                f"amount={transaction['amount']:.2f}",
                flush=True,
            )

            if args.delay:
                time.sleep(args.delay)

    finally:
        producer.flush()
        producer.close()

    print(f"Finished: published {sent} transactions.", flush=True)


if __name__ == "__main__":
    main()
