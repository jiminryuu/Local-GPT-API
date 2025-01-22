#!/bin/bash

# URL of the API endpoint
URL="http://127.0.0.1:8000/generate"

# JSON payload
PAYLOAD='{
  "prompt": "My day was not great, my girlfriend just dumped me, what should I do?",
  "max_length": 40,
  "temperature": 0.1
}'

# Loop to send 20 requests
for i in {1..20}
do
  echo "Sending request $i..."
  RESPONSE=$(curl -s -X POST "$URL" -H "Content-Type: application/json" -d "$PAYLOAD")
  echo "Response $i: $RESPONSE"
done

echo "All 20 requests sent."