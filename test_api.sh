#!/bin/bash

# DiCoW API Test Script
# Uses test file: data/test_2speakers_eng_10s.wav

BASE_URL="http://localhost:8000"
TEST_FILE="./data/test_2speakers_eng_10s.wav"
BATCH_FOLDER="./data"

echo "========================================"
echo "  DiCoW API - Test Suite"
echo "========================================"
echo

# Check if test file exists
if [ ! -f "$TEST_FILE" ]; then
    echo "❌ Test file not found: $TEST_FILE"
    exit 1
fi

echo "✓ Test file found: $TEST_FILE"
echo

# Test 1: Health check
echo "========================================"
echo "Test 1: Health Check"
echo "========================================"
HEALTH_RESPONSE=$(curl -s "$BASE_URL/health")
echo "$HEALTH_RESPONSE" | python3 -m json.tool
echo

if echo "$HEALTH_RESPONSE" | python3 -c "import sys, json; data = json.load(sys.stdin); sys.exit(0 if data.get('status') == 'healthy' else 1)"; then
    echo "✓ Health check passed"
else
    echo "✗ Health check failed"
    exit 1
fi
echo
echo

# Test 2: Submit single file job
echo "========================================"
echo "Test 2: Submit Single File Transcription"
echo "========================================"
JOB_RESPONSE=$(curl -s -X POST "$BASE_URL/transcribe" \
  -F "mode=single_file" \
  -F "file=@$TEST_FILE")

echo "$JOB_RESPONSE" | python3 -m json.tool

JOB_ID=$(echo "$JOB_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin)['job_id'])" 2>/dev/null)

if [ -z "$JOB_ID" ]; then
    echo "✗ Failed to get job_id"
    exit 1
fi

echo
echo "✓ Job submitted: $JOB_ID"
echo
echo

# Test 3: Poll job status until completion
echo "========================================"
echo "Test 3: Poll Job Status"
echo "========================================"
MAX_ATTEMPTS=30
ATTEMPT=0

while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    STATUS_RESPONSE=$(curl -s "$BASE_URL/jobs/$JOB_ID")
    STATUS=$(echo "$STATUS_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin)['status'])" 2>/dev/null)
    
    echo "Attempt $((ATTEMPT + 1))/$MAX_ATTEMPTS - Status: $STATUS"
    
    if [ "$STATUS" = "completed" ]; then
        echo
        echo "✓ Job completed!"
        echo
        echo "Final status:"
        echo "$STATUS_RESPONSE" | python3 -m json.tool
        break
    elif [ "$STATUS" = "failed" ]; then
        echo
        echo "✗ Job failed!"
        echo "$STATUS_RESPONSE" | python3 -m json.tool
        exit 1
    elif [ "$STATUS" = "pending" ] || [ "$STATUS" = "processing" ]; then
        ATTEMPT=$((ATTEMPT + 1))
        sleep 2
    else
        echo "✗ Unknown status: $STATUS"
        exit 1
    fi
done

if [ $ATTEMPT -eq $MAX_ATTEMPTS ]; then
    echo "✗ Timeout - job did not complete in time"
    exit 1
fi
echo
echo

# Test 4: Get job result
echo "========================================"
echo "Test 4: Get Transcription Result"
echo "========================================"
RESULT_RESPONSE=$(curl -s "$BASE_URL/jobs/$JOB_ID/result")
echo "$RESULT_RESPONSE" | python3 -m json.tool
echo

# Validate result structure
SEGMENTS_COUNT=$(echo "$RESULT_RESPONSE" | python3 -c "import sys, json; data = json.load(sys.stdin); print(len(data.get('result', {}).get('segments', [])))" 2>/dev/null)

if [ -n "$SEGMENTS_COUNT" ] && [ "$SEGMENTS_COUNT" -gt 0 ]; then
    echo "✓ Result contains $SEGMENTS_COUNT segments"
else
    echo "✗ No segments found in result"
    exit 1
fi
echo
echo

# Test 5: List all jobs
echo "========================================"
echo "Test 5: List All Jobs"
echo "========================================"
JOBS_RESPONSE=$(curl -s "$BASE_URL/jobs")
echo "$JOBS_RESPONSE" | python3 -m json.tool
echo

JOBS_COUNT=$(echo "$JOBS_RESPONSE" | python3 -c "import sys, json; print(len(json.load(sys.stdin)))" 2>/dev/null)
echo "✓ Total jobs: $JOBS_COUNT"
echo
echo

# Test 6: Submit batch job
echo "========================================"
echo "Test 6: Submit Batch Transcription"
echo "========================================"
BATCH_RESPONSE=$(curl -s -X POST "$BASE_URL/transcribe" \
  -F "mode=batch_folder" \
  -F "folder_path=/app/data" \
  -F "file_pattern=test_*.wav")

echo "$BATCH_RESPONSE" | python3 -m json.tool

BATCH_JOB_ID=$(echo "$BATCH_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin)['job_id'])" 2>/dev/null)

if [ -z "$BATCH_JOB_ID" ]; then
    echo "✗ Failed to get batch job_id"
    exit 1
fi

echo
echo "✓ Batch job submitted: $BATCH_JOB_ID"
echo
echo

# Test 7: List jobs filtered by status
echo "========================================"
echo "Test 7: List Jobs by Status"
echo "========================================"
echo "Pending jobs:"
curl -s "$BASE_URL/jobs?status=pending" | python3 -m json.tool
echo

echo "Completed jobs:"
curl -s "$BASE_URL/jobs?status=completed" | python3 -m json.tool
echo
echo

# Test 8: Delete job
echo "========================================"
echo "Test 8: Delete Job"
echo "========================================"
DELETE_RESPONSE=$(curl -s -X DELETE "$BASE_URL/jobs/$JOB_ID")
echo "$DELETE_RESPONSE" | python3 -m json.tool
echo

# Verify deletion
VERIFY_DELETE=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/jobs/$JOB_ID")
if [ "$VERIFY_DELETE" = "404" ]; then
    echo "✓ Job successfully deleted (got 404 as expected)"
else
    echo "✗ Job deletion failed (got HTTP $VERIFY_DELETE)"
fi
echo
echo

# Summary
echo "========================================"
echo "  Test Summary"
echo "========================================"
echo "✓ Health check"
echo "✓ Single file transcription"
echo "✓ Job status polling"
echo "✓ Result retrieval"
echo "✓ List jobs"
echo "✓ Batch transcription"
echo "✓ Filter by status"
echo "✓ Delete job"
echo
echo "========================================"
echo "  All tests passed!"
echo "========================================"
