# Voice API Test Script
# PowerShell

$ApiBase = "http://localhost:8000"
$SampleUrl = "https://www.kozco.tv/tech/piano2.wav"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Voice API Local Test" -ForegroundColor Cyan
Write-Host "API: $ApiBase" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# 1. Health Check
Write-Host "[1/4] Health Check..." -ForegroundColor Yellow
try {
    $r = Invoke-WebRequest -Uri "$ApiBase/health" -UseBasicParsing | ConvertFrom-Json
    Write-Host "Status: $($r.status)" -ForegroundColor Green
    Write-Host "Model loaded: $($r.model_loaded)`n" -ForegroundColor Green
} catch {
    Write-Host "ERROR: API not responding" -ForegroundColor Red
    exit 1
}

# 2. Extract employee vector
Write-Host "[2/4] Extracting employee vector from $SampleUrl ..." -ForegroundColor Yellow
try {
    $body = @{ sample_url = $SampleUrl } | ConvertTo-Json
    $r = Invoke-WebRequest -Uri "$ApiBase/extract" -Method POST -ContentType "application/json" -Body $body | ConvertFrom-Json
    $embedding = $r.embedding
    Write-Host "Vector extracted! Length: $($embedding.Count)" -ForegroundColor Green
    Write-Host "Shape: $($r.embedding_shape | ConvertTo-Json)`n" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Extract failed" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}

# 3. Prepare identify request
Write-Host "[3/4] Prepare Identify Request" -ForegroundColor Yellow
Write-Host "NOTE: /identify requires a STEREO call recording (2 channels)." -ForegroundColor Magenta
Write-Host "The test URL above is mono. For a real test, replace `$CallUrl with your stereo WAV/MP3.`n" -ForegroundColor Magenta

$CallUrl = $SampleUrl  # <-- REPLACE THIS WITH YOUR STEREO CALL RECORDING URL

$CallbackUrl = "https://webhook.site/44d2c3fe-6ef2-4fe6-b2ab-8ea08fd27aef"
Write-Host "Using webhook URL: $CallbackUrl" -ForegroundColor Cyan

$identifyBody = @{
    call_id = "test-call-001"
    call_url = $CallUrl
    callback_url = $CallbackUrl
    employee_channel = 1  # 0=left (client), 1=right (employee)
    employee_vectors = @(
        @{
            id = "emp1"
            name = "Ivan Testov"
            embedding = $embedding
        }
    )
} | ConvertTo-Json -Depth 10

# 4. Send identify request
Write-Host "`n[4/4] Sending /identify request..." -ForegroundColor Yellow
try {
    $r = Invoke-WebRequest -Uri "$ApiBase/identify" -Method POST -ContentType "application/json" -Body $identifyBody | ConvertFrom-Json
    Write-Host "Request accepted!" -ForegroundColor Green
    Write-Host "Status: $($r.status)" -ForegroundColor Green
    Write-Host "Call ID: $($r.call_id)" -ForegroundColor Green
    Write-Host "Message: $($r.message)`n" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Identify request failed" -ForegroundColor Red
    Write-Host $_.Exception.Message -ForegroundColor Red
    exit 1
}

Write-Host "========================================" -ForegroundColor Green
Write-Host "DONE! Check your webhook.site page for the result." -ForegroundColor Green
Write-Host "It usually arrives in 5-30 seconds." -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
