# Voice ID API — Stateless (векторы в Bubble)

## Архитектура
- **Bubble.io** — хранит компании, сотрудников, **векторы голоса** (192 float)
- **Railway** — только считает: извлекает векторы из семплов, сравнивает записи звонков
- **АТС Т2** — stereo-запись звонка (2 канала)

## Flow
1. **Один раз:** Bubble шлёт URL семпла → `/extract` → получает вектор `[0.12, -0.05, ...]` → сохраняет в поле сотрудника
2. **Каждый звонок:** Bubble шлёт векторы сотрудников + URL звонка → `/identify` → Railway callback с результатом

## Деплой на Railway

### 1. GitHub
```bash
git init
git add .
git commit -m "init"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/voice-api.git
git push -u origin main
```

### 2. Railway
New Project → Deploy from GitHub repo → выбери репозиторий.

### 3. Домен
Settings → Networking → Generate Domain.

## Эндпоинты

### POST /extract
Извлекает вектор из URL семпла. Bubble вызывает один раз.

**Request:**
```json
{"sample_url": "https://.../ivan_sample.wav"}
```

**Response:**
```json
{
  "status": "ok",
  "embedding": [0.1234, -0.0567, 0.0891, ...],  // 192 числа
  "embedding_shape": [192]
}
```

### POST /extract/file
То же самое, но загружаешь файл напрямую (multipart/form-data).

### POST /identify
Идентификация по готовым векторам из БД Bubble.

**Request:**
```json
{
  "call_id": "call_001",
  "call_url": "https://.../call_stereo.wav",
  "callback_url": "https://yourapp.bubbleapps.io/api/1.1/wf/voice-callback",
  "employee_vectors": [
    {"id": "emp_001", "name": "Иван Петров", "embedding": [0.12, -0.05, ...]},
    {"id": "emp_002", "name": "Мария Сидорова", "embedding": [0.08, 0.03, ...]}
  ],
  "employee_channel": 1
}
```

**Response (сразу):**
```json
{
  "status": "processing",
  "call_id": "call_001",
  "message": "Результат будет отправлен на callback_url после обработки"
}
```

**Callback (через 2–5 секунд):**
```json
{
  "call_id": "call_001",
  "result": {
    "identified_employee_id": "emp_001",
    "identified_employee_name": "Иван Петров",
    "confidence": 0.9234,
    "is_match": true,
    "threshold": 0.8,
    "employee_channel": 1,
    "top_scores": [
      {"employee_id": "emp_001", "employee_name": "Иван Петров", "score": 0.9234},
      {"employee_id": "emp_002", "employee_name": "Мария Сидорова", "score": 0.6123}
    ]
  },
  "processed_at": "2026-04-20T14:30:00+00:00"
}
```

## Настройка Bubble.io

### Тип данных Employee
Добавь поле:
- `Voice Embedding` (type: **List of numbers**) — сюда сохраняешь 192 float из `/extract`

### Workflow 1: Загрузка семпла
1. Пользователь загружает файл → FileUploader
2. Button "Сохранить семпл"
3. Step 1: API Connector `/extract` → `sample_url` = `FileUploader's URL`
4. Step 2: `Make changes to Employee` → `Voice Embedding` = `Result of Step 1's embedding` (List of numbers)

### Workflow 2: Идентификация звонка
1. Звонок поступил → запись сохранена
2. Backend WF: API Connector `/identify`
   - `call_id` = `This Call's unique id`
   - `call_url` = `This Call's Recording URL`
   - `callback_url` = `https://yourapp.bubbleapps.io/api/1.1/wf/voice-callback`
   - `employee_vectors` = `Search for Employees` → `:formatted as text`:
     - Content: `{"id":"This Employee's unique id","name":"This Employee's Name","embedding":This Employee's Voice Embedding}`
     - Delimiter: `,`
     - Обернуть в `[` `]` через Calculate formula
   - `employee_channel` = `1`

### Backend Workflow: voice-callback
Параметры: `call_id` (text), `result` (JSON)

Действия:
- `Make changes to Call` (where `unique id = call_id`)
  - `Identified Employee` = `result's identified_employee_name`
  - `Confidence` = `result's confidence`
  - `Voice Status` = `result's is_match` → "Определено" / "Не определено"

## Важно
- **Вектор = 192 float чисел.** В Bubble храни как `List of numbers` (не текст!)
- **employee_channel:** `0` = left (клиент), `1` = right (сотрудник)
- **Первый запрос** на Railway может занять 10–15 сек (загрузка модели SpeechBrain)
- **Порог уверенности:** `0.80`. Если `confidence < 0.80` → `is_match = false`

## Стек
- FastAPI + BackgroundTasks
- SpeechBrain (ECAPA-TDNN)
- PyTorch / torchaudio
- Docker
- Railway
