# Руководство по развёртыванию

## Содержание

- [Предварительные требования](#предварительные-требования)
- [Локальная разработка](#локальная-разработка)
- [Docker Compose](#docker-compose)
- [Production развёртывание](#production-развёртывание)
- [M Kubernetes](#kubernetes)
- [Мониторинг и логирование](#мониторинг-и-логирование)
- [Backup и восстановление](#backup-и-восстановление)
- [Troubleshooting](#troubleshooting)
- [Оффлайн-развёртывание](#оффлайн-развёртывание)
  - [Когда использовать оффлайн-режим](#когда-использовать-оффлайн-режим)
  - [Что переносится на сервер](#что-переносится-на-сервер)
  - [Предварительные требования](#предварительные-требования-1)
  - [1. Сборка образа на машине с интернетом](#1-сборка-образа-на-машине-с-интернетом)
  - [2. Экспорт образов в tar-архив](#2-экспорт-образов-в-tar-архив)
  - [3. Перенос на целевой сервер](#3-перенос-на-целевой-сервер)
  - [4. Импорт образов на целевом сервере](#4-импорт-образов-на-целевом-сервере)
  - [5. Настройка и запуск](#5-настройка-и-запуск)
  - [6. Проверка работоспособности](#6-проверка-работоспособности)
  - [7. Остановка и обновление](#7-остановка-и-обновление)
  - [Замена модели эмбеддингов](#замена-модели-эмбеддингов)
  - [Решение проблем при оффлайн-развёртывании](#решение-проблем-при-оффлайн-развёртывании)

---

## Предварительные требования

### Минимальные требования

| Ресурс | Development | Production (Small) | Production (Large) |
|--------|-------------|-------------------|-------------------|
| CPU | 2 cores | 4 cores | 8 cores |
| RAM | 4 GB | 8 GB | 16 GB |
| Disk | 10 GB | 50 GB | 200 GB |
| GPU | Опционально | NVIDIA GPU (4GB+) | NVIDIA GPU (8GB+) |

### Зависимости

- Docker Engine 20.10+
- Docker Compose v2+
- (Опционально) NVIDIA Container Toolkit для GPU

---

## Локальная разработка

### 1. Клонирование и настройка

```bash
git clone <repository-url>
cd rag

# Копирование конфигурации
cp .env.example .env
```

### 2. Настройка переменных окружения

Отредактируйте `.env`:

```env
# Минимальная конфигурация
QDRANT_URL=http://localhost:6333
RAG_SERVICE_API_KEY=dev-api-key-12345
EMBEDDING_MODEL=BAAI/bge-m3
USE_GPU=false
LOG_LEVEL=DEBUG
```

### 3. Запуск Qdrant

```bash
docker run -d --name qdrant-dev \
  -p 6333:6333 \
  -p 6334:6334 \
  -v ./qdrant_data:/qdrant/storage \
  qdrant/qdrant:latest
```

### 4. Запуск приложения

**Вариант A: С виртуальным окружением**

```bash
# Создание venv
python -m venv venv

# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

# Установка зависимостей
pip install -r requirements.txt

# Запуск
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Вариант B: Docker (CPU)**

```bash
docker build -t rag-service:dev-cpu -f Dockerfile.cpu .
docker run -d --name rag-dev \
  --env-file .env \
  -p 8000:8000 \
  -v ./uploads:/app/uploads \
  rag-service:dev-cpu
```

### 5. Проверка

```bash
# Health check
curl http://localhost:8000/health

# Swagger UI
open http://localhost:8000/docs
```

---

## Docker Compose

### Development

```bash
docker-compose up -d
```

### Production

```bash
# Сборка
docker-compose build

# Запуск
docker-compose up -d

# Проверка
docker-compose ps
docker-compose logs -f app
```

### GPU развёртывание

Раскомментируйте секцию `deploy` в `docker-compose.yml`:

```yaml
services:
  app:
    # ...
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

Запуск с GPU:

```bash
docker-compose up -d
```

Проверка использования GPU:

```bash
docker exec -it rag-app-1 nvidia-smi
```

---

## Production развёртывание

### 1. Подготовка сервера

#### Ubuntu/Debian

```bash
# Обновление системы
sudo apt update && sudo apt upgrade -y

# Установка Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Установка Docker Compose
sudo apt install docker-compose-plugin -y

# (Опционально) Установка NVIDIA Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update
sudo apt install nvidia-container-toolkit -y
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 2. Настройка приложения

```bash
# Клонирование репозитория
git clone <repository-url>
cd rag

# Создание .env файла
cat > .env << EOF
# Production конфигурация
QDRANT_URL=http://qdrant:6333
QDRANT_API_KEY=<your-qdrant-api-key>
RAG_SERVICE_API_KEY=<your-secret-api-key>

# Модель
EMBEDDING_MODEL=BAAI/bge-m3
USE_GPU=true

# Оптимизация
LOG_LEVEL=WARNING
MCP_AUTH_ENABLED=true

# Service
SERVICE_HOST=0.0.0.0
SERVICE_PORT=8000
EOF
```

### 3. Настройка firewall

```bash
# Ubuntu (UFW)
sudo ufw allow 8000/tcp  # RAG Service
sudo ufw allow 6333/tcp  # Qdrant (если нужен доступ)
sudo ufw enable
```

### 4. Развёртывание

```bash
# Сборка образов
docker-compose build

# Запуск сервисов
docker-compose up -d

# Проверка
docker-compose ps
```

### 5. Настройка reverse proxy (Nginx)

```nginx
# /etc/nginx/sites-available/rag-service
server {
    listen 80;
    server_name rag.example.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
    }

    # Лимит размера для загрузки файлов
    client_max_body_size 100M;
}
```

```bash
# Включение сайта
sudo ln -s /etc/nginx/sites-available/rag-service /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### 6. SSL (Let's Encrypt)

```bash
# Установка Certbot
sudo apt install certbot python3-certbot-nginx -y

# Получение сертификата
sudo certbot --nginx -d rag.example.com

# Автоматическое обновление
sudo crontab -e
# Добавить: 0 0 * * * certbot renew --quiet
```

---

## Kubernetes

### 1. Deployment Manifest

```yaml
# rag-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-service
  namespace: rag
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rag-service
  template:
    metadata:
      labels:
        app: rag-service
    spec:
      containers:
      - name: rag-service
        image: rag-service:latest
        ports:
        - containerPort: 8000
        env:
        - name: QDRANT_URL
          value: "http://qdrant-service.qdrant:6333"
        - name: RAG_SERVICE_API_KEY
          valueFrom:
            secretKeyRef:
              name: rag-secrets
              key: api-key
        resources:
          requests:
            memory: "4Gi"
            cpu: "2000m"
          limits:
            memory: "8Gi"
            cpu: "4000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: rag-service
  namespace: rag
spec:
  selector:
    app: rag-service
  ports:
  - port: 80
    targetPort: 8000
  type: ClusterIP
```

### 2. Qdrant StatefulSet

```yaml
# qdrant-statefulset.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: qdrant
  namespace: qdrant
spec:
  serviceName: qdrant
  replicas: 1
  selector:
    matchLabels:
      app: qdrant
  template:
    metadata:
      labels:
        app: qdrant
    spec:
      containers:
      - name: qdrant
        image: qdrant/qdrant:latest
        ports:
        - containerPort: 6333
        - containerPort: 6334
        volumeMounts:
        - name: qdrant-storage
          mountPath: /qdrant/storage
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
  volumeClaimTemplates:
  - metadata:
      name: qdrant-storage
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 50Gi
```

### 3. Ingress

```yaml
# rag-ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: rag-ingress
  namespace: rag
  annotations:
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/proxy-body-size: "100m"
spec:
  tls:
  - hosts:
    - rag.example.com
    secretName: rag-tls
  rules:
  - host: rag.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: rag-service
            port:
              number: 80
```

### 4. Apply

```bash
# Создание namespace
kubectl create namespace rag
kubectl create namespace qdrant

# Применение конфигураций
kubectl apply -f rag-secrets.yaml
kubectl apply -f qdrant-statefulset.yaml
kubectl apply -f rag-deployment.yaml
kubectl apply -f rag-ingress.yaml

# Проверка
kubectl get pods -n rag
kubectl get pods -n qdrant
```

---

## Мониторинг и логирование

### Логи

```bash
# Docker Compose логи
docker-compose logs -f app

# Kubernetes логи
kubectl logs -f deployment/rag-service -n rag

# Просмотр логов за последний час
docker-compose logs --since=1h app
```

### Metrics (опционально)

Добавьте Prometheus metrics endpoint:

```python
# app/metrics.py
from prometheus_fastapi_instrumentator import Instrumentator

# В app/main.py:
from app.metrics import Instrumentator
Instrumentator().instrument(app).expose(app, endpoint="/metrics")
```

### Health Check

```bash
# Проверка здоровья
curl http://localhost:8000/health

# Мониторинг через cron
while true; do
  curl -f http://localhost:8000/health || echo "Service is down!"
  sleep 30
done
```

---

## Backup и восстановление

### Backup Qdrant

```bash
# Остановка Qdrant
docker stop qdrant

# Создание бэкапа
tar -czf qdrant_backup_$(date +%Y%m%d).tar.gz qdrant_data/

# Восстановление
tar -xzf qdrant_backup_YYYYMMDD.tar.gz
docker start qdrant
```

### Backup документов

```bash
# Backup uploads директории
tar -czf uploads_backup_$(date +%Y%m%d).tar.gz uploads/
```

### Automated Backup (cron)

```bash
# Добавить в crontab
0 2 * * * /opt/rag/backup.sh

# backup.sh
#!/bin/bash
BACKUP_DIR="/opt/backups"
DATE=$(date +%Y%m%d)

# Backup Qdrant
docker exec qdrant tar czf /tmp/qdrant_backup_${DATE}.tar.gz /qdrant/storage
docker cp qdrant:/tmp/qdrant_backup_${DATE}.tar.gz ${BACKUP_DIR}/

# Backup uploads
tar -czf ${BACKUP_DIR}/uploads_${DATE}.tar.gz uploads/

# Удаление старых бэкапов (30 дней)
find ${BACKUP_DIR} -name "*.tar.gz" -mtime +30 -delete
```

---

## Troubleshooting

###常见问题

#### 1. Service не запускается

```bash
# Проверка логов
docker-compose logs app

# Проверка зависимостей
docker-compose logs qdrant

# Проверка порта
netstat -tulpn | grep 8000
```

#### 2. Ошибка подключения к Qdrant

```bash
# Проверка доступности
curl http://localhost:6333/health

# Проверка Docker сети
docker network ls
docker network inspect rag_dlds

# Перезапуск Qdrant
docker-compose restart qdrant
```

#### 3. GPU не используется

```bash
# Проверка NVIDIA Docker
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

# Проверка переменных окружения
docker exec -it rag-app-1 env | grep USE_GPU

# Проверка логов
docker-compose logs app | grep -i gpu
```

#### 4. Ошибка "Collection not found"

```bash
# Создание коллекции через API
curl -X POST http://localhost:8000/api/admin/collections \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"action": "create", "collection_name": "documents"}'
```

#### 5. Высокое потребление памяти

```bash
# Уменьшение batch size
UPLOAD_BATCH_SIZE=10  # вместо 32

# Уменьшение chunk size
CHUNK_SIZE=512  # вместо 1024

# Перезапуск
docker-compose restart app
```

#### 6. Ошибка загрузки файлов

```bash
# Проверка размера файлов
ls -lh uploads/

# Проверка прав доступа
ls -ld uploads/
chmod 755 uploads/

# Проверка диска
df -h
```

### Получение диагностической информации

```bash
# Системная информация
docker version
docker-compose version
nvidia-smi  # если используется GPU

# Состояние сервисов
docker-compose ps

# Использование ресурсов
docker stats

# Сеть
docker network inspect rag_dlds
```

---

## Чеклист перед production

- [ ] Установлены все переменные окружения в `.env`
- [ ] `RAG_SERVICE_API_KEY` установлен и безопасен
- [ ] `QDRANT_API_KEY` установлен (если требуется)
- [ ] `USE_GPU=true` (если используется GPU)
- [ ] `LOG_LEVEL=WARNING` или `ERROR`
- [ ] `MCP_AUTH_ENABLED=true`
- [ ] Настроен reverse proxy (Nginx)
- [ ] Настроены SSL сертификаты
- [ ] Настроен firewall
- [ ] Настроены backup скрипты
- [ ] Настроен мониторинг
- [ ] Проведено нагрузочное тестирование
- [ ] Документация обновлена
- [ ] Доступы переданы команде поддержки

---

## Оффлайн-развёртывание

Этот раздел описывает создание оффлайн-дистрибутива и развёртывание сервиса на машинах
без доступа к интернету (air-gapped environments).

### Когда использовать оффлайн-режим

- Серверы без доступа к интернету
- Изолированные корпоративные сети с требованиями безопасности
- Среды, где запрещено скачивание образов и моделей извне

### Что переносится на сервер

Код приложения, модель эмбеддингов `BAAI/bge-m3` и модели Docling (layout, TableFormer)
запечены внутрь образа, поэтому клонировать репозиторий на сервере не нужно. Достаточно
трёх файлов:

| Файл | Назначение |
|------|-----------|
| `images.tar.gz` | образы `dds-app` и `qdrant/qdrant` |
| `docker-compose.offline.yml` | запуск без сборки и без bind-mount'ов кода |
| `.env` | обязателен: указан в `env_file`, без него Compose не стартует |

`docker-compose.offline.yml` намеренно сделан самодостаточным, а не override-файлом: при
слиянии нескольких `-f` Compose **складывает** списки `volumes`, поэтому убрать наслоением
bind-mount'ы `./app` и `./model_cache` из основного `docker-compose.yml` невозможно — они
перекрыли бы запечённые в образ код и модели.

### Предварительные требования

- Машина с доступом к интернету и Docker с BuildKit (используется `--mount=type=cache`)
- Совпадающая архитектура сборочной и целевой машины (обычно `linux/amd64`)
- Целевая машина: Docker Engine 20.10+ и Docker Compose v2+
- Свободное место на сервере: примерно двойной размер образа (архив + распакованные слои)

---

### 1. Сборка образа на машине с интернетом

На этом шаге скачиваются `BAAI/bge-m3` (стадия `model-downloader`) и модели Docling
(стадия `docling-models`). В финальном образе выставлен `HF_HUB_OFFLINE=1`, поэтому
дозагрузить что-либо в рантайме уже нельзя — всё должно попасть в образ при сборке.

#### Вариант A: CPU-only (рекомендуется)

```bash
cd /path/to/rag
docker build -f Dockerfile.cpu -t dds-app:latest .
```

#### Вариант B: основной `Dockerfile`

```bash
docker build -t dds-app:latest .
```

Разница только в зависимостях: `requirements.txt` тянет `qdrant-client[fastembed-gpu]`
(onnxruntime с CUDA), а `requirements-cpu.txt` — обычный `qdrant-client`. **PyTorch в обоих
файлах зафиксирован как `torch==2.6.0+cpu`**, то есть модель эмбеддингов считается на CPU в
любом случае, и NVIDIA Container Toolkit на сервере не нужен. Для настоящего GPU-инференса
понадобится отдельная сборка с CUDA-версией torch — в проекте её сейчас нет.

> Тег `dds-app:latest` прописан в `docker-compose.offline.yml`. Если собрали под другим
> именем, переименуйте образ перед экспортом:
> ```bash
> docker tag dds-app-cpu:latest dds-app:latest
> ```

### 2. Экспорт образов в tar-архив

Оба образа удобно сохранить в один архив:

```bash
mkdir -p distrib/docker-images

docker pull qdrant/qdrant:latest
docker save -o distrib/docker-images/images.tar dds-app:latest qdrant/qdrant:latest
gzip distrib/docker-images/images.tar
```

#### Размеры

Измеряйте, а не оценивайте — размер зависит от варианта сборки:

```bash
docker images dds-app:latest qdrant/qdrant:latest
ls -lh distrib/docker-images/
```

Основной вклад дают модели: кэш `bge-m3` ~4.3 ГБ (Hugging Face скачивает и
`pytorch_model.bin`, и `model.safetensors`), модели Docling ~0.7 ГБ, PyTorch CPU с
зависимостями ~2 ГБ. Сжатие gzip обычно уменьшает архив на 30–50 %.

> Не перенаправляйте `docker save` в stdout из PowerShell — бинарный поток портится.
> Используйте `-o`, а `gzip` запускайте отдельной командой.

### Замена модели эмбеддингов

Имя `BAAI/bge-m3` захардкожено в стадии `model-downloader` обоих Dockerfile'ов, поэтому
одной переменной в `.env` его не сменить. Нужны два согласованных изменения:

1. В `Dockerfile` (или `Dockerfile.cpu`) — строка `SentenceTransformer('BAAI/bge-m3', ...)`
2. В `.env` на сервере — `EMBEDDING_MODEL=<та же модель>`

Если значения разойдутся, приложение попытается скачать отсутствующую модель и упадёт:
в образе стоит `HF_HUB_OFFLINE=1`.

Отдельно учтите размерность вектора: у другой модели она почти наверняка другая, а коллекция
в Qdrant создаётся под конкретную размерность. После смены модели коллекции нужно пересоздать
и переиндексировать документы. Для русскоязычных документов также проверяйте, что новая
модель многоязычная — англоязычные модели заметно теряют качество.

### 3. Перенос на целевой сервер

Скопируйте архив и два текстовых файла любым доступным способом:

```bash
# Через scp
scp distrib/docker-images/images.tar.gz docker-compose.offline.yml .env user@server:/opt/rag/

# Через rsync
rsync -avz distrib/docker-images/images.tar.gz docker-compose.offline.yml .env user@server:/opt/rag/

# Через физический носитель (USB, NAS) — скопируйте те же три файла
```

### 4. Импорт образов на целевом сервере

```bash
cd /opt/rag

gunzip -c images.tar.gz | docker load
# если архив без сжатия:
# docker load -i images.tar

# Проверка загруженных образов
docker images | grep -E 'dds-app|qdrant'

# Ожидаемый вывод:
# dds-app        latest    <image-id>    <size>    <date>
# qdrant/qdrant  latest    <image-id>    <size>    <date>
```

Тег должен быть ровно `dds-app:latest` — именно его ищет `docker-compose.offline.yml`.

### 6. Настройка и запуск

Создайте файл `.env` в корневой директории проекта:

```bash
cat > .env << EOF
# Сервис
SERVICE_PORT=8000
SERVICE_HOST=0.0.0.0

# Qdrant
QDRANT_URL=http://qdrant:6333
QDRANT_API_KEY=

# Модель
EMBEDDING_MODEL=BAAI/bge-m3

# Аутентификация
RAG_SERVICE_API_KEY=<ваш-секретный-api-key>

# Логирование
LOG_LEVEL=INFO

# MCP (если нужен)
MCP_AUTH_ENABLED=true
EOF
```

Запуск через offline-компоу:

```bash
# Запуск сервисов
docker compose -f docker-compose.offline.yml up -d

# Проверка статуса
docker compose ps

# Ожидаемый вывод:
# NAME                STATUS
# rag-qdrant-1        Up (healthy)
# rag-app-1           Up (healthy)

# Проверка здоровья приложения
curl http://localhost:8000/health

# Ожидаемый ответ:
# {"status": "healthy"}

# Просмотр логов
docker compose logs -f app
```

### 7. Проверка работоспособности

```bash
# Health check
curl http://localhost:8000/health

# Swagger UI (в браузере)
# http://localhost:8000/docs

# Загрузка тестового документа (опционально)
curl -X POST http://localhost:8000/api/documents/upload \
  -H "Authorization: Bearer <ваш-api-key>" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_document.pdf"

# Поиск по документам
curl -X POST http://localhost:8000/api/documents/search \
  -H "Authorization: Bearer <ваш-api-key>" \
  -H "Content-Type: application/json" \
  -d '{"query": "тестовый запрос", "top_k": 5}'
```

### 8. Остановка и обновление

```bash
# Остановка
docker compose -f docker-compose.offline.yml down

# Остановка с очисткой volumes
docker compose -f docker-compose.offline.yml down -v

# Обновление (повторная загрузка нового tar)
docker load -i new-dds-app-latest.tar
docker compose -f docker-compose.offline.yml up -d --no-deps app
```

### Решение проблем при оффлайн-развёртывании

#### Проблема: Ошибка «image not found»

```bash
# Проверка, загружен ли образ
docker images | grep dds-app

# Если образа нет — повторите docker load
docker load -i distrib/docker-images/dds-app-cpu_latest.tar
```

#### Проблема: Приложение не запускается

```bash
# Проверка логов
docker compose -f docker-compose.offline.yml logs app

# Проверка сети
docker network ls
docker network inspect rag_dds_network

# Проверка портов
netstat -tulpn | grep 8000
```

#### Проблема: Qdrant не подключается

```bash
# Проверка Qdrant
curl http://localhost:6333/health

# Перезапуск Qdrant
docker compose -f docker-compose.offline.yml restart qdrant
```

#### Проблема: Нехватка места на диске

```bash
# Проверка используемого места
docker system df

# Очистка неиспользуемых образов
docker image prune -a

# Проверка диска
df -h
```

---

## Контакты поддержки

[Добавить контактную информацию команды поддержки]
