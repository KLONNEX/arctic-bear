# arctic-bear

## Запуск докера

Для создания докер-образа следует выполнить команду из корневой папки проекта:

```bash
docker build -t IMAGE_NAME .
```

- IMAGE_NAME - название докер образа (любое название, чтобы задать имя).

Для запуска докер контейнера следует выполнить команду:

```bash
# Для пользователей Linux
bash run_docker.sh IMAGE_NAME CONTAINER_NAME

# Для пользователей Windows
docker run -it --name CONTAINER_NAME -p 7777:7777 --runtime=nvidia --privileged=true IMAGE_NAME
```

- IMAGE_NAME - название докер образа, которое вы задали на предыдущем шаге.
- CONTAINER_NAME - название докер контейнера (любое название, чтобы задать имя).

## Запуск сервера

```bash
bash run_server.py
```

Сервер запустится на порте 7777, подключение к нему через localhost:7777