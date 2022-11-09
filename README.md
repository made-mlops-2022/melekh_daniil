# MADE MLOps

## Инструкции

Запуск скриптов выполнятся от корня проекта

Необходимо указать корректный PYTHONPATH

Windows:

    set PYTHONPATH=..\made_mlops

Linux:

    export PYTHONPATH=../made_mlops

### Обучение модели

#### Пример с логистической регрессией

    python ml_project/train_pipeline.py config/logreg_train_config.yaml
    
#### Пример с SVM

    python ml_project/train_pipeline.py config/svc_train_config.yaml

### Прогноз модели

#### Пример с логистической регрессией

    python ml_project/predict_pipeline.py config/logreg_predict_config.yaml

#### Пример с SVM

    python ml_project/predict_pipeline.py config/svc_predict_config.yaml

## Тесты

    export PYTHONPATH=../made_mlops
    pytest -v
