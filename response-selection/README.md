# Dialogue Graph for Multimodal dataset

Построение мультимодального диалогового графа для датасета [IMAD](https://arxiv.org/abs/2305.10512) на основе метода [DGAC](https://link.springer.com/chapter/10.1007/978-3-031-19032-2_52).

## Построение чатбота

Построение с нуля:
1. Создать папку `persona_chat_stuff` или `imad_stuff`
2. В случае IMAD дополнительно создать `imad_images/train` и `imad_images/test` с загруженными картинками
3. Запустить `bash pipeline.sh --task <name>`, где `<name>` --- это `imad` или `persona_chat`
4. API реализовано в виде класса `ChatBot` в файле `my_dff.py`

Посчитано и предобучено: [архив](https://drive.google.com/file/d/1XAdEtz7fdu2VE9u2c0fZO5bxr_17AI9c/view?usp=sharing)

## Процесс построения чатбота

`parse_dataset.py`: качает исходный датасет с hugging face и парсит в различные `.json`, которые используются на следующих этапах:
- `train_dialogues.json` --- извлечение диалогов в виде `list[list[str]]`, где `list[str]` --- отдельный диалог
- `train_rle.json` --- список числа реплик в каждом диалоге (типа run-length encoding всего датасета диалогов)
- `train_utterances.json` --- список всех реплик из датасета
- все то же самое для `test`, `val`

`sentence_encoding.py`: получает sentence-эмбеддинги всех реплик в датасете
- `train_embeddings.npy` --- эмбеддинги всех фраз из `train_utterances.json`
- `train_response_embeddings.npy` --- эмбеддинги фраз, которые будут использоваться ботом для ответов (в случае `persona_chat` они совпадают с `train_embeddings.py`)
- все то же самое для `test`, `val`

`dgac_clustering.py`: реализация статьи DGAC
- сохраняет модель в `clusterer.pickle`
- модель сохраняет полезные аттрибуты после обучения: w2v-эмбеддинги кластеров, центроиды кластеров, кластерные метки реплик из трейна

`linking_prediction.py`: обучает `CatBoostClassifier` для предсказания следующего кластера по контексту из последних реплик
- более детальное описание см. в докстринге функции `make_data()`

`response_selection.py`: обучает два проектора на задачу матчинга контекста и ответа
- сохраняет чекпоинты в `lightning_logs`, необходимо вручную выбрать один из четырех чекпоинтов `.ckpt`, перенести его в папку `*_stuff` и переименовать в `response_selector.ckpt`

## Пример работы с чатботом и пример инференса

```python
from dgac_clustering import Clusters
from my_dff import ChatBot


bot = ChatBot(task='persona_chat')
bot.send('Hello! Do yo like shopping?')
bot.respond()
```

Пример инференса можно посмотреть в файле `acc_of_response_selector.py`

## Метрики

Для `persona_chat`:
- linking prediction
    - не сохранил
Dialogue Systems
- response selection:
    - `acc@1 = 10.28%`
    - `acc@3 = 24.04%`
    - `acc@5 = 33.77%`
    - `acc@10 = 49.13%`
    - `acc@20 = 63.30%`
    - `acc@50 = 81.91%`

Для `imad`:
- linking prediction
    - `acc@1 = 11.82%`
    - `acc@3 = 30.40%`
    - `acc@5 = 43.93%`
    - `acc@10 = 60.90%`
- response selection
    - не считал

## Недостатки проекта

- Это бейзлайновое решение, наивное, простое, нехлопотное (сделано интерном меньше чем за неделю)  
- Скрипт `acc_of_response_selector.py` работает очень медленно (на Ryzen 7 5800H + Mobile RTX 3060 считает 15 минут для `perona_chat`)
- KMeans считается очень долго
- Нет некоторых метрики (см. выше)
- Код весьма прилизанный (на мой взгляд), но нет предела совершенству
- Не проанализирована работоспособность бота. Известно лишь то, что он плохо переходит от темы к теме.

## Идеи для улучшения

- Заменить KMeans из sklearn на аналог из [`cuML`](https://github.com/rapidsai/cuml), т.к. он на GPU (может, будет быстрее)

## Идеи для исследования

Используя данный пайплайн, сравнить мультимодальные кодировщики с текстовыми. Главный вопрос: насколько мультимодальные кодировщики уступают в text only диалогах, и насколько они выигрывают в мультимодальных диалогах.
