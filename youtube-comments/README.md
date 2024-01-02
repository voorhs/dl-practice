# Clustering YouTube Comments

YouTube Data API usage for scrapping comments and later analytics.

Анализ комментариев с канала [SonixFan](https://www.youtube.com/c/SonixFan).

интерактивная визуализация:
- кластеризация комментариев (KMeans, DBScan)
- учесть в визуализации кластеров дату комментария, видео комментария и все такое

приложения:
- наиболее похожие комменты к данному
- файнтюн GPT-2

статистика:
- анализ сентиментов всех комментариев
- самое популярное слово
- число уникальных комментаторов
- кто написал больше всех комментариев
- аналитика по времени

**Nearest comments demo.**
Query: `sonic is for shadow`.

Feed:
```
['sonic x shadow',
 'Sonic and shadow',
 'Sonic shadow is the best character.',
 'Why did they make shadow better than sonic',
 'classic adventure shadow > generations modern sonic',
 'Sonic number 2 shadow number 1',
 'How come shadow has all of his attacks but not sonic',
 'Shadow might as well be the Doom Slayer of the Sonic universe.',
 'there not shadow in the classic sonic\n\nedit:i mean bruh its a fan game sorry 😁😁',
 'I like my shadow,sonic']
```
