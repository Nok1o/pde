# Сопоставление full/layers

Отношения времени и памяти показаны только для полностью завершённых пар с одинаковыми значениями, остановкой и работой. RSS измерен на процесс со всеми повторами и JIT.

| Этап | Случай | Метод | Одинаковые результат/работа | Обе серии полны | Время layers/full | Буферы layers/full | RSS layers/full | Сокращено записей |
|---|---|---|---|---|---|---|---|---|
| history-n8 | point-4-layers-kind1 | lod | True | True | 0.89657 | 0.96952 | 1.0016 | 40 |
| history-n8 | point-4-layers-kind1 | block | True | True | 1.041 | 0.99482 | 1.0104 | 40 |
| history-n8 | point-4-layers-kind1 | mc | True | True | 0.99363 | 1.003 | 1.0027 | 0 |
| history-n8 | point-4-layers-kind1 | mc_control | True | True | 0.99541 | 1.0001 | 0.99967 | 0 |
| history-n8 | point-4-layers-kind1 | mc_grouped | True | True | 0.98607 | 1.0029 | 1.0051 | 0 |
| history-n8 | point-4-layers-kind1 | mc_control_grouped | True | True | 1.0272 | 1.0001 | 1.0038 | 0 |
| history-n8 | field-4-layers-kind1 | lod | True | True | 0.9862 | 0.97174 | 1.0009 | 40 |
| history-n8 | field-4-layers-kind1 | block | True | True | 0.99588 | 0.99489 | 1 | 40 |
| history-n8 | field-4-layers-kind1 | mc | True | True | 1.0059 | 1.0029 | 0.99685 | 0 |
| history-n8 | field-4-layers-kind1 | mc_control | True | True | 0.99596 | 1.0001 | 0.99816 | 0 |
| history-n8 | field-4-layers-kind1 | mc_grouped | True | True | 1.0055 | 1.0028 | 0.99642 | 0 |
| history-n8 | field-4-layers-kind1 | mc_control_grouped | True | True | 0.99669 | 1.0001 | 0.99721 | 0 |
| history-n16 | point-16-layers-all3 | lod | True | True | 0.99526 | 0.97257 | 0.99359 | 240 |
| history-n16 | point-16-layers-all3 | block | True | True | 0.96196 | 0.99832 | 1.0016 | 240 |
| history-n16 | point-16-layers-all3 | mc | True | False | — | — | — | 35 |
| history-n16 | point-16-layers-all3 | mc_control | False | False | — | — | — | 0 |
| history-n16 | point-16-layers-all3 | mc_grouped | True | False | — | — | — | 35 |
| history-n16 | point-16-layers-all3 | mc_control_grouped | False | False | — | — | — | 0 |
| history-n16 | field-16-layers-all3 | lod | True | True | 1.0816 | 0.97542 | 0.9983 | 240 |
| history-n16 | field-16-layers-all3 | block | True | True | 0.97122 | 0.99833 | 0.99379 | 240 |
| history-n16 | field-16-layers-all3 | mc | True | False | — | — | — | 35 |
| history-n16 | field-16-layers-all3 | mc_control | True | False | — | — | — | 0 |
| history-n16 | field-16-layers-all3 | mc_grouped | True | False | — | — | — | 35 |
| history-n16 | field-16-layers-all3 | mc_control_grouped | False | False | — | — | — | 0 |
| history-long | n4-layers200-point | lod | True | True | 1.02 | 0.5556 | 0.98944 | 1000 |
| history-long | n4-layers1000-field | lod | True | True | 1.0075 | 0.51873 | 0.98559 | 5000 |
| history-long | n8-layers200-field | lod | True | True | 0.99678 | 0.70695 | 1.003 | 1000 |
| history-long | n8-layers1000-point | lod | True | True | 1.0102 | 0.56291 | 0.99137 | 5000 |
