# Predators & Preys
### Что нужно делать, чтобы пользоваться
Склонировать, затем в терминале в корне каталога выполнить `bash making.sh`. 

На `Linux` -- с помощью `gcc` скомпилируются динамические библиотеки `entity.so` и `game.so`.

На `macOS` -- с помощью `clang` библиотеки `entity.dylib` и `game.dylib`. 

Также должен запуститься `test_run.py` с 25 играми и простыми агентами. После этого `making.sh` можно удалить.

### В чем отличие от исходного
Два файла: `entity.py` и `game.py` заменяются соответствующими динамическими библиотеками. Каждая из них затем подгружается в `emb_game.py`, который таким образом предоставляет обертку над быстрыми шагами внутри среды. Среда на каждом шаге также возвращает reward для каждого объекта: минимальное расстояние до противника из другой группы домноженное на 0.1 + штраф за смерть/награда за убийство(для охотников и жертв это 10); для охотников эта величина берется с отрицательным знаком, а для мертвых жертв reward = 0. В остальном каждый из файлов-заменителей: `entity.c` и `game.c` строка-в-строку повторяет исходные `.py` файлы, что не должно вызывать вопросов о соответствии первоначальной среде.  
Исходный            |  Представленный здесь
:-------------------------:|:-------------------------:
![](https://github.com/rtyasdf/Predators-and-Preys/blob/v2/images/default_scheme.jpg)  |  ![](https://github.com/rtyasdf/Predators-and-Preys/blob/v2/images/new_scheme.jpg)

На левой картинке цвета пунктирных линий соответствуют цветам на гистограмме ниже.

### Это быстрее?
Если брать в качестве показателя среднее время одного шага внутри среды, то можно получить следующие результаты(смотрите не на конкретные значения, а на порядки отношений).
![](https://github.com/rtyasdf/Predators-and-Preys/blob/v2/images/hist10.png)

Здесь для построения гистограмм брались:
1. Для исходной среды 100 игр (синяя гистограмма)
2. Для среды из `baseline branch` также 100 игр (желтая гистограмма)
3. Для среды только c переписанным `entity.py` 250 игр (красная гистограмма)
4. Для среды здесь(с переписанным `entity.py` и `game.py`) 1000 игр (зеленая гистограмма)
