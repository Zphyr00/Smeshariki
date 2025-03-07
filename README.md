![Скриншот к проекту](https://github.com/Zphyr00/Smeshariki/blob/main/image_2025-03-06_12-19-26.png)

# Smeshariki Simulation

Проект **Smeshariki Simulation** представляет собой симуляцию взаимодействия персонажей, вдохновлённую популярным мультсериалом «Смешарики». Программа моделирует перемещения, диалоги, действия и поведение персонажей в динамической среде, используя современные возможности генерации текста через модели трансформеров и вычислительные возможности PyTorch. Дополнительно реализован графический интерфейс на PyQt5 для удобного наблюдения за симуляцией.

## Описание проекта

**Smeshariki Simulation** — это экспериментальный проект, демонстрирующий возможности моделирования поведения виртуальных персонажей в динамичной среде. Основные задачи проекта:
- **Симуляция персонажей:** Каждый герой обладает уникальной личностью, характеристиками и набором действий, что позволяет генерировать разнообразные диалоги и ситуации.
- **Генерация текста:** Используется модель на базе Transformers (в данном случае модель "ruadapt_qwen2.5_3B_ext_u48_instruct_v4") для генерации ответов и реплик персонажей.
- **Многопоточность и сервер инференса:** Реализован сервер для параллельной обработки запросов на генерацию текста/действия/перемещения, что позволяет не блокировать основной поток симуляции.
- **Графический интерфейс:** Приложение имеет интерфейс на PyQt5, позволяющий наблюдать за ходом симуляции.

## Особенности

- **Динамическая симуляция:** Персонажи перемещаются между локациями, общаются, выполняют действия и реагируют на изменения окружения.
- **Управление ресурсами:** В симуляции есть ресурсы (урожай, рыба, грибы, дрова), что влияет на поведение персонажей.
- **Генерация диалогов:** Диалоги и действия генерируются с учётом истории событий, что создаёт атмосферу.
- **Сохранение и загрузка состояния:** Возможность сохранять текущее состояние симуляции и затем его восстанавливать для продолжения наблюдения.
- **Поддержка сезонности:** Симуляция учитывает смену времён года, влияющую на поведение и действия персонажей.

## Требования

- Python
- GPU: Для ускорения работы модели рекомендуется использовать видеокарту с поддержкой CUDA.
- [PyTorch](https://pytorch.org/)
- [Transformers](https://huggingface.co/docs/transformers/)
- [PyQt5](https://pypi.org/project/PyQt5/)
- [ruadapt_qwen2.5_3B_ext_u48_instruct_v4](https://huggingface.co/RefalMachine/ruadapt_qwen2.5_3B_ext_u48_instruct_v4)

### Шаги установки

1. **Клонируйте репозиторий:**

   ```bash
   git clone https://github.com/Zphyr00/Smeshariki.git
   cd Smeshariki
   ```

2. **Создайте виртуальное окружение (опционально):**


   ```bash
   python -m venv venv
   source venv/bin/activate   # Для Unix-систем
   venv\Scripts\activate      # Для Windows
   ```

3. **Установите зависимости:**

   ```bash
   pip install -r requirements.txt
   ```

Убедитесь, что PyTorch распознаёт ваше устройство.

### Запуск проекта

Запустить симуляцию можно с помощью следующей команды:

   ```bash
   python smeshariki.py
   ```

После запуска откроется графический интерфейс, через который можно наблюдать за симуляцией, а также сохранять или загружать состояние.

### Использование и функционал
Основные возможности:

- Интерфейс PyQt5: Позволяет визуально отслеживать действия персонажей, текущие локации и события симуляции.
- Генерация реплик: При помощи модели Transformers генерируются уникальные диалоги персонажей, что делает общение более разнообразным.
- Симуляция событий: Каждый персонаж действует согласно своим характеристикам (возраст, пол, личностные особенности). Действия включают перемещения, взаимодействие с ресурсами и другими персонажами, а также выполнение индивидуальных сценариев (игры, занятия, лечение и т.д.).
- Управление ресурсами: Симуляция отслеживает состояние ресурсов в различных локациях, что влияет на действия персонажей (например, сбор урожая, доставка дров).
- Сохранение и загрузка состояния: Реализованы функции сохранения текущего состояния симуляции в файл и его последующей загрузки. Это позволяет приостановить симуляцию и продолжить её в дальнейшем.
- Многопоточность: Для генерации реплик и обработки запросов используется отдельный поток, что позволяет поддерживать плавное функционирование основного цикла симуляции.

### Архитектура и внутреннее устройство
Основные компоненты:
- Модуль генерации текста: Использует библиотеку Transformers и модель ruadapt_qwen2.5_3B_ext_u48_instruct_v4 для создания динамичных и контекстно-зависимых ответов.
- Логика симуляции: Включает обработку перемещений персонажей, изменение их состояний (энергия, голод, здоровье) и выполнение действий, соответствующих текущей локации и времени года.
- Обработка событий: Каждое действие персонажа сопровождается записью события в историю локации. Эти записи используются для формирования диалогов и реакции других персонажей.
- Графический интерфейс: Реализован с помощью PyQt5. Интерфейс предоставляет пользователю возможность наблюдать за событиями симуляции в реальном времени.
- Инференс сервер: Сервер инференса работает в отдельном потоке и обрабатывает запросы на генерацию текста, что позволяет не блокировать основной цикл симуляции.
