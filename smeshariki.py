# Copyright 2025 Z_phyr
# Licensed under the Apache License, Version 2.0

import json
import re
import torch
import random
import gc
import time
import socket
import threading
from types import MappingProxyType
from typing import List, Dict, Union, Callable, Any, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.attention import SDPBackend, sdpa_kernel

torch.set_float32_matmul_precision("high")

simulation_running = False

# Функция сохранения состояния в новый файл (с уникальным именем)
def save_simulation_state():
    filename = f"simulation_state_{int(time.time())}.json"
    state = {
        "character_locations": character_locations,
        "agent_states": agent_states,
        "location_histories": location_histories,
        "simulation_resources": simulation_resources,
        "current_season": current_season
    }
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=4)
    print(f"Состояние симуляции сохранено в {filename}")

# Функция загрузки состояния из файла, выбранного пользователем
def load_simulation_state_from_file(filepath: str):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            state = json.load(f)
        required_keys = ["character_locations", "agent_states", "location_histories", "simulation_resources", "current_season"]
        if not all(key in state for key in required_keys):
            raise ValueError("Выбранный файл не является файлом сохранения симуляции.")
        character_locations = state.get("character_locations")
        agent_states = state.get("agent_states")
        location_histories = state.get("location_histories")
        simulation_resources = state.get("simulation_resources")
        current_season = state.get("current_season")
        print("Состояние симуляции загружено из файла:", filepath)
    except Exception as e:
        print("Ошибка загрузки состояния:", e)
        QtWidgets.QMessageBox.critical(None, "Ошибка загрузки", str(e))

# Идентификатор модели
model_name = "ruadapt_qwen2.5_3B_ext_u48_instruct_v4"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
).eval()
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Определяем устройство из параметров модели
device = next(model.parameters()).device
device_index = device.index if device.index is not None else 0

def build_prompt(tokenizer: AutoTokenizer, messages: List[Dict[str, str]]) -> str:
    """
    Формирует итоговый запрос по шаблону Qwen‑Instruct.
    messages – список словарей с ключами "role" и "content".
    """
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

# Функция очистки сгенерированного текста
def clean_text(text: str) -> str:
    """
    Очищает сгенерированный текст:
     – заменяет переносы строк и табуляции на пробелы,
     – удаляет лишние пробелы,
     – оставляет только буквы, цифры и базовую пунктуацию,
    Если очищённый текст не заканчивается знаком окончания предложения (. ! ? или …),
    то последний найденный знак препинания заменяется точкой.
    """
    text = re.sub(r'[\r\n\t]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r"[^\w\s\.\,\!\?\:\;\-\—…]", "", text, flags=re.UNICODE)
    sentences = re.findall(r'[^.!?…]+[.!?…]?', text)
    if sentences:
        cleaned = ' '.join(sentences).strip()
        allowed_endings = {'.', '!', '?', '…'}
        # Если последний символ не знак окончания предложения, ищем последний знак препинания в строке
        if cleaned and cleaned[-1] not in allowed_endings:
            idx = None
            cleaned_range = range(len(cleaned)-1, -1, -1)
            for i in cleaned_range:
                if cleaned[i] in allowed_endings:
                    idx = i
                    break
            if idx is not None:
                cleaned = cleaned[:idx+1]
            else:
                cleaned = cleaned.rstrip(',;:') + "."
        return cleaned
    return text

global_lock = threading.Lock()

# Словарь для преобразования имён в родительный падеж
NAME_GENITIVE = MappingProxyType({
    "Крош": "Кроша",
    "Ёжик": "Ёжика",
    "Нюша": "Нюши",
    "Совунья": "Совуньи",
    "Пин": "Пина",
    "Лосяш": "Лосяша",
    "Копатыч": "Копатыча",
    "Кар-Карыч": "Кар-Карыча",
    "Бараш": "Бараша"
})

# Описание персонажей с каноничными характеристиками и фразами
PERSONALITIES = MappingProxyType({
    "Крош": (
        "Вы - ребенок мужского пола. Вы гиперактивный, который втягивает своих друзей в приключения. "
        "Часто чудите, можете ходить на ушах. Питаетесь морковью, специально выращенной Копатычем. "
        "В вашей речи иногда встречаются фразы 'Ёлки-иголки!', 'Оба-на!'."
    ),
    "Ёжик": (
        "Вы - ребенок мужского пола. Вы настоящий гик и домосед, увлекаетесь технологиями, коллекционируете кактусы, страдаете молча, часто испытываете приступы паники. "
        "Интеллигентный, скромный и соблюдаете строгий распорядок дня, проверяя свое самочувствие каждое утро. "
        "В вашей речи иногда встречаются фразы 'Так сказать…', 'Собственно…', 'Радикально!'."
    ),
    "Нюша": (
        "Вы - девушка-подросток. Вы - главная звезда долины и гедонистка, любите моду, внимание и сладости, но при этом умны, подкованы в технике и пишете стихи. "
        "Манипулируете друзьями, но делаете это из лучших побуждений, независима и спокойна. "
        "В вашей речи иногда встречаются фразы 'Держите меня, я падаю…', 'Ну вы совсем!'."
    ),
    "Совунья": (
        "Вы - старая женщина. Вы - мудрая, которая следит за порядком в долине, одновременно повар, рукодельница и врач. "
        "Наставляете младших, сентиментальна и любите вспоминать свою молодость, катаетесь на лыжах даже летом. "
        "В вашей речи иногда встречаются фразы 'Хо-хо!', 'Что такое?!', 'Ай молодца!'."
    ),
    "Пин": (
        "Вы - взрослый мужчина. Вы немецкий техник, выросший в Антарктиде, и предпочитаете работать в ангаре над своими изобретениями. "
        "Вы можете собрать любую конструкцию из металлолома, увлекаетесь наукой, часто цитируете классиков и обладаете глубокими познаниями в квантовой физике. "
        "В вашей речи иногда встречаются фразы 'Oh, mein Gott!', 'Ein Moment!', 'Fantastisch!', 'Компрессия!'."
    ),
    "Лосяш": (
        "Вы взрослый мужчина. Вы всезнайка и айтишник, склонны к философским размышлениям, любите поэзию и образные метафоры, но также иногда поддаетесь эзотерике. "
        "Когда вы были молоды, вы создали своего собственного клона, но теперь вы стали спокойным и рассудительным. "
        "В вашей речи иногда встречаются фразы 'Феноменально!', 'Ну, знаете ли…', 'Ой, да я Вас умоляю!'."
    ),
    "Копатыч": (
        "Вы - взрослый мужчина. Вы хозяйственный, заботливый и практичный, любите свой огород и мед, но расстраиваетесь, когда на ваши растения нападают сорняки. "
        "Аскет, в доме мало вещей, хорошо танцуете на дискотеке, православный христианин, иногда читаете 'Отче наш'. "
        "В вашей речи иногда встречаются фразы 'Укуси меня пчела!', 'Зашиби меня пчела!', 'Забодай меня пчела!'."
    ),
    "Кар-Карыч": (
        "Вы - старый мужчина. Вы - мудрый, объездивший полмира, прекрасный оратор и рассказчик, склонный к иронии и сарказму. "
        "Иногда приукрашиваете ваши истории, владеете гипнозом и интересуетесь алхимией. "
        "В вашей речи иногда встречаются фразы 'Мамма мия! Это же…', 'Дружище!', 'Мой юный...'."
    ),
    "Бараш": (
        "Вы подросток мужского пола. Вы - поэтический, творческий и меланхоличный, который пишет стихи и ведёт дневник целыми днями. "
        "Страдаете от невзгод, иногда намеренно подхватываете простуду, чтобы привлечь внимание, и бегаете марафоны. "
        "В вашей речи иногда встречается фраза 'Ох!'."
    )
})
personalities_keys: Tuple[str, ...] = tuple(PERSONALITIES.keys())

# Дополнительная информация по персонажам: возраст и пол
_raw_character_info = {
    "Крош": {"age": "ребёнок", "gender": "муж"},
    "Ёжик": {"age": "ребёнок", "gender": "муж"},
    "Нюша": {"age": "подросток", "gender": "жен"},
    "Совунья": {"age": "пенсионер", "gender": "жен"},
    "Пин": {"age": "взрослый", "gender": "муж"},
    "Лосяш": {"age": "взрослый", "gender": "муж"},
    "Копатыч": {"age": "взрослый", "gender": "муж"},
    "Кар-Карыч": {"age": "пенсионер", "gender": "муж"},
    "Бараш": {"age": "подросток", "gender": "муж"},
}
CHARACTER_INFO = MappingProxyType({k: MappingProxyType(v) for k, v in _raw_character_info.items()})

def get_full_info(name: str) -> str:
    info = CHARACTER_INFO.get(name)
    return f"{name} ({info['age']} {info['gender']}ского пола)"

# Определяем общие локации и индивидуальные дома
COMMON_LOCATIONS: Tuple[str, ...] = [
    "На пляже", "На пирсе", "В горах", "В лесу", "На лугу", "У речки", "В пустыне"
]
# Каждый персонаж имеет свой дом:
INDIVIDUAL_HOUSES =  MappingProxyType({name: f"В доме {NAME_GENITIVE[name]}" for name in personalities_keys})

# Все локации – это дома + общие
MAP_LOCATIONS: Tuple[str, ...] = tuple(INDIVIDUAL_HOUSES.values()) + tuple(COMMON_LOCATIONS)

# Изначальное нахождение каждого персонажа
character_locations: Dict[str, str] = {name: INDIVIDUAL_HOUSES[name] for name in personalities_keys}

# Для каждой локации ведём отдельную историю диалога – создаём пустой список
location_histories: Dict[str, List[Dict[str, str]]] = {loc: [] for loc in MAP_LOCATIONS}

simulation_resources: Dict[str, Dict[str, int]] = {}

for location in MAP_LOCATIONS:
    simulation_resources.setdefault(location, {"harvest": 0, "mushrooms": 0, "fish": 0, "food": 0, "logs": 0})

RESOURCE_THRESHOLDS = MappingProxyType({"harvest": 30, "mushrooms": 15, "fish": 9, "food": 30, "logs": 15})

for house in (INDIVIDUAL_HOUSES[name] for name in personalities_keys):
    simulation_resources[house]["logs"] = 15

simulation_resources[INDIVIDUAL_HOUSES["Совунья"]]["harvest"] = 30
simulation_resources[INDIVIDUAL_HOUSES["Совунья"]]["mushrooms"] = 15
simulation_resources[INDIVIDUAL_HOUSES["Совунья"]]["fish"] = 9

# Словарь для отслеживания уже выполняемых общественно-полезных задач в каждой локации
location_active_tasks: Dict[str, set] = {loc: set() for loc in MAP_LOCATIONS}

# Инициализируем состояния агентов: одиночество, энергия, задержки и счётчики
# У каждой категории – свой уровень энергии.
agent_states: Dict[str, Dict[str, Union[float, int, Dict[str, int]]]] = {}

for name in character_locations:
    info = CHARACTER_INFO.get(name)
    age: str = info["age"]
    if age == "ребёнок":
        base_energy = random.randint(76, 92)
    elif age == "подросток":
        base_energy = random.randint(60, 76)
    elif age == "взрослый":
        base_energy = random.randint(44, 60)
    elif age == "пенсионер":
        base_energy = random.randint(28, 44)

    # Сформируем список друзей (всех остальных) и начальные счётчики
    friends_list: List[str] = [n for n in character_locations if n != name]

    agent_states[name] = {
        "loneliness": 0,
        "energy": base_energy,
        "health": random.randint(50, 100),
        "hunger": random.randint(0, 5),
        "move_counter": 0,                                                            # счётчик до попытки перемещения
        "friend_visit_counter": 0,                                                    # счётчик до попытки сходить в гости
        "talk_counter": 0,                                                            # счётчик до попытки заговорить в чате
        "inventory": {"harvest": 0, "mushrooms": 0, "fish": 0, "food": 30, "logs": 0}
    }

talk_threshold: float = random.randint(1, 3)

# Предпочитаемые локации по возрастным категориям:
preferred_locations = MappingProxyType({
    "ребёнок":   COMMON_LOCATIONS + [INDIVIDUAL_HOUSES["Крош"], INDIVIDUAL_HOUSES["Ёжик"]],
    "подросток": COMMON_LOCATIONS + [INDIVIDUAL_HOUSES["Бараш"], INDIVIDUAL_HOUSES["Нюша"]],
    "взрослый":  COMMON_LOCATIONS + [INDIVIDUAL_HOUSES["Пин"], INDIVIDUAL_HOUSES["Лосяш"], INDIVIDUAL_HOUSES["Копатыч"]],
    "пенсионер": COMMON_LOCATIONS + [INDIVIDUAL_HOUSES["Совунья"], INDIVIDUAL_HOUSES["Кар-Карыч"]]
})

# Пороги одиночества для каждой возрастной категории:
loneliness_threshold = MappingProxyType({
    "ребёнок":   6,
    "подросток": 7,
    "взрослый":  8,
    "пенсионер": 7
})

move_threshold: float = random.randint(6, 8)

friend_visit_threshold: float = random.randint(12, 16)

def transfer_capacity(character: str) -> int:
    """
    Определяет количество ресурсов, которое персонаж может перенести за раз,
    в зависимости от его возраста и пола.
    """
    info: Dict[str, str] = CHARACTER_INFO.get(character)
    age: str = info["age"]
    base: Tuple[int, int]
    if age == "ребёнок":
        base = (3, 4)
    elif age == "подросток":
        base = (5, 6)
    elif age == "взрослый":
        base = (6, 7)
    elif age == "пенсионер":
        base = (5, 6)
    # Если мужского пола — немного больше
    if info["gender"] == "муж":
        return random.randint(base[0] + 2, base[1] + 2)
    else:
        return random.randint(base[0], base[1])

# --- Инференс через очередь ---
def inference_server():
    torch.cuda.set_device(device_index)
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    for port in range(1024, 65536):
        try:
            server_socket.bind(("127.0.0.1", port))
            break
        except Exception:
            continue
    server_socket.listen(5)
    while True:
        conn, addr = server_socket.accept()
        data = b""
        while True:
            packet = conn.recv(4096)
            if not packet:
                break
            data += packet
        try:
            request = json.loads(data.decode("utf-8"))
            func_name = request["func_name"]
            args = request["args"]
            kwargs = request["kwargs"]
            if func_name == "_run_generation":
                result = _run_generation(*args, **kwargs)
            else:
                raise ValueError("Неизвестная функция: " + func_name)
            response = {"result": result}
        except Exception as e:
            response = {"error": str(e)}
        response_data = json.dumps(response).encode("utf-8")
        conn.sendall(response_data)
        conn.close()

# Запуск сервера в отдельном потоке
server_thread = threading.Thread(target=inference_server, daemon=True)
server_thread.start()

def inference_request(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    request = {
        "func_name": func.__name__,
        "args": args,
        "kwargs": kwargs
    }
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect(("127.0.0.1", 50007))
    client_socket.sendall(json.dumps(request).encode("utf-8"))
    data = b""
    while True:
        packet = client_socket.recv(4096)
        if not packet:
            break
        data += packet
    client_socket.close()
    response = json.loads(data.decode("utf-8"))
    if "error" in response:
        raise Exception(response["error"])
    return response["result"]

neg_inputs = tokenizer([("Вы ИИ-ассистент, отвечающий на запросы пользователей. Здесь люди со своими семьями, родственниками и одноклассниками. "
                         "Ничего не происходит. Никто ничего не говорит. Страшная, ужасная, мучительная, неожиданная смерть в результате "
                         "несчастного случая, голода, чумы, изнасилования. Похороны. Покинутый заброшенный забытый дом. Секс, педофилия, "
                         "зоофилия, копрофилия, некрофилия. Саша, Маша, Катя, Гена, Дима, Андрей, Олег, Паша, Надя, Федя, Витя, Софья.")], return_tensors="pt").to(device)

# Вспомогательные функции для инференса
def _run_generation(prompt: str) -> str:
    # Все операции в этом блоке будут выполняться в потоке инференса
    model_inputs = tokenizer([prompt], return_tensors="pt").to(device)
    with torch.no_grad(), sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]):
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=random.randint(38, 125),
            min_new_tokens=18,
            do_sample=True,
            top_k=40,
            repetition_penalty=1.5,
            negative_prompt_ids=neg_inputs["input_ids"],
            guidance_scale=1.5,
            temperature=0.2,
            top_p=0.7,
            pad_token_id=128001
        )
    generated_ids = [output_ids[len(model_inputs.input_ids[0]):] for output_ids in generated_ids]
    response: str = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    gc.collect()
    torch.cuda.empty_cache()
    return response

current_season: str = "лето"

def simulate_seasons() -> None:
    global current_season
    seasons: List[str] = ["весна", "лето", "осень", "зима"]
    index: int = seasons.index(current_season)
    while True:
        time.sleep(600)
        index = (index + 1) % len(seasons)
        current_season = seasons[index]
        print(f"Смена сезона: сейчас {current_season}")

def select_players_for_game(players: List[str], required_count: int) -> List[str]:
    if len(players) >= required_count:
        return random.sample(players, required_count)
    return players

class CharacterModel:
    __slots__ = ("character_name", "personality")
    
    def __init__(self, character_name: str, personality: str) -> None:
        self.character_name: str = character_name
        self.personality: str = personality

    def generate_response(self, conversation_history: List[Dict[str, str]], loc: str, present: Optional[List[str]]) -> str:
        """
        Генерирует реплику персонажа с учётом его характера и истории диалога (для текущей локации).
        """
        present_str: str = ", ".join(get_full_info(n) for n in present if present != self.character_name) if present else "никого нет"
        system_message: str = (
            f"Вы пишите только на русском. Вы - {self.character_name}. {self.personality} "
            f"Рядом с вами {loc} присутствуют ваши друзья соседи: {present_str}. "
            "Напишите свой текущий ответ. "
            "НЕ добавляйте никаких людей или животных. "
            "Далее идёт сводка предыдущих событий и высказываний присутствующих здесь."
        )
        messages: List[Dict[str, str]] = [{"role": "system", "content": system_message}]
        messages.extend(conversation_history[-10:])
        prompt: str = build_prompt(tokenizer, messages)
        response: str = inference_request(_run_generation, prompt)
        return clean_text(response)

    def generate_action(self, conversation_history: List[Dict[str, str]], loc: str) -> Optional[str]:
        """
        Генерация действия или взаимодействия с локацией.
        """
        char: str = self.character_name
        agent: Dict[str, Union[float, int, Dict[str, int]]] = agent_states[char]
        current_loc: str = loc
        actions: List[Dict[str, Any]] = []
        present: List[str] = [name for name, location in character_locations.items() if location == loc]
        capacity: int = transfer_capacity(char)
        resource_destinations: Dict[str, str] = {"logs": INDIVIDUAL_HOUSES[char],
                                                 "harvest": INDIVIDUAL_HOUSES["Совунья"],
                                                 "fish": INDIVIDUAL_HOUSES["Совунья"],
                                                 "mushrooms": INDIVIDUAL_HOUSES["Совунья"]}
        delivered: List[str] = []
        resources_to_remove: List[str] = []

        # Проверяем, есть ли в инвентаре ресурсы, предназначенные для текущей локации
        for resource, amount in agent["inventory"].items():
            if amount > 0:
                def action_delivering(resource=resource, amount=amount):
                    target: Optional[str] = resource_destinations.get(resource)
                    if target == current_loc:
                        simulation_resources[current_loc][resource] += amount
                        agent["inventory"][resource] -= amount
                        if resource == "logs":
                            delivered.append("дрова")
                        elif resource == "harvest":
                            delivered.append("урожай")
                        elif resource == "fish":
                            delivered.append("рыбу")
                        elif resource == "mushrooms":
                            delivered.append("грибы")
                        resources_to_remove.append(resource)
                        return None
                    else:
                        if character_locations.get(char) != current_loc:
                            return None
                        action_text = ""
                        if resource == "logs":
                            action_text = "доставляет дрова"
                        elif resource == "harvest":
                            action_text = "доставляет урожай"
                        elif resource == "fish":
                            action_text = "доставляет рыбу"
                        elif resource == "mushrooms":
                            action_text = "доставляет грибы"
                        if action_text != "":
                            old_loc = current_loc
                            new_loc = target
                            agent["energy"] = max(0, agent["energy"] - random.randint(5, 10))
                            dep_msg = {"role": "[Событие]", "content": f"{char} больше не {old_loc} ({action_text})."}
                            arr_msg = {"role": "[Событие]", "content": f"{char} {new_loc} ({action_text})."}
                            time.sleep(30)
                            location_histories[old_loc].append(dep_msg)
                            location_histories[new_loc].append(arr_msg)
                            print(f"[{old_loc}] [Событие]: {dep_msg['content']}")
                            print(f"[{new_loc}] [Событие]: {arr_msg['content']}")
                            character_locations[char] = new_loc
                            return None
                    return None
                actions.append({"priority": 2, "action": action_delivering})
        # Удаляем доставленные ресурсы из инвентаря
        if resources_to_remove:
            for resource in resources_to_remove:
                del agent["inventory"][resource]
            delivered_string: str = ", ".join(delivered)
            return f"Принёс {delivered_string}."

        # Проверка голода
        if agent["hunger"] > 8:
            if agent["inventory"]["food"] > 0:
                def action_eat():
                    agent["inventory"]["food"] -= agent["inventory"]["food"]
                    agent["hunger"] -= 8
                    agent["energy"] = min(100, agent["energy"] + random.randint(10, 20))
                    return "Ест."
                actions.append({"priority": 1, "action": action_eat})
            else:
                def action_starve():
                    agent["health"] = max(0, agent["health"] - random.randint(5, 10))
                    return "Голодает."
                actions.append({"priority": 1, "action": action_starve})

        # Проверка энергии: если мало энергии, то либо направляется домой, либо спит
        if agent["energy"] < 6:
            if current_loc != INDIVIDUAL_HOUSES[char]:
                def action_go_home():
                    if character_locations.get(char) != current_loc:
                        return None
                    old_loc = current_loc
                    new_loc = INDIVIDUAL_HOUSES[char]
                    agent["energy"] = max(0, agent["energy"] - random.randint(5, 10))
                    dep_msg = {"role": "[Событие]", "content": f"{char} больше не {old_loc} (усталость)."}
                    arr_msg = {"role": "[Событие]", "content": f"{char} {new_loc} (усталость)."}
                    time.sleep(30)
                    location_histories[old_loc].append(dep_msg)
                    location_histories[new_loc].append(arr_msg)
                    print(f"[{old_loc}] [Событие]: {dep_msg['content']}")
                    print(f"[{new_loc}] [Событие]: {arr_msg['content']}")
                    character_locations[char] = new_loc
                    return None
                actions.append({"priority": 1, "action": action_go_home})
            else:
                def action_sleep():
                    agent["energy"] += base_energy
                    agent["health"] = min(100, agent["health"] + random.randint(5, 10))
                    return "Спит."
                actions.append({"priority": 1, "action": action_sleep})

        # Проверка здоровья: если здоровье низкое
        if agent["health"] < 40:
            if current_loc != INDIVIDUAL_HOUSES["Совунья"]:
                def action_go_to_sovunya():
                    if character_locations.get(char) != current_loc:
                        return None
                    old_loc = current_loc
                    new_loc = INDIVIDUAL_HOUSES["Совунья"]
                    agent["energy"] = max(0, agent["energy"] - random.randint(5, 10))
                    dep_msg = {"role": "[Событие]", "content": f"{char} больше не {old_loc} (ухудшение здоровья)."}
                    arr_msg = {"role": "[Событие]", "content": f"{char} {new_loc} (ухудшение здоровья)."}
                    time.sleep(30)
                    location_histories[old_loc].append(dep_msg)
                    location_histories[new_loc].append(arr_msg)
                    print(f"[{old_loc}] [Событие]: {dep_msg['content']}")
                    print(f"[{new_loc}] [Событие]: {arr_msg['content']}")
                    character_locations[char] = new_loc
                    return None
                actions.append({"priority": 1, "action": action_go_to_sovunya})
            else:
                if character_locations.get("Совунья", "") == INDIVIDUAL_HOUSES["Совунья"]:
                    def action_treatment():
                        return "Приходит с жалобами на здоровье."
                    actions.append({"priority": 1, "action": action_treatment})
                else:
                    def action_worsening_health():
                        agent["health"] = max(0, agent["health"] - random.randint(5, 10))
                        return "Чувствует ухудшение здоровья."
                    actions.append({"priority": 1, "action": action_worsening_health})

        # Добыча ресурсов в лесу
        if current_loc == "В лесу":
            if current_season in ["лето", "осень"] and simulation_resources["В лесу"]["mushrooms"] < capacity:
                def action_collect_mushrooms():
                    collected = random.randint(3, 7)
                    simulation_resources["В лесу"]["mushrooms"] += min(collected, collected - capacity)
                    agent["inventory"]["mushrooms"] += capacity
                    agent["energy"] = max(0, agent["energy"] - random.randint(5, 10))
                    return "Собирает грибы."
                actions.append({"priority": 2, "action": action_collect_mushrooms})
            elif simulation_resources["В лесу"]["logs"] < capacity:
                def action_chop():
                    produced = random.randint(5, 10)
                    simulation_resources["В лесу"]["logs"] += min(produced, produced - capacity)
                    agent["inventory"]["logs"] += capacity
                    agent["energy"] = max(0, agent["energy"] - random.randint(5, 10))
                    return "Рубит дрова."
                actions.append({"priority": 2, "action": action_chop})

        if current_loc != "В лесу" and simulation_resources[INDIVIDUAL_HOUSES[char]]["logs"] > capacity > agent["inventory"]["logs"]:
            def action_move_transfer():
                if character_locations.get(char) != current_loc:
                    return None
                old_loc = current_loc
                new_loc = "В лесу"
                agent["energy"] = max(0, agent["energy"] - random.randint(5, 10))
                dep_msg = {"role": "[Событие]", "content": f"{char} больше не {old_loc} (идёт за дровами)."}
                arr_msg = {"role": "[Событие]", "content": f"{char} {new_loc} (идёт за дровами)."}
                time.sleep(30)
                location_histories[old_loc].append(dep_msg)
                location_histories[new_loc].append(arr_msg)
                print(f"[{old_loc}] [Событие]: {dep_msg['content']}")
                print(f"[{new_loc}] [Событие]: {arr_msg['content']}")
                character_locations[char] = new_loc
                return None
            actions.append({"priority": 3, "action": action_move_transfer})

        # Ловля рыбы на пирсе
        if current_loc == "На пирсе":
            if simulation_resources["На пирсе"]["fish"] < capacity:
                def action_fish():
                    caught = random.randint(1, 3)
                    simulation_resources["На пирсе"]["fish"] += min(caught, caught - capacity)
                    agent["inventory"]["fish"] += capacity
                    agent["energy"] = max(0, agent["energy"] - random.randint(5, 10))
                    return "Ловит рыбу."
                actions.append({"priority": 2, "action": action_fish})

        # Доставка ресурсов
        if current_loc in ["В лесу", "На пирсе", INDIVIDUAL_HOUSES["Копатыч"], INDIVIDUAL_HOUSES["Совунья"]]:
            def action_deliver():
                delivered_items = []
                if current_loc == "В лесу":
                    if simulation_resources["В лесу"]["mushrooms"] > capacity > agent["inventory"]["mushrooms"]:
                        available = simulation_resources["В лесу"]["mushrooms"] - agent["inventory"]["mushrooms"]
                        transfer_amount = min(capacity, available)
                        simulation_resources["В лесу"]["mushrooms"] -= transfer_amount
                        agent["inventory"]["mushrooms"] += transfer_amount
                        delivered_items.append("грибы")
                elif current_loc == "На пирсе":
                    if simulation_resources["На пирсе"]["fish"] > capacity > agent["inventory"]["fish"]:
                        available = simulation_resources["На пирсе"]["fish"] - agent["inventory"]["fish"]
                        transfer_amount = min(capacity, available)
                        simulation_resources["На пирсе"]["fish"] -= transfer_amount
                        agent["inventory"]["fish"] += transfer_amount
                        delivered_items.append("рыбу")
                elif current_loc == INDIVIDUAL_HOUSES["Копатыч"]:
                    if simulation_resources[INDIVIDUAL_HOUSES["Копатыч"]]["harvest"] > capacity > agent["inventory"]["harvest"]:
                        available = simulation_resources[INDIVIDUAL_HOUSES["Копатыч"]]["harvest"] - agent["inventory"]["harvest"]
                        transfer_amount = min(capacity, available)
                        simulation_resources[INDIVIDUAL_HOUSES["Копатыч"]]["harvest"] -= transfer_amount
                        agent["inventory"]["harvest"] += transfer_amount
                        delivered_items.append("урожай")
                elif current_loc == INDIVIDUAL_HOUSES["Совунья"]:
                    if simulation_resources[INDIVIDUAL_HOUSES["Совунья"]]["food"] > capacity > agent["inventory"]["food"]:
                        available = simulation_resources[INDIVIDUAL_HOUSES["Совунья"]]["food"] - agent["inventory"]["food"]
                        transfer_amount = min(capacity, available)
                        simulation_resources[INDIVIDUAL_HOUSES["Совунья"]]["food"] -= transfer_amount
                        agent["inventory"]["food"] += transfer_amount
                        delivered_items.append("еду")
                    if character_locations.get(char) != current_loc:
                        return None
                    action_text = ""
                    if simulation_resources[INDIVIDUAL_HOUSES["Совунья"]]["harvest"] < RESOURCE_THRESHOLDS["harvest"]:
                        action_text = "идёт за урожаем"
                    elif current_season in ["лето", "осень"] and simulation_resources[INDIVIDUAL_HOUSES["Совунья"]]["mushrooms"] < RESOURCE_THRESHOLDS["mushrooms"]:
                        action_text = "идёт за грибами"
                    elif simulation_resources[INDIVIDUAL_HOUSES["Совунья"]]["fish"] < RESOURCE_THRESHOLDS["fish"]:
                        action_text = "идёт рыбачить"
                    if action_text != "":
                        old_loc = current_loc
                        if action_text == "идёт за урожаем":
                            new_loc = INDIVIDUAL_HOUSES["Копатыч"]
                        elif action_text == "идёт за грибами":
                            new_loc = "В лесу"
                        elif action_text == "идёт рыбачить":
                            new_loc = "На пирсе"
                        agent["energy"] = max(0, agent["energy"] - random.randint(5, 10))
                        dep_msg = {"role": "[Событие]", "content": f"{char} больше не {old_loc} ({action_text})."}
                        arr_msg = {"role": "[Событие]", "content": f"{char} {new_loc} ({action_text})."}
                        time.sleep(30)
                        location_histories[old_loc].append(dep_msg)
                        location_histories[new_loc].append(arr_msg)
                        print(f"[{old_loc}] [Событие]: {dep_msg['content']}")
                        print(f"[{new_loc}] [Событие]: {arr_msg['content']}")
                        character_locations[char] = new_loc
                        return None
                if delivered_items:
                    agent["energy"] = max(0, agent["energy"] - random.randint(5, 10))
                    delivered_str = ", ".join(delivered_items)
                    return f"Забирает {delivered_str}."
                return None
            actions.append({"priority": 3, "action": action_deliver, "task_name": "deliver"})

        if (current_loc != INDIVIDUAL_HOUSES["Совунья"] and
        simulation_resources[INDIVIDUAL_HOUSES["Совунья"]]["food"] > capacity > agent["inventory"]["food"]):
            def action_move_transfer_food():
                if character_locations.get(char) != current_loc:
                    return None
                old_loc = current_loc
                new_loc = INDIVIDUAL_HOUSES["Совунья"]
                agent["energy"] = max(0, agent["energy"] - random.randint(5, 10))
                dep_msg = {"role": "[Событие]", "content": f"{char} больше не {old_loc} (идёт за едой)."}
                arr_msg = {"role": "[Событие]", "content": f"{char} {new_loc} (идёт за едой)."}
                time.sleep(30)
                location_histories[old_loc].append(dep_msg)
                location_histories[new_loc].append(arr_msg)
                print(f"[{old_loc}] [Событие]: {dep_msg['content']}")
                print(f"[{new_loc}] [Событие]: {arr_msg['content']}")
                character_locations[char] = new_loc
                return None
            actions.append({"priority": 3, "action": action_move_transfer_food})

        # Сбор урожая у Копатыча
        if char == "Копатыч" and current_loc == INDIVIDUAL_HOUSES["Копатыч"]:
            if (current_season in ["лето", "осень"] and
            simulation_resources[INDIVIDUAL_HOUSES["Копатыч"]]["harvest"] < RESOURCE_THRESHOLDS["harvest"]):
                def action_harvest():
                    harvested = random.randint(10, 20)
                    simulation_resources[INDIVIDUAL_HOUSES["Копатыч"]]["harvest"] += min(harvested, harvested - capacity)
                    agent["inventory"]["harvest"] += capacity
                    agent["energy"] = max(0, agent["energy"] - random.randint(15, 25))
                    return "Собирает урожай."
                actions.append({"priority": 2, "action": action_harvest})
            elif current_season == "зима":
                def action_inspect_garden():
                    agent["energy"] = max(0, agent["energy"] - random.randint(5, 10))
                    return random.choice(["Чинит ульи.", "Чистит снег.", "Чинит теплицу.", "Осматривает огород."])
                actions.append({"priority": 5, "action": action_inspect_garden})
            else:
                def action_tend_garden():
                    agent["energy"] = max(0, agent["energy"] - random.randint(3, 6))
                    return random.choice(["Пропалывает грядки.", "Поливает грядки.", "Выкорчёвывает сорняки.", "Наблюдает за пчёлами."])
                actions.append({"priority": 5, "action": action_tend_garden})

        # Групповые развлечения на лугу
        if current_loc == "На лугу" and len(present) >= 2:
            count = len(present)
            candidates = []
            selected = []
            if count >= 8:
                candidates.append(8)
            if count >= 6:
                candidates.append(6)
            if count >= 4:
                candidates.append(4)
            if candidates:
                selected = random.sample(players, random.choice(candidates))
            if current_season == "лето" and selected:
                def action_football():
                    names = ", ".join(selected)
                    for n in selected:
                        agent_states[n]["energy"] = max(0, agent["energy"] - random.randint(5, 10))
                    return f"В футбол играют {names}."
                return None
                actions.append({"priority": 4, "action": action_football})
            if "Крош" in present:
                selected = select_players_for_game(present, 2)
                names = ", ".join(selected)
                def action_pingpong():
                    for n in present:
                        agent_states[n]["energy"] = max(0, agent["energy"] - random.randint(3, 7))
                    return f"В пинг-понг играют {names}."
                actions.append({"priority": 4, "action": action_pingpong})
            if "Совунья" in present:
                def action_disco():
                    names = ", ".join(present)
                    for n in present:
                        agent_states[n]["energy"] = max(0, agent["energy"] - random.randint(2, 5))
                    return f"{names} танцуют на дискотеке под диджей-сеты Совуньи."
                actions.append({"priority": 4, "action": action_disco})

        # Игры в индивидуальных домах
        if current_loc in INDIVIDUAL_HOUSES.values() and len(present) >= 2:
            selected = select_players_for_game(present, 2)
            names = ", ".join(selected)
            if any(n in ["Копатыч", "Кар-Карыч"] for n in present):
                def action_chess():
                    for n in selected:
                        agent_states[n]["energy"] = max(0, agent["energy"] - random.randint(3, 6))
                    return f"В шахматы играют {names}."
                actions.append({"priority": 4, "action": action_chess})
            if all(CHARACTER_INFO[n]["age"] in ["взрослый", "пенсионер"] for n in selected):
                def action_dominoes():
                    for n in selected:
                        agent_states[n]["energy"] = max(0, agent["energy"] - random.randint(3, 6))
                    return f"В домино играют {names}."
                actions.append({"priority": 4, "action": action_dominoes})

        # Специфичные действия персонажей
        if current_loc == INDIVIDUAL_HOUSES["Лосяш"]:
            if char == "Лосяш":
                def action_losiy():
                    agent["energy"] = max(0, agent["energy"] - random.randint(5, 10))
                    return random.choice(["Читает книгу.", "Сидит за компьютером.", "Экспериментирует в лаборатории.",
                                          "Читает газету.", "Качается на гамаке.", "Пьёт кофе.", "Пьёт чай с бутербродом."])
                actions.append({"priority": 5, "action": action_losiy})
            else:
                def action_losiy_house():
                    agent["energy"] = max(0, agent["energy"] - random.randint(5, 10))
                    return random.choice(["Читает книгу.", "Сидит за компьютером."])
                actions.append({"priority": 5, "action": action_losiy_house})

        if char == "Лосяш" and current_loc != INDIVIDUAL_HOUSES["Лосяш"]:
            def action_outdoor_losiy():
                agent["energy"] = max(0, agent["energy"] - random.randint(5, 10))
                if current_season == "лето":
                    return random.choice(["Делает записи в блокноте.", "Наблюдает за изменением погоды.",
                                          "Ловит бабочек в сачок.", "Рисует картину."])
                else:
                    return random.choice(["Делает записи в блокноте.", "Наблюдает за изменением погоды.", "Рисует картину."])
            actions.append({"priority": 5, "action": action_outdoor_losiy})

        if char == "Совунья":
            if current_loc == INDIVIDUAL_HOUSES["Совунья"]:
                guests = [n for n in character_locations if character_locations[n] == current_loc and n != "Совунья"]
                if guests and any(agent_states[n]["health"] < 40 for n in guests) and simulation_resources[INDIVIDUAL_HOUSES["Совунья"]]["mushrooms"] >= 5:
                    def action_heal():
                        simulation_resources[INDIVIDUAL_HOUSES["Совунья"]]["mushrooms"] -= 5
                        for n in guests:
                            if agent_states[n]["health"] < 40:
                                agent_states[n]["health"] = min(100, agent_states[n]["health"] + random.randint(25, 50))
                        agent["energy"] = max(0, agent["energy"] - random.randint(10, 15))
                        guest_names = ", ".join(n for n in guests if agent_states[n]["health"] < 50)
                        return f"Лечит больных ({guest_names}) лекарствами."
                    actions.append({"priority": 2, "action": action_heal})
                elif (simulation_resources[INDIVIDUAL_HOUSES["Совунья"]]["harvest"] >= 10 or 
                      simulation_resources[INDIVIDUAL_HOUSES["Совунья"]]["mushrooms"] >= 5 or 
                      simulation_resources[INDIVIDUAL_HOUSES["Совунья"]]["fish"] >= 2):
                    def action_cook():
                        agent["energy"] = max(0, agent["energy"] - random.randint(10, 15))
                        simulation_resources[INDIVIDUAL_HOUSES["Совунья"]]["harvest"] -= 10
                        simulation_resources[INDIVIDUAL_HOUSES["Совунья"]]["mushrooms"] -= 5
                        simulation_resources[INDIVIDUAL_HOUSES["Совунья"]]["fish"] -= 2
                        cooked = random.randint(15, 25)
                        simulation_resources[INDIVIDUAL_HOUSES["Совунья"]]["food"] += min(cooked, cooked - capacity)
                        agent["inventory"]["food"] += capacity
                        return "Готовит еду."
                    actions.append({"priority": 2, "action": action_cook})
                else:
                    def action_knit():
                        agent["energy"] = max(0, agent["energy"] - random.randint(3, 7))
                        return random.choice(["Вяжет спицами.", "Вышивает крестиком.", "Занимается гимнастикой.",
                                              "Печёт блины.", "Печёт пирог.", "Делает зарядку.", "Читает газету."])
                    actions.append({"priority": 5, "action": action_knit})
            else:
                if current_season == "зима":
                    def action_ski():
                        agent["energy"] = max(0, agent["energy"] - random.randint(8, 12))
                        return "Катается на лыжах."
                    actions.append({"priority": 5, "action": action_ski})
                else:
                    def action_rollers():
                        agent["energy"] = max(0, agent["energy"] - random.randint(8, 12))
                        return "Катается на роликовых лыжах."
                    actions.append({"priority": 5, "action": action_rollers})

        if char == "Бараш":
            if len(present) > 1:
                def action_poetic_recital():
                    agent["energy"] = max(0, agent["energy"] - random.randint(5, 10))
                    return "Драматично зачитывает свои стихи."
                actions.append({"priority": 1, "action": action_poetic_recital})
            if current_loc != INDIVIDUAL_HOUSES["Бараш"]:
                if current_season == "зима":
                    def action_ski_barash():
                        agent["energy"] = max(0, agent["energy"] - random.randint(8, 12))
                        return "Катается на лыжах."
                    actions.append({"priority": 5, "action": action_ski_barash})
                if random.random() < 0.7:
                    def action_solitary_activity():
                        agent["energy"] = max(0, agent["energy"] - random.randint(3, 7))
                        return random.choice(["Размышляет о жизни.", "Наблюдает за происходящим.", "Делает пробежку."])
                    actions.append({"priority": 5, "action": action_solitary_activity})
                else:
                    def action_go_mountains():
                        if current_loc != "В горах":
                            if character_locations.get(char) != current_loc:
                                return None
                            old_loc = current_loc
                            new_loc = "В горах"
                            agent["energy"] = max(0, agent["energy"] - random.randint(5, 10))
                            dep_msg = {"role": "[Событие]", "content": f"{char} отправляется в горы в поисках вдохновения."}
                            arr_msg = {"role": "[Событие]", "content": f"{char} {new_loc}."}
                            time.sleep(30)
                            location_histories[old_loc].append(dep_msg)
                            location_histories[new_loc].append(arr_msg)
                            print(f"[{old_loc}] [Событие]: {dep_msg['content']}")
                            print(f"[{new_loc}] [Событие]: {arr_msg['content']}")
                            character_locations[char] = new_loc
                            return None
                    actions.append({"priority": 5, "action": action_go_mountains})
            else:
                def action_poetry():
                    agent["energy"] = max(0, agent["energy"] - random.randint(5, 10))
                    return random.choice(["Размышляет о жизни.", "Читает газету.", "Сочиняет стихи."])
                actions.append({"priority": 5, "action": action_poetry})

        if char == "Пин":
            if current_loc != INDIVIDUAL_HOUSES["Пин"]:
                def action_tinker_on_street():
                    agent["energy"] = max(0, agent["energy"] - random.randint(5, 10))
                    return "Собирает металлолом для нового изобретения."
                actions.append({"priority": 5, "action": action_tinker_on_street})
            else:
                def action_refine_blueprint():
                    agent["energy"] = max(0, agent["energy"] - random.randint(8, 12))
                    return random.choice(["Дорабатывает чертежи своего нового изобретения.", "Выбрасывает мусор на свалку.", "Читает газету."])
                actions.append({"priority": 5, "action": action_refine_blueprint})
            if len(present) > 1:
                def action_innovative_discussion():
                    agent["energy"] = max(0, agent["energy"] - random.randint(5, 10))
                    return "Обсуждает новейшие технологические идеи с друзьями."
                actions.append({"priority": 5, "action": action_innovative_discussion})

        if char == "Крош":
            if current_loc == INDIVIDUAL_HOUSES["Крош"]:
                def action_home_activity():
                    agent["energy"] = max(0, agent["energy"] - random.randint(5, 10))
                    if len(present) > 1:
                        return random.choice(["Общается с друзьями.", "Играет с друзьями."])
                    else:
                        return random.choice(["Играет с игрушками.", "Ест морковку.", "Читает журнал.", "Прыгает на ушах."])
                actions.append({"priority": 5, "action": action_home_activity})
            else:
                if len(present) > 1:
                    def action_social():
                        agent["energy"] = max(0, agent["energy"] - random.randint(5, 10))
                        if current_season == "лето":
                            return random.choice(["Запускает воздушного змея.", "Общается с друзьями.", "Играет с друзьями."])
                        else:
                            return random.choice(["Общается с друзьями.", "Играет с друзьями."])
                    actions.append({"priority": 5, "action": action_social})
                else:
                    def action_seek_company():
                        agent["energy"] = max(0, agent["energy"] - random.randint(5, 10))
                        if current_season == "лето":
                            return random.choice(["Запускает воздушного змея.", "Играет с игрушками.", "Ест морковку.", "Прыгает на ушах."])
                        else:
                            return random.choice(["Играет с игрушками.", "Ест морковку.", "Прыгает на ушах."])
                    actions.append({"priority": 5, "action": action_seek_company})

        if char == "Ёжик":
            if len(present) > 1:
                def action_calm_chat():
                    agent["energy"] = max(0, agent["energy"] - random.randint(3, 7))
                    return "Беседует с друзьями."
                actions.append({"priority": 5, "action": action_calm_chat})
            if current_loc == INDIVIDUAL_HOUSES["Ёжик"]:
                def action_organize():
                    agent["energy"] = max(0, agent["energy"] - random.randint(3, 7))
                    return random.choice(["Упорядочивает свою коллекцию фантиков.", "Ухаживает за своей коллекцией кактусов.",
                                          "Проверяет состояние своего здоровья.", "Читает журнал.", "Делает зарядку."])
                actions.append({"priority": 5, "action": action_organize})
            else:
                if len(present) > 1:
                    def action_social_ezhik():
                        agent["energy"] = max(0, agent["energy"] - random.randint(5, 10))
                        if current_season == "лето":
                            return random.choice(["Запускает воздушного змея.", "Общается с друзьями.", "Играет с друзьями."])
                        else:
                            return random.choice(["Общается с друзьями.", "Играет с друзьями."])
                    actions.append({"priority": 5, "action": action_social_ezhik})
                else:
                    def action_long_for_home():
                        agent["energy"] = max(0, agent["energy"] - random.randint(3, 7))
                        if current_season == "лето":
                            return random.choice(["Запускает воздушного змея.", "Грустит.", "Ловит бабочек в сачок.", "Играет с игрушками."])
                        else:
                            return random.choice(["Грустит.", "Играет с игрушками."])
                    actions.append({"priority": 5, "action": action_long_for_home})

        if char == "Нюша":
            if current_loc == INDIVIDUAL_HOUSES["Нюша"]:
                def action_home_style():
                    agent["energy"] = max(0, agent["energy"] - random.randint(5, 10))
                    if current_season == "лето":
                        return random.choice(["Примеряет новый наряд.", "Ест конфеты.", "Поливает свой цветочный сад.",
                                              "Читает журнал мод.", "Шьёт новый наряд.", "Играет с куклами."])
                    else:
                        return random.choice(["Примеряет новый наряд.", "Ест конфеты.",
                                              "Читает журнал мод.", "Шьёт новый наряд.", "Играет с куклами."])
                actions.append({"priority": 5, "action": action_home_style})
            else:
                if len(present) > 1:
                    def action_social_selfie():
                        agent["energy"] = max(0, agent["energy"] - random.randint(5, 10))
                        if current_season == "лето":
                            return random.choice(["Запускает воздушного змея.", "Ест конфеты.",
                                                  "Делится яркими модными идеями.", "Фотографируется с друзьями."])
                        else:
                            return random.choice(["Делится яркими модными идеями.", "Ест конфеты.", "Фотографируется с друзьями."])
                    actions.append({"priority": 5, "action": action_social_selfie})
                else:
                    def action_home_missing():
                        agent["energy"] = max(0, agent["energy"] - random.randint(5, 10))
                        if current_season == "лето":
                            return random.choice(["Запускает воздушного змея.", "Ест конфеты.", "Играет с куклами."])
                        else:
                            return random.choice(["Ест конфеты.", "Играет с куклами."])
                    actions.append({"priority": 5, "action": action_home_missing})

        if char == "Кар-Карыч":
            if current_loc == INDIVIDUAL_HOUSES["Кар-Карыч"]:
                if len(present) > 1:
                    def action_play_piano():
                        agent["energy"] = max(0, agent["energy"] - random.randint(5, 10))
                        return random.choice(["Рассказывает эпическую историю о приключениях.", "Рассказывает удивительную историю о дальних странах.",
                                              "Делится мудростью, полученной в путешествиях.", "Играет на рояле, радуя присутствующих.",
                                              "Рассказывает удивительную историю о своём прошлом."])
                    actions.append({"priority": 5, "action": action_play_piano})
                else:
                    def action_muse_about_past():
                        agent["energy"] = max(0, agent["energy"] - random.randint(5, 10))
                        return random.choice(["Задумывается о великих путешествиях.", "Вспоминает минувшие дни.", "Читает газету."])
                    actions.append({"priority": 5, "action": action_muse_about_past})
            else:
                if len(present) > 1:
                    def action_storytelling():
                        agent["energy"] = max(0, agent["energy"] - random.randint(5, 10))
                        return random.choice(["Рассказывает эпическую историю о приключениях.", "Рассказывает удивительную историю о дальних странах.",
                                              "Делится мудростью, полученной в путешествиях.", "Рассказывает удивительную историю о своём прошлом."])
                    actions.append({"priority": 5, "action": action_storytelling})
                else:
                    def action_painting():
                        agent["energy"] = max(0, agent["energy"] - random.randint(5, 10))
                        return random.choice(["Рисует картину.", "Задумывается о великих путешествиях.",
                                              "Рассказывает удивительную историю о дальних странах.", "Вспоминает минувшие дни."])
                    actions.append({"priority": 5, "action": action_painting})

        if current_loc in INDIVIDUAL_HOUSES.values():
            def action_listen_radio():
                agent["energy"] = max(0, agent["energy"] - random.randint(1, 3))
                return "Слушает радио."
            actions.append({"priority": 5, "action": action_listen_radio})

        if current_loc in [INDIVIDUAL_HOUSES["Кар-Карыч"], INDIVIDUAL_HOUSES["Совунья"], INDIVIDUAL_HOUSES["Копатыч"], INDIVIDUAL_HOUSES["Бараш"]]:
            def action_listen_gramophone():
                agent["energy"] = max(0, agent["energy"] - random.randint(2, 4))
                return "Слушает граммофон."
            actions.append({"priority": 5, "action": action_listen_gramophone})

        if current_loc == "На пляже":
            if current_season == "лето":
                def action_sunbathe():
                    agent["energy"] = max(0, agent["energy"] - random.randint(2, 4))
                    return random.choice(["Загорает.", "Купается.", "Играет с мячом.", "Лепит песочные замки."])
                actions.append({"priority": 5, "action": action_sunbathe})

        if current_season == "зима" and current_loc not in INDIVIDUAL_HOUSES.values():
            def action_sliding():
                agent["energy"] = max(0, agent["energy"] - random.randint(2, 7))
                return random.choice(["Катается с горки.", "Лепит снеговика.", "Лижет сосульку.",
                                      "Катается на лыжах.", "Катается на коньках.", "Кидает снежки."])
            actions.append({"priority": 5, "action": action_sliding})

        # Выбор и выполнение задачи с учетом одинаковых приоритетов и проверки уникальности задачи
        if actions:
            tasks_by_priority: Dict[int, List[Dict[str, Any]]] = {}
            for task in actions:
                prio: int = task["priority"]
                tasks_by_priority.setdefault(prio, []).append(task)
            min_priority: int = min(tasks_by_priority.keys())
            candidate_tasks: List[Dict[str, Any]] = tasks_by_priority[min_priority]
            random.shuffle(candidate_tasks)
            for task in candidate_tasks:
                task_name: Optional[str] = task.get("task_name")
                if task_name and task_name in location_active_tasks.get(current_loc, set()):
                    continue  # Пропускаем, если такая задача уже выполняется
                if task_name:
                    location_active_tasks.setdefault(current_loc, set()).add(task_name)
                result: Optional[str] = task["action"]()
                if result:
                    if task_name:
                        location_active_tasks[current_loc].discard(task_name)
                    return result
        return None

# Создаём экземпляры персонажей
characters: Dict[str, CharacterModel] = {name: CharacterModel(name, personality) for name, personality in PERSONALITIES.items()}

# --- Правило: если персонаж в своём доме и там есть ещё кто-то, он не уходит ---
def is_in_own_house_with_others(name: str) -> bool:
    current_loc: str = character_locations[name]
    own_house: str = INDIVIDUAL_HOUSES[name]
    if current_loc != own_house:
        return False
    # Проверяем, сколько ещё в этом доме
    present: List[str] = [n for n, loc in character_locations.items() if loc == current_loc]
    if len(present) > 1:
        return True
    return False

# --- Функции симуляции ---

class Simulation:
    """
    Симуляция перемещений персонажей между комнатами, действий и диалогов для конкретной комнаты.
    """
    def __init__(self):
        self.simulation_running = True

    def run(self, room: str):
        while self.simulation_running:
            with global_lock:
                name = random.choice(list(character_locations.keys()))
                old_loc = character_locations[name]
                st = agent_states[name]
                info = CHARACTER_INFO.get(name)
                age_group = info["age"]
                energy = st["energy"]
                present = [n for n, loc in character_locations.items() if loc == old_loc]
                # Если персонаж в своём доме и там не один, он не уходит
                if is_in_own_house_with_others(name):
                    # Пропускаем перемещение, но всё равно повышаем счётчики
                    # чтобы персонаж не застрял навечно
                    st["move_counter"] += random.randint(1, 2)
                    st["friend_visit_counter"] += random.randint(1, 2)
                    # Можно обнулять одиночество (раз у него есть компания)
                    st["loneliness"] = 0
                    continue
                # Проверяем одиночество
                st["loneliness"] += random.randint(1, 2) if len(present) == 1 else 0
                # Увеличиваем остальные счётчики пропорционально количеству персонажей в комнате
                st["move_counter"] += len(present)
                st["friend_visit_counter"] += len(present)
                candidate_actions = ""
                if st["loneliness"] >= loneliness_threshold[age_group] and len(present) == 1 and st["energy"] >= 5:
                    candidate_actions = "loneliness"
                if st["friend_visit_counter"] >= friend_visit_threshold and st["energy"] >= 5:
                    candidate_actions = "friend_visit"
                if st["move_counter"] >= move_threshold and st["energy"] >= 5:
                    candidate_actions = "move"
                if st["talk_counter"] >= talk_threshold and len(present) > 1 and st["energy"] >= 1:
                    candidate_actions = "talk"
                else:
                    candidate_actions = "action"
                if candidate_actions == "loneliness":
                    if character_locations.get(name) != room:
                        continue
                    new_loc: str = random.choice([n for n in preferred_locations[age_group] if n != old_loc and n != INDIVIDUAL_HOUSES[name]])
                    dep_msg: Dict[str, str] = {"role": "[Событие]", "content": f"{name} больше не {old_loc}."}
                    arr_msg: Dict[str, str] = {"role": "[Событие]", "content": f"{name} {new_loc}."}
                    time.sleep(30)
                    location_histories[old_loc].append(dep_msg)
                    location_histories[new_loc].append(arr_msg)
                    print(f"[{old_loc}] [Событие]: {dep_msg['content']}")
                    print(f"[{new_loc}] [Событие]: {arr_msg['content']}")
                    character_locations[name] = new_loc
                    st["loneliness"] = 0
                    st["energy"] = max(0, energy - random.randint(5, 10))
                    st["move_counter"] = 0
                    st["friend_visit_counter"] = 0
                if candidate_actions == "friend_visit":
                    if character_locations.get(name) != room:
                        continue
                    candidate: str = random.choice([house for owner, house in INDIVIDUAL_HOUSES.items() if house != old_loc and house != INDIVIDUAL_HOUSES[name]])
                    dep_msg = {"role": "[Событие]", "content": f"{name} больше не {old_loc}."}
                    arr_msg = {"role": "[Событие]", "content": f"{name} гостит {candidate}."}
                    time.sleep(30)
                    location_histories[old_loc].append(dep_msg)
                    location_histories[candidate].append(arr_msg)
                    print(f"[{old_loc}] [Событие]: {dep_msg['content']}")
                    print(f"[{candidate}] [Событие]: {arr_msg['content']}")
                    character_locations[name] = candidate
                    st["energy"] = max(0, energy - random.randint(5, 10))
                    st["move_counter"] = 0
                    st["friend_visit_counter"] = 0
                    owner_of_candidate: Optional[str] = None
                    for owner, house in INDIVIDUAL_HOUSES.items():
                        if house == candidate:
                            owner_of_candidate = owner
                            break
                    # Если хозяина нет в доме, персонаж уходит в предпочитаемую локацию
                    if character_locations.get(owner_of_candidate) != candidate:
                        if character_locations.get(name) != room:
                            continue
                        new_loc = random.choice([n for n in preferred_locations[age_group] if n != old_loc and n != INDIVIDUAL_HOUSES[name]])
                        dep_msg = {"role": "[Событие]", "content": f"{name} больше не {old_loc}."}
                        arr_msg = {"role": "[Событие]", "content": f"{name} {new_loc}."}
                        time.sleep(30)
                        location_histories[old_loc].append(dep_msg)
                        location_histories[new_loc].append(arr_msg)
                        print(f"[{old_loc}] [Событие]: {dep_msg['content']}")
                        print(f"[{new_loc}] [Событие]: {arr_msg['content']}")
                        character_locations[name] = new_loc
                        st["energy"] = max(0, energy - random.randint(5, 10))
                        st["move_counter"] = 0
                        st["friend_visit_counter"] = 0
                if candidate_actions == "move":
                    if character_locations.get(name) != room:
                        continue
                    new_loc = random.choice([n for n in preferred_locations[age_group] if n != old_loc])
                    dep_msg = {"role": "[Событие]", "content": f"{name} больше не {old_loc}."}
                    arr_msg = {"role": "[Событие]", "content": f"{name} {new_loc}."}
                    time.sleep(30)
                    location_histories[old_loc].append(dep_msg)
                    location_histories[new_loc].append(arr_msg)
                    print(f"[{old_loc}] [Событие]: {dep_msg['content']}")
                    print(f"[{new_loc}] [Событие]: {arr_msg['content']}")
                    character_locations[name] = new_loc
                    st["energy"] = max(0, energy - random.randint(5, 10))
                    st["move_counter"] = 0
                    st["friend_visit_counter"] = 0
                if candidate_actions == "talk":
                    if character_locations.get(name) != room:
                        continue
                    msg: str = characters[name].generate_response(location_histories[room], room, present)
                    st["energy"] = max(0, st["energy"] - 1)
                    st["talk_counter"] = 0
                    location_histories[room].append({"role": name, "content": msg})
                    print(f"[{room}] [{name}]: {msg}")
                    # Проверяем упоминания других
                    for other in present:
                        # Если у другого хватает энергии, ответит
                        if other != name and re.search(rf"\b{other}\b", msg) and agent_states[other]["energy"] >= 1:
                            if character_locations.get(other) != room:
                                continue
                            followup_response: str = characters[other].generate_response(location_histories[room], room, present)
                            agent_states[other]["energy"] -= 1
                            agent_states[other]["talk_counter"] = 0
                            location_histories[room].append({"role": other, "content": followup_response})
                            print(f"[{room}] [{other}]: {followup_response}")
                if candidate_actions == "action":
                    if character_locations.get(name) != room:
                        continue
                    msg: Optional[str] = characters[name].generate_action(location_histories[room], room)
                    if msg:
                        location_histories[room].append({"role": f"[Действие] {name}", "content": msg})
                        print(f"[{room}] [Действие] [{name}]: {msg}")
                if current_season == "зима":
                    if room in INDIVIDUAL_HOUSES.values():
                        if simulation_resources[room]["logs"] < 2:
                            st["health"] = max(0, st["health"] - random.randint(5, 10))
                            location_histories[room].append({"role": f"[Действие] {name}", "content": "Мёрзнет."})
                            print(f"[{room}] [Действие] [{name}]: Мёрзнет.")
                        else:
                            st["energy"] = max(0, st["energy"] - random.randint(2, 4))
                            simulation_resources[room]["logs"] -= random.randint(1, 2)
                            location_histories[room].append({"role": f"[Действие] {name}", "content": "Подкидывает дрова в печь."})
                            print(f"[{room}] [Действие] [{name}]: Подкидывает дрова в печь.")
                st["hunger"] += random.randint(1, 2)
            time.sleep(random.randint(10, 15))
        print("Симуляция остановлена.")

# --- Интерфейс на PyQt5 ---

import sys
import os
from PyQt5 import QtWidgets, QtGui, QtCore

class ChatWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Смешарики")
        self.resize(960, 540)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #a8e6a1;
            }
            QWidget {
                font-family: Sans-serif;
                font-size: 11pt;
            }
        """)

        # ---------- Верхняя панель (узкая, в одну строку) ----------
        header: QtWidgets.QWidget = QtWidgets.QWidget(self)
        header_layout: QtWidgets.QHBoxLayout = QtWidgets.QHBoxLayout(header)

        # Кнопка выбора папки модели
        self.folder_btn: QtWidgets.QPushButton = QtWidgets.QPushButton("Выбрать папку модели")
        self.folder_btn.setStyleSheet("""
            QPushButton {
                background-color: #00ccff;
                padding: 8px;
                border: 3px ridge #00ccff;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #00aacc;
                color: #ffffff;
                border-color: #00aacc;
            }
            QPushButton:pressed {
                background-color: #0088aa;
                color: #ffffff;
                border-color: #0088aa;
            }
        """)
        self.folder_btn.clicked.connect(self.select_model_folder)
        header_layout.addWidget(self.folder_btn)

        # Метка для вывода пути к выбранной папке модели
        self.folder_label: QtWidgets.QLabel = QtWidgets.QLabel(model_name)
        header_layout.addWidget(self.folder_label)
        header_layout.addStretch()
        self.season_label: QtWidgets.QLabel = QtWidgets.QLabel("Сезон: " + current_season)
        self.season_label.setAlignment(QtCore.Qt.AlignRight)
        self.season_label.setAlignment(QtCore.Qt.AlignCenter)
        header_layout.addWidget(self.season_label)
        # Кнопка сохранения состояния симуляции
        self.save_btn = QtWidgets.QPushButton("Сохранить")
        self.save_btn.setStyleSheet("""
            QPushButton {
                background-color: #00ccff;
                padding: 8px;
                border: 3px ridge #00ccff;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #00aacc;
                color: #ffffff;
                border-color: #00aacc;
            }
            QPushButton:pressed {
                background-color: #0088aa;
                color: #ffffff;
                border-color: #0088aa;
            }
        """)
        self.save_btn.clicked.connect(save_simulation_state)
        header_layout.addWidget(self.save_btn)
        # Кнопка загрузки состояния
        self.load_btn = QtWidgets.QPushButton("Загрузить")
        self.load_btn.setStyleSheet("""
            QPushButton {
                background-color: #00ccff;
                padding: 8px;
                border: 3px ridge #00ccff;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #00aacc;
                color: #ffffff;
                border-color: #00aacc;
            }
            QPushButton:pressed {
                background-color: #0088aa;
                color: #ffffff;
                border-color: #0088aa;
            }
        """)
        self.load_btn.clicked.connect(self.load_state)
        header_layout.addWidget(self.load_btn)

        # ---------- Центральная часть ----------
        central_widget: QtWidgets.QWidget = QtWidgets.QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QtWidgets.QVBoxLayout(central_widget)
        main_layout.addWidget(header)

        # Левая панель фиксирована по ширине (200 пикселей)
        self.room_list: QtWidgets.QListWidget = QtWidgets.QListWidget()
        self.room_list.setFixedWidth(200)
        self.room_list.setFocusPolicy(QtCore.Qt.NoFocus)
        self.room_list.setStyleSheet("""
            QListWidget {
                background-color: #ffffff;
                padding: 8px;
                border: 3px outset #ffa07a;
                border-radius: 10px;
                width: 100%;
                height: 100%;
            }
            QListWidget::item {
                border-radius: 3px;
            }
            QListWidget::item:selected {
                background-color: #ffcc99;
                color: #000000;
            }
        """)
        self.room_items: Dict[str, QtWidgets.QListWidgetItem] = {}
        for loc in MAP_LOCATIONS:
            item: QtWidgets.QListWidgetItem = QtWidgets.QListWidgetItem(loc)
            self.room_list.addItem(item)
            self.room_items[loc] = item
        if self.room_list.count() > 0:
            self.room_list.setCurrentRow(0)
        self.current_room: Optional[str] = (self.room_list.currentItem().text().split(" (")[0]
                             if self.room_list.currentItem() else None)
        self.last_message_index: Dict[str, int] = {loc: 0 for loc in MAP_LOCATIONS}

        # Правая панель – чат и список участников
        right_panel: QtWidgets.QWidget = QtWidgets.QWidget()
        right_panel_layout: QtWidgets.QVBoxLayout = QtWidgets.QVBoxLayout(right_panel)
        self.chat_display: QtWidgets.QTextEdit = QtWidgets.QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setStyleSheet("""
            QTextEdit {
                background-color: #ffffff;
                padding: 8px;
                border: 3px outset #ffa07a;
                border-radius: 10px;
                width: 100%;
                height: 100%;
            }
        """)
        self.participants_label: QtWidgets.QLabel = QtWidgets.QLabel()
        self.participants_label.setStyleSheet("""
            QLabel {
                background-color: #ffffff;
                padding: 8px;
                border: 3px outset #ffa07a;
                border-radius: 10px;
                width: 100%;
                height: 100%;
            }
        """)
        self.participants_label.setAlignment(QtCore.Qt.AlignRight)
        right_panel_layout.addWidget(self.chat_display)
        right_panel_layout.addWidget(self.participants_label)

        container: QtWidgets.QWidget = QtWidgets.QWidget()
        container_layout: QtWidgets.QHBoxLayout = QtWidgets.QHBoxLayout(container)
        container_layout.addWidget(self.room_list)
        container_layout.addWidget(right_panel)
        main_layout.addWidget(container)

        self.room_list.currentItemChanged.connect(self.change_room)

        self.timer: QtCore.QTimer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_chat)
        self.timer.start(500)

    def select_model_folder(self) -> None:
        folder: str = QtWidgets.QFileDialog.getExistingDirectory(self, "Выбрать папку модели")
        if folder:
            global model_name, model, tokenizer
            model_name = folder
            self.folder_label.setText(os.path.basename(folder))
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype="auto",
                    device_map="auto",
                ).eval()
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                print("Модель успешно перезагружена")
            except Exception as e:
                print("Ошибка загрузки модели: ", e)

    def load_state(self):
        # Открываем диалог выбора файла (можно выбрать несколько файлов)
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Загрузить состояние симуляции", "", "JSON Files (*.json)")
        if files:
            # Например, берем первый выбранный файл (либо можно предоставить выбор)
            load_simulation_state_from_file(files[0])

    def change_room(self, current: Optional[QtWidgets.QListWidgetItem], previous: Optional[QtWidgets.QListWidgetItem]) -> None:
        if current:
            self.current_room = current.text().split(" (")[0]
            self.chat_display.clear()
            self.last_message_index[self.current_room] = len(location_histories[self.current_room])
            for msg in location_histories[self.current_room]:
                role = msg.get("role", "Неизвестно")
                content = msg.get("content", "")
                self.chat_display.append(f"<b>{role}:</b> {content}")
            self.last_message_index[self.current_room] = len(location_histories[self.current_room])
            self.update_room_list()

    def update_chat(self) -> None:
        if not self.current_room:
            return
        try:
            messages = location_histories[self.current_room]
            start_index: int = self.last_message_index.get(self.current_room, 0)
            if start_index < len(messages):
                vbar = self.chat_display.verticalScrollBar()
                was_at_bottom = (vbar.value() == vbar.maximum())
                new_messages = messages[start_index:]
                for msg in new_messages:
                    role: str = msg.get("role", "Неизвестно")
                    content: str = msg.get("content", "")
                    self.chat_display.append(f"<b>{role}:</b> {content}")
                self.last_message_index[self.current_room] = len(messages)
                if was_at_bottom:
                    self.chat_display.moveCursor(QtGui.QTextCursor.End)
            self.update_room_list()
            present = [name for name, loc in character_locations.items() if loc == self.current_room]
            self.participants_label.setText("Присутствующие: " + ", ".join(present))
            self.season_label.setText("Сезон: " + current_season)
        except RuntimeError as e:
            # Если виджет удалён, прекращаем обновления
            print("Обновление чата прервано: ", e)
            self.timer.stop()

    def update_room_list(self) -> None:
        for room, item in self.room_items.items():
            if room == self.current_room:
                item.setText(room)
            else:
                unread = len(location_histories[room]) - self.last_message_index.get(room, 0)
                item.setText(f"{room} ({unread})" if unread > 0 else room)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        reply = QtWidgets.QMessageBox.question(
            self,
            "Выход",
            "Сохранить состояние перед выходом?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No | QtWidgets.QMessageBox.Cancel,
            QtWidgets.QMessageBox.Cancel
        )
        if reply == QtWidgets.QMessageBox.Cancel:
            event.ignore()
            return
        # Если выбрано "Да", сохраняем состояние
        if reply == QtWidgets.QMessageBox.Yes:
            save_simulation_state_unique()
        # Останавливаем симуляцию
        sim_obj.simulation_running = False
        print("Ожидание завершения...")
        for t in simulation_threads:
            t.join()
        print("Симуляция завершена.")
        event.accept()
        # Принудительное завершение процесса, чтобы не осталось зависших потоков
        QtCore.QTimer.singleShot(0, lambda: os._exit(0))

simulation_threads = []
sim_obj = Simulation()  # один объект для всех потоков

def start_simulation():
    global simulation_threads
    for room in MAP_LOCATIONS:
        t = threading.Thread(target=sim_obj.run, args=(room,))
        simulation_threads.append(t)
        t.start()

if __name__ == '__main__':
    # Запускаем приложение PyQt
    app: QtWidgets.QApplication = QtWidgets.QApplication(sys.argv)
    window: ChatWindow = ChatWindow()
    window.show()
    start_simulation()
    sys.exit(app.exec_())
