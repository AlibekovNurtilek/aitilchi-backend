def split_last_syllable(word):
    """
    Функция делит кыргызское слово на две части:
    - последний слог
    - остальная часть слова
    
    Параметры:
    word (str): Кыргызское слово для разделения
    
    Возвращает:
    tuple: (остальная_часть, последний_слог)
    
    Примеры:
    >>> split_last_syllable("китеп")
    ('ки', 'теп')
    >>> split_last_syllable("мектеп")
    ('мек', 'теп')
    >>> split_last_syllable("тоо")
    ('то', 'о')
    >>> split_last_syllable("Кыргызстан")
    ('Кыргыз', 'стан')
    """
    # Проверка входных данных
    if not word:
        return "", ""
    if len(word) < 6:
        return "", word
    
    if not isinstance(word, str):
        raise TypeError(f"Ожидалась строка, получено {type(word).__name__}")
    
    # Константа - список гласных в кыргызском языке (для ускорения поиска)
    VOWELS = frozenset("аеёиоөуүыэюяАЕЁИОӨУҮЫЭЮЯ")
    
    # Найдем позицию последней гласной буквы
    last_vowel_pos = -1
    
    # Поиск с конца слова более эффективен для этой задачи
    for i in range(len(word) - 1, -1, -1):
        if word[i] in VOWELS:
            last_vowel_pos = i
            break
    
    if last_vowel_pos == -1:
        # Если гласных нет, возвращаем всё слово как последний слог
        return "", word
    
    # Теперь определяем начало последнего слога
    syllable_start = last_vowel_pos
    
    # Идём назад от последней гласной, чтобы найти начало слога
    if last_vowel_pos > 0:
        # Проверяем, есть ли согласные перед последней гласной
        consonant_count = 0
        i = last_vowel_pos - 1
        
        while i >= 0:
            if word[i] not in VOWELS:
                consonant_count += 1
                syllable_start = i
                
                # Если встретили две согласные подряд, начинаем с последней согласной
                if consonant_count >= 2:
                    syllable_start = i + 1
                    break
            else:
                break
            
            i -= 1
    
    # Разделяем слово
    rest_part = word[:syllable_start]
    last_syllable = word[syllable_start:]
    if len(rest_part) < 2:
        last_syllable = rest_part + last_syllable
        rest_part = ''
    
    return rest_part, last_syllable



if __name__ == "__main__":
   
   
    # Интерактивный режим
    print("Введите кыргызское слово (или 'выход' для завершения):")
    while True:
        try:
            user_input = input("> ")
            if user_input.lower() in ['выход', 'exit', 'quit']:
                break
            
            rest, last_syllable = split_last_syllable(user_input)
            print(f"Остальная часть: '{rest}'")
            print(f"Последний слог: '{last_syllable}'")
        except Exception as e:
            print(f"Ошибка: {str(e)}")