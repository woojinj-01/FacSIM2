# import sys

from configparser import ConfigParser


typo_history_path = "./typo_history.ini"

length_compare_threshold = 3
hamming_dist_threshold = 0.25

area_seoul = ['seoul']

area_capital = ['hwaseong', 'bucheon', 'suwon', 'ansan',
                'anseong', 'osan', 'yongin',
                'goyang', 'anyang', 'gyeonggi-do',
                'seongnam', '_x0008_suwon']

area_metro = ['songdo', 'incheon', 'daegu', 'gwangju', 'daejeon', 'ulsan', 'busan', 'pusan']

area_others = ['miryang', 'pohang', 'jeonju',
               'changwon', 'asan', 'gunsan', 'yeongju',
               'jeju', 'gongju', 'goesan',
               'jecheon', 'chuncheon', 'kyungsan',
               'gumi', 'masan', 'muan',
               'cheongju', 'yeosu', 'wonju',
               'cheonan', 'chuncheon-si', 'kunsan', 'sejong',
               'gyeongsan', 'jinju', 'suncheon', 'gangneung',
               'mokpo', 'cheonanl', 'sunchon', 'chungju',
               'gyeongju', 'kongju', 'andong',
               'gimhae', 'iksan', 'namwon']

inst_ists = ['korea advanced institute of science and technology,daejeon,KR',
             'gwangju institute of science and technology,gwangju,KR',
             'daegu gyeongbuk institute of science and technology,daegu,KR',
             'korea advanced institute of science and technology munji camus,daejeon,KR',
             'ulsan national institute of science and technology,ulsan,KR']

con_america = ["US", "CA"]
con_asia = ["KR", "JP", "IN", "INDIA", "KOREA", "UZ", "IL", "PK", "SINGAPORE", "MONGOLIA", "CN", "SG"]
con_europe = ["SE", "HU", "UK", "ES", "FR", "IT", "DE", "SZ", "AT", "NL", "RU", "DK", "PL", "WE"]
con_ocenaia = ["NZ", "AU"]


def are_similar_texts(str_1, str_2):    # str_2 is reference (more reliable)

    if (any(type(elem) is not str for elem in [str_1, str_2])):
        return None
    
    if (str_1 == str_2):
        return str_2
    
    from_history = _read_typo_history(str_1, str_2)
    
    if (from_history is not None):
        return from_history
    
    str_1_compact = str_1.replace(" ", "")
    str_2_compact = str_2.replace(" ", "")

    if (str_1_compact.lower() == str_2_compact.lower()):
        return user_selects_string(str_1, str_2)
    
    elif (str_1_compact.lower() in str_2_compact.lower()):
        return user_selects_string(str_1, str_2)
    
    elif (str_1_compact.lower() in str_2_compact.lower()):
        return user_selects_string(str_1, str_2)
    
    elif (str_1_compact in str_2_compact):
        return user_selects_string(str_1, str_2)
    
    elif (str_1_compact in str_2_compact):
        return user_selects_string(str_1, str_2)
    
    str_1_hamming = str_1_compact.replace("university", "")
    str_2_hamming = str_2_compact.replace("university", "")

    if (length_compare_threshold <= abs(len(str_1_hamming) - len(str_2_hamming))):
        return None

    length_to_compare = min([len(str_1_hamming), len(str_2_hamming)])
    hamming_dist = 0 

    for index in range(length_to_compare):
        if (str_1_hamming[index] != str_2_hamming[index]):
            hamming_dist += 1

    if (int((hamming_dist / length_to_compare) <= hamming_dist_threshold)):
        return user_selects_string(str_1, str_2)
    
    return None


def _read_typo_history(str_1, str_2):

    return None
    
    # config = ConfigParser(allow_no_value=True)
    # config.read(typo_history_path, encoding='utf-8')

    # if (config.has_section(str_1) and config.has_option(str_1, str_2)):
    #     print("Reading history")
    #     return config.get(str_1, str_2)
    # else:
    #     return None
    
    
def _write_typo_history(str_1, str_2, str_chosen):
    
    config = ConfigParser(allow_no_value=True)

    if (not config.has_section(str_1)):
        config.add_section(str_1)

    config.set(str_1, str_2, str_chosen)

    print("Local write:", str_1, str_2, str_chosen)

    with open(typo_history_path, 'a') as history:
        print("Writing history")
        config.write(history)


def user_selects_string(str_1, str_2):

    return str_2

    # original_stream = sys.stdout
    # sys.stdout = sys.__stdout__
    
    # print("=" * 10 + "Typo Correction" + "=" * 10)
        
    # while (1):

    #     print("Press 1 to Use", str_1)
    #     print("Press 2 to Use", str_2)

    #     select = int(input())

    #     str_chosen = str_1 if 1 == select else str_2 if 2 == select else None

    #     if (str_chosen is None):
    #         print("Wrong Key. Press 1 or 2")

    #     else:
    #         _write_typo_history(str_1, str_2, str_chosen)

    #         sys.stdout = original_stream
    #         return str_chosen


def normalize_inst_name(text):

    if (type(text) is not str or text is None):
        return None
    
    text.lstrip()
    text.rstrip()

    text_replaced = text.replace(", ", ",")

    text_splitted = text_replaced.split(",")

    if (3 != len(text_splitted)):
        return None
    elif (any("" == word for word in text_splitted)):
        return None
    
    inst_name = text_splitted[0].lower()
    region_code = text_splitted[1].lower()
    country_code = text_splitted[2].upper()

    return ",".join([inst_name, region_code, country_code])


def get_country_code(text):

    text_normalized = normalize_inst_name(text)

    if (text_normalized is None):
        return None
    
    return text_normalized.split(',')[2]


def get_region(text):

    text_normalized = normalize_inst_name(text)

    if (text_normalized is None):
        return None
    
    return text_normalized.split(',')[1]


if (__name__ == "__main__"):

    target = ["KAIST, Daejeon, kR", "University of Chicago, Chicago", "Yonsei University,,KR"]

    for str in target:
        print(normalize_inst_name(str))