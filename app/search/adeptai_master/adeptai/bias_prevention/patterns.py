# Bias Prevention Patterns - Focused on Race, Culture, and Religion Only

# RACE PATTERNS - Ethnic names, cultural identifiers, racial indicators
RACE_PATTERNS = [
    # Ethnic surnames and prefixes
    r'\b(von|van|de la|del|ibn|bin|singh|kumar|patel|o\'|mc|mac|al-|el-|ben-|bat-|bar-)\w+\b',
    r'\b(abdul|ahmed|ali|hassan|hussein|mohammed|muhammad|ibrahim|ismail|yusuf)\b',
    r'\b(zhang|wang|li|liu|chen|yang|huang|zhao|wu|zhou|sun|ma|zhu|guo|lin)\b',
    r'\b(kim|lee|park|choi|jung|kang|yoon|cho|yoo|song|han|lim|shin|bae|ahn)\b',
    r'\b(patel|shah|kumar|singh|verma|gupta|sharma|yadav|kumar|tiwari|reddy)\b',
    r'\b(rodriguez|gonzalez|hernandez|lopez|garcia|martinez|perez|sanchez|ramirez|torres)\b',
    r'\b(johnson|williams|brown|jones|garcia|miller|davis|rodriguez|martinez|hernandez)\b',
    r'\b(nguyen|tran|le|pham|hoang|vu|dang|bui|do|ho|duong|ly|ngo|truong)\b',
    r'\b(anderson|wilson|taylor|thomas|jackson|white|harris|martin|thompson|garcia)\b',
    r'\b(ng|chan|wong|lau|yeung|cheung|chow|lam|liu|tam|wu|chen|ho|lee)\b',
    
    # Cultural identifiers
    r'\b(african|asian|hispanic|latino|caucasian|white|black|indian|middle\s+eastern)\b',
    r'\b(native\s+american|indigenous|aboriginal|first\s+nations|inuit|metis)\b',
    r'\b(pacific\s+islander|polynesian|melanesian|micronesian|hawaiian|maori)\b',
    r'\b(caribbean|west\s+indian|jamaican|barbadian|trinidadian|haitian)\b',
    r'\b(eastern\s+european|slavic|baltic|balkan|mediterranean|nordic)\b'
]

# CULTURE PATTERNS - Nationalities, cultural practices, traditional names
CULTURE_PATTERNS = [
    # Nationality indicators
    r'\b(american|canadian|british|australian|german|french|italian|spanish|portuguese)\b',
    r'\b(chinese|japanese|korean|vietnamese|thai|indonesian|malaysian|filipino|singaporean)\b',
    r'\b(indian|pakistani|bangladeshi|sri\s+lankan|nepali|bhutanese|maldivian)\b',
    r'\b(iranian|iraqi|syrian|lebanese|jordanian|palestinian|egyptian|moroccan|tunisian)\b',
    r'\b(nigerian|ghanaian|kenyan|ugandan|tanzanian|ethiopian|somali|sudanese)\b',
    r'\b(mexican|brazilian|argentine|chilean|colombian|peruvian|venezuelan|ecuadorian)\b',
    r'\b(russian|ukrainian|polish|czech|slovak|hungarian|romanian|bulgarian|serbian)\b',
    
    # Cultural practices and traditions
    r'\b(diwali|holi|ramadan|eid|christmas|easter|hanukkah|passover|vesak|vesakha)\b',
    r'\b(chinese\s+new\s+year|lunar\s+new\s+year|tet|songkran|nyepi|galungan)\b',
    r'\b(quinceanera|bar\s+mitzvah|bat\s+mitzvah|confirmation|first\s+communion)\b',
    r'\b(henna|mehndi|bindis|saris|kimonos|hanboks|ao\s+dai|cheongsam)\b',
    r'\b(bollywood|kollywood|ollywood|nollywood|hollywood|european\s+cinema)\b',
    
    # Traditional names and titles
    r'\b(raja|rani|sultan|emir|sheikh|shah|khan|begum|bibi|nawab)\b',
    r'\b(sensei|sifu|guru|acharya|pandit|imam|rabbi|priest|monk|nun)\b',
    r'\b(lao\s+shi|xiansheng|xiaojie|xiansheng|laoshi|tongzhi|tongxue)\b',
    r'\b(san|kun|chan|sama|senpai|kohai|sensei|shihan|renshi|kyoshi)\b',
    r'\b(sir|madam|lord|lady|duke|duchess|prince|princess|emperor|empress)\b'
]

# RELIGION PATTERNS - Religious titles, practices, institutions
RELIGION_PATTERNS = [
    # Religious titles and honorifics
    r'\b(rabbi|imam|pastor|father|sister|brother|priest|monk|nun|bishop|archbishop)\b',
    r'\b(pope|cardinal|deacon|minister|reverend|canon|vicar|curate|chaplain)\b',
    r'\b(swami|guru|acharya|pandit|brahmin|kshatriya|vaishya|shudra)\b',
    r'\b(lama|rinpoche|tulku|geshe|khenpo|chogyal|dorje|tashi|tenzin)\b',
    r'\b(ayatollah|mullah|sheikh|qadi|mufti|hafiz|alim|ustadh|ustadha)\b',
    
    # Religious practices and observances
    r'\b(prayer|worship|meditation|fasting|pilgrimage|baptism|communion|confession)\b',
    r'\b(halal|kosher|vegetarian|vegan|jain|buddhist\s+diet|hindu\s+diet)\b',
    r'\b(sabbath|shabbat|jummah|puja|aarti|bhajan|kirtan|qawwali)\b',
    r'\b(rosary|novena|liturgy|mass|service|ceremony|ritual|tradition)\b',
    
    # Religious institutions and organizations
    r'\b(church|mosque|synagogue|temple|cathedral|chapel|shrine|monastery|convent)\b',
    r'\b(gurdwara|mandir|pagoda|stupa|vihara|ashram|math|peeth|dargah)\b',
    r'\b(christian|muslim|jewish|hindu|buddhist|sikh|jain|zoroastrian)\s+(society|association|center|institute)\b',
    r'\b(youth\s+group|bible\s+study|quran\s+study|sunday\s+school|madrasa|yeshiva)\b',
    
    # Religious holidays and festivals
    r'\b(ramadan|eid\s+al-fitr|eid\s+al-adha|mawlid|ashura|laylat\s+al-qadr)\b',
    r'\b(passover|shavuot|rosh\s+hashanah|yom\s+kippur|sukkot|purim|hanukkah)\b',
    r'\b(christmas|easter|pentecost|epiphany|all\s+saints|assumption|immaculate\s+conception)\b',
    r'\b(diwali|holi|ram\s+navami|krishna\s+janmashtami|ganesh\s+chaturthi|navratri)\b',
    r'\b(vesak|magha\s+puja|asala\s+puja|kathina|ulambana|songkran|loy\s+krathong)\b'
]

# Combined patterns for comprehensive detection
PROTECTED_CHARACTERISTIC_PATTERNS = {
    'race': RACE_PATTERNS,
    'culture': CULTURE_PATTERNS,
    'religion': RELIGION_PATTERNS
}

# Legacy pattern names for backward compatibility (deprecated)
NAME_PATTERNS = RACE_PATTERNS
LOCATION_PATTERNS = []  # Removed - not race/culture/religion related
DEMOGRAPHIC_PATTERNS = []  # Removed - not race/culture/religion related
ASSOCIATION_PATTERNS = []  # Removed - not race/culture/religion related
