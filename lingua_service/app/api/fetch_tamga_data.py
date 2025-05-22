import requests
from bs4 import BeautifulSoup, Tag, NavigableString
from typing import List, Dict, Union, Tuple
import re


def extract_text_parts(tag: Tag) -> Tuple[List[Dict], List[Dict]]:
    """Разбивает содержимое <td> или <th> по <br> или вложенным тегам"""
    parts = []
    for content in tag.contents:
        if isinstance(content, NavigableString):
            text = content.strip()
            if text:
                parts.append(text)
        elif isinstance(content, Tag):
            if content.name == "br":
                parts.append("<BR>")  # временная метка для последующего сплита
            else:
                subtext = content.get_text(strip=True)
                if subtext:
                    parts.append(subtext)
    joined = " ".join(parts).replace("\xa0", " ").replace(" ", " ")
    return [part.strip() for part in joined.split("<BR>") if part.strip()]


def fetch_tamga_data(word: str) -> Union[List[Dict], Dict]:
    url = f"https://tamgasoft.kg/morfo/ru/{word}"
    response = requests.get(url)
    if response.status_code != 200:
        return [], {"error": f"Failed to fetch data for word: {word}"}

    soup = BeautifulSoup(response.text, "html.parser")

    # === 1. Проверка: найдено ли слово ===
    if soup.find("div", class_="alert-danger"):
        return [], {"error": f"No morphological data found for word: {word}"}
    
    suggestions = []
    success_div = soup.find("div", class_="alert-success")
    if success_div:
        h5_elements = success_div.find_all("h5")
        for h5 in h5_elements:
            b = h5.find("b")
            if b:
                word = b.get_text(strip=True)
                # Текст после <b> — обычно вида " - Зат атооч"
                remaining_text = h5.get_text(strip=True).replace(word, "", 1).lstrip(" -–—").strip()
                suggestions.append({
                    "word": word,
                    "tag": remaining_text
                })

    # === 2. Вкладки: nav-tabs ===
    tabs = []
    for li in soup.select("ul.nav-tabs li"):
        a = li.find("a")
        if not a:
            continue
        href = a.get("href", "").lstrip("#")
        raw_label = a.get_text(separator=" ").strip()
        label = re.sub(r'\s*-\s*', ' - ', raw_label)
        tabs.append({"id": href, "label": label})

    # === 3. Содержимое вкладок ===
    result = []
    for tab in tabs:
        tab_id = tab["id"]
        tab_label = tab["label"]
        tab_div = soup.find("div", {"id": tab_id})
        if not tab_div:
            continue
        
        first_element = next(
            (child for child in tab_div.children if isinstance(child, Tag)),
            None
        )

        info = None
        if first_element and first_element.name == "div" and "bs-callout" in first_element.get("class", []):
            # Извлекаем основной текст
            lemma = first_element.find("h4").get_text(strip=True) if first_element.find("h4") else ""
            
            # Удалим <h4>, чтобы не дублировалось
            first_element_copy = BeautifulSoup(str(first_element), "html.parser")
            h4_copy = first_element_copy.find("h4")
            if h4_copy:
                h4_copy.decompose()

            # Извлекаем весь остальной текст
            description = first_element_copy.get_text(separator=" ", strip=True)

            info = f"{lemma}: {description}" if description else lemma

        blocks = []
        current_block = None

        for child in tab_div.children:
            if isinstance(child, NavigableString):
                continue
            if not isinstance(child, Tag):
                continue

            # Если это <div class="row">, ищем внутри h4
            if child.name == "div" and "row" in child.get("class", []):
                # Ищем все <div> внутри <div class="row">
                inner_divs = child.find_all("div", recursive=False)
                for inner_div in inner_divs:
                    # Убедимся, что это нужная колонка
                    if "col-md-6" in inner_div.get("class", []) or "col-sm-6" in inner_div.get("class", []):
                        h4 = inner_div.find("h4")
                        if h4:
                            current_block = {
                                "title": '',
                                "tables": []
                            }
                            blocks.append(current_block)
            
            # Новый блок — <h4>
            if child.name == "h4":
                current_block = {
                    "title": child.get_text(strip=True),
                    "tables": []
                }
                blocks.append(current_block)

            

            # Таблицы: могут быть внутри div или сразу
            elif child.name in {"div", "table"} and current_block:
                tables = child.find_all("table", class_="grm") if child.name == "div" else [child]
                for table in tables:
                    # === Заголовок таблицы (перед таблицей) ===
                    table_title_tag = table.find_previous_sibling("h4")
                    table_title = table_title_tag.get_text(strip=True) if table_title_tag else ""

                    # === Заголовки ===
                    header_row = table.find("tr")
                    headers = []
                    if header_row:
                        headers = [extract_text_parts(th) for th in header_row.find_all("th")]
                        headers = [h if len(h) > 1 else h[0] if h else "" for h in headers]

                    # === Строки ===
                    rows = []
                    for tr in table.find_all("tr")[1:]:
                        row = []
                        for td in tr.find_all("td"):
                            parts = extract_text_parts(td)
                            if len(parts) > 1:
                                row.append(parts)
                            elif parts:
                                row.append(parts[0])
                            else:
                                row.append("")
                        rows.append(row)

                    current_block["tables"].append({
                        "table_title": table_title,
                        "headers": headers,
                        "rows": rows
                    })

        result.append({
            "id": tab_id,
            "label": tab_label,
            "info": info,
            "blocks": blocks
        })

    return suggestions, result
