import re
import requests
from lxml import etree

s2 = 'Total income is around $750,000, pretty good income'
s3 = '<a href="https://www.baidu.com">Baidu<.a>'

# 2-7 basics
print(re.findall("income", s2))
print(re.findall("income$", s2)) # "$": end of sentence
print(re.findall("^income", s2)) # "^": start of sentence

print(re.findall("\$\d+",s2)) # "\": changes the meaning of character
print(re.findall("\$\d?",s2)) # "+": once or multiple times; "?": once or none
print(re.findall("\d{3,}",s2))# "{n,m}": min times versus max times
print(re.findall("href=\".*\"", s3)) # ".": any character; "*": zero or multiple times
print(re.findall("href=\"(.*)\"", s3)) # "()": within the pattern only

print(re.findall("[^0-9]+", s2)) # "^" in "[]" means excluding.
print(re.findall('.*?>(.*?)<',s3)) # "?" is not greedy.

print(re.sub("(?<=href=\").*?(?=\")", "https://www.tencent.com", s3)) # (?<=pattern) sub (?=pattern)


# 8. XPATH in Domain Operating Model (DOM)
url = "https://en.wikipedia.org/wiki/Steve_Jobs"
res = requests.get(url)

# with open("steve_jobs.html","w", encoding="utf-8") as f:
#     f.write(res.text)

with open("steve_jobs.html","r",encoding="utf-8") as f:
    c = f.read()

tree = etree.HTML(c)
table_element = tree.xpath("//table[@class='infobox biography vcard']")
print(table_element)
table_rows = tree.xpath("//table[@class='infobox biography vcard'][1]/tbody/tr")
print(table_rows)

pattern_attrib = re.compile('<.*?>')
for row in table_rows:
    try:
        thead = row.xpath('th')[0]
        title = etree.tostring(thead).decode('utf-8')
        title = pattern_attrib.sub('',title)

        desc = row.xpath('td')[0]
        desc = etree.tostring(desc).decode('utf-8')
        desc = pattern_attrib.sub('',desc)
        print(title, ":", desc)

    except Exception as err:
        print("error: ", err)
        pass

content = tree.xpath("//div[@id='mw-content-text'][1]//*[self::h2 or self::p]")
for line in content:
    print(line.text)
