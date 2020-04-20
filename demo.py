import re
s2 = 'Total income is around $750,000, pretty good income'
s3 = '<a href="https://www.baidu.com">Baidu<.a>'
print(re.findall("income", s2))
print(re.findall("income$", s2)) # "$": end of sentence
print(re.findall("^income", s2)) # "^": start of sentence

print(re.findall("\$\d+",s2)) # "\": changes the meaning of character
print(re.findall("\$\d?",s2)) # "+": once or multiple times; "?": once or none
print(re.findall("\d{3,}",s2))# "{n,m}": min times versus max times
print(re.findall("href=\".*\"", s3)) # ".": any character; "*": zero or multiple times
print(re.findall("href=\"(.*)\"", s3)) # "()": within the pattern only

print(re.findall("[^0-9]+", s2)) #"^" in "[]" means excluding.