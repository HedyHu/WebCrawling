### WebCrawling

###### course notes from chinahadoop - WebCrawling basics
1. Software Installation
* Visual Studio Code (VS Code): In addition to Anaconda, VS Code is needed for this course as Integrated Development Environment(IDE). 
Stable VS Code software can be found under https://code.visualstudio.com/. Install customized extension for Python on the very first page under Tools and Languages. 
Extension name: Python extension for Visual Studio Code.
* Third Party Packages for Python: pip install -i https://pypi.tuna.tsinghua.edu.cn/simple some-package. 
some-package refers to package names in requirements.txt under https://github.com/suneri/junior_spider

2. Find HTML-a and table
* The full name for HTML is Hyper Text Markup Language. 
\<a href=\"\****\"></a\>": "<a" stands for an anchor to locate our interested contents. 
\<table></table\> defines a table. \<tr></tr\> stands for a line. \<th></th\> refers to the header cells and \<td>/</td\> refers to other regular cells.

3. dom attributes
* id is unique and class may include many elements. 

4. Cascading Style Sheet(CSS)
* CSS can be saved in HTML file under certain elements. Also, it can be stored as ".css" file while HTML triggers the css file.
<link rel="stylesheet" type="text/css" href="myindex.css">

5. requests, url, headers
* js and css files may not be saved when request webpage. Then, source page file has to replace "//" to "https://".

6. encoding
* HTML has "charset" to locate the web page's character coding style. 
* res.encoding="utf-8" can be used to reset the coding into "utf8". 
* res.content => "w+b" res.text => "w"
* hash function can be triggered under hashlib package for md5 encoding. 

7. Parse
* Regular Expression and LXML (based on DOM tree)
* re.findall(pattern, string): For details, refer to demo.py
* "?": once or none so that it is not greedy.
* (?<=pattern) sub (?=pattern): two rules find the start and the end of pattern, and substitution happens within the pattern. For details, refer to demo.py

8. XPATH in Domain Operating Model (DOM)