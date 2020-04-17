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
* <a href="****"></a>: "<a" stands for an anchor to locate our interested contents. 
* <table></table> defines a table. <tr></tr> stands for a line. <th></th> refers to the header cells and <td>/</td> refers to other regular cells.

3. dom attributes
* id is unique and class may include many elements. 

4. Cascading Style Sheet(CSS)
* CSS can be saved in HTML file under certain elements. Also, it can be stored as ".css" file while HTML triggers the css file.
<link rel="stylesheet" type="text/css" href="myindex.css">