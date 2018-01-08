import web
import os
from HTMLParser import HTMLParser

urls = (
    '/', 'home',
    '/test', 'test'
)

class home:
    def GET(self):
        return '''<html>
<head>
<title>Cat Cat</title>
</head>
<style type="text/css">
  .top {width:100%;height:38%}
  .mid {margin-bottom:30px;text-align:center}
  .mid input {width:40%;height:35px;font-size:15px;padding-left:7px}
  .mid button {height:35px;font-size:15px}
  .content {padding-left:121px; width:538px;margin-bottom:14px}
  .content div {font-size:13px}
  .content div em {font-style:normal;color:#c00}
  .content div a {font-size:13px;color:green}
  .content a {font-size:1.17em;color:blue}
  .content a em {font-style:normal;color:#c00;text-decoration:underline}
</style>
<body>
    <div id='top' class='top'>
    </div>
    <div id='mid' class='mid'>
        <input type='text' id='key'><button onclick='search()'>search</button>
    </div>
    <div id='content'>
    </div>
</body>
<script>
    function createXMLHttpRequest() {
        var xmlHttp;
        if (window.XMLHttpRequest) {
            xmlHttp = new XMLHttpRequest();
            if (xmlHttp.overrideMimeType)
                xmlHttp.overrideMimeType('text/xml');
        } else if (window.ActiveXObject) {
            try {
                xmlHttp = new ActiveXObject("Msxml2.XMLHTTP");
            } catch (e) {
                try {
                    xmlHttp = new ActiveXObject("Microsoft.XMLHTTP");
                } catch (e) {}
            }
        }
        return xmlHttp;
    }
    function searchback(){
        if(xmlHttp.readyState == 4 && xmlHttp.status == 200){
            var result = xmlHttp.responseText;
            data = JSON.parse(result);
            text = "";
            for(i = 0; i < data.length; i++){
                text = text + "<div class='content'>"+
                    "<a href='"+data[i].url+"'>"+ data[i].title +"</a>"+
                    "<div>" + data[i].body + "</div>" +
                    "<div> <a href='" + data[i].url + "'>"+ data[i].url +"</a></div>" +
                    "</div>";
            }
            document.getElementById('content').innerHTML = text;
        }
    }
    function search(){
        input = document.getElementById('key');
        key = input.value;
        key = key.trim();
        if(key != ''){
            document.getElementById('top').style.height="8px";
            document.getElementById('mid').style.textAlign = "left"
            document.getElementById('mid').style.paddingLeft="121px";
            
        }else{
            document.getElementById('top').style.height="38%";
            document.getElementById('mid').style.paddingLeft="30%";
            document.getElementById('content').innerHTML = "";
        }
        xmlHttp = createXMLHttpRequest();
        xmlHttp.onreadystatechange = searchback;
        var url = '/test?key=' + key
        xmlHttp.open("GET", url, true);
        xmlHttp.setRequestHeader("Content-Type",
            "application/x-www-form-urlencoded;");
        xmlHttp.setRequestHeader("Access-Control-Allow-Origin", "*");
        xmlHttp.send();
    }
</script>
</html>'''

class test:
    def GET(self):
        i = web.input()
        key = i.key
        if key == '':
            return ''
        else:
        #key = HTMLParser().unescape(key)
            comm = 'scrapy crawl cat -a key=\''+key + '\''
            print(comm)
            os.system(comm)
            data = open('/root/clim/scrapy/tutorial/data/data.json', 'r').read()
            l = len(data) - 1
            res = '[' + data[0:l] + ']'
            return res

if __name__ == "__main__":
    app = web.application(urls, globals())
    app.run()
