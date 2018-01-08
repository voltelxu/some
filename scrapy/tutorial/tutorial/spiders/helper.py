import re

class HTML_Tool:
    BgnCharToNoneRex = re.compile("(\t|\n|<a.*?>|<img.*?>)")
    EndCharToNoneRex = re.compile("<.*?>")
    BgnPartRex = re.compile("<p.*?>")
    CharToNewLineRex = re.compile("(<br/>|</p>|<tr>|<div>|</div>|<br>|<BR>)")
    CharToNextTabRex = re.compile("<td>")
    replaceTab = [("&lt;","<"),("&gt;",">"),("&amp;","&"),("&amp;","\""),("&nbsp;"," "),("&quot;","\""),("\r\n"," "),("\t"," "),("\r"," ")]
    def Replace_Char(self,x):
        x = self.BgnCharToNoneRex.sub(" ",x)
        x = self.BgnPartRex.sub(" ",x)
        x = self.CharToNewLineRex.sub(" ",x)
        x = self.CharToNextTabRex.sub(" ",x)
        x = self.EndCharToNoneRex.sub("",x)

        for t in self.replaceTab:
            x = x.replace(t[0],t[1])
        x = re.sub('[ ]+', ' ', x)
        x = x.strip(' ')
        return x

    def Check_URL(self, url):
        #pattern = re.compile(r'(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]')
        pattern = re.compile(r'((https?|ftp|file|/)(://|)[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|])')
        match = pattern.match(url)
        if match:
            return True
        else:
            return False
    def Remove_A(self, text):
        text = re.sub(r'<a (.*?)>([\s\S]*?)</a>', '', text)
        text = re.sub(r'<script(.*?)>([\s\S]*?)</script>', '', text)
        text = re.sub(r'<SCRIPT(.*?)>([\s\S]*?)</SCRIPT>', '', text)
        text = re.sub(r'<style(.*?)>([\s\S]*?)</style>', '', text)
        text = re.sub(r'<!--([\s\S]*?)-->', '', text)
        return text

    def Get_Domain(self, url):
        pattern = re.compile(r'^((http://)|(https://))?([a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,6}')
        domain = pattern.match(url).group()
        return domain

