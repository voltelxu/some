import scrapy
import sys
import hashlib
from scrapy.selector import Selector
from tutorial.items import TutorialItem
import re

sys.path.append('/root/clim/scrapy/tutorial/tutorial/spiders/')
import helper
import bodyparse
import checkurl
import used
sys.path.append('/root/clim/scrapy/tutorial/tutorial/spiders/cat/')
import charparse

class CatCool(scrapy.Spider):
    name = "cat"
    maxpage = 0
    tool = helper.HTML_Tool()
    bodyparse = bodyparse.Body_Parse()
    usedurl = used.Url_Used()
    checkurl = checkurl.Check_Url()
    wordparse = charparse.Word_Parse()
    keyword = ''

    def __init__(self, key=None, *args, **kwargs):
        keyword = self.wordparse.parse(key)
        self.keyword = keyword

    def start_requests(self):
        start_urls = open('./tutorial/spiders/urls', 'r')
        urls = []
        i = 0
        for line in start_urls:
            i = i + 1
            urls.append(line.strip('\n'))
#            if i == 100:
#                break
        urls = ['http://www.163.com','http://www.sohu.com','http://www.sina.com.cn/','http://www.ifeng.com/']
        for url in urls:
            if url not in self.usedurl.url:
                self.usedurl.url.append(url)
                yield scrapy.Request(url = url, meta={"root":True}, callback = self.parse)

    def parse(self, response):
        isroot = response.meta['root']
        #domain = self.tool.Get_Domain(response.url)
        hrefs = response.css('a').extract()
        print('========')
        #title = ''.join(response.xpath('//title/text()').extract())
        #print(title)
        #body_data = self.bodyparse.parse(response.text)
        next_urls = self.checkurl.check(hrefs, self.keyword)
        for fl in next_urls:
            if self.maxpage > 200:
                break
            if fl not in self.usedurl.url:
                self.usedurl.url.append(fl)
                self.maxpage = self.maxpage + 1
                yield response.follow(fl, self.follow_parse)
        #print(len(next_urls))
        #print(len(text))

    def follow_parse(self, response):
        item = TutorialItem()
        url = response.url
        title = ''.join(response.xpath('//title/text()').extract())
        body_data = self.bodyparse.parse(response.text)
        filename = hashlib.md5(url).hexdigest()
        item['name'] = filename
        item['url'] = url
        item['title'] = title.encode('utf-8')
        if len(item['title']) > 60:
            item['title'] = item['title'][0:36] + "..."
        item['body'] = body_data.encode('utf-8')
        if len(item['body']) > 400:
            item['body'] = body_data.encode('utf-8')[0:390] + "..."
        yield item
        #print('title')
        #print(title)

