import sys
sys.path.append('/root/clim/scrapy/tutorial/tutorial/spiders/')
import helper
from scrapy.selector import Selector
sys.path.append('/root/clim/scrapy/tutorial/tutorial/spiders/cat/')
import judge


class Check_Url:
    tool = helper.HTML_Tool()
    def check(self, hrefs, keyword):
        url = []
        print('---------------------')
        print(keyword)
        data = []
        for i in range(1,len(hrefs)):
            href = Selector(text=hrefs[i]).xpath('//a//@href').extract()
            href = ''.join(href)
            isurl = self.tool.Check_URL(href)
            if isurl == False:
                continue
            text = Selector(text=hrefs[i]).xpath("//a//text()").extract()
            #print(data)
            text = ''.join(text)
            text = self.tool.Replace_Char(text)
            if text == '':
                continue
            if href in url:
                index = url.index(href)
                data[index] = data[index] + " " + text
            else:
                url.append(href)
                data.append(text)
                #print(text)
        #data.append('')
        true = self.judge(url, data, keyword)
        return true

    def judge(self, href, data, keyword):
        true_urls = []
        ju = judge.Judge()
        for i in range(len(href)):
            flow = ju.judge(data[i],keyword)
            if flow == True:
                true_urls.append(href[i])
        return true_urls

