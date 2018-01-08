import sys
sys.path.append('/root/clim/scrapy/tutorial/tutorial/spiders/')
import helper
from scrapy.selector import Selector

class Body_Parse:
    tool = helper.HTML_Tool()
    def parse(self, data):
        body = self.tool.Remove_A(data)
        body_data = Selector(text=body).xpath('//body//text()').extract()
        body_data = ' '.join(body_data)
        body_data = self.tool.Replace_Char(body_data)
        return body_data
