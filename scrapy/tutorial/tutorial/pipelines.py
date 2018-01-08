# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: http://doc.scrapy.org/en/latest/topics/item-pipeline.html

import codecs
import json
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

class TutorialPipeline(object):
    def __init__(self):
        self.file = codecs.open('/root/clim/scrapy/tutorial/data/data.json', mode='wb', encoding='utf-8')
    def process_item(self, item, spider):
        ts = item['title'].split(' ')
        item['title'] = ""
        for i in range(len(ts) - 1):
            item['title'] = item['title'] + " " + ts[i]
        bs = item['body'].split(' ')
        item['body'] = ""
        for i in range(len(bs) - 1):
            item['body'] = item['body'] + " " + bs[i]
        text = {"url":item['url'],"title":item['title'],"body":item['body']}
        line = json.dumps(text, ensure_ascii=False) + ','
        self.file.write(line)

    def close_spide(self, spider):
        self.file.close()
