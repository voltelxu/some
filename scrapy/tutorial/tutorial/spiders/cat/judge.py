import lable

class Judge():
    fe = lable.Lable()
    def judge(self, text, keyword):
        keys = keyword.strip(' ').split(' ')
        if len(keys) == 1:
             if keys[0] in text:
                 return True
             
        return False
