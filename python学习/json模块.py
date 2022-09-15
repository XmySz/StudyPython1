import json

"""
    JSON（JavaScript Object Notation）是一种轻量级的数据交换格式，遵循欧洲计算机协会制定的 JavaScript 规范（简称 ECMAScript）。
    JSON 易于人阅读和编写，同时也易于机器解析和生成，能够有效的提升网信息的传输效率，因此它常被作为网络、程序之间传递信息的标准语言，
    比如客户端与服务器之间信息交互就是以 JSON 格式传递的。
    
    jons.loads()：该方法可以将json格式的字符串转换成Python对象（比如列表、字典、元组、整型以及浮点型），其中最常用的是转换为字典类型。
    json.dump()：它可以将Python对象（字典、列表等）转换为json字符串，并将转换后的数据写入到json格式的文件中，因此该方法必须操作文件流对象。
        json.dump(object,f,inden=0，ensure_ascii=False)
            object：Python 数据对象，比如字典，列表等
            f：文件流对象，即文件句柄。
            indent：格式化存储数据，使 JSON 字符串更易阅读。
            ensure_ascii：是否使用 ascii 编码，当数据中出现中文的时候，需要将其设置为 False。
    json.load(): 它表示从json文件中读取JSON字符串，并将读取内容转换为Python对象。
    json.dumps(): 该方法可以将Python对象转换成 JSON 字符串。

"""
# loads方法的示例
website_info = '{"name" : "c语言中文网","PV" : "50万","UV" : "20万","create_time" : "2010年"}'
py_dict = json.loads(website_info)
print("python字典数据格式：%s；数据类型：%s" % (py_dict, type(py_dict)))

# dump方法的示例
ditc_info = {"name": "c语言中文网", "PV": "50万", "UV": "20万", "create_time": "2010年"}
with open("./data/web.josn", "w") as f:
    json.dump(ditc_info, f, ensure_ascii=False)

# load方法的示例
with open('./data/web.josn', 'r') as f:
    print(json.load(f))

# dumps方法的示例
item = {'website': 'C语言中文网', 'rank': 1}
item = json.dumps(item, ensure_ascii=False)
print('转换之后的数据类型为：', type(item))
print(item)
