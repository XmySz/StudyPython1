import xlrd  # 从 Excel 中读取数据，支持 xls。
import xlwt  # 从 Excel 中写入数据，支持 xls 。
import xlutils  # 提供了一些 Excel 的实用操作，比如复制、拆分、过滤等，通常与 xlrd、xlwt 一起使用。
from xlutils.copy import copy
import xlsxwriter  # 向 Excel 中写入数据, 生成图表，支持 xlsx。
import openpyxl  # 用于读写 Excel，支持 xlsx。


def _write_use_xlwt():
    """
    使用xlwt向excel写入数据
    :return:
    """
    wb = xlwt.Workbook()  # 创建工作簿
    sh = wb.add_sheet('test')  # 创建表单
    font = xlwt.Font()  # 创建字体对象
    font.bold = True  # 字体加粗
    alm = xlwt.Alignment()  #
    alm.horz = 0x01  # 设置左对齐
    style1 = xlwt.XFStyle()  # 创建样式对象
    style2 = xlwt.XFStyle()
    style1.font = font
    style2.alignment = alm
    # write 方法参数1：行，参数2：列，参数3：内容
    sh.write(0, 1, '姓名', style1)
    sh.write(0, 2, '年龄', style1)
    sh.write(1, 1, '张三')
    sh.write(1, 2, 50, style2)
    sh.write(2, 1, '李四')
    sh.write(2, 2, 30, style2)
    sh.write(3, 1, '王五')
    sh.write(3, 2, 40, style2)
    sh.write(4, 1, '赵六')
    sh.write(4, 2, 60, style2)
    sh.write(5, 0, '平均年龄', style1)

    wb.save('./data/test.xls')  # 保存


def _write_use_xlsxwriter():
    """
    使用xlsxwriter向xlsx写入数据绘制图表
    :return:
    """
    wk = xlsxwriter.Workbook('./data/test_xlsxwriter.xlsx')       # 创建工作簿
    sh = wk.add_worksheet('test')    # 创建表单
    fmt1 = wk.add_format()
    fmt2 = wk.add_format()
    fmt1.set_bold(True)     # 字体加粗
    fmt2.set_align('left')      # 设置左对齐
    # 数据
    data = [
        ['', '姓名', '年龄'],
        ['', '张三', 50],
        ['', '李四', 30],
        ['', '王五', 40],
        ['', '赵六', 60],
        ['平均年龄', '', ]
    ]
    sh.write_row('A1', data[0], fmt1)
    sh.write_row('A2', data[1], fmt2)
    sh.write_row('A3', data[2], fmt2)
    sh.write_row('A4', data[3], fmt2)
    sh.write_row('A5', data[4], fmt2)
    sh.write_row('A6', data[5], fmt1)
    '''
    area：面积图
    bar：直方图
    column：柱状图
    line：折线图
    pie：饼图
    doughnut：环形图
    radar：雷达图
    '''
    chart = wk.add_chart({'type': 'line'})  # 创建图表
    chart.add_series(
        {
            'name': '=test!$B$1',
            'categories': '=test!$B$2:$B$5',
            'values': '=test!$C$2:$C$5'
        }
    )
    chart.set_title({'name': '用户年龄折线图'})
    chart.set_x_axis({'name': '姓名'})
    chart.set_y_axis({'name': '年龄'})
    sh.insert_chart('A9', chart)
    wk.close()


def _read_use_xlrd():
    """
    使用xlrd读取数据
    :return:
    """
    wb = xlrd.open_workbook("./data/test.xls")
    print('sheet名称:', wb.sheet_names())
    print('sheet数量:', wb.nsheets)
    sh = wb.sheet_by_index(0)    # 根据索引获得sheet
    sh = wb.sheet_by_name('test')    # 根据名称获得sheet
    print(u'sheet %s 有 %d 行' % (sh.name, sh.nrows))
    print(u'sheet %s 有 %d 列' % (sh.name, sh.ncols))
    print('第二行内容:', sh.row_values(1))
    print('第三列内容:', sh.col_values(2))
    print('第二行第三列的值为:', sh.cell_value(1, 2))
    print('第二行第三列值的类型为:', type(sh.cell_value(1, 2)))


def _modify_use_xlutils():
    """
    使用xlutils修改excel
    :return:
    """
    def avg(list):
        return sum(list)//len(list)

    wb = xlrd.open_workbook("./data/test.xls", formatting_info=True)    # formatting_info 为 True 表示保留原格式
    wbc = copy(wb)  # 复制
    sh = wb.sheet_by_index(0)
    age_list = sh.col_values(2)
    age_list = age_list[1:len(age_list) - 1]
    avg_age = avg(age_list)
    sh = wbc.get_sheet(0)
    alm = xlwt.Alignment()      # 设置左对齐
    alm.horz = 0x01
    style = xlwt.XFStyle()
    style.alignment = alm
    sh.write(5, 2, avg_age, style)
    wbc.save('test.xls')

_modify_use_xlutils()