"""
        SMTP 全称 Simple Mail Transfer Protocol，中文译为简单邮件传输协议，它能跨越网络传输邮件，
    可实现相同网络处理进程之间的邮件传输，也可通过中继器或网关实现进程与其他网络之间的邮件传输。
    Python 的 smtplib 模块对 SMTP 做了封装，可以很方便的实现邮件的发送，同时提供了 email 模块来构造邮件。
        POP3 全称 Post Office Protocol - Version 3，3 是版本号，中文译为邮局协议，该协议可以实现邮件的收取，
    Python 的 poplib 模块实现了该协议。

    构建SMTP对象:
        smtplib.SMTP(host='', port=0, local_hostname=None, [timeout, ]source_address=None)
            host：SMTP服务器主机。
            port：端口号。
            local_hostname：如果 SMTP 在本机，只需指定服务器地址为 localhost 即可。
            timeout：指定超时时间，可选。
            source_address：该参数允许绑定到具有多个网络接口的计算机中的某些特定源地址或某些特定源 TCP 端口。
    发送邮件:
        SMTP.sendmail(from_addr, to_addrs, msg, mail_options=(), rcpt_options=())
            from_addr：邮件发送者地址。
            to_addrs：邮件接收者地址。
            msg：邮件内容。

    收取邮件:
        实现pop3的类
        poplib.POP3(host, port=POP3_PORT[, timeout])
            host：主机。
            port：端口号
            timeout：可选参数，超时时间。

        POP3 通过 SSL 加密套接字连接到服务器的子类,自版本 3.6 起不推荐使用 keyfile 和 certfile，而推荐使用 context。
        poplib.POP3_SSL(host, port=POP3_SSL_PORT, keyfile=None, certfile=None, timeout=None, context=None)

"""
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage


import poplib
from email.parser import Parser
from email.header import decode_header,Header
from email.utils import parseaddr


def simple_mail():
    """
    简单邮件发送
    :return:
    """
    senderMail = "1845543526@qq.com"  # 发送者邮箱地址
    anthCode = "zohyelzbyrezcfjf"  # 授权码
    receiverMail = "sssuperzyn@gmail.com"  # 接收者邮箱地址
    subject = "简单邮件"  # 邮件主题
    content = "Hi,What do you do?"  # 邮件内容
    msg = MIMEText(content, 'plain', 'utf-8')
    msg["Subject"] = subject
    msg["From"] = senderMail
    msg["To"] = receiverMail
    try:
        server = smtplib.SMTP_SSL('smtp.qq.com', smtplib.SMTP_SSL_PORT)
        print("成功连接到邮件服务器")
        server.login(senderMail, anthCode)
        print("成功登录")
        server.sendmail(senderMail, receiverMail, msg.as_string())
        print("邮件发送成功")
    except smtplib.SMTPException as e:
        print("邮件发送异常")
    finally:
        server.quit()


def complex_mail():
    """
    复杂邮件发送
    :return:
    """
    senderMail = "1845543526@qq.com"  # 发送者邮箱地址
    anthCode = "zohyelzbyrezcfjf"  # 授权码
    receiverMail = "sssuperzyn@gmail.com"  # 接收者邮箱地址
    subject = "复杂邮件测试"
    msgRoot = MIMEMultipart('related')
    msgRoot['Subject'] = subject
    msgRoot['From'] = senderMail
    msgRoot['To'] = receiverMail
    msgAtv = MIMEMultipart('alternative')
    msgRoot.attach(msgAtv)  # 附加内容
    # html
    html_content = """
        <h1>复杂邮件测试</h1>
        <p>我的博客地址：</p>
        <p><a href='https://blog.csdn.net/ityard'>点击进入我的CSDN</a></p>
        <p><img src="cid:image"></p>
    """
    html = MIMEText(html_content, 'html', 'utf-8')
    msgAtv.attach(html)
    with open("./data/demo.jpg", 'rb') as f:
        msgImage = MIMEImage(f.read())
    msgImage.add_header('Content-ID', '<image>')
    msgRoot.attach(msgImage)
    # 添加附件
    annex = MIMEText(open('./data/test.txt', 'rb').read(), 'base64', 'utf-8')
    annex['Content-Type'] = 'application/octet-stream'
    annex['Content-Disposition'] = 'attachment; filename="test.txt"'
    msgRoot.attach(annex)
    try:
        server = smtplib.SMTP_SSL('smtp.qq.com', smtplib.SMTP_SSL_PORT)
        print("成功连接到邮件服务器")
        server.login(senderMail, anthCode)
        print("成功登录")
        server.sendmail(senderMail, receiverMail, msgRoot.as_string())
        print("邮件发送成功")
    except smtplib.SMTPException as e:
        print("邮件发送异常")
    finally:
        server.quit()


def gather_mail():
    """
        使用pop3协议接收邮件
    :return:
    """
    def print_email(msg):
        """
        打印邮件内容
        :param msg:已解析的邮件内容
        :return:
        """
        for header in ['From', 'To', 'Subject']:
            value = msg.get(header, '')
            if value:
                if header == 'Subject':
                    value = decode_str(value)
                else:
                    hdr, addr = parseaddr(value)
                    name = decode_str(hdr)
                    value = u'%s <%s>' % (name, addr)
            print('%s: %s' % (header, value))
            # 获取邮件主体信息
        attachment_files = []
        for part in msg.walk():
            # 获取附件名称类型
            file_name = part.get_filename()
            # 获取数据类型
            contentType = part.get_content_type()
            # 获取编码格式
            mycode = part.get_content_charset()
            if file_name:
                h = Header(file_name)
                # 对附件名称进行解码
                dh = decode_header(h)
                filename = dh[0][0]
                if dh[0][1]:
                    # 将附件名称可读化
                    filename = decode_str(str(filename, dh[0][1]))
                attachment_files.append(filename)
                # 下载附件
                data = part.get_payload(decode=True)
                # 在当前目录下创建文件
                with open(filename, 'wb') as f:
                    # 保存附件
                    f.write(data)
            elif contentType == 'text/plain':
                data = part.get_payload(decode=True)
                content = data.decode(mycode)
                print('正文：', content)
            elif contentType == 'text/html':
                data = part.get_payload(decode=True)
                content = data.decode(mycode)
                print('正文：', content)
        print('附件名列表：', attachment_files)

    def decode_str(s):
        value, charset = decode_header(s)[0]
        if charset:
            value = value.decode(charset)
        return value

    receiverMail = "sssuperzyn@gmail.com"
    authCode = ""   # 接收者邮箱授权码
    pop3_server = "pop.qq.com"  # pop3服务器
    server = poplib.POP3_SSL(pop3_server, 995)
    server.user(receiverMail)   # 身份验证
    server.pass_(authCode)
    print('邮件数量:%s  占用空间:%s' % server.stat())   # 返回邮件数量和占用空间
    resp, mails, octets = server.list() # list() 返回所有邮件的编号，
    index = len(mails)
    resp, lines, octets = server.retr(index)    # 获取一封最新的邮件
    msg_content = b'\r\n'.join(lines).decode('utf-8')   # lines存储了邮件的原始文本的每一行
    msg = Parser().parsestr(msg_content)    # 解析邮件
    print_email(msg)    # 打印邮件内容
    server.dele(1)  # 根据索引号直接从服务器删除邮件
    server.quit()   # 关闭与服务器的连接