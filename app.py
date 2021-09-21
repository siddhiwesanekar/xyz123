import socket
import threading


from flask import Flask, render_template, request
from test import response1

app = Flask(__name__, template_folder='./templates')

app.static_folder = './static'
app.temp_dict = {}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    msg5 = response1(userText)
    if userText.isnumeric():
        if int(userText) in app.temp_dict.keys():
            userText = app.temp_dict[int(userText)]
            userText = ''.join(userText)

    if 'form' in msg5:
        import webbrowser

        webbrowser.open("D:/PLM Nordic/UOM_request_form.docx")
    if '/' in msg5:
        if 'open' in userText:
            import webbrowser

            webbrowser.open(msg5)

            return "File opened successfully!"


        if 'delete' in userText:
            import os
            if os.path.exists(msg5):
                os.remove(msg5)
                msg5 = "File Deleted Successfully!"
                return msg5

            else:
                msg5 = "File Dose Not Exist!"
                return msg5
    if (type(msg5) is list) == True:


        res = [''.join(ele) for ele in msg5]

        temp_num = 0
        for i in res[1:]:
            temp_num = temp_num + 1
            app.temp_dict[temp_num] = []

            app.temp_dict[temp_num].append(str(i))

        return app.temp_dict



    return str(response1(userText))

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=80)
