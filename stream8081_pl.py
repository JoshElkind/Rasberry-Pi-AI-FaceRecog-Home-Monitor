import http.server as hs
import socketserver
import sys
import os
import time

url = "http://raspberrypi.local:8081"
file_object = open('/test_sh.txt', 'a')
# Append 'hello' at the end of file
file_object.write('--4--')
# Close the file
file_object.close()

class RedirectHandler(hs.SimpleHTTPRequestHandler):
    print("ping")
    def do_GET(self):
        if os.system("ps -ef | grep -v grep | grep motion | wc -l") == 0:
            os.system('sudo systemctl start motion')
            print("ps -ef entered")
        print(os.system("ps -ef | grep -v grep | grep motion | wc -l"))
        print("connection")
        time.sleep(2)
        self.send_response(301)
        self.send_header('Location', url)
        self.end_headers()
        time.sleep(3)
        file_read = open("/home/pi/VSCode/Python/m_store.txt", "r")
        file_content = file_read.read()
        file_read.close()
        print(file_content)
        if file_content == '' or int(file_content) == 0:
            os.system("sudo python3 /home/pi/VSCode/Python/managers8.py")
            


theport = 81
Handler = RedirectHandler
pywebserver = socketserver.TCPServer(("", theport), Handler)
pywebserver.serve_forever()
