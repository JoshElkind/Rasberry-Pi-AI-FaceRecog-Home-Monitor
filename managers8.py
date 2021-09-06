import os, time

var = 0

'''file = open("m_store.txt", "w")
file.write("0")
file.close()
time.sleep(10)
while True:
    
    file = open("m_store.txt", "w")
    file.write("1")
    file.close()
    connections_8081 = os.system('netstat -tn src :8081')
    print(connections_8081)
    if connections_8081 != 0:
        time.sleep(5)
    else:
        break
file = open("m_store.txt", "w")
file.write("0")
file.close()
os.system('sudo systemctl stop motion')'''
file = open("/home/pi/VSCode/Python/m_store.txt", "w")
file.write("0")
file.close()

while True:
    
    file = open("/home/pi/VSCode/Python/m_store.txt", "w")
    file.write("1")
    file.close()
    time.sleep(1)
    var += 1
    if var == 32:
       break 

os.system('sudo systemctl stop motion')

file = open("/home/pi/VSCode/Python/m_store.txt", "w")
file.write("0")
file.close()
