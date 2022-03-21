import os
import subprocess

"""
p = subprocess.run('ls -la', shell = True)
print(p)

p1 = subprocess.run(['ls', '-la'])
print(p1)
print(p1.args)
print(p1.returncode)


p2 = subprocess.run(['ls', '-la'], capture_output=True, text=True)
print(p2.stdout)
"""

p3= subprocess.Popen("ls", cwd="/")
print(p3)

subprocess.Popen("ls", cwd="/")

"""
os.chdir('../../../darknet')

darknet_path = '../../../darknet'
sonuc = os.listdir()
print(sonuc)


./darknet detector train ../opt/project/resources/train-datasett/datasett/class.data  ../opt/project/resources/train-datasett/yolov3.cfg  ../opt/project/resources/train-datasett/darknet53.conv.74 -dont_show

../opt/project/resources/train-datasett/class.data


"""
