# blender --background --python  /home/lch/Downloads/bpy-visualization-utils-master/render_binvox.py --
# --binvox /home/lch/Downloads/3DSVT/output/real/chair14.binvox --output /home/lch/Downloads/3DSVT/output/real/chair14.png

#run command format
import os
import threading
 

cmd = " blender --background --python  /home/lch/Downloads/bpy-visualization-utils-master/render_binvox.py -- --binvox /home/lch/Downloads/3DSVT/output/real/chair14.binvox --output /home/lch/Downloads/3DSVT/output/real/chair14.png"
lock=threading.Lock()
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QFileDialog
folder_path = None
class ExampleApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
 
    def initUI(self):
        # 创建一个按钮
        self.button = QPushButton('选择文件夹', self)
        self.button.clicked.connect(self.openFolder)  # 当按钮被点击时，连接到openFolder方法
 
        # 设置窗口布局
        layout = QVBoxLayout()
        layout.addWidget(self.button)
        self.setLayout(layout)
 
        # 设置窗口大小和标题
        self.setGeometry(300, 300, 300, 150)
        self.setWindowTitle('点击按钮选择文件夹')
        self.show()
 
    def openFolder(self):
        # 使用QFileDialog打开文件夹选择对话框
        folder_path = QFileDialog.getExistingDirectory(self, '选择文件夹', '/')
        return folder_path
def voxvisual(folder_path):
    file_list = []    
    if(folder_path != None):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                file_list.append(file_path)
        for ele in file_list:
            if ele.endswith(".binvox"):
                print(ele) 
                global cmd
                with lock:    
                    cmd = " blender --background --python  /home/lch/Downloads/bpy-visualization-utils-master/render_binvox.py"+" -- --" "binvox "+ele +" --output " +ele+".png"
                #print(cmd)
                os.system(cmd)
        folder_path = None    
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ExampleApp()
    folder_path = ex.openFolder()
    threads = []
    for _ in range(6):
        t = threading.Thread(target=voxvisual(folder_path))
        t.start()
        threads.append(t)
for t in threads:
    t.join()        
    
    sys.exit(app.exec_()) 
 
    
# 调用函数

