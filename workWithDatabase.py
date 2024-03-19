import mysql.connector

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from emailForm import form

class DatabaseConnector:
    def __init__(self) -> None:
        # Kết nối đến cơ sở dữ liệu
        self.conn = mysql.connector.connect(
            host="localhost",
            user="root",  
            password=None,  
            database="VietNamVehicle" 
        )
        # Kết nối đến cơ sở dữ liệu
        self.conn = mysql.connector.connect(
            host="localhost",
            user="root",  
            password=None,  
            database="VietNamVehicle" 
        )
        self.plateNumber = "";
        # Thông tin SMTP server và thông tin đăng nhập
        self.smtp_server = 'smtp.office365.com'
        self.smtp_port = 587  # Cổng SMTP, thường là 587 hoặc 465
        self.smtp_username = 'hoangtrhien203@gmail.com'
        self.smtp_password = '22022003Hth$$'
        
        
    def queryPlate(self, plateFileName):
        with open(plateFileName, 'r') as f:
            self.plateNumber = f.read().strip();
        if (self.plateNumber):
            # Tạo một đối tượng cursor để thực thi các truy vấn SQL
            cursor = self.conn.cursor()
            # Lấy dữ liệu từ bảng
            cursor.execute("SELECT * FROM Motobike Where BienSoXe = '" + self.plateNumber + "';")
            rows = cursor.fetchall()
            # Commit các thay đổi và đóng kết nối
            self.conn.commit()
            self.conn.close()
            if(rows):
                print(rows[0])
                server = smtplib.SMTP(self.smtp_server, 587)
                server.starttls()
                server.login(self.smtp_username, self.smtp_password)
                email = rows[0][10] 
                # thay đổi content trong form email được import từ emailForm
                html = form.replace("&biensoxe", rows[0][0])
                html = html.replace("&chuxe", rows[0][3])
                message = MIMEMultipart('html')
                message['From'] = "hoangtrhien203@gmail.com"
                message['To'] = email
                message['Subject'] = "Email thông báo vi phạm"
                message.attach(MIMEText(html, 'html', 'utf-8'))
                with open('./VehicleImageData/' + rows[0][9], 'rb') as attachment:
                    image_part = MIMEImage(attachment.read(), name='Anh-Vi-Pham.jpg')
                    message.attach(image_part)
                server.sendmail("hoangtrhien203@gmail.com", email, message.as_string())
                server.quit()










