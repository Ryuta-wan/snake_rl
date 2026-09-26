import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os


class Linear_QNet(nn.Module):
    """
    Neural Network ง่ายๆ 2 ชั้น ทำหน้าที่เป็น 'สมอง' ของ agent
    รับ state (11 ค่า) เข้ามา แล้วทำนายว่า action ไหน (ตรง/ขวา/ซ้าย)
    น่าจะให้ผลตอบแทน (Q-value) สูงที่สุด
    """

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # ชั้นที่ 1: จาก input (11 ค่า) -> hidden layer
        self.linear1 = nn.Linear(input_size, hidden_size)
        # ชั้นที่ 2: จาก hidden layer -> output (3 ค่า = 3 action)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # ข้อมูลไหลผ่านชั้นที่ 1 แล้วผ่าน activation function (ReLU)
        x = F.relu(self.linear1(x))
        # ผ่านชั้นที่ 2 ออกมาเป็นค่า Q ของแต่ละ action (ไม่ต้องมี activation ตรงนี้)
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        # บันทึกน้ำหนักของโมเดลลงไฟล์ ไว้เรียกกลับมาใช้ทีหลังโดยไม่ต้องเทรนใหม่
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)
        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

    def load(self, file_name='model.pth'):
        """Load model weights saved by ``save``."""
        # This file is a local checkpoint created by this project. PyTorch 2.6
        # defaults to weights_only=True, which rejects older serialized files.
        self.load_state_dict(torch.load(
            file_name, map_location='cpu', weights_only=False
        ))


class QTrainer:
    """
    ตัวจัดการการเทรนโมเดล คำนวณ loss และอัปเดตน้ำหนักของ neural network
    ตามสูตร Bellman equation
    """

    def __init__(self, model, lr, gamma):
        self.lr = lr          # learning rate: ปรับน้ำหนักทีละเท่าไหร่ในแต่ละรอบ
        self.gamma = gamma    # discount rate: ให้ความสำคัญกับ reward ในอนาคตแค่ไหน
        self.model = model
        # optimizer: ตัวช่วยปรับน้ำหนักของโมเดลให้ loss ต่ำลงเรื่อยๆ
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        # loss function: วัดว่าโมเดลทำนายผิดไปแค่ไหน
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        # แปลงข้อมูลให้เป็น tensor (รูปแบบที่ PyTorch ใช้งานได้)
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        # ถ้าข้อมูลมาแค่ 1 ชุด (ไม่ใช่ batch) ให้เพิ่มมิติเข้าไปให้ตรงรูปแบบ
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1. ทำนายค่า Q จาก state ปัจจุบัน
        pred = self.model(state)

        # 2. คำนวณค่า Q ใหม่ตามสูตร Bellman equation:
        #    Q_new = reward + gamma * max(Q(next_state))   ถ้ายังไม่จบเกม
        #    Q_new = reward                                  ถ้าจบเกมแล้ว
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action[idx]).item()] = Q_new

        # 3. คำนวณ loss ระหว่างค่าที่ทำนายได้ (pred) กับค่าเป้าหมาย (target)
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        # 4. Backpropagation: ปรับน้ำหนักของโมเดลให้ loss ลดลง
        loss.backward()
        self.optimizer.step()