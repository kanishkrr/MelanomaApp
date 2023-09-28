import os
import flask
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image

class NeuralNet(nn.Module):
    
    def __init__(self):
        super(NeuralNet, self).__init__() 
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(11, 11), padding=4, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2, dilation=1)
        )
            
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2, dilation=1)
        )
            
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2, dilation=1)
        )
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=2, dilation=1)
        )
        
        self.flat = nn.Flatten(1)
        self.linear1 = nn.Linear(256 * 5 * 5, 4000)
        self.drop1 = nn.Dropout(p=0.5, inplace=False)
        self.linear2 = nn.Linear(4000, 2000)
        self.drop2 = nn.Dropout(p=0.5, inplace=False)
        self.linear3 = nn.Linear(2000, 2)
        
            
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.flat(x)
        x = self.linear1(x)
        x = self.drop1(x)
        x = self.linear2(x)
        x = self.drop2(x)
        x = self.linear3(x)
        return x 
    
model = NeuralNet()
model.load_state_dict(torch.load('/Users/kanishk/Downloads/vscode/python/machine learning/Melanoma/model.pt'))

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = '/Users/kanishk/Downloads/vscode/python/machine learning/Melanoma/uploads'

transform = transforms.Compose([transforms.Resize((200, 200)), transforms.ToTensor()])

print('running')


@app.route("/", methods=["GET", "POST"])
def upload_image():

    uploaded_filename = None

    print('got')
    

    if request.method == "POST":

        file = request.files["file"]

        uploaded_filename = file.filename

        file.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename)))

        img = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename)))
        
        image = transform(img)

        yhat = model(image.unsqueeze(0))

        _, label = torch.max(yhat, 1)
        
        if label.item() == 0:

            return redirect(url_for('negative'))
        
        elif label.item() == 1:

            return redirect(url_for('positive'))

    else:
        return flask.render_template("index.html", uploaded_filename=uploaded_filename)
    

@app.route('/negative')
def negative():
    return flask.render_template('negative.html')


@app.route('/positive')
def positive():
    return flask.render_template('positive.html')



if __name__ == "__main__":
    app.run(debug=True)