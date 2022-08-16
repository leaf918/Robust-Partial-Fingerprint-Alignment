
import torchvision
import torchsummary

# torchsummary(torchvision.models.mobilenet_v2())
torchsummary.summary(torchvision.models.mobilenet_v2(),
                     input_size=(3, 128, 128))
