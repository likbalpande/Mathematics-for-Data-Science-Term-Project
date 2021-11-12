#To reconstruct the image using imageio
import imageio
from numpy.core.fromnumeric import amin
x_red = npy.load('ans_x_red.npy')
x_green = npy.load('ans_x_green.npy')
x_blue = npy.load('ans_x_blue.npy')
x_red = x_red.reshape((100, 100))
x_green = x_green.reshape((100, 100))
x_blue = x_blue.reshape((100, 100))
oldRange_red = npy.amax(x_red)-npy.amin(x_red)
newRange_red = 255
oldRange_green = npy.amax(x_green)-npy.amin(x_green)
newRange_green = 255
oldRange_blue = npy.amax(x_blue)-npy.amin(x_blue)
newRange_blue = 255
# Creating(x,y,z)
x = (((x_red-npy.amin(x_red))*newRange_red)/oldRange_red)+0
x = x.astype(npy.uint8)
y = (((x_green-npy.amin(x_green))*newRange_green)/oldRange_green)+0
y = y.astype(npy.uint8)
z = (((x_blue-npy.amin(x_blue))*newRange_blue)/oldRange_blue)+0
z = z.astype(npy.uint8)
final = np.zeros((100,100,3))
final[:,:,0] = x.T
final[:,:,1] = y.T
final[:,:,2] = z.T
imageio.imwrite("Complete_image.png", final)
#done
