# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.pyplot import draw
# # from IPython.display import clear_output
#
# import time
#
# fig = plt.figure()
# im = plt.imshow(np.random.random((16, 512)), aspect='auto')
# for i in range(5):
#     im.set_data(np.random.random((16, 512)))
#     plt.show()
#     # clear_output(wait=True)
#     plt.clf()
#     time.sleep(.05)

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True
fig, ax = plt.subplots()
image = np.random.rand(16, 512)
img = ax.imshow(image, cmap='jet', aspect='auto')
axcolor = 'yellow'
ax_slider = plt.axes([0.20, 0.01, 0.65, 0.03], facecolor=axcolor)
slider = Slider(ax_slider, 'Slide->', 0.1, 30.0, valinit=2)
def update(val):
   ax.imshow(np.random.rand(16, 512),cmap='jet', aspect='auto')
   fig.canvas.draw_idle()
slider.on_changed(update)
plt.show()