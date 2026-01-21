from multiprocessing import Process, Queue
from ui import start_gui
from Agent.phyplan_vis_launcher import start_training

queue = Queue()
train_proc = Process(target=start_training, args=(queue,))
train_proc.start()

start_gui(queue, train_proc)

train_proc.join()