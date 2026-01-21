from multiprocessing import Queue
import sys
import ui_bridge

def start_training(queue):
    def update_label(label_name, data):
        queue.put((label_name, data))
    
    ui_bridge.update_label = update_label
    
    script_args = sys.argv[1:]
    sys.argv = ["gflownet_tool_selector_specific_perm_joint.py"] + script_args + ["--vis"] + ["True"]
    
    global_vars = {
        "__name__": "__main__"
    }
    with open("gflownet_tool_selector_specific_perm_joint.py", "rb") as f:
        code = compile(f.read(), "train.py", "exec")
        exec(code, global_vars)