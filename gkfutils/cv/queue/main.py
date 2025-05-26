import multiprocessing
import time
import cv2

def put_data(queue):
    # for i in range(5):
    #     print(f"Putting {i} into the queue.")
    #     queue.put(i)
    #     # time.sleep(1)
    i= 0
    while True:
        print(f"Putting {i} into the queue.")
        queue.put(i)
        i += 1



def get_data(queue):
    while True:
        if not queue.empty():
            item = queue.get()
            print(f"Getting {item} from the queue.")


def main1():
    queue = multiprocessing.Queue()

    # 创建两个进程
    p1 = multiprocessing.Process(target=put_data, args=(queue,))
    p2 = multiprocessing.Process(target=get_data, args=(queue,))

    # 启动进程
    p1.start()
    p2.start()

    # 等待进程结束
    p1.join()
    p2.join()

    print("All processes have finished.")


# ----------------------------------------------------------------
def simulate_camera(q):
    img_path = r"G:\Gosion\data\006.Belt_Torn_Det\data\ningmei\data51\20250513\src\1747102683.874673.jpg"
    img = cv2.imread(img_path)
    while True:
        q.put(img)


def simulate_detector(q1, q2):
    while True:
        if not q1.empty():
            img = q1.get()
            cv2.rectangle(img, (100, 100), (200, 200), (0, 255, 0), 2)
            q2.put(img)
            print("Detector: Image processed and sent to the next process.")
            cv2.imwrite(r"G:\Gosion\data\006.Belt_Torn_Det\data\ningmei\others\testq.jpg", img)
            time.sleep(100)


def main2():
    q1 = multiprocessing.Queue()
    q2 = multiprocessing.Queue()

    

    # 创建两个进程
    p1 = multiprocessing.Process(target=simulate_camera, args=(q1,))
    p2 = multiprocessing.Process(target=simulate_detector, args=(q1, q2,))

    # 启动进程
    p1.start()
    p2.start()

    # 等待进程结束
    p1.join()
    p2.join()

    print("All processes have finished.")




if __name__ == "__main__":
    # main1()
    main2()





    