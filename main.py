import tensorflow as tf
import cv2
import numpy as np
import multiprocessing as _mp
from utils import load_graph, detect_hands, predict, is_in_triangle, ORANGE, RED, GREEN, CYAN, BLUE
from pyKey import pressKey, releaseKey, press #inspired by https://github.com/andohuman/pyKey for controlling keyboard keys
import keyboard

width = 640
height = 480
threshold = 0.6
alpha = 0.3
pre_trained_model_path = "model/pretrained_model.pb"

def mario_main():
    graph, sess = load_graph(pre_trained_model_path)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    mp = _mp.get_context("spawn")
    v = mp.Value('i', 0)
    lock = mp.Lock()

    while True:
        key = cv2.waitKey(10)
        if key == ord("q"):
            break
        _, frame = cap.read()
        
        frame = cv2.flip(frame, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, scores, classes = detect_hands(frame, graph, sess)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        results = predict(boxes, scores, classes, threshold, width, height)

        if len(results) == 1:
            x_min, x_max, y_min, y_max, category = results[0]
            x = int((x_min + x_max) / 2)
            y = int((y_min + y_max) / 2)
            cv2.circle(frame, (x, y), 5, RED, -1)

            if category == "Open" and x <= width / 3:
                action = 7  # Left jump
                text = "Jump left"
                releaseKey("LEFT")
                press('LEFT', 0.15)
                pressKey("UP")
                
            elif category == "Closed" and x <= width / 3:
                action = 6  # Left
                text = "Run left"
                releaseKey('UP')
                pressKey("LEFT")
                
            elif category == "Open" and width / 3 < x <= 2 * width / 3:
                action = 5  # Jump
                releaseKey('LEFT')
                releaseKey("RIGHT")
                pressKey("UP")
                text = "Jump"
               
            elif category == "Closed" and width / 3 < x <= 2 * width / 3:
                action = 0  # Do nothing
                releaseKey('LEFT')
                releaseKey("RIGHT")
                releaseKey('UP')
                keyboard.press_and_release('shift')
                text = "Stay"
                
            elif category == "Open" and x > 2 * width / 3:
                action = 2  # Right jump
                text = "Jump right"
                releaseKey("RIGHT")
                press("RIGHT", 0.15)
                pressKey('UP')
                
            elif category == "Closed" and x > 2 * width / 3:
                action = 1  # Right
                text = "Run right"
                releaseKey("UP")
                pressKey(key = 'RIGHT')
                
            else:
                action = 0
                text = "Stay"
            
            with lock:
                v.value = action
            
            cv2.putText(frame, "{}".format(text), (x_min, y_min - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2)
               
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (int(width / 3), height), ORANGE, -1)
        cv2.rectangle(overlay, (int(2 * width / 3), 0), (width, height), ORANGE, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.imshow('Detection', frame)

    cap.release()
    cv2.destroyAllWindows()
    pass

def dinosaur_main():
    # Your Dinosaur game code here
    graph, sess = load_graph(pre_trained_model_path)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    mp = _mp.get_context("spawn")
    v = mp.Value('i', 0)
    lock = mp.Lock()

    while True:
        key = cv2.waitKey(10)
        if key == ord("q"):
            break
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, scores, classes = detect_hands(frame, graph, sess)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        results = predict(boxes, scores, classes, threshold, width, height)

        if len(results) == 1:
            x_min, x_max, y_min, y_max, category = results[0]
            x = int((x_min + x_max) / 2)
            y = int((y_min + y_max) / 2)
            cv2.circle(frame, (x, y), 5, RED, -1)
            
            if category == "Closed":
                action = 0  # Do nothing
                text = "Run"
            
            elif category == "Open" and y < height/2:
                action = 1 # Jump
                pressKey('UP')
                text = "Jump"
            
            elif category == "Open" and y > height/2:
                action = 2
                pressKey('DOWN')
                text = "Duck"
       
            with lock:
                v.value = action
            cv2.putText(frame, "{}".format(text), (x_min, y_min - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2)
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, int(height / 2)), ORANGE, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.imshow('Detection', frame)

    cap.release()
    cv2.destroyAllWindows()
    pass

def temple_run_main():
    graph, sess = load_graph(pre_trained_model_path)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    mp = _mp.get_context("spawn")
    v = mp.Value('i', 0)
    lock = mp.Lock()

    x_center = int(width / 2)
    y_center = int(height / 2)
    radius = int(min(width, height) / 12)
    while True:
        key = cv2.waitKey(10)
        if key == ord("q"):
            break
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, scores, classes = detect_hands(frame, graph, sess)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        results = predict(boxes, scores, classes, threshold, width, height)
        if len(results) == 1:
            x_min, x_max, y_min, y_max, category = results[0]
            x = int((x_min + x_max) / 2)
            y = int((y_min + y_max) / 2)
            cv2.circle(frame, (x, y), 5, RED, -1)
            
            if category == "Open" and np.linalg.norm((x - x_center, y - y_center)) <= radius:
                action = 0 # Stay
                text = "Stay"
            
            elif category == "Open" and is_in_triangle((x, y), [(0, 0), (width, 0),(x_center, y_center)]):
                action = 1  # Up
                text = "Up"
                keyboard.press_and_release('up')
                
                
            elif category == "Open" and is_in_triangle((x, y), [(0, height), (width, height), (x_center, y_center)]):
                action = 2  # Down
                text = "Down"
                keyboard.press_and_release('down')
                
            elif category == "Open" and is_in_triangle((x, y), [(0, 0), (0, height), (x_center, y_center)]):
                action = 3  # Left
                text = "Left"
                keyboard.press_and_release('left')
                
            elif category == "Open" and is_in_triangle((x, y), [(width, 0), (width, height), (x_center, y_center)]):
                action = 4  # Right
                text = "Right"
                keyboard.press_and_release('right')
            
            else:
                action = 0
                text = "Stay"
                
            with lock:
                v.value = action
            cv2.putText(frame, "{}".format(text), (x_min, y_min - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2)

        overlay = frame.copy()
        cv2.drawContours(overlay, [np.array([(0, 0), (width, 0), (x_center, y_center)])], 0, CYAN, -1)
        cv2.drawContours(overlay, [np.array([(0, height), (width, height), (x_center, y_center)])], 0, CYAN, -1)
        cv2.drawContours(overlay, [np.array([(0, 0), (0, height), (x_center, y_center)])], 0, (0, 255, 255), -1)
        cv2.drawContours(overlay, [np.array([(width, 0), (width, height), (x_center, y_center)])], 0, (0, 255, 255), -1)
        cv2.circle(overlay, (x_center, y_center), radius, BLUE, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.imshow('Detection', frame)

    cap.release()
    cv2.destroyAllWindows()

    # Your Temple Run game code here
    pass

def car_racing_main():
    # Your Car Racing game code here
    
    graph, sess = load_graph(pre_trained_model_path)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    mp = _mp.get_context("spawn")
    v = mp.Value('i', 0)
    lock = mp.Lock()
    
    while True:
        key = cv2.waitKey(10)
        if key == ord("q"):
            break
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, scores, classes = detect_hands(frame, graph, sess)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        results = predict(boxes, scores, classes, threshold, width, height)

        if len(results) == 1:
            x_min, x_max, y_min, y_max, category = results[0]
            x = int((x_min + x_max) / 2)
            y = int((y_min + y_max) / 2)
            cv2.circle(frame, (x, y), 5, RED, -1)

            if category == "Open" and x <= width / 3:
                action = 7  # Left move
                text = "Left move"
                releaseKey("LEFT")
                press('LEFT', 0.15)
                pressKey("UP")
            
            elif category == "Open" and width / 3 < x <= 2 * width / 3:
                action = 5  # Straight
                releaseKey('LEFT')
                releaseKey("RIGHT")
                pressKey("UP")
                text = "Straight"
            
            elif category == "Open" and x > 2 * width / 3:
                action = 2  # Right move
                text = "Right move"
                releaseKey("RIGHT")
                press("RIGHT", 0.15)
                pressKey('UP')
                
            else:
                action = 0
                text = "Go"
            
            with lock:
                v.value = action
            cv2.putText(frame, "{}".format(text), (x_min, y_min - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2)
            
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (int(width / 3), height), ORANGE, -1)
        cv2.rectangle(overlay, (int(2 * width / 3), 0), (width, height), ORANGE, -1)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.imshow('Detection', frame)
        
    cap.release()
    cv2.destroyAllWindows()
    pass

def main():
    while True:
        print("Choose a game to play:")
        print("1. Mario")
        print("2. Dinosaur")
        print("3. Temple Run")
        print("4. Car Racing")
        print("Press 'q' to quit")
        
        choice = input("Enter your choice: ")

        if choice == '1':
            print("Starting Mario...")
            mario_main()
        elif choice == '2':
            print("Starting Dinosaur...")
            dinosaur_main()
        elif choice == '3':
            print("Starting Temple Run...")
            temple_run_main()
        elif choice == '4':
            print("Starting Car Racing...")
            car_racing_main()
        elif choice == 'q':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please enter a valid option.")

if __name__ == '__main__':
    main()