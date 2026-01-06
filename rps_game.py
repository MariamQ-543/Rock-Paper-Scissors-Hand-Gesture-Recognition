# ============================================
# ROCK–PAPER–SCISSORS GAME 
# ============================================

# https://www.geeksforgeeks.org/python/python-opencv-capture-video-from-camera/
# This tutorial helped me understand basic OpenCV VideoCapture concepts:
# - cv2.VideoCapture(0) for camera access
# - cap.read() for frame capture
# - cv2.imshow() for display
# - cv2.waitKey() for key input handling

# https://medium.com/beyondlabsey/creating-a-simple-region-of-interest-roi-inside-a-video-streaming-using-opencv-in-python-30fd671350e0
# Used to understand ROI rectangle creation and frame cropping for focused hand gesture detection

# https://labex.io/tutorials/python-how-to-use-counter-for-frequency-424183
# I used this tutorial to understand Python's Counter class for frequency analysis.
# The Smart AI tracks player move history and uses Counter.most_common() to identify the most frequently played gesture in the last 10 moves. This allows the AI to predict player patterns and play strategic counter-moves, demonstrating opponent modeling beyond basic image classification.

# Using OpenCV for webcam capture and real-time gesture recognition
# ROI (Region of Interest) approach for focused hand detection
# This game allows testing all three trained models
# It captures hand gestures from webcam in real-time
# The smart ai learns from your playing patterns and adapts its strategy
# I implemented this to demonstrate the models working in a real application

import cv2
import numpy as np
import time
import random
from collections import Counter

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ============================================
# CONFIGURATION
# ============================================
# these settings match exactly what i used during training

# class order must exactly match training (alphabetical order)
# if i change this order the predictions will be wrong
CLASS_NAMES = ['None', 'Paper', 'Rock', 'Scissors']

# only valid game gestures (exclude none)
# none is used in training but not valid for playing the game
GAME_GESTURES = ['Rock', 'Paper', 'Scissors']

# region of interest (roi) box coordinates
# this defines where the player should place their hand
# only this area is sent to the model for prediction
# helps reduce background noise and focus on the hand
ROI_X1, ROI_Y1 = 200, 100  # top-left corner
ROI_X2, ROI_Y2 = 440, 340  # bottom-right corner

# ============================================
# MODEL SELECTION MENU
# ============================================

def select_model():
    """
    Displays menu for user to select which model to use for the game
    
    Returns:
        tuple: (model, model_name, use_preprocess_input)
    """
    print("=" * 60)
    print("ROCK-PAPER-SCISSORS GAME - MODEL SELECTION")
    print("=" * 60)
    print("\nSelect which model to use:")
    print("1. Custom CNN (85.51% accuracy)")
    print("2. MobileNetV2 Frozen (97.10% accuracy)")
    print("3. MobileNetV2 Fine-tuned (97.10% accuracy)")
    print("=" * 60)
    
    while True:
        choice = input("\nEnter your choice (1/2/3): ").strip()
        
        if choice == '1':
            print("\nLoading Custom CNN model...")
            model = load_model("Models/cnn_model.h5")
            model_name = "Custom CNN"
            use_preprocess = False
            print("Custom CNN loaded successfully")
            break
        elif choice == '2':
            print("\nLoading MobileNetV2 (Frozen) model...")
            model = load_model("Models/transfer_model_frozen.h5")
            model_name = "MobileNetV2 (Frozen)"
            use_preprocess = True
            print("MobileNetV2 (Frozen) loaded successfully")
            break
        elif choice == '3':
            print("\nLoading MobileNetV2 (Fine-tuned) model...")
            model = load_model("Models/transfer_model_finetuned.h5")
            model_name = "MobileNetV2 (Fine-tuned)"
            use_preprocess = True
            print("MobileNetV2 (Fine-tuned) loaded successfully")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")
    
    print("=" * 60)
    return model, model_name, use_preprocess

# ============================================
# GAME LOGIC
# ============================================

def determine_winner(player, ai):
    """
    determines the winner of a rock-paper-scissors round
    
    rock beats scissors
    paper beats rock
    scissors beats paper
    same gesture = draw
    
    args:
        player: what the player played (string)
        ai: what the ai played (string)
    
    returns:
        string: 'player', 'ai', or 'draw'
    """
    # if both played the same thing its a draw
    if player == ai:
        return "draw"

    # dictionary of all winning combinations
    # format: (player_move, ai_move): who wins
    rules = {
        ('Rock', 'Scissors'): 'player',  # rock beats scissors
        ('Paper', 'Rock'): 'player',     # paper beats rock
        ('Scissors', 'Paper'): 'player', # scissors beats paper
        ('Scissors', 'Rock'): 'ai',      # rock beats scissors
        ('Rock', 'Paper'): 'ai',         # paper beats rock
        ('Paper', 'Scissors'): 'ai'      # scissors beats paper
    }

    # look up who wins and return result
    return rules.get((player, ai), 'draw')

# ============================================
# IMAGE PREPROCESSING
# ============================================

def preprocess_frame(frame, use_preprocess_input=True):
    """
    preprocesses webcam roi for model inference
    
    this follows the exact same preprocessing i used during training
    steps:
    1. crop to roi (hand area only)
    2. resize to 224x224 (model input size)
    3. convert to numpy array
    4. add batch dimension (model expects batches)
    5. apply appropriate preprocessing (preprocess_input or rescale)
    
    args:
        frame: raw webcam frame
        use_preprocess_input: if True, use MobileNetV2 preprocessing
                             if False, use simple rescaling for CNN
    
    returns:
        preprocessed image ready for model prediction
    """
    # crop roi - only the hand area inside the green box
    roi = frame[ROI_Y1:ROI_Y2, ROI_X1:ROI_X2]
    
    # resize to 224x224 which is what all models expect
    img = cv2.resize(roi, (224, 224))
    
    # convert opencv image to numpy array
    img = img_to_array(img)
    
    # add batch dimension: (224, 224, 3) becomes (1, 224, 224, 3)
    img = np.expand_dims(img, axis=0)
    
    # apply appropriate preprocessing based on model type
    if use_preprocess_input:
        # MobileNetV2 specific preprocessing
        img = preprocess_input(img)
    else:
        # Simple rescaling for CNN
        img = img / 255.0
    
    return img

def predict_gesture(frame, model, use_preprocess_input):
    """
    predicts hand gesture from webcam frame
    
    takes the current webcam frame and returns what gesture
    the model thinks it is along with how confident it is
    
    args:
        frame: current webcam image
        model: the loaded model to use for prediction
        use_preprocess_input: whether to use MobileNetV2 preprocessing
    
    returns:
        tuple: (predicted_class_name, confidence_score)
    """
    # preprocess the frame using appropriate preprocessing
    processed = preprocess_frame(frame, use_preprocess_input)
    
    # get model prediction
    probs = model.predict(processed, verbose=0)[0]
    
    # find which class has highest probability
    idx = np.argmax(probs)
    
    # return the class name and its confidence score
    return CLASS_NAMES[idx], probs[idx]

# ============================================
# SMART AI OPPONENT
# ============================================

class SmartAI:
    """
    intelligent ai opponent that learns from player patterns
    
    this is the innovation part of my project
    instead of playing random moves the ai:
    - tracks every move the player makes
    - analyzes patterns in recent history
    - predicts what the player will do next
    - plays the counter-move to gain advantage
    
    this demonstrates pattern learning beyond just image classification
    the ai gets better at beating you the more rounds you play
    """

    def __init__(self):
        """initialize with empty move history"""
        self.player_history = []

    def record(self, move):
        """
        stores player move for learning
        
        called after each round to remember what the player did
        this builds up the history that the ai uses to find patterns
        
        args:
            move: what the player just played (rock/paper/scissors)
        """
        self.player_history.append(move)

    def predict_player(self):
        """
        predicts what the player will play next based on their history
        
        this is where the pattern learning happens
        strategy:
        - if less than 3 moves: not enough data so play random
        - otherwise: look at last 10 moves and find most common
        - assumes player will repeat their favorite move
        
        returns:
            string: predicted player move (rock/paper/scissors)
        """
        # needs at least 3 moves to start detecting patterns
        if len(self.player_history) < 3:
            return random.choice(GAME_GESTURES)

        # looks at recent moves (last 10)
        recent = self.player_history[-10:]
        
        # counts how often each gesture appears
        return Counter(recent).most_common(1)[0][0]

    def make_move(self):
        """
        selects ai move that beats predicted player move
        
        this is where the ai becomes smart
        instead of playing random it:
        1. predicts what you'll play based on your history
        2. looks up what beats that prediction
        3. plays the counter-move
        
        returns:
            tuple: (ai_move, predicted_player_move)
        """
        # predict what the player will do
        predicted = self.predict_player()
        
        # dictionary of what beats what
        counters = {
            'Rock': 'Paper',
            'Paper': 'Scissors',
            'Scissors': 'Rock'
        }
        
        # return the counter-move and what we predicted
        return counters[predicted], predicted

# ============================================
# MAIN GAME LOOP
# ============================================

def play_game(model, model_name, use_preprocess_input):
    """
    runs the real-time rock-paper-scissors game using webcam input
    
    game flow:
    1. open webcam and show live feed
    2. player positions hand in green box
    3. press space to capture and play round
    4. model predicts gesture
    5. smart ai makes strategic move
    6. determine winner and update scores
    7. repeat until q pressed
    
    the ai learns from every round so it gets harder to beat
    """
    print("\n" + "=" * 60)
    print(f"ROCK-PAPER-SCISSORS - {model_name}")
    print("=" * 60)
    print("Controls:")
    print(" SPACE - capture gesture and play round")
    print(" Q     - quit game")
    print("=" * 60)

    # open webcam
    cap = cv2.VideoCapture(0)
    
    # checks if the webcam opened successfully
    if not cap.isOpened():
        print("ERROR: Webcam not accessible")
        print("Make sure no other program is using the camera")
        return

    # create smart ai opponent
    smart_ai = SmartAI()
    
    # initialise all game scores to zero
    player_score = ai_score = draws = rounds = 0

    # main game loop - runs continuously until q pressed
    while True:
        # read one frame from webcam
        ret, frame = cap.read()
        
        if not ret:
            break

        # create copy of frame to draw on
        display = frame.copy()

        # draw roi guide box where player should place hand
        cv2.rectangle(display, (ROI_X1, ROI_Y1), (ROI_X2, ROI_Y2),
                      (0, 255, 0), 2)
        
        # add instruction text above the box
        cv2.putText(display, "Place hand inside box",
                    (ROI_X1, ROI_Y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)

        # draw model name at top
        cv2.putText(display, f"Model: {model_name}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 0), 2)

        # draw scoreboard
        cv2.putText(display,
                    f"Rounds:{rounds}  You:{player_score}  AI:{ai_score}  Draws:{draws}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)

        # add controls reminder at bottom
        cv2.putText(display,
                    "SPACE=Play | Q=Quit",
                    (10, 460),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

        # show the frame
        cv2.imshow("RPS Game", display)
        
        # wait for key press
        key = cv2.waitKey(1) & 0xFF

        # if q pressed exit game
        if key == ord('q'):
            break

        # if space pressed start a new round
        if key == ord(' '):
            rounds += 1

            # get prediction from model
            player_move, conf = predict_gesture(frame, model, use_preprocess_input)
            
            # print what the model predicted
            print(f"\nRound {rounds}")
            print(f"Player: {player_move} ({conf*100:.1f}% confidence)")

            # check if prediction is valid game gesture
            if player_move not in GAME_GESTURES:
                print("Invalid gesture detected - retry")
                print("Make sure your hand is clearly visible in the box")
                rounds -= 1
                continue

            # ai makes its move using smart strategy
            ai_move, predicted = smart_ai.make_move()
            
            # print what the ai was thinking
            print(f"Smart AI predicted: {predicted}")
            print(f"AI plays: {ai_move}")

            # determine who won this round
            result = determine_winner(player_move, ai_move)

            # update scores based on who won
            if result == 'player':
                print("YOU WIN!")
                player_score += 1
            elif result == 'ai':
                print("AI WINS!")
                ai_score += 1
            else:
                print("DRAW!")
                draws += 1

            # record the players move so ai learns
            smart_ai.record(player_move)
            
            # pause so the player can see the result
            time.sleep(1.5)

    # cleanup when game ends
    cap.release()
    cv2.destroyAllWindows()

    # print final game statistics
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Rounds: {rounds}")
    print(f"Player wins: {player_score}")
    print(f"AI wins: {ai_score}")
    print(f"Draws: {draws}")
    print("=" * 60)
    
    # calculate win percentages if any rounds were played
    if rounds > 0:
        player_pct = (player_score / rounds) * 100
        ai_pct = (ai_score / rounds) * 100
        print(f"\nPlayer win rate: {player_pct:.1f}%")
        print(f"AI win rate: {ai_pct:.1f}%")

# ============================================
# RUN THE GAME
# ============================================

if __name__ == "__main__":
    # let user select which model to use
    model, model_name, use_preprocess = select_model()
    
    # run the game with selected model
    play_game(model, model_name, use_preprocess)